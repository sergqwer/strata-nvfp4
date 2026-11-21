// src/kernels/cuda/kv_stream.cu - see include/strata/kernels/kv_stream.hpp.
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"

#include "strata/core/dispatch_stats.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_stream: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int RT = 1024;   // the resolve block

// The per-block byte runs of the (up to four) pool arrays: block b of array i is bytes [b * len, (b + 1) * len).
struct Runs {
    const uint8_t* src[4];
    uint8_t* dst[4];
    int len[4];
    int n;
};

Runs runs_of(const QsaAttnPools& slots, const KvHostPools& host, int fmt, const QsaShapes& s) {
    const int rows = (int) (s.n_head_kv * s.page_size);
    Runs r{};
    if (fmt == kKvHybrid) {   // K8V4: int8 K codes, their scales, rotated q4_0 V
        const int codes = rows * (int) s.head_dim, scales = rows * (int) (s.head_dim / KV_Q8_GROUP) * 2;
        const int v = rows * (int) kv_q4_bytes_per_head((int) s.head_dim);
        r.src[0] = (const uint8_t*) host.k_q;     r.dst[0] = (uint8_t*) slots.k_q;     r.len[0] = codes;
        r.src[1] = (const uint8_t*) host.k_scale; r.dst[1] = (uint8_t*) slots.k_scale; r.len[1] = scales;
        r.src[2] = (const uint8_t*) host.v_q4;    r.dst[2] = (uint8_t*) slots.v_q4;    r.len[2] = v;
        r.n = 3;
    } else if (fmt == kKvQ4) {
        const int bytes = rows * (int) kv_q4_bytes_per_head((int) s.head_dim);
        r.src[0] = (const uint8_t*) host.k_q4; r.dst[0] = (uint8_t*) slots.k_q4; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_q4; r.dst[1] = (uint8_t*) slots.v_q4; r.len[1] = bytes;
        r.n = 2;
    } else if (fmt == kKvInt8) {
        const int codes = rows * (int) s.head_dim, scales = rows * (int) (s.head_dim / KV_Q8_GROUP) * 2;
        r.src[0] = (const uint8_t*) host.k_q;     r.dst[0] = (uint8_t*) slots.k_q;     r.len[0] = codes;
        r.src[1] = (const uint8_t*) host.v_q;     r.dst[1] = (uint8_t*) slots.v_q;     r.len[1] = codes;
        r.src[2] = (const uint8_t*) host.k_scale; r.dst[2] = (uint8_t*) slots.k_scale; r.len[2] = scales;
        r.src[3] = (const uint8_t*) host.v_scale; r.dst[3] = (uint8_t*) slots.v_scale; r.len[3] = scales;
        r.n = 4;
    } else {
        const int bytes = rows * (int) s.head_dim * 2;
        r.src[0] = (const uint8_t*) host.k_pool; r.dst[0] = (uint8_t*) slots.k_pool; r.len[0] = bytes;
        r.src[1] = (const uint8_t*) host.v_pool; r.dst[1] = (uint8_t*) slots.v_pool; r.len[1] = bytes;
        r.n = 2;
    }
    return r;
}

// Block-wide exclusive prefix sum of one int per thread (RT threads); `total` is the sum over the block.
__device__ int block_scan(int v, int* warp_sums, int& total) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_sums[w] = x;
    __syncthreads();
    if (w == 0) {
        int t = warp_sums[lane];
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, t, o);
            if (lane >= o) t += y;
        }
        warp_sums[lane] = t;
    }
    __syncthreads();
    total = warp_sums[31];
    const int excl = x - v + (w > 0 ? warp_sums[w - 1] : 0);
    __syncthreads();
    return excl;
}

// dec3 D5 STRATA_KV_RESOLVE_ILP (default on, =0 the old loop): step 1 for RQ queries at a time with every id load,
// then every page-table load, then the stamps / claims - not one dependent ids -> table round trip per (query, pass).
// The same lookups, stamps and claims (a claim is an atomicCAS either way; the miss list order was never fixed).
constexpr int RK = 2;   // passes per query in a round (ids past RK * RT: the plain loop after it)
// r5 F5 STRATA_KV_RESOLVE_PROF=1 (diagnostic, default off): %globaltimer per phase into ctl (kv_stream_counters).
// u64 view c = ctl + 4: [3] lookups ns [4] sweep ns [5] table ns [6] sweep steps [7] calls with misses [8] copy ns
// [9] resolve end -> copy start ns [10] resolve end (scratch) [11] copy start (scratch) [12] fresh misses; ctl[30] the
// copy's CTA count.
__device__ __forceinline__ unsigned long long kv_gtime() {
#if defined(__HIPCC__)
    return 0ull;
#else
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
#endif
}
// r5 F5 STRATA_KV_RESIDENT_FAST (default on, =0 the full walk every call): ctl[kKvCtlResident] = W says blocks
// [0, W) are resident with current bytes (kv_stream_reset_resident maps a new sequence's first W blocks to slots
// 0..W-1, so every append writes them in VRAM as well; only a clock eviction of a block below W lowers it).  The ids
// of each query are ascending, so its last id names its largest block: when every query's last block is below W the
// whole selection is resident and the call ends there - no lookups, stamps, sweep or copy.  The attention reads the
// same slots either way; the stamps and reference bits it skips only steer which slot a later eviction takes.
template <int RQ, bool PROF, bool FAST>   // RQ queries per round (4 or 8); 0 = the plain loop
__global__ void __launch_bounds__(RT) resolve_kernel(KvStreamMap m, const int32_t* __restrict__ ids,
                                                     const int32_t* __restrict__ steps, int n_q, int cap, int page_size) {
    __shared__ int s_nmiss, s_lookups, s_cut, s_fresh;
    __shared__ int warp_sums[32];
    unsigned long long t0 = 0, t1 = 0, t2 = 0;
    if (PROF && threadIdx.x == 0) t0 = kv_gtime();
    const int epoch = m.ctl[0] + 1;
    const int resident = FAST ? m.ctl[kKvCtlResident] : 0;
    if (FAST && resident > 0) {
        bool in = true;
        for (int q = threadIdx.x; q < n_q; q += RT) {
            const int w = steps[q * kStepCount + kStepWidth];
            if (w > 0 && ids[(long long) q * cap + w - 1] / page_size >= resident) in = false;
        }
        if (__syncthreads_and(in)) {
            if (threadIdx.x == 0) {
                m.ctl[0] = epoch;
                m.ctl[2] = 0;
                unsigned long long* c = reinterpret_cast<unsigned long long*>(m.ctl + 4);
                c[2] += 1ull;
                c[14] += 1ull;
                if (PROF) {
                    const unsigned long long t3 = kv_gtime();
                    c[3] += t3 - t0;
                    c[10] = t3;
                    c[11] = ~0ull;
                    m.ctl[30] = 0;
                }
            }
            return;
        }
    }
    if (threadIdx.x == 0) { s_nmiss = 0; s_lookups = 0; s_fresh = 0; }
    __syncthreads();
    // 1. hits take this epoch and their reference bit; a missing block is claimed exactly once (-1 -> -2)
    int lookups = 0;
    auto look = [&](const int32_t* qi, int i) {   // one id of step 1 (the plain loop's body)
        const int b = qi[i] / page_size;
        if (i > 0 && qi[i - 1] / page_size == b) return;   // ids are ascending: one lookup per block
        ++lookups;
        const int sl = m.page_table[b];
        if (sl >= 0) {
            m.slot_stamp[sl] = epoch;
            m.slot_ref[sl] = 1;
        } else if (sl == -1 && atomicCAS(&m.page_table[b], -1, -2) == -1) {
            m.miss_block[atomicAdd(&s_nmiss, 1)] = b;
        }
    };
    if (RQ > 0) for (int q0 = 0; q0 < n_q; q0 += (RQ > 0 ? RQ : 1)) {
        constexpr int Q = RQ > 0 ? RQ : 1;
        int width[Q], cur[Q][RK], prv[Q][RK], bb[Q][RK], sl[Q][RK];
        bool lk[Q][RK];
#pragma unroll
        for (int a = 0; a < Q; ++a) width[a] = 0;
#pragma unroll
        for (int a = 0; a < Q; ++a)
            if (q0 + a < n_q) width[a] = steps[(q0 + a) * kStepCount + kStepWidth];
#pragma unroll
        for (int a = 0; a < Q; ++a) {   // the loads only, no use in between: all in flight at once
            const int32_t* qi = ids + (long long) (q0 + a) * cap;
#pragma unroll
            for (int k = 0; k < RK; ++k) {
                const int i = threadIdx.x + k * RT;
                cur[a][k] = 0;
                prv[a][k] = 0;
                if (i < width[a]) cur[a][k] = qi[i];
                if (i < width[a] && i > 0) prv[a][k] = qi[i - 1];
            }
        }
#pragma unroll
        for (int a = 0; a < Q; ++a)
#pragma unroll
            for (int k = 0; k < RK; ++k) {
                const int i = threadIdx.x + k * RT;
                const int b = cur[a][k] / page_size;
                // ids are ascending: one lookup per block
                bb[a][k] = b;
                lk[a][k] = i < width[a] && !(i > 0 && prv[a][k] / page_size == b);
            }
#pragma unroll
        for (int a = 0; a < Q; ++a)
#pragma unroll
            for (int k = 0; k < RK; ++k) sl[a][k] = lk[a][k] ? m.page_table[bb[a][k]] : 0;
#pragma unroll
        for (int a = 0; a < Q; ++a)
#pragma unroll
            for (int k = 0; k < RK; ++k) {
                const int b = bb[a][k];
                if (!lk[a][k]) continue;
                ++lookups;
                if (sl[a][k] >= 0) {
                    m.slot_stamp[sl[a][k]] = epoch;
                    m.slot_ref[sl[a][k]] = 1;
                } else if (sl[a][k] == -1 && atomicCAS(&m.page_table[b], -1, -2) == -1) {
                    m.miss_block[atomicAdd(&s_nmiss, 1)] = b;
                }
            }
    }
    // ILP: the ids past the round's RK passes (a long context: threads 0..2 at width 2,051); else every id
    for (int q = 0; q < n_q; ++q) {
        const int width = steps[q * kStepCount + kStepWidth];
        const int32_t* qi = ids + (long long) q * cap;
        for (int i = threadIdx.x + (RQ > 0 ? RK * RT : 0); i < width; i += RT) look(qi, i);
    }
    if (lookups > 0) atomicAdd(&s_lookups, lookups);   // (#783, stuchapin909) most threads see none: skip the shared atomic
    __syncthreads();
    if (PROF && threadIdx.x == 0) t1 = kv_gtime();
    int sweeps = 0;
    // 2. one victim per miss: a clock sweep from the hand. A slot this call uses (stamp == epoch) is never taken;
    //    a referenced one loses its bit as the hand passes it and is taken on the next pass.
    const int need = s_nmiss, n = (int) m.n_slots;
    int hand = m.ctl[1], got = 0;
    for (int scanned = 0; got < need && scanned < 3 * n; scanned += RT) {
        const int j = (int) (((long long) hand + threadIdx.x) % n);
        const bool mine = m.slot_stamp[j] == epoch;
        const bool cand = !mine && (m.slot_block[j] < 0 || m.slot_ref[j] == 0);
        int total = 0;
        const int rank = block_scan(cand ? 1 : 0, warp_sums, total);
        const int want = need - got;
        if (threadIdx.x == 0) s_cut = RT;
        __syncthreads();
        if (cand && rank == want - 1) s_cut = threadIdx.x + 1;   // the hand stops just past the last slot taken
        __syncthreads();
        const int cut = s_cut;
        if (cand && rank < want) {
            m.miss_slot[got + rank] = j;
            m.slot_stamp[j] = epoch;   // taken: a sweep that wraps around must not take it twice
        } else if (threadIdx.x < cut && !mine) {
            m.slot_ref[j] = 0;
        }
        got += total < want ? total : want;
        hand = (int) (((long long) hand + cut) % n);
        ++sweeps;
        __syncthreads();
    }
    if (PROF && threadIdx.x == 0) t2 = kv_gtime();
    // (PROF: misses of blocks holding the last 16 positions before the window's first query - just appended)
    const int fresh_from = PROF ? (steps[kStepPos] - 16) / page_size : 0;
    // 3. re-point the table; the copy kernel fills the slots
    const int placed = got < need ? got : need;
    for (int k = threadIdx.x; k < need; k += RT) {
        const int b = m.miss_block[k];
        if (k >= placed) { m.page_table[b] = -1; continue; }   // overflow: never happens with a legal n_slots
        const int sl = m.miss_slot[k];
        const int old = m.slot_block[sl];
        if (old >= 0) m.page_table[old] = -1;
        if (FAST && old >= 0 && old < resident) atomicMin(&m.ctl[kKvCtlResident], old);   // [0, W) no longer resident
        m.slot_block[sl] = b;
        m.slot_stamp[sl] = epoch;
        m.slot_ref[sl] = 1;
        m.page_table[b] = sl;
        if (PROF && b >= fresh_from) atomicAdd(&s_fresh, 1);
    }
    if (PROF) __syncthreads();
    if (threadIdx.x == 0) {
        m.ctl[0] = epoch;
        m.ctl[1] = hand;
        m.ctl[2] = placed;
        if (placed < need) m.ctl[3] = 1;
        unsigned long long* c = reinterpret_cast<unsigned long long*>(m.ctl + 4);
        c[0] += (unsigned long long) placed;
        c[1] += (unsigned long long) s_lookups;
        c[2] += 1ull;
        if (PROF) {
            const unsigned long long t3 = kv_gtime();
            c[3] += t1 - t0;
            c[4] += t2 - t1;
            c[5] += t3 - t2;
            c[6] += (unsigned long long) sweeps;
            c[7] += need > 0 ? 1ull : 0ull;
            c[10] = t3;
            c[11] = ~0ull;
            c[12] += (unsigned long long) s_fresh;
            m.ctl[30] = 0;
        }
    }
}

// PROF: the copy's first CTA start -> last CTA end, and the resolve's end -> that start (c[8], c[9])
template <bool PROF>
__device__ __forceinline__ void copy_prof_begin(const KvStreamMap& m) {
    if (PROF && threadIdx.x == 0) atomicMin(reinterpret_cast<unsigned long long*>(m.ctl + 4) + 11, kv_gtime());
}
template <bool PROF>
__device__ __forceinline__ void copy_prof_end(const KvStreamMap& m) {
    if (!PROF) return;
    __syncthreads();
    if (threadIdx.x != 0) return;
    __threadfence();
    if (atomicAdd(reinterpret_cast<unsigned int*>(m.ctl + 30), 1u) != gridDim.x - 1) return;
    __threadfence();
    unsigned long long* c = reinterpret_cast<unsigned long long*>(m.ctl + 4);
    const unsigned long long e = kv_gtime(), s0 = atomicAdd(c + 11, 0ull);
    c[8] += e - s0;
    c[9] += s0 - c[10];
}

// One block per missed block (grid-stride): copy its runs from the host copy into its slot, 16 B per thread.
template <bool PROF>
__global__ void copy_kernel(KvStreamMap m, Runs r) {
    copy_prof_begin<PROF>(m);
    const int need = m.ctl[2];
    for (int k = blockIdx.x; k < need; k += gridDim.x) {
        const long long b = m.miss_block[k], sl = m.miss_slot[k];
        for (int a = 0; a < r.n; ++a) {
            const uint4* src = reinterpret_cast<const uint4*>(r.src[a] + b * r.len[a]);
            uint4* dst = reinterpret_cast<uint4*>(r.dst[a] + sl * r.len[a]);
            for (int i = threadIdx.x; i < r.len[a] / 16; i += blockDim.x) dst[i] = src[i];
        }
    }
    copy_prof_end<PROF>(m);
}

// r5 F5: a new sequence's map with blocks [0, w) resident in slots 0..w-1 (nothing is written yet, so every slot is
// current: appends write a resident block's slot); the rest as reset_kernel leaves it
__global__ void reset_resident_kernel(KvStreamMap m, int w) {
    const long long i0 = (long long) blockIdx.x * blockDim.x + threadIdx.x, st = (long long) gridDim.x * blockDim.x;
    for (long long i = i0; i < m.n_blocks; i += st) m.page_table[i] = i < w ? (int32_t) i : -1;
    for (long long i = i0; i < m.n_slots; i += st) {
        m.slot_block[i] = i < w ? (int32_t) i : -1;
        m.slot_stamp[i] = -1;
        m.slot_ref[i] = 0;
    }
    if (i0 < kKvCtlInts) m.ctl[i0] = i0 == kKvCtlResident ? w : 0;
}

__global__ void reset_kernel(KvStreamMap m) {
    const long long i0 = (long long) blockIdx.x * blockDim.x + threadIdx.x, st = (long long) gridDim.x * blockDim.x;
    for (long long i = i0; i < m.n_blocks; i += st) m.page_table[i] = -1;
    for (long long i = i0; i < m.n_slots; i += st) {
        m.slot_block[i] = -1;
        m.slot_stamp[i] = -1;
        m.slot_ref[i] = 0;
    }
    if (i0 < kKvCtlInts) m.ctl[i0] = 0;
}

// r5 F5 test hook (qsa_front_bench): the listed blocks evicted, their slots free
__global__ void test_evict_kernel(KvStreamMap m, const int32_t* blocks, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const int b = blocks[i], sl = m.page_table[b];
        if (sl < 0) continue;
        m.page_table[b] = -1;
        m.slot_block[sl] = -1;
        m.slot_ref[sl] = 0;
    }
}

__global__ void ring_kernel(int32_t* table, long long n_blocks, long long n_slots) {
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n_blocks;
         i += (long long) gridDim.x * blockDim.x)
        table[i] = (int32_t) (i % n_slots);
}

int g_resolve_ilp = -1;   // kv_stream_set_resolve_ilp: -1 = STRATA_KV_RESOLVE_ILP (default on)
bool resolve_ilp_on() {
    if (g_resolve_ilp >= 0) return g_resolve_ilp != 0;
    static const bool on = [] { const char* v = std::getenv("STRATA_KV_RESOLVE_ILP"); return v == nullptr || std::atoi(v) != 0; }();
    return on;
}

int env_int(const char* name, int dflt) {   // (an empty value = unset)
    const char* v = std::getenv(name);
    return v == nullptr || *v == 0 ? dflt : std::atoi(v);
}
bool resolve_prof_on() {
    static const bool on = env_int("STRATA_KV_RESOLVE_PROF", 0) != 0;
    return on;
}

template <int RQ>
void launch_resolve(bool prof, const KvStreamMap& m, const int32_t* ids, const int32_t* steps, int n_q, int cap,
                    int page_size, cudaStream_t st) {
    const bool fast = kv_stream_resident_fast();
    if (fast) strata::dispatch::note("kv stream: resolve, the resident-prefix fast path (STRATA_KV_RESIDENT_FAST)");
    if (prof && fast) resolve_kernel<RQ, true, true><<<1, RT, 0, st>>>(m, ids, steps, n_q, cap, page_size);
    else if (prof) resolve_kernel<RQ, true, false><<<1, RT, 0, st>>>(m, ids, steps, n_q, cap, page_size);
    else if (fast) resolve_kernel<RQ, false, true><<<1, RT, 0, st>>>(m, ids, steps, n_q, cap, page_size);
    else resolve_kernel<RQ, false, false><<<1, RT, 0, st>>>(m, ids, steps, n_q, cap, page_size);
}

}  // namespace

int g_resident_fast = -1;   // kv_stream_set_resident_fast: -1 = STRATA_KV_RESIDENT_FAST (default on)
bool kv_stream_resident_fast() {
    if (g_resident_fast >= 0) return g_resident_fast != 0;
    static const bool on = env_int("STRATA_KV_RESIDENT_FAST", 1) != 0;
    return on;
}
void kv_stream_set_resident_fast(int on) { g_resident_fast = on; }

void kv_stream_set_resolve_ilp(int on) { g_resolve_ilp = on; }

uint64_t kv_block_bytes(const QsaShapes& s, int fmt) {
    const uint64_t rows = (uint64_t) (s.n_head_kv * s.page_size);
    if (fmt == kKvHybrid)
        return rows * (uint64_t) s.head_dim + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 +
               rows * kv_q4_bytes_per_head((int) s.head_dim);
    if (fmt == kKvQ4) return rows * kv_q4_bytes_per_head((int) s.head_dim) * 2;
    return fmt == kKvInt8 ? rows * (uint64_t) s.head_dim * 2 + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 * 2
                : rows * (uint64_t) s.head_dim * 2 * 2;
}

void kv_stream_reset(const KvStreamMap& m, void* stream) {
    reset_kernel<<<128, 256, 0, (cudaStream_t) stream>>>(m);
    check("reset");
}

int64_t kv_stream_reset_resident(const KvStreamMap& m, int64_t blocks, void* stream) {
    if (!kv_stream_resident_fast()) {
        kv_stream_reset(m, stream);
        return 0;
    }
    const int64_t w = std::min<int64_t>(std::min<int64_t>(blocks, m.n_slots), m.n_blocks);
    reset_resident_kernel<<<128, 256, 0, (cudaStream_t) stream>>>(m, (int) w);
    check("reset resident");
    return w;
}

void kv_stream_resolve(const KvStreamMap& m, const QsaAttnPools& slots, const KvHostPools& host, int fmt,
                       const int32_t* ids, const int32_t* steps, int64_t n_q, int64_t cap, const QsaShapes& s,
                       void* stream) {
    if (n_q <= 0) return;
    if (s.n_head_kv * s.page_size * (s.head_dim / KV_Q8_GROUP) * 2 % 16 != 0) {
        std::fprintf(stderr, "kv_stream: a block's scale run must be a multiple of 16 bytes\n");
        std::exit(1);
    }
    // one sweep step looks at RT consecutive slots `(hand + thread) % n_slots`; with fewer slots than RT two
    // threads see the same slot and may both take it for two different misses.  The engine never streams with fewer
    // than qsa_kv_resident_min() / page_size = 5,120 slots, so this is a guard, not a limit.
    if (m.n_slots < RT) {
        std::fprintf(stderr, "kv_stream: %lld slots is fewer than the resolve block (%d): the clock sweep would take a "
                             "slot twice\n", (long long) m.n_slots, RT);
        std::exit(1);
    }
    // dec3 D5: the batched lookups from 3 queries (bench: T 2 is 0.3 us slower with them), 4 a round up to 4, else 8
    const cudaStream_t st = (cudaStream_t) stream;
    const bool prof = resolve_prof_on();
    const int nq = (int) n_q, cp = (int) cap, ps = (int) s.page_size;
    if (prof) strata::dispatch::note("kv stream: resolve phase timers (STRATA_KV_RESOLVE_PROF)");
    if (resolve_ilp_on() && n_q >= 3 && n_q <= 4) {
        strata::dispatch::note("kv stream: resolve, batched lookups 4 (STRATA_KV_RESOLVE_ILP)");
        launch_resolve<4>(prof, m, ids, steps, nq, cp, ps, st);
    } else if (resolve_ilp_on() && n_q >= 5) {
        strata::dispatch::note("kv stream: resolve, batched lookups 8 (STRATA_KV_RESOLVE_ILP)");
        launch_resolve<8>(prof, m, ids, steps, nq, cp, ps, st);
    } else
        launch_resolve<0>(prof, m, ids, steps, nq, cp, ps, st);
    check("resolve");
    const Runs r = runs_of(slots, host, fmt, s);
    if (prof) copy_kernel<true><<<96, 128, 0, st>>>(m, r);
    else copy_kernel<false><<<96, 128, 0, st>>>(m, r);
    check("copy");
}

void kv_stream_test_evict(const KvStreamMap& m, const int32_t* blocks, int n, void* stream) {
    if (n <= 0) return;
    test_evict_kernel<<<(n + 127) / 128, 128, 0, (cudaStream_t) stream>>>(m, blocks, n);
    check("test evict");
}

void kv_ring_table(int32_t* page_table, int64_t n_blocks, int64_t n_slots, void* stream) {
    ring_kernel<<<64, 256, 0, (cudaStream_t) stream>>>(page_table, n_blocks, n_slots);
    check("ring table");
}

void kv_ring_restore(const QsaAttnPools& slots, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                     int64_t n_slots, const QsaShapes& s, void* stream) {
    const Runs r = runs_of(slots, host, fmt, s);
    for (int64_t b = b0; b < b1;) {
        const int64_t sl = b % n_slots, run = std::min<int64_t>(b1 - b, n_slots - sl);   // up to the ring's end
        for (int a = 0; a < r.n; ++a)
            if (cudaMemcpyAsync(r.dst[a] + sl * r.len[a], r.src[a] + b * r.len[a], (size_t) (run * r.len[a]),
                                cudaMemcpyDefault, (cudaStream_t) stream) != cudaSuccess)
                check("ring restore");
        b += run;
    }
}

void kv_stage_from_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t n_blocks,
                        const QsaShapes& s, void* stream) {
    if (n_blocks <= 0) return;
    const Runs r = runs_of(stage, host, fmt, s);
    for (int a = 0; a < r.n; ++a)
        if (cudaMemcpyAsync(r.dst[a], r.src[a], (size_t) (n_blocks * r.len[a]), cudaMemcpyDefault,
                            (cudaStream_t) stream) != cudaSuccess)
            check("stage");
}

void kv_unstage_to_host(const QsaAttnPools& stage, const KvHostPools& host, int fmt, int64_t b0, int64_t b1,
                        const QsaShapes& s, void* stream) {
    if (b1 <= b0) return;
    const Runs r = runs_of(stage, host, fmt, s);   // src: the host copy, dst: the staging pool (identity layout both)
    for (int a = 0; a < r.n; ++a)
        if (cudaMemcpyAsync((void*) (r.src[a] + b0 * r.len[a]), r.dst[a] + b0 * r.len[a], (size_t) ((b1 - b0) * r.len[a]),
                            cudaMemcpyDefault, (cudaStream_t) stream) != cudaSuccess)
            check("unstage");
}

KvStreamCounters kv_stream_counters(const KvStreamMap& m) {
    int32_t c[kKvCtlInts] = {};
    KvStreamCounters r;
    if (m.ctl == nullptr || cudaMemcpy(c, m.ctl, sizeof(c), cudaMemcpyDeviceToHost) != cudaSuccess) return r;
    const unsigned long long* u = reinterpret_cast<const unsigned long long*>(c + 4);
    r.misses = u[0];
    r.lookups = u[1];
    r.calls = u[2];
    r.overflow = c[3] != 0;
    r.lookup_ns = u[3]; r.sweep_ns = u[4]; r.table_ns = u[5]; r.sweep_steps = u[6]; r.miss_calls = u[7];
    r.copy_ns = u[8]; r.gap_ns = u[9]; r.fresh = u[12];
    r.fast_calls = u[14];
    r.resident = c[kKvCtlResident];
    return r;
}

}  // namespace strata::kernels
