// src/kernels/cuda/qsa_decode_attn.cu - see include/strata/kernels/qsa_decode_attn.hpp.
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/core/sm75_series.hpp"
#include "strata/core/sm120_series.hpp"
#include "strata/core/geometry.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"
#if defined(STRATA_HIP_GFX906)
#include "strata/kernels/gfx_arch.hpp"
#endif

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

constexpr int HD = 256;          // head_dim
constexpr int G = strata::core::kCompiledGeometry.n_head / strata::core::kCompiledGeometry.n_head_kv;   // query heads per KV head (12 for Flash-Next; 1..16, a multiple of 4)
static_assert(G % 4 == 0 && G <= 16, "qsa_decode_attn: query heads per KV head must be 4, 8, 12 or 16 (float4 probability rows, the 16-lane score reduce)");
static_assert(HD == strata::core::kCompiledGeometry.head_dim && G == strata::core::kCompiledGeometry.n_head / strata::core::kCompiledGeometry.n_head_kv, "qsa_decode_attn is compiled for core/geometry.hpp");
constexpr int CHUNK = 64;        // cells per block
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// 8 consecutive values of one cell's key or value row for KV head `kvh`, dimensions [d0, d0+8).
// Per-format bodies; `load8` below is the KV_MODE dispatcher. value=false is the K side, true the V side.
__device__ __forceinline__ void load8_f16(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    const uint16_t* base = (value ? p.v_pool : p.k_pool) + row * HD + d0;
    const uint4 raw = *reinterpret_cast<const uint4*>(base);
    const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __half22float2(h2[j]);
        out[2 * j] = f.x;
        out[2 * j + 1] = f.y;
    }
}
__device__ __forceinline__ void load8_q8(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    const int8_t* codes = (value ? p.v_q : p.k_q) + row * HD + d0;
    const uint16_t sbits = (value ? p.v_scale : p.k_scale)[row * (HD / KV_Q8_GROUP) + d0 / KV_Q8_GROUP];
    const float sc = __half2float(__ushort_as_half(sbits));
    const uint2 raw = *reinterpret_cast<const uint2*>(codes);
    const int8_t* c = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = (float) c[j] * sc;
}
__device__ __forceinline__ void load8_q4(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
    const int b = d0 / QK4_0;
    const int rem = d0 % QK4_0;
    const block_q4_0* blk = reinterpret_cast<const block_q4_0*>((value ? p.v_q4 : p.k_q4) + row * bytes_per_head) + b;
    const float d = __half2float(__ushort_as_half(blk->d));
    const int j = (rem == 0 || rem == 16) ? 0 : 8;
    const uint8_t* bytes = blk->qs + j;
    if (rem < 16) {
#pragma unroll
        for (int k = 0; k < 8; ++k) out[k] = (float) ((int)(bytes[k] & 0x0F) - 8) * d;
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) out[k] = (float) ((int)(bytes[k] >> 4) - 8) * d;
    }
}
template <int KV_MODE>
__device__ __forceinline__ void load8(const QsaAttnPools& p, bool value, long long row, int d0, float* out) {
    if constexpr (KV_MODE == 0) load8_f16(p, value, row, d0, out);
    else if constexpr (KV_MODE == 1) load8_q8(p, value, row, d0, out);
    else if constexpr (KV_MODE == 3) {
        // hybrid K8V4: both sides are defined - K unrotated INT8, V rotated Q4_0 - so a value=true call
        // reads the Q4_0 pool instead of dereferencing the null v_q (no call site does today; PR review)
        if (value) load8_q4(p, value, row, d0, out);
        else load8_q8(p, value, row, d0, out);
    } else load8_q4(p, value, row, d0, out);
}

// One cell's G probabilities (G/4 broadcast float4 loads) into the G accumulators: every accumulator takes one fmaf,
// so the sums are the per-head loop's, bit for bit, for any G.
template <int GG> __device__ __forceinline__ void fold_cell(float (&acc)[GG], const float* w, float v) {
    const float4* w4 = reinterpret_cast<const float4*>(w);
    if constexpr (GG == 12) {   // Flash-Next: the expressions this replaced, so its device code is unchanged
        const float4 w0 = w4[0], w1 = w4[1], w2 = w4[2];
        acc[0] = fmaf(w0.x, v, acc[0]);  acc[1] = fmaf(w0.y, v, acc[1]);
        acc[2] = fmaf(w0.z, v, acc[2]);  acc[3] = fmaf(w0.w, v, acc[3]);
        acc[4] = fmaf(w1.x, v, acc[4]);  acc[5] = fmaf(w1.y, v, acc[5]);
        acc[6] = fmaf(w1.z, v, acc[6]);  acc[7] = fmaf(w1.w, v, acc[7]);
        acc[8] = fmaf(w2.x, v, acc[8]);  acc[9] = fmaf(w2.y, v, acc[9]);
        acc[10] = fmaf(w2.z, v, acc[10]); acc[11] = fmaf(w2.w, v, acc[11]);
        return;
    }
#pragma unroll
    for (int i = 0; i < GG / 4; ++i) {
        const float4 x = w4[i];
        acc[4 * i + 0] = fmaf(x.x, v, acc[4 * i + 0]);
        acc[4 * i + 1] = fmaf(x.y, v, acc[4 * i + 1]);
        acc[4 * i + 2] = fmaf(x.z, v, acc[4 * i + 2]);
        acc[4 * i + 3] = fmaf(x.w, v, acc[4 * i + 3]);
    }
}

// the q8_v3 kernel's value loop: Flash-Next (12 heads per KV head) keeps the expressions this kernel always had (its device code is
// unchanged); any other G folds the cell with fold_cell.  A template, so the branch not taken is not instantiated (G is not 12 in a
// non-Flash-Next build, and the 12-element initialiser below would not compile).
template <int GG> __device__ __forceinline__ void fold_cell_v3(float (&acc)[GG], const float* w, float v) {
    if constexpr (GG == 12) {
        const float4 p0 = *reinterpret_cast<const float4*>(&w[0]);
        const float4 p1 = *reinterpret_cast<const float4*>(&w[4]);
        const float4 p2 = *reinterpret_cast<const float4*>(&w[8]);
        const float pr[GG] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w, p2.x, p2.y, p2.z, p2.w};
#pragma unroll
        for (int h = 0; h < GG; ++h) acc[h] = fmaf(pr[h], v, acc[h]);
    } else {
        fold_cell<GG>(acc, w, v);
    }
}


// Exact-intent score reduce-scatter: the same butterfly pairs as twelve warp_sum calls,
// with 16 shuffle exchanges instead of 60. Used only by the opt-in gfx906 INT8
// query-swizzled specialization; softmax, value accumulation and merge are unchanged.
__device__ __forceinline__ float reduce12_gfx906(const float (&part)[16], int lane) {
    float r8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool hi = (lane & 16) != 0;
        r8[i] = (hi ? part[i + 8] : part[i]) + __shfl_xor_sync(0xffffffffu, hi ? part[i] : part[i + 8], 16);
    }
    float r4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool hi = (lane & 8) != 0;
        r4[i] = (hi ? r8[i + 4] : r8[i]) + __shfl_xor_sync(0xffffffffu, hi ? r8[i] : r8[i + 4], 8);
    }
    float r2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool hi = (lane & 4) != 0;
        r2[i] = (hi ? r4[i + 2] : r4[i]) + __shfl_xor_sync(0xffffffffu, hi ? r4[i] : r4[i + 2], 4);
    }
    const bool hi2 = (lane & 2) != 0;
    const float r1 = (hi2 ? r2[1] : r2[0]) + __shfl_xor_sync(0xffffffffu, hi2 ? r2[0] : r2[1], 2);
    return r1 + __shfl_xor_sync(0xffffffffu, r1, 1);
}
__device__ __forceinline__ int reduce12_head_of_lane(int lane) {
    return ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
}

// The chunk kernel's body; `p` is the query's KV pools (the call's one set, or blockIdx.z's slot's set).
// (the body is a device function so the batch window's rows kernel below can run it over each query's own pools without
// the default kernel's signature, or the function-pointer types the pre-sm_75 / HIP selections build, changing)
template <int KV_MODE, bool LANE_CELL, bool QUERY_SWIZZLE, bool REDUCE12>
__device__ __forceinline__ void attn_chunk_body(const float* __restrict__ q, QsaAttnPools p,
                                                const int32_t* __restrict__ ids, const int32_t* __restrict__ step,
                                                int n_kv_heads, int page_size, float scale,
                                                float* __restrict__ part_acc, float* __restrict__ part_m,
                                                float* __restrict__ part_l, int n_chunks, int cap,
                                                long long scratch_stride) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[G][HD];     // 12 KB: this KV head's query heads
    __shared__ float sp[G][CHUNK];                // scores, then probabilities
    __shared__ long long srow[CHUNK];             // pool row of each cell (page, kv head, slot)
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    // Opt-in gfx906 batched attention: a bijective XOR of query dimensions in LDS only. It swaps the two
    // float4 halves of lanes whose bit 2 is set, retaining 16-byte alignment and the 12 KB tile.
    // The scoring reads undo this permutation; FP operands and operation order are unchanged.
    static_assert(!QUERY_SWIZZLE || !LANE_CELL, "query swizzle is for warp scoring only");
    for (int i = t; i < G * HD; i += THREADS) {
        const int d = i % HD;
        const int sd = QUERY_SWIZZLE ? (d ^ ((d >> 3) & 4)) : d;
        sq[i / HD][sd] = q[(size_t) (kvh * G) * HD + i];
    }
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            // a block the KV streaming could not make resident keeps page -1 (ctl[3]); its cells are masked
            // (score -FLT_MAX, weight 0) instead of being read from before the pool.
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    __syncthreads();
    if constexpr (LANE_CELL) {
        // S25 (STRATA_ATTN_LANECELL=1): thread t scores cell t % 64 for heads 3 (t / 64) .. +2 by itself - the same
        // 8-dimension partial per former lane (the identical expression), then the former warp_sum's butterfly as
        // lane 0 ran it (a[l] += a[l + o] for o = 16 .. 1), so every score is bit-identical without 12 shuffle
        // reductions per cell
        constexpr int HPT = G * CHUNK / THREADS;   // heads per thread (3)
        const int c = t % CHUNK, h0 = (t / CHUNK) * HPT;
        if (c >= n_here || srow[c] < 0) {
#pragma unroll
            for (int j = 0; j < HPT; ++j) sp[h0 + j][c] = -FLT_MAX;
        } else {
            float a[HPT][32];
#pragma unroll
            for (int l = 0; l < 32; ++l) {
                float k8[8];
                load8<KV_MODE>(p, false, srow[c], l * 8, k8);
#pragma unroll
                for (int j = 0; j < HPT; ++j) {
                    const float4 qa = *reinterpret_cast<const float4*>(&sq[h0 + j][l * 8]);
                    const float4 qb = *reinterpret_cast<const float4*>(&sq[h0 + j][l * 8 + 4]);
                    float s = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                              k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
                    a[j][l] = s;
                }
            }
#pragma unroll
            for (int j = 0; j < HPT; ++j) {
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)
#pragma unroll
                    for (int l = 0; l < o; ++l) a[j][l] = a[j][l] + a[j][l + o];
                sp[h0 + j][c] = a[j][0] * scale;
            }
        }
    } else
    // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.
    for (int c = warp; c < CHUNK; c += WARPS) {
        if (c >= n_here || srow[c] < 0) {
            if (lane < G) sp[lane][c] = -FLT_MAX;
            continue;
        }
        if constexpr (REDUCE12) {
        float k8[8];
        load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
        float part[16];
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const int qflip = QUERY_SWIZZLE ? (lane & 4) : 0;
            const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + qflip]);
            const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + (qflip ^ 4)]);
            float s = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                      k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
            part[h] = s;
        }
#pragma unroll
        for (int h = G; h < 16; ++h) part[h] = 0.0f;
        const float score = reduce12_gfx906(part, lane);
        const int head = reduce12_head_of_lane(lane);
        if ((lane & 1) == 0 && head < G) sp[head][c] = score * scale;
        } else {
        float k8[8];
        load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const int qflip = QUERY_SWIZZLE ? (lane & 4) : 0;
            const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + qflip]);
            const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + (qflip ^ 4)]);
            float s = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                      k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
            s = warp_sum(s);
            if (lane == 0) sp[h][c] = s * scale;
        }
        }
    }
    __syncthreads();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8.
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        sp[h][lane] = ea;
        sp[h][lane + 32] = eb;
        const float l = warp_sum(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();
    // values: thread t owns dimension t for all 12 heads.
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    for (int c = 0; c < n_here; ++c) {
        if (srow[c] < 0) continue;   // masked above, weight 0
        float v;
        if constexpr (KV_MODE == 0) {
            v = __half2float(__ushort_as_half(p.v_pool[srow[c] * HD + t]));
        } else if constexpr (KV_MODE == 1) {
            const float sc = __half2float(__ushort_as_half(p.v_scale[srow[c] * (HD / KV_Q8_GROUP) + t / KV_Q8_GROUP]));
            v = (float) p.v_q[srow[c] * HD + t] * sc;
        } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
            constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
            const int b = t / QK4_0;
            const int rem = t % QK4_0;
            const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + srow[c] * bytes_per_head) + b;
            const float d = __half2float(__ushort_as_half(blk->d));
            const int j = rem < 16 ? rem : (rem - 16);
            const uint8_t byte = blk->qs[j];
            const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
            v = (float) nibble * d;
        }
#pragma unroll
        for (int h = 0; h < G; ++h) acc[h] = fmaf(sp[h][c], v, acc[h]);
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}

template <int KV_MODE, bool LANE_CELL = false, bool QUERY_SWIZZLE = false, bool REDUCE12 = false>
__global__ void __launch_bounds__(THREADS) attn_chunk_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                             const int32_t* __restrict__ ids,
                                                             const int32_t* __restrict__ step, int n_kv_heads,
                                                             int page_size, float scale, float* __restrict__ part_acc,
                                                             float* __restrict__ part_m, float* __restrict__ part_l,
                                                             int n_chunks, int cap = 0, long long scratch_stride = 0) {
    attn_chunk_body<KV_MODE, LANE_CELL, QUERY_SWIZZLE, REDUCE12>(q, p, ids, step, n_kv_heads, page_size, scale,
                                                                 part_acc, part_m, part_l, n_chunks, cap,
                                                                 scratch_stride);
}

// The same blocks for queries of different slots: query blockIdx.z reads its own slot's pools, everything else is
// attn_chunk_kernel's batched form (qsa_decode_attn_slots).
template <int KV_MODE, bool LANE_CELL>
__global__ void __launch_bounds__(THREADS) attn_chunk_slots_kernel(const float* __restrict__ q, QsaAttnSlotPools ps,
                                                                   const int32_t* __restrict__ ids,
                                                                   const int32_t* __restrict__ step, int n_kv_heads,
                                                                   int page_size, float scale,
                                                                   float* __restrict__ part_acc,
                                                                   float* __restrict__ part_m,
                                                                   float* __restrict__ part_l, int n_chunks, int cap,
                                                                   long long scratch_stride) {
    attn_chunk_body<KV_MODE, LANE_CELL, false, false>(q, ps.p[blockIdx.z], ids, step, n_kv_heads, page_size, scale,
                                                      part_acc, part_m, part_l, n_chunks, cap, scratch_stride);
}

// Batch windows: query blockIdx.z attends over its own slot's pools (rows.p[z]); the default kernel otherwise.
template <int KV_MODE>
__global__ void __launch_bounds__(THREADS) attn_chunk_rows_kernel(const float* __restrict__ q, QsaAttnPoolsRows rows,
                                                                  const int32_t* __restrict__ ids,
                                                                  const int32_t* __restrict__ step, int n_kv_heads,
                                                                  int page_size, float scale,
                                                                  float* __restrict__ part_acc,
                                                                  float* __restrict__ part_m,
                                                                  float* __restrict__ part_l, int n_chunks, int cap,
                                                                  long long scratch_stride) {
    attn_chunk_body<KV_MODE, false, false, false>(q, rows.p[blockIdx.z], ids, step, n_kv_heads, page_size, scale,
                                                  part_acc, part_m, part_l, n_chunks, cap, scratch_stride);
}

#if !defined(__HIPCC__)
// Turing (INT8 pools): attn_chunk_kernel<1> bit for bit, rearranged for the SM's shared-memory pipe, which the
// default kernel saturates (its time grew with the queries even with the SMs half empty).  Three changes, none to an
// operand or the order of an operation:
//   - the lane's slice of the 12 query heads (8 dimensions each) is read from shared memory once, into registers,
//     instead of two 16-byte reads per head per cell (with a two-way bank conflict at lane * 8);
//   - the 12 heads' lane sums are reduced with reduce12_gfx906 (16 shuffles, each sum warp_sum's pairs) instead of
//     12 warp_sums (60 shuffles);
//   - the probabilities are stored cell-major, so the value loop reads a cell's 12 as three 16-byte broadcasts.
// RTX 2080 Ti, 2051 selected cells: 1 / 2 / 3 / 4 queries 51.6 / 78.5 / 110 / 146 -> 34.5 / 41.0 / 73.0 / 84.9 us
// (with the merge); 600 cells 45.9 / 74.6 / 105 / 137 -> 28.6 / 32.8 / 33.5 / 33.3 us.  STRATA_ATTN_V3=0 keeps the
// default kernel; =1 takes this one on any CUDA card.
__global__ void __launch_bounds__(THREADS) attn_chunk_q8_v3_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                   const int32_t* __restrict__ ids,
                                                                   const int32_t* __restrict__ step, int n_kv_heads,
                                                                   int page_size, float scale, float* __restrict__ part_acc,
                                                                   float* __restrict__ part_m, float* __restrict__ part_l,
                                                                   int n_chunks, int cap = 0, long long scratch_stride = 0) {
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[G][HD];
    __shared__ __align__(16) float spt[CHUNK][G];   // cell-major: one cell's 12 heads are 3 float4
    __shared__ long long srow[CHUNK];
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    // stored with the gfx906 swizzle (the two float4 halves of lanes with bit 2 swapped): fewer conflicts on the one read
    for (int i = t; i < G * HD; i += THREADS) {
        const int d = i % HD;
        sq[i / HD][d ^ ((d >> 3) & 4)] = q[(size_t) (kvh * G) * HD + i];
    }
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    __syncthreads();
    float qr[G][8];
#pragma unroll
    for (int h = 0; h < G; ++h) {
        const int qflip = lane & 4;
        const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + qflip]);
        const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + (qflip ^ 4)]);
        qr[h][0] = qa.x; qr[h][1] = qa.y; qr[h][2] = qa.z; qr[h][3] = qa.w;
        qr[h][4] = qb.x; qr[h][5] = qb.y; qr[h][6] = qb.z; qr[h][7] = qb.w;
    }
    for (int c = warp; c < CHUNK; c += WARPS) {
        if (c >= n_here || srow[c] < 0) {
            if (lane < G) spt[c][lane] = -FLT_MAX;
            continue;
        }
        float k8[8];
        load8<1>(p, false, srow[c], lane * 8, k8);
        float part[16];
#pragma unroll
        for (int h = 0; h < G; ++h)
            part[h] = k8[0] * qr[h][0] + k8[1] * qr[h][1] + k8[2] * qr[h][2] + k8[3] * qr[h][3] +
                      k8[4] * qr[h][4] + k8[5] * qr[h][5] + k8[6] * qr[h][6] + k8[7] * qr[h][7];
#pragma unroll
        for (int h = G; h < 16; ++h) part[h] = 0.0f;
        const float score = reduce12_gfx906(part, lane);
        const int head = reduce12_head_of_lane(lane);
        if ((lane & 1) == 0 && head < G) spt[c][head] = score * scale;
    }
    __syncthreads();
    for (int h = warp; h < G; h += WARPS) {
        const float a = spt[lane][h], b = spt[lane + 32][h];
        const float m = warp_max(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        spt[lane][h] = ea;
        spt[lane + 32][h] = eb;
        const float l = warp_sum(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    for (int c = 0; c < n_here; ++c) {
        if (srow[c] < 0) continue;   // masked above, weight 0
        const float sc = __half2float(__ushort_as_half(p.v_scale[srow[c] * (HD / KV_Q8_GROUP) + t / KV_Q8_GROUP]));
        const float v = (float) p.v_q[srow[c] * HD + t] * sc;
        fold_cell_v3<G>(acc, &spt[c][0], v);
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}
bool attn_v3_on() {
    static int on[16] = {};   // by device ordinal: 0 unknown, 1 yes, -1 no
    static const int env = [] { const char* v = std::getenv("STRATA_ATTN_V3"); return v ? (v[0] == '0' ? 0 : 1) : -1; }();
    if (env >= 0) return env == 1;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return false;
    if (on[dev] == 0) {
        int maj = 0, mnr = 0;
        const bool ok = cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                        cudaDeviceGetAttribute(&mnr, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess;
        if (!ok) cudaGetLastError();
        on[dev] = ok && strata::sm75_default(maj, mnr) ? 1 : -1;   // Turing only, with STRATA_SM75_PERF=1
    }
    return on[dev] > 0;
}
#endif

// PR #540 (sskver): the same attention with fewer shuffles and less shared-memory traffic, bit for bit the kernel
// above (same operands, same order); +7% prompt speed on a V100, where this kernel reads every prompt chunk.  It
// uses more registers (80 vs 38 on sm_70), so on CUDA it runs on cards below sm_75 only, and exists only in the
// experimental build (-DSTRATA_EXPERIMENTAL_SM60=ON).  HIP: built for wave32 cards (not the gfx906 wave64 backend) and
// run on gfx103x (RDNA2), with 8 cells per step and the butterfly's lane exchanges in registers (DPP / permlanex16)
// instead of ds_bpermute: there the score loop is bound by the LDS pipe (the q reads and ds_bpermute share it), not by
// occupancy - 8 cells per step at 7 waves/SIMD beat 1 cell at 9.  Every other card keeps exactly the kernel above.
// Fork (lab/upsync): built on every CUDA build too, for STRATA_ATTN_DAILY=1 (see daily_attn below).  With NC 2 and
// no DPP it is the Daily's (be4cf1c) attn_chunk_kernel line for line - the same reduce12 scores, the same cell-major
// probabilities and the same four-cell value fold - except that `spt` shares `sq`'s storage here (no arithmetic
// change).  Without the variable nothing selects it on a card at or above sm_75.
#if defined(STRATA_EXPERIMENTAL_SM60) || !defined(__HIPCC__) || !defined(STRATA_HIP_GFX906)
#define STRATA_ATTN_PRE75_BUILT 1
#endif
#if defined(STRATA_ATTN_PRE75_BUILT)
// A butterfly step's lane exchange, `M` = 1, 2, 4, 8 or 16.  DPP (HIP, wave32): in registers - DPP row_xmask within a
// 16-lane row, v_permlanex16 across the two rows - instead of ds_bpermute; the same value moves, so every sum is
// unchanged bit for bit.
template <bool DPP, int M> __device__ __forceinline__ float shfl_x(float v) {
#if defined(__HIPCC__)
    if constexpr (DPP) {
        const int x = __float_as_int(v);
        if constexpr (M == 16) return __int_as_float(__builtin_amdgcn_permlanex16(x, x, 0x76543210, 0xfedcba98, false, false));
        else return __int_as_float(__builtin_amdgcn_mov_dpp(x, 0x160 | M, 0xF, 0xF, false));   // row_xmask:M
    }
#endif
    return __shfl_xor_sync(0xffffffffu, v, M);
}
// warp_sum / warp_max with the same pairing order (xor 16, 8, 4, 2, 1)
template <bool DPP> __device__ __forceinline__ float warp_sum_x(float v) {
    v += shfl_x<DPP, 16>(v); v += shfl_x<DPP, 8>(v); v += shfl_x<DPP, 4>(v); v += shfl_x<DPP, 2>(v); v += shfl_x<DPP, 1>(v);
    return v;
}
template <bool DPP> __device__ __forceinline__ float warp_max_x(float v) {
    v = fmaxf(v, shfl_x<DPP, 16>(v)); v = fmaxf(v, shfl_x<DPP, 8>(v)); v = fmaxf(v, shfl_x<DPP, 4>(v));
    v = fmaxf(v, shfl_x<DPP, 2>(v)); v = fmaxf(v, shfl_x<DPP, 1>(v));
    return v;
}
// The 12 heads' lane sums (`part[12..15]` zero), reduce-scattered with warp_sum's pairing order (xor 16, 8, 4, 2, 1):
// every output adds the same two operands at every level, so each sum is bit for bit warp_sum's (float add commutes),
// for 16 shuffles instead of 12 x 5.  Lane l ends holding head head_of_lane(l); lanes l and l^1 agree.
template <bool DPP> __device__ __forceinline__ float reduce12(const float (&part)[16], int lane) {
    float r8[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const bool hi = (lane & 16) != 0;
        r8[i] = (hi ? part[i + 8] : part[i]) + shfl_x<DPP, 16>(hi ? part[i] : part[i + 8]);
    }
    float r4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const bool hi = (lane & 8) != 0;
        r4[i] = (hi ? r8[i + 4] : r8[i]) + shfl_x<DPP, 8>(hi ? r8[i] : r8[i + 4]);
    }
    float r2[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool hi = (lane & 4) != 0;
        r2[i] = (hi ? r4[i + 2] : r4[i]) + shfl_x<DPP, 4>(hi ? r4[i] : r4[i + 2]);
    }
    const bool hi2 = (lane & 2) != 0;
    const float r1 = (hi2 ? r2[1] : r2[0]) + shfl_x<DPP, 2>(hi2 ? r2[0] : r2[1]);
    return r1 + shfl_x<DPP, 1>(r1);
}
__device__ __forceinline__ int head_of_lane(int lane) {
    return ((lane >> 4) & 1) * 8 + ((lane >> 3) & 1) * 4 + ((lane >> 2) & 1) * 2 + ((lane >> 1) & 1);
}

// One V element of pool row `row`, dimension `d` (the value loop's own thread index), per KV format.
template <int KV_MODE>
__device__ __forceinline__ float load_v1(const QsaAttnPools& p, long long row, int d) {
    if constexpr (KV_MODE == 0) {
        return __half2float(__ushort_as_half(p.v_pool[row * HD + d]));
    } else if constexpr (KV_MODE == 1) {
        const float sc = __half2float(__ushort_as_half(p.v_scale[row * (HD / KV_Q8_GROUP) + d / KV_Q8_GROUP]));
        return (float) p.v_q[row * HD + d] * sc;
    } else {   // modes 2 and 3: V is rotated Q4_0 (kv_q4.hpp); the caller rotates the output back
        constexpr int bytes_per_head = (HD / QK4_0) * sizeof(block_q4_0);
        const int b = d / QK4_0;
        const int rem = d % QK4_0;
        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + row * bytes_per_head) + b;
        const float dd = __half2float(__ushort_as_half(blk->d));
        const int j = rem < 16 ? rem : (rem - 16);
        const uint8_t byte = blk->qs[j];
        const int nibble = (rem < 16) ? ((byte & 0x0F) - 8) : ((byte >> 4) - 8);
        return (float) nibble * dd;
    }
}

template <int KV_MODE, int NC_ = 2, bool DPP_ = false>
__global__ void __launch_bounds__(THREADS) attn_chunk_kernel_pre75(const float* __restrict__ q, QsaAttnPools p,
                                                             const int32_t* __restrict__ ids,
                                                             const int32_t* __restrict__ step, int n_kv_heads,
                                                             int page_size, float scale, float* __restrict__ part_acc,
                                                             float* __restrict__ part_m, float* __restrict__ part_l,
                                                             int n_chunks, int cap = 0, long long scratch_stride = 0) {
    // batched form: query blockIdx.z, with its own q row, selection, step and scratch
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    __shared__ __align__(16) float sq[G][HD];     // 12 KB: this KV head's query heads (dead once the scores are done)
    __shared__ float sp[G][CHUNK];                // scores
    __shared__ long long srow[CHUNK];             // pool row of each cell (page, kv head, slot)
    // probabilities, cell-major: the value loop reads one cell's 12 as three float4.  They are first written after the
    // __syncthreads that ends the score phase, the last reader of sq, so they share its storage (64 x 12 of its 256 x 12
    // floats) and the block's shared memory stays what it was before the cell-major layout.
    float (*const spt)[G] = reinterpret_cast<float (*)[G]>(&sq[0][0]);
    static_assert(CHUNK * G <= G * HD, "spt must fit in sq");
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    for (int i = t; i < G * HD; i += THREADS) sq[i / HD][i % HD] = q[(size_t) (kvh * G) * HD + i];
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            // a block the KV streaming could not make resident keeps page -1 (ctl[3]); its cells are masked
            // (score -FLT_MAX, weight 0) instead of being read from before the pool.
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        srow[t] = r;
    }
    __syncthreads();
    // scores: each warp takes cells warp, warp+8, ...; each lane holds 8 of the 256 dimensions.  NC cells per step
    // (warp + 8 * (i + j)): a lane's q slice comes out of shared memory once for all of them, which cuts the traffic
    // that bounds this phase (24 x 16 B per lane per cell, against the FMAs) by NC; every dot product is the same
    // expression as for a single cell, so the scores are unchanged.
    constexpr int NC = NC_;   // CUDA 2 (4 measured slower on a V100: the registers cost occupancy); HIP gfx103x 8
    const int hd = head_of_lane(lane);
    for (int i = 0; i < CHUNK / WARPS; i += NC) {
        int cc[NC];
        bool all_ok = true;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            cc[j] = warp + WARPS * (i + j);
            all_ok = all_ok && cc[j] < n_here && srow[cc[j]] >= 0;
        }
        if (all_ok) {
            float kk[NC][8];
#pragma unroll
            for (int j = 0; j < NC; ++j) load8<KV_MODE>(p, false, srow[cc[j]], lane * 8, kk[j]);
            float pp[NC][16];
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
#pragma unroll
                for (int j = 0; j < NC; ++j)
                    pp[j][h] = kk[j][0] * qa.x + kk[j][1] * qa.y + kk[j][2] * qa.z + kk[j][3] * qa.w +
                               kk[j][4] * qb.x + kk[j][5] * qb.y + kk[j][6] * qb.z + kk[j][7] * qb.w;
            }
#pragma unroll
            for (int j = 0; j < NC; ++j) {
#pragma unroll
                for (int h = G; h < 16; ++h) pp[j][h] = 0.0f;
                const float sj = reduce12<DPP_>(pp[j], lane);
                if ((lane & 1) == 0 && hd < G) sp[hd][cc[j]] = sj * scale;
            }
        } else {
            for (int k = 0; k < NC; ++k) {   // a masked or out-of-range cell in the group: one cell at a time
                const int c = cc[k];
                if (c >= n_here || srow[c] < 0) {
                    if (lane < G) sp[lane][c] = -FLT_MAX;
                    continue;
                }
                float k8[8];
                load8<KV_MODE>(p, false, srow[c], lane * 8, k8);
                float part[16];
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                    const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
                    part[h] = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w +
                              k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
                }
#pragma unroll
                for (int h = G; h < 16; ++h) part[h] = 0.0f;
                const float s = reduce12<DPP_>(part, lane);
                if ((lane & 1) == 0 && hd < G) sp[hd][c] = s * scale;
            }
        }
    }
    __syncthreads();
    // per-head chunk max and exp-sum: warp w handles heads w and w+8.
    for (int h = warp; h < G; h += WARPS) {
        const float a = sp[h][lane], b = sp[h][lane + 32];
        const float m = warp_max_x<DPP_>(fmaxf(a, b));
        const float ea = (lane < n_here && srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        spt[lane][h] = ea;
        spt[lane + 32][h] = eb;
        const float l = warp_sum_x<DPP_>(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();
    // values: thread t owns dimension t for all 12 heads.
    float acc[G];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;
    // One cell's 12 weights (three broadcast float4 loads) into the 12 accumulators; the cells are folded in
    // ascending order, so the sums are the plain loop's, bit for bit.
    const auto fold = [&](int c, float v) {
        fold_cell<G>(acc, spt[c], v);
    };
    int c = 0;
    // four cells at a time with their V loads issued together (the loop was one dependent load chain per cell);
    // a group holding a masked cell (page -1, rare) leaves this loop and the scalar one below finishes the chunk.
    for (; c + 4 <= n_here; c += 4) {
        const long long r0 = srow[c], r1 = srow[c + 1], r2 = srow[c + 2], r3 = srow[c + 3];
        if ((r0 | r1 | r2 | r3) < 0) break;
        const float v0 = load_v1<KV_MODE>(p, r0, t), v1 = load_v1<KV_MODE>(p, r1, t);
        const float v2 = load_v1<KV_MODE>(p, r2, t), v3 = load_v1<KV_MODE>(p, r3, t);
        fold(c, v0);
        fold(c + 1, v1);
        fold(c + 2, v2);
        fold(c + 3, v3);
    }
    for (; c < n_here; ++c) {
        if (srow[c] < 0) continue;   // masked above, weight 0
        fold(c, load_v1<KV_MODE>(p, srow[c], t));
    }
#pragma unroll
    for (int h = 0; h < G; ++h) part_acc[((size_t) slot * G + h) * HD + t] = acc[h];
}
#if !defined(__HIPCC__)
// The drafter's one-query attention over INT8 pools (qsa_decode_attn_draft): attn_chunk_kernel_pre75<1, 2, false>'s
// arithmetic line for line (the same scores, reduce12, softmax partials and ascending fold; every chunk kernel here
// writes the same partials), with the chunk's K and V rows and their scales copied into shared memory by cp.async up
// front.  The plain kernel reads them as it goes: ~20 dependent global round trips per block (4 score passes, 16 value
// groups), which is what its 16 us per draft step was (latency, not bandwidth: a step reads ~2 MB).
struct DraftAttnSmem {
    float sq[G][HD];                              // the query heads, then (after the scores) the cell-major weights
    float sp[G][CHUNK];
    long long srow[CHUNK];
    int8_t k[CHUNK][HD];
    int8_t v[CHUNK][HD];
    uint16_t ks[CHUNK][HD / KV_Q8_GROUP];
    uint16_t vs[CHUNK][HD / KV_Q8_GROUP];
};
static_assert(HD % 16 == 0 && (HD / KV_Q8_GROUP) * 2 == 8, "draft attention staging: 16-byte rows, 8-byte scale rows");
__device__ __forceinline__ void dattn_cp16(void* s, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((unsigned) __cvta_generic_to_shared(s)), "l"(g));
}
__device__ __forceinline__ void dattn_cp8(void* s, const void* g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"((unsigned) __cvta_generic_to_shared(s)), "l"(g));
}
// The draft kernel's threads: HD (a thread a dimension, every head; measured 9.2 us a struct step) or 2 * HD (two a
// dimension, half the heads each: 9.7 us); the same bits either way.
constexpr int DT_THREADS = HD;
constexpr int DT_WARPS = DT_THREADS / 32;
constexpr int DT_HPT = G * HD / DT_THREADS;   // heads a thread folds
static_assert(G * HD % DT_THREADS == 0 && CHUNK % DT_WARPS == 0 && (CHUNK / DT_WARPS) % 2 == 0 && G <= DT_WARPS * 2,
              "draft attention geometry");
__global__ void __launch_bounds__(DT_THREADS) attn_chunk_draft_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                   const int32_t* __restrict__ ids,
                                                                   const int32_t* __restrict__ step, int n_kv_heads,
                                                                   int page_size, float scale, float* __restrict__ part_acc,
                                                                   float* __restrict__ part_m, float* __restrict__ part_l,
                                                                   int n_chunks, int bounded) {
    extern __shared__ __align__(16) unsigned char dattn_raw[];
    DraftAttnSmem& sm = *reinterpret_cast<DraftAttnSmem*>(dattn_raw);
    float (*const spt)[G] = reinterpret_cast<float (*)[G]>(&sm.sq[0][0]);
    const int n_ids = __ldg(step + kStepWidth);
    const int kvh = blockIdx.x;   // the KV head on x: blocks are dispatched x-fastest, so every head's live chunks come first
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    // a block takes chunks blockIdx.y, + gridDim.y, ... (one wave of blocks, sized by the launch; each chunk's partials
    // are computed exactly as by its own block).  `bounded`: the merge reads only the chunks below the width
    // (attn_merge_v2_kernel), so the empty ones past it are not written.
    for (int chunk = blockIdx.y; chunk < n_chunks; chunk += gridDim.y) {
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (bounded) break;
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        continue;
    }
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = ids[c0 + t];
            const long long page = (long long) p.page_table[cell / page_size];
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        sm.srow[t] = r;
    }
    __syncthreads();
    constexpr int NSEG = HD / 16;   // 16-byte pieces of a row
    for (int i = t; i < CHUNK * NSEG; i += DT_THREADS) {
        const int c = i / NSEG, o = (i % NSEG) * 16;
        const long long r = sm.srow[c];
        if (r >= 0) {
            dattn_cp16(&sm.k[c][o], p.k_q + r * HD + o);
            dattn_cp16(&sm.v[c][o], p.v_q + r * HD + o);
        }
    }
    if (t < CHUNK && sm.srow[t] >= 0) {
        dattn_cp8(&sm.ks[t][0], p.k_scale + sm.srow[t] * (HD / KV_Q8_GROUP));
        dattn_cp8(&sm.vs[t][0], p.v_scale + sm.srow[t] * (HD / KV_Q8_GROUP));
    }
    asm volatile("cp.async.commit_group;\n" ::);
    for (int i = t; i < G * HD; i += DT_THREADS) sm.sq[i / HD][i % HD] = q[(size_t) (kvh * G) * HD + i];
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    // load8_q8's expression on the staged row
    auto k8 = [&](int c, int d0, float* out) {
        const float sc = __half2float(__ushort_as_half(sm.ks[c][d0 / KV_Q8_GROUP]));
        const uint2 raw = *reinterpret_cast<const uint2*>(&sm.k[c][d0]);
        const int8_t* cc = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
        for (int j = 0; j < 8; ++j) out[j] = (float) cc[j] * sc;
    };
    constexpr int NC = 2;
    const int hd = head_of_lane(lane);
    for (int i = 0; i < CHUNK / DT_WARPS; i += NC) {
        int cc[NC];
        bool all_ok = true;
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            cc[j] = warp + DT_WARPS * (i + j);
            all_ok = all_ok && cc[j] < n_here && sm.srow[cc[j]] >= 0;
        }
        if (all_ok) {
            float kk[NC][8];
#pragma unroll
            for (int j = 0; j < NC; ++j) k8(cc[j], lane * 8, kk[j]);
            float pp[NC][16];
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float4 qa = *reinterpret_cast<const float4*>(&sm.sq[h][lane * 8]);
                const float4 qb = *reinterpret_cast<const float4*>(&sm.sq[h][lane * 8 + 4]);
#pragma unroll
                for (int j = 0; j < NC; ++j)
                    pp[j][h] = kk[j][0] * qa.x + kk[j][1] * qa.y + kk[j][2] * qa.z + kk[j][3] * qa.w +
                               kk[j][4] * qb.x + kk[j][5] * qb.y + kk[j][6] * qb.z + kk[j][7] * qb.w;
            }
#pragma unroll
            for (int j = 0; j < NC; ++j) {
#pragma unroll
                for (int h = G; h < 16; ++h) pp[j][h] = 0.0f;
                const float sj = reduce12<false>(pp[j], lane);
                if ((lane & 1) == 0 && hd < G) sm.sp[hd][cc[j]] = sj * scale;
            }
        } else {
            for (int k = 0; k < NC; ++k) {
                const int c = cc[k];
                if (c >= n_here || sm.srow[c] < 0) {
                    if (lane < G) sm.sp[lane][c] = -FLT_MAX;
                    continue;
                }
                float kv8[8];
                k8(c, lane * 8, kv8);
                float part[16];
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    const float4 qa = *reinterpret_cast<const float4*>(&sm.sq[h][lane * 8]);
                    const float4 qb = *reinterpret_cast<const float4*>(&sm.sq[h][lane * 8 + 4]);
                    part[h] = kv8[0] * qa.x + kv8[1] * qa.y + kv8[2] * qa.z + kv8[3] * qa.w +
                              kv8[4] * qb.x + kv8[5] * qb.y + kv8[6] * qb.z + kv8[7] * qb.w;
                }
#pragma unroll
                for (int h = G; h < 16; ++h) part[h] = 0.0f;
                const float s = reduce12<false>(part, lane);
                if ((lane & 1) == 0 && hd < G) sm.sp[hd][c] = s * scale;
            }
        }
    }
    __syncthreads();
    for (int h = warp; h < G; h += DT_WARPS) {
        const float a = sm.sp[h][lane], b = sm.sp[h][lane + 32];
        const float m = warp_max_x<false>(fmaxf(a, b));
        const float ea = (lane < n_here && sm.srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && sm.srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        spt[lane][h] = ea;
        spt[lane + 32][h] = eb;
        const float l = warp_sum_x<false>(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();
    // dimension d of DT_HPT heads a thread (all G with DT_THREADS = HD, as the plain kernel): load_v1<1>'s expression on the
    // staged rows, each head's fmaf as fold_cell's, in ascending cell order (a masked cell is skipped, weight 0)
    const int d = t & (HD - 1), h0 = (t / HD) * DT_HPT;
    float acc[DT_HPT];
#pragma unroll
    for (int h = 0; h < DT_HPT; ++h) acc[h] = 0.0f;
    for (int c = 0; c < n_here; ++c) {
        if (sm.srow[c] < 0) continue;
        const float sc = __half2float(__ushort_as_half(sm.vs[c][d / KV_Q8_GROUP]));
        const float v = (float) sm.v[c][d] * sc;
        const float* w = &spt[c][h0];
#pragma unroll
        for (int h = 0; h < DT_HPT; ++h) acc[h] = fmaf(w[h], v, acc[h]);
    }
#pragma unroll
    for (int h = 0; h < DT_HPT; ++h) part_acc[((size_t) slot * G + h0 + h) * HD + d] = acc[h];
    __syncthreads();   // the next chunk restages the shared memory
    }
}
// dec/int (STRATA_ATTN_STAGED=1): the verify window's INT8 chunk pass as Basalt's attn_chunk_v2 runs it -
// attn_chunk_kernel_pre75<1, 2, false>'s operands and order (the same 8-dimension dot products, reduce12, softmax
// partials and ascending value fold), with the chunk's K and V rows and their scales staged in shared memory by
// cp.async up front (one memory round trip per block instead of ~20 dependent ones), the lane's query slice read
// straight into registers, and two blocks per SM (40 KB each), so a T~3 window's 2 x 3 x n_chunks blocks run at once.
// A masked or missing cell is zero in the staged V rows with weight 0, so the fold needs no branch: fmaf(0, 0, acc)
// leaves acc unchanged (acc starts at +0 and is never -0).  Each block is one query's, so a row's result does not
// depend on how many rows the window has.
struct StagedAttnSmem {
    alignas(16) int8_t k[CHUNK][HD];
    alignas(16) int8_t v[CHUNK][HD];
    alignas(16) uint16_t ks[CHUNK][HD / KV_Q8_GROUP];
    alignas(16) uint16_t vs[CHUNK][HD / KV_Q8_GROUP];
    float sp[G][CHUNK];               // scores
    alignas(16) float spt[CHUNK][G];  // probabilities, cell-major
    long long srow[CHUNK];
};
static_assert(sizeof(StagedAttnSmem) <= 48 * 1024, "staged attention: static shared memory");
__global__ void __launch_bounds__(THREADS, 2) attn_chunk_staged_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                     const int32_t* __restrict__ ids,
                                                                     const int32_t* __restrict__ step, int n_kv_heads,
                                                                     int page_size, float scale, float* __restrict__ part_acc,
                                                                     float* __restrict__ part_m, float* __restrict__ part_l,
                                                                     int n_chunks, int cap = 0, long long scratch_stride = 0) {
    __shared__ StagedAttnSmem sm;
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.z * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.z * (size_t) scratch_stride;
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int c0 = chunk * CHUNK;
    // the cell id is requested before the width is known (in bounds; unused past the width)
    const int cell = (t < CHUNK && c0 + t < cap) ? __ldg(ids + c0 + t) : 0;
    const int n_ids = __ldg(step + kStepWidth);
    const int n_here = min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const long long page = (long long) p.page_table[cell / page_size];
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        sm.srow[t] = r;
    }
    __syncthreads();
    {
        constexpr int PER = HD / 16;
        const unsigned char* kb = reinterpret_cast<const unsigned char*>(p.k_q);
        const unsigned char* vb = reinterpret_cast<const unsigned char*>(p.v_q);
        for (int i = t; i < CHUNK * PER; i += THREADS) {
            const int c = i / PER, j = i % PER;
            const long long r = sm.srow[c];
            if (r < 0) {
                *reinterpret_cast<uint4*>(&sm.v[c][j * 16]) = make_uint4(0, 0, 0, 0);
                continue;
            }
            dattn_cp16(&sm.k[c][j * 16], kb + r * HD + j * 16);
            dattn_cp16(&sm.v[c][j * 16], vb + r * HD + j * 16);
        }
        if (t < 2 * CHUNK) {
            const int c = t & (CHUNK - 1);
            const long long r = sm.srow[c];
            if (r >= 0) {
                if (t < CHUNK) dattn_cp8(&sm.ks[c][0], p.k_scale + r * (HD / KV_Q8_GROUP));
                else dattn_cp8(&sm.vs[c][0], p.v_scale + r * (HD / KV_Q8_GROUP));
            } else if (t >= CHUNK) {
                *reinterpret_cast<uint2*>(&sm.vs[c][0]) = make_uint2(0, 0);
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
    }
    // the lane's 8 dimensions of the 12 query heads, while the copies are in flight
    float qr[G][8];
#pragma unroll
    for (int h = 0; h < G; ++h) {
        const float4 qa = __ldg(reinterpret_cast<const float4*>(q + (size_t) (kvh * G + h) * HD + lane * 8));
        const float4 qb = __ldg(reinterpret_cast<const float4*>(q + (size_t) (kvh * G + h) * HD + lane * 8 + 4));
        qr[h][0] = qa.x; qr[h][1] = qa.y; qr[h][2] = qa.z; qr[h][3] = qa.w;
        qr[h][4] = qb.x; qr[h][5] = qb.y; qr[h][6] = qb.z; qr[h][7] = qb.w;
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    const int hd = head_of_lane(lane);
    constexpr int NC = 2;
#pragma unroll 1
    for (int i = 0; i < CHUNK / WARPS; i += NC) {
        int cc[NC];
        bool ok[NC];
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            cc[j] = warp + WARPS * (i + j);
            ok[j] = cc[j] < n_here && sm.srow[cc[j]] >= 0;
        }
        float kk[NC][8];
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            // load8_q8's expression on the staged row
            const float sc = __half2float(__ushort_as_half(sm.ks[cc[j]][lane * 8 / KV_Q8_GROUP]));
            const uint2 raw = *reinterpret_cast<const uint2*>(&sm.k[cc[j]][lane * 8]);
            const int8_t* cv = reinterpret_cast<const int8_t*>(&raw);
#pragma unroll
            for (int e = 0; e < 8; ++e) kk[j][e] = ok[j] ? (float) cv[e] * sc : 0.0f;
        }
        float pp[NC][16];
#pragma unroll
        for (int h = 0; h < G; ++h) {
#pragma unroll
            for (int j = 0; j < NC; ++j)
                pp[j][h] = kk[j][0] * qr[h][0] + kk[j][1] * qr[h][1] + kk[j][2] * qr[h][2] + kk[j][3] * qr[h][3] +
                           kk[j][4] * qr[h][4] + kk[j][5] * qr[h][5] + kk[j][6] * qr[h][6] + kk[j][7] * qr[h][7];
        }
#pragma unroll
        for (int j = 0; j < NC; ++j) {
#pragma unroll
            for (int h = G; h < 16; ++h) pp[j][h] = 0.0f;
            const float sj = reduce12<false>(pp[j], lane);
            if (ok[j]) {
                if ((lane & 1) == 0 && hd < G) sm.sp[hd][cc[j]] = sj * scale;
            } else if (lane < G) {
                sm.sp[lane][cc[j]] = -FLT_MAX;
            }
        }
    }
    __syncthreads();
    for (int h = warp; h < G; h += WARPS) {
        const float a = sm.sp[h][lane], b = sm.sp[h][lane + 32];
        const float m = warp_max(fmaxf(a, b));
        const float ea = (lane < n_here && sm.srow[lane] >= 0) ? __expf(a - m) : 0.0f;
        const float eb = (lane + 32 < n_here && sm.srow[lane + 32] >= 0) ? __expf(b - m) : 0.0f;
        sm.spt[lane][h] = ea;
        sm.spt[lane + 32][h] = eb;
        const float l = warp_sum(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();
    // values: thread t < 128 owns dimensions 2t, 2t+1 of every head; the cells in ascending order
    if (t >= HD / 2) return;
    float acc[G][2];
#pragma unroll
    for (int h = 0; h < G; ++h) acc[h][0] = acc[h][1] = 0.0f;
    const int lim = n_here;
#pragma unroll 4
    for (int c = 0; c < lim; ++c) {
        const float sc = __half2float(__ushort_as_half(sm.vs[c][2 * t / KV_Q8_GROUP]));
        const unsigned short raw = *reinterpret_cast<const unsigned short*>(&sm.v[c][2 * t]);
        const float v0 = (float) (int8_t) (raw & 0xff) * sc;
        const float v1 = (float) (int8_t) (raw >> 8) * sc;
        float w[G];
#pragma unroll
        for (int i = 0; i < G / 4; ++i) {
            const float4 x = *reinterpret_cast<const float4*>(&sm.spt[c][4 * i]);
            w[4 * i] = x.x; w[4 * i + 1] = x.y; w[4 * i + 2] = x.z; w[4 * i + 3] = x.w;
        }
#pragma unroll
        for (int h = 0; h < G; ++h) {
            acc[h][0] = fmaf(w[h], v0, acc[h][0]);
            acc[h][1] = fmaf(w[h], v1, acc[h][1]);
        }
    }
#pragma unroll
    for (int h = 0; h < G; ++h)
        *reinterpret_cast<float2*>(&part_acc[((size_t) slot * G + h) * HD + 2 * t]) = make_float2(acc[h][0], acc[h][1]);
}
#endif
#endif  // STRATA_ATTN_PRE75_BUILT

__global__ void __launch_bounds__(HD) attn_merge_kernel(const float* __restrict__ part_acc,
                                                        const float* __restrict__ part_m,
                                                        const float* __restrict__ part_l, int n_chunks,
                                                        float* __restrict__ attn, long long scratch_stride = 0) {
    part_acc += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.y * (size_t) scratch_stride;
    attn += (size_t) blockIdx.y * (size_t) gridDim.x * HD;
    const int h = blockIdx.x;                 // global query head
    const int kvh = h / G, hl = h % G;
    const int d = threadIdx.x;
    float M = -FLT_MAX;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, part_m[(kvh * n_chunks + c) * G + hl]);
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const int slot = kvh * n_chunks + c;
        const float m = part_m[slot * G + hl];
        if (m == -FLT_MAX) continue;
        const float w = __expf(m - M);
        L = fmaf(part_l[slot * G + hl], w, L);
        acc = fmaf(part_acc[((size_t) slot * G + hl) * HD + d], w, acc);
    }
    attn[(size_t) h * HD + d] = L > 0.0f ? acc / L : 0.0f;
}

// attn_merge_kernel's arithmetic in the same chunk order, with two changes that leave every output bit as it was
// (the Daily's 3effa9d5, from Hardin22; qsa_merge_daily_parity compares the two kernels on the same partials, all four
// KV formats):
//
// (1) Bounded by the width.  The chunk kernel writes m = -FLT_MAX, l = 0 for every chunk with c0 >= step[kStepWidth]
//     (its `n_here <= 0` exit) and the merge skips exactly those, so it can stop at n_act = ceil(width / CHUNK):
//     fmaxf with -FLT_MAX never changes M, and a skipped chunk never touches L or acc.  The drafter's merge walks its
//     whole capacity (--mtp-window 32768: 512 chunks) at every context, where a short context fills one or two.
// (2) Loads batched.  attn_merge_kernel loads part_l / part_acc behind its `m == -FLT_MAX` branch, so the compiler
//     issues each chunk's l and acc loads inside that chunk's branch: a few dependent L2 round trips per chunk.  Here
//     the m, l and acc of U chunks are loaded first, guarded only by the chunk index, then consumed in chunk order.
//     The skip is a select on the loaded m instead of a branch, so the loads cannot move back behind it.  The values
//     selected are the very fmaf's attn_merge_kernel computes (same operands, same order), and a skipped chunk leaves
//     L and acc as they were, as its `continue` does.
constexpr int kMergeU = 16;

template <int U, bool TAIL>
__device__ __forceinline__ void merge_max_block(const float* __restrict__ pm, int c0, int n_act, float& M) {
    float mm[U];
#pragma unroll
    for (int u = 0; u < U; ++u) mm[u] = (!TAIL || c0 + u < n_act) ? pm[(size_t) (c0 + u) * G] : -FLT_MAX;
#pragma unroll
    for (int u = 0; u < U; ++u)
        if (!TAIL || c0 + u < n_act) M = fmaxf(M, mm[u]);
}

template <int U, bool TAIL>
__device__ __forceinline__ void merge_sum_block(const float* __restrict__ pm, const float* __restrict__ pl,
                                                const float* __restrict__ pa, int c0, int n_act, float M, float& L,
                                                float& acc) {
    float mm[U], ll[U], aa[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const bool in = !TAIL || c0 + u < n_act;
        mm[u] = in ? pm[(size_t) (c0 + u) * G] : -FLT_MAX;
        ll[u] = in ? pl[(size_t) (c0 + u) * G] : 0.0f;
        aa[u] = in ? pa[(size_t) (c0 + u) * G * HD] : 0.0f;
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const float w = __expf(mm[u] - M);
        const float L_next = fmaf(ll[u], w, L);
        const float acc_next = fmaf(aa[u], w, acc);
        const bool live = !(mm[u] == -FLT_MAX);   // attn_merge_kernel: `if (m == -FLT_MAX) continue;`
        L = live ? L_next : L;
        acc = live ? acc_next : acc;
    }
}

template <int U, bool ROT = false>
__global__ void __launch_bounds__(HD) attn_merge_v2_kernel(const float* __restrict__ part_acc,
                                                           const float* __restrict__ part_m,
                                                           const float* __restrict__ part_l, int n_chunks,
                                                           const int32_t* __restrict__ steps,
                                                           float* __restrict__ attn, long long scratch_stride = 0) {
    part_acc += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_m += (size_t) blockIdx.y * (size_t) scratch_stride;
    part_l += (size_t) blockIdx.y * (size_t) scratch_stride;
    attn += (size_t) blockIdx.y * (size_t) gridDim.x * HD;
    // the step record the chunk kernel read for this query (attn_chunk_kernel: step + blockIdx.z * kStepCount)
    const int width = __ldg(steps + (size_t) blockIdx.y * kStepCount + kStepWidth);
    const int n_act = width > 0 ? min(n_chunks, (width - 1) / CHUNK + 1) : 0;
    const int h = blockIdx.x;                 // global query head
    const int kvh = h / G, hl = h % G;
    const int d = threadIdx.x;
    // chunk c of this head: part_m / part_l [(kvh * n_chunks + c) * G + hl], part_acc [that * HD + d]
    const size_t base = (size_t) kvh * (size_t) n_chunks * G + (size_t) hl;
    const float* pm = part_m + base;
    const float* pl = part_l + base;
    const float* pa = part_acc + base * HD + (size_t) d;
    const int n_full = n_act - n_act % U;
    float M = -FLT_MAX;
    for (int c0 = 0; c0 < n_full; c0 += U) merge_max_block<U, false>(pm, c0, n_act, M);
    if (n_full < n_act) merge_max_block<U, true>(pm, n_full, n_act, M);
    float L = 0.0f, acc = 0.0f;
    for (int c0 = 0; c0 < n_full; c0 += U) merge_sum_block<U, false>(pm, pl, pa, c0, n_act, M, L, acc);
    if (n_full < n_act) merge_sum_block<U, true>(pm, pl, pa, n_full, n_act, M, L, acc);
    float o = L > 0.0f ? acc / L : 0.0f;
    if constexpr (ROT) {
        // dec/int (STRATA_KV_POST): fwht256_cuda's inverse rotation of the head's row (kv_q4.cu, scale 1/16): its lane
        // stages on each warp's 32 values, then its register stages (value bits 5..7 = the warp) through shared memory -
        // the same operations in the same order (the drafter's attn_merge_draft_gq_kernel does the same)
        __shared__ float fw[HD];
        const int lane = d & 31, w = d >> 5;
        float v = o * (1.0f / 16.0f);
#pragma unroll
        for (int hh = 1; hh < 32; hh *= 2) {
            const float v2 = __shfl_xor_sync(0xffffffffu, v, hh, 32);
            v = (lane & hh) == 0 ? v + v2 : v2 - v;
        }
#pragma unroll
        for (int st = 1; st < HD / 32; st *= 2) {
            fw[d] = v;
            __syncthreads();
            v = (w & st) == 0 ? fw[d] + fw[d + 32 * st] : fw[d - 32 * st] - fw[d];
            __syncthreads();
        }
        o = v;
    }
    attn[(size_t) h * HD + d] = o;
}

#if !defined(__HIPCC__)
// ---- the drafter's merge with the output gate and o_proj's q8_1 input folded in (STRATA_MTP_ATTN_GQ, mtp.cpp): the merge
// is attn_merge_v2_kernel's arithmetic (this file's build), then native_qsa_gate_apply's and native_quantize_q8_1's, whose
// files are built with --use_fast_math, written as the instructions that build emits - the three kernels' outputs bit for bit.
__device__ __forceinline__ float dgq_gate(float x, float raw) {   // x * (1 / (1 + expf(-raw))), fast math
    float y;
    asm("{\n\t.reg .f32 t, e, d, r;\n\t"
        "mul.ftz.f32 t, %2, 0fBFB8AA3B;\n\t"
        "ex2.approx.ftz.f32 e, t;\n\t"
        "add.ftz.f32 d, e, 0f3F800000;\n\t"
        "rcp.approx.ftz.f32 r, d;\n\t"
        "mul.ftz.f32 %0, %1, r;\n\t}"
        : "=f"(y) : "f"(x), "f"(raw));
    return y;
}
__device__ __forceinline__ float dgq_max(float a, float b) { float y; asm("max.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float dgq_abs(float a) { float y; asm("abs.ftz.f32 %0, %1;" : "=f"(y) : "f"(a)); return y; }
__device__ __forceinline__ float dgq_add(float a, float b) { float y; asm("add.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float dgq_div(float a, float b) { float y; asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
struct DgqQ81 { half2 ds; int8_t qs[32]; };
static_assert(sizeof(DgqQ81) == 36 && HD % 32 == 0, "q8_1 blocks of 32 inside a head");

template <int U>
__global__ void __launch_bounds__(HD) attn_merge_draft_gq_kernel(const float* __restrict__ part_acc,
                                                                 const float* __restrict__ part_m,
                                                                 const float* __restrict__ part_l, int n_chunks,
                                                                 const int32_t* __restrict__ steps, float* __restrict__ attn,
                                                                 const float* __restrict__ q_full, float* __restrict__ gated,
                                                                 DgqQ81* __restrict__ xq, int rot) {
    // attn_merge_v2_kernel for the one query
    const int width = __ldg(steps + kStepWidth);
    const int n_act = width > 0 ? min(n_chunks, (width - 1) / CHUNK + 1) : 0;
    const int h = blockIdx.x;
    const int kvh = h / G, hl = h % G;
    const int d = threadIdx.x;
    const size_t base = (size_t) kvh * (size_t) n_chunks * G + (size_t) hl;
    const float* pm = part_m + base;
    const float* pl = part_l + base;
    const float* pa = part_acc + base * HD + (size_t) d;
    const int n_full = n_act - n_act % U;
    float M = -FLT_MAX;
    for (int c0 = 0; c0 < n_full; c0 += U) merge_max_block<U, false>(pm, c0, n_act, M);
    if (n_full < n_act) merge_max_block<U, true>(pm, n_full, n_act, M);
    float L = 0.0f, acc = 0.0f;
    for (int c0 = 0; c0 < n_full; c0 += U) merge_sum_block<U, false>(pm, pl, pa, c0, n_act, M, L, acc);
    if (n_full < n_act) merge_sum_block<U, true>(pm, pl, pa, n_full, n_act, M, L, acc);
    float o = L > 0.0f ? acc / L : 0.0f;
    if (rot) {
        // fwht256_cuda (kv_q4.cu, scale 1/16) on the head's row: its lane stages on the warp's 32 values, then its register
        // stages (value index bits 5..7 = the warp) through shared memory - the same additions in the same order
        __shared__ float fw[HD];
        const int lane = d & 31, w = d >> 5;
        float v = o * (1.0f / 16.0f);
#pragma unroll
        for (int hh = 1; hh < 32; hh *= 2) {
            const float v2 = __shfl_xor_sync(0xffffffffu, v, hh, 32);
            v = (lane & hh) == 0 ? v + v2 : v2 - v;
        }
#pragma unroll
        for (int st = 1; st < HD / 32; st *= 2) {
            fw[d] = v;
            __syncthreads();
            v = (w & st) == 0 ? fw[d] + fw[d + 32 * st] : fw[d - 32 * st] - fw[d];
            __syncthreads();
        }
        o = v;
    }
    const size_t i = (size_t) h * HD + d;
    attn[i] = o;
    // native_qsa_gate_apply (head h, channel d)
    const float gv = dgq_gate(o, q_full[(size_t) h * 2 * HD + HD + d]);
    gated[i] = gv;
    // native_quantize_q8_1_kernel: lane d & 31 holds value d & 31 of block i / 32
    float amax = dgq_abs(gv), sum = gv;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = dgq_max(amax, __shfl_xor_sync(0xffffffffu, amax, off, 32));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum = dgq_add(sum, __shfl_xor_sync(0xffffffffu, sum, off, 32));
    const float dd = fabsf(dgq_div(amax, 127.0f)) > 65504.0f ? copysignf(65504.0f, dgq_div(amax, 127.0f)) : dgq_div(amax, 127.0f);
    int8_t code = 0;
    if (amax != 0.0f) {
        const float r = roundf(dgq_div(gv, dd));
        code = (int8_t) (r > 127.0f ? 127.0f : r < -127.0f ? -127.0f : r);
    }
    xq[i / 32].qs[i % 32] = code;
    if (i % 32 == 0) {
        const float s2 = fabsf(sum) > 65504.0f ? copysignf(65504.0f, sum) : sum;
        const float d2 = fabsf(dd) > 65504.0f ? copysignf(65504.0f, dd) : dd;
        xq[i / 32].ds = make_half2(__float2half(d2), __float2half(s2));
    }
}
#endif

#if defined(STRATA_HIP_GFX906)
// The compat backend can target other architectures. Check the calling thread's current device,
// caching only its last ordinal so a layer split or another thread cannot reuse the wrong answer.
bool query_swizzle_gfx906_device() {
    static thread_local int cached_device = -1;
    static thread_local bool cached_supported = false;
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess || device < 0) {
        cached_device = -1;
        cached_supported = false;
        (void) hipGetLastError();
        return false;
    }
    if (device != cached_device) {
        cached_device = -1;
        cached_supported = false;
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, device) != hipSuccess) {
            (void) hipGetLastError();
            return false;
        }
        cached_supported = gfx_arch_is(prop.gcnArchName, "gfx906") && prop.warpSize == 64;
        cached_device = device;
    }
    return cached_supported;
}
#endif

// Fork (lab/upsync): STRATA_ATTN_DAILY=1 runs the Daily's (be4cf1c) decode attention on every CUDA card: the chunk
// pass on PR #540's kernel (attn_chunk_kernel_pre75<M, 2, false>, the Daily's attn_chunk_kernel), bit for bit the
// default kernel's partials.  CUDA only; unset or anything but "1" keeps upstream 0.1.42's kernels.
// STRATA_ATTN_PRE75=0 still turns the chunk kernel off (A/B).
// 147-verify: unset, it is on for a compute capability 12.0 card (sm120_series.hpp; RTX PRO 5000, a 4-token verify round
// -2.6%, bitwise); STRATA_ATTN_DAILY=0 (anything but "1") keeps upstream's kernels there too.
[[maybe_unused]] bool daily_attn() {
#if defined(__HIPCC__)
    return false;
#else
    static const int env = [] {
        const char* v = std::getenv("STRATA_ATTN_DAILY");
        if (v == nullptr) return -1;
        return v[0] == '1' && v[1] == '\0' ? 1 : 0;
    }();
    if (env >= 0) return env != 0;
    static int per_dev[64];   // 0 unknown, 1 no, 2 yes
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    if (!per_dev[dev]) {
        int maj = 0, mnr = 0;
        const bool ok = cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
                        cudaDeviceGetAttribute(&mnr, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess;
        if (!ok) cudaGetLastError();
        per_dev[dev] = ok && strata::sm120_default(maj, mnr) ? 2 : 1;
    }
    return per_dev[dev] == 2;
#endif
}

#if defined(STRATA_ATTN_PRE75_BUILT)
// CUDA: the current device is below sm_75 (experimental build) or STRATA_ATTN_DAILY=1 (any card); HIP: it is gfx103x
// (per device: a layer split can mix cards).
// STRATA_ATTN_PRE75=0 turns PR #540's kernel off (A/B); on HIP =1 also runs it on another wave32 card.
bool pre75_attn() {
    static const int env = [] {
        const char* e = std::getenv("STRATA_ATTN_PRE75");
        return e == nullptr ? -1 : e[0] == '0' ? 0 : e[0] == '1' ? 1 : -1;
    }();
    if (env == 0) return false;
#if !defined(__HIPCC__) && !defined(STRATA_EXPERIMENTAL_SM60)
    return daily_attn();   // the ready-made engine: upstream's attn_chunk_kernel unless STRATA_ATTN_DAILY=1
#else
#if !defined(__HIPCC__)
    if (daily_attn()) return true;
#endif
    static int cc[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
#if defined(__HIPCC__)
    if (cc[dev] == 0) {   // 2: gfx103x, 1: another card
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, dev) != hipSuccess) { (void) hipGetLastError(); return false; }
        cc[dev] = std::strncmp(prop.gcnArchName, "gfx103", 6) == 0 ? 2 : 1;
    }
    return env == 1 || cc[dev] == 2;
#else
    if (cc[dev] == 0) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        cc[dev] = 10 * major + minor;
    }
    return cc[dev] < 75;
#endif
#endif  // ready-made CUDA engine
}
// `PRE` picks PR #540's kernel; the environment's choice is STRATA_ATTN_CHUNK.
#if defined(__HIPCC__)
#define STRATA_ATTN_CHUNK_SEL(M, PRE) ((PRE) ? attn_chunk_kernel_pre75<M, 8, true> : attn_chunk_kernel<M>)
#else
#define STRATA_ATTN_CHUNK_SEL(M, PRE) ((PRE) ? attn_chunk_kernel_pre75<M> : attn_chunk_kernel<M>)
#endif
#define STRATA_ATTN_CHUNK(M) STRATA_ATTN_CHUNK_SEL(M, pre75_attn())
#else
#define STRATA_ATTN_CHUNK_SEL(M, PRE) attn_chunk_kernel<M>
#define STRATA_ATTN_CHUNK(M) attn_chunk_kernel<M>
#endif

// dec/int: STRATA_ATTN_STAGED=1 - the INT8 chunk pass on attn_chunk_staged_kernel (bitwise the Daily's partials)
bool staged_attn() {
#if defined(__HIPCC__)
    return false;
#else
    static const bool on = [] { const char* v = std::getenv("STRATA_ATTN_STAGED"); return v && v[0] == '1'; }();
    return on;
#endif
}

// S25: STRATA_ATTN_LANECELL=1 - the score phase one cell per thread (bit-identical scores, no shuffle reductions)
bool lane_cell_attn() {
    static const bool on = [] { const char* v = std::getenv("STRATA_ATTN_LANECELL"); return v && v[0] == '1'; }();
    return on;
}

// 0 fp16, 1 int8, 2 q4_0, 3 int8 K with q4_0 V
int attn_kv_mode(const QsaAttnPools& p) {
    return p.k_q4 != nullptr ? 2 : (p.k_q != nullptr && p.v_q4 != nullptr ? 3 : (p.k_q != nullptr ? 1 : 0));
}

// STRATA_ATTN_MERGE_V2: 1 merges with attn_merge_v2_kernel, 0 with attn_merge_kernel.  Unset it follows
// STRATA_ATTN_DAILY (the Daily had it on by default on NVIDIA), so upstream's merger stays the default.
int attn_merge_v2() {
    static const int env = [] {
        const char* v = std::getenv("STRATA_ATTN_MERGE_V2");
        return v != nullptr && *v != '\0' ? (std::atoi(v) != 0 ? 1 : 0) : -1;
    }();
    return env >= 0 ? env : (daily_attn() ? 1 : 0);
}

// the merge of `n_q` queries' partials: v2 = 0 attn_merge_kernel, 1 attn_merge_v2_kernel (bitwise the same output)
void launch_merge(int v2, const float* part_acc, const float* part_m, const float* part_l, int n_chunks,
                  const int32_t* steps, float* attn, long long stride, unsigned n_head, unsigned n_q, cudaStream_t st) {
    if (v2 == 2)   // dec/int: + the inverse rotation (attn_merge_v2_kernel<U, true>)
        attn_merge_v2_kernel<kMergeU, true><<<dim3(n_head, n_q), HD, 0, st>>>(part_acc, part_m, part_l, n_chunks, steps,
                                                                               attn, stride);
    else if (v2)
        attn_merge_v2_kernel<kMergeU><<<dim3(n_head, n_q), HD, 0, st>>>(part_acc, part_m, part_l, n_chunks, steps,
                                                                         attn, stride);
    else
        attn_merge_kernel<<<dim3(n_head, n_q), HD, 0, st>>>(part_acc, part_m, part_l, n_chunks, attn, stride);
}

// qsa_decode_attn_batch's chunk pass: `n_q` queries' partials into `scratch` (one stride each).  `kernel` < 0 is the
// environment's choice (lane cell, gfx906 swizzle, STRATA_ATTN_DAILY / PR #540); 0 is attn_chunk_kernel and 1 PR #540's
// kernel where it is built (the parity harness: never the environment).
void launch_chunks_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                         int64_t cap, const QsaShapes& s, float* scratch, int64_t n_q, cudaStream_t st, int kernel) {
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    // per query: [acc: n_chunks*n_head*HD][m: n_chunks*n_head][l: n_chunks*n_head], all offsets from one stride
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q);
#if defined(STRATA_ATTN_PRE75_BUILT)
    const bool pre = kernel < 0 ? pre75_attn() : kernel == 1;
#endif
    // S25: STRATA_ATTN_LANECELL=1 - the score phase one cell per thread (bit-identical scores, no shuffle reductions)
    const bool lane_cell = kernel < 0 && lane_cell_attn();
#if defined(STRATA_HIP_GFX906)
    static const bool query_swizzle = [] {
        const char* v = std::getenv("STRATA_GFX906_ATTN_QUERY_SWIZZLE"); return v && v[0] == '1';
    }();
    const bool use_query_swizzle = kernel < 0 && query_swizzle && !lane_cell && kv_mode == 1 &&
                                   query_swizzle_gfx906_device();
    static const bool query_swizzle_trace = std::getenv("STRATA_GFX906_ATTN_QUERY_SWIZZLE_TRACE") != nullptr;
    static thread_local bool query_swizzle_said = false;
    if (query_swizzle_trace && !query_swizzle_said) {
        query_swizzle_said = true;
        std::fprintf(stderr, "strata attn-query-swizzle: active=%d nq=%lld cap=%lld kv_mode=%d lane_cell=%d\n",
                     use_query_swizzle, (long long) n_q, (long long) cap, kv_mode, lane_cell);
    }
#endif
#define STRATA_ATTN_LC(M) attn_chunk_kernel<M, true><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride)
    if (lane_cell) {
        if (kv_mode == 3) STRATA_ATTN_LC(3);
        else if (kv_mode == 2) STRATA_ATTN_LC(2);
        else if (kv_mode == 1) STRATA_ATTN_LC(1);
        else STRATA_ATTN_LC(0);
    } else
#undef STRATA_ATTN_LC
#if defined(STRATA_HIP_GFX906)
    // Opt-in INT8 batched attention: prefill, verification and MTP (including n_q=1).
    // Other pool formats and lane-cell scoring keep their paths; qsa_decode_attn_step is unchanged.
    // A k8v4 model's MTP drafter uses INT8 pools and can therefore take this path.
    if (use_query_swizzle) {
        // use_query_swizzle already requires INT8, warp scoring, and the calling
        // thread's current device to be exactly gfx906 with a 64-lane wavefront.
        static const bool reduce12 = [] {
            const char* v = std::getenv("STRATA_GFX906_ATTN_REDUCE12");
            return v && v[0] == '1' && v[1] == '\0';
        }();
        if (reduce12) {
            attn_chunk_kernel<1, false, true, true><<<grid, THREADS, 0, st>>>(q, pools, ids, steps,
                (int) s.n_head_kv, (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
            static const bool trace = [] {
                const char* v = std::getenv("STRATA_GFX906_ATTN_REDUCE12_TRACE");
                return v && v[0] == '1' && v[1] == '\0';
            }();
            static thread_local bool said = false;
            if (trace && !said) {
                said = true;
                std::fprintf(stderr, "strata attn-reduce12: selected=1 query_swizzle=1 kv_mode=1 nq=%lld cap=%lld\n",
                             (long long) n_q, (long long) cap);
            }
        } else {
            attn_chunk_kernel<1, false, true><<<grid, THREADS, 0, st>>>(q, pools, ids, steps,
                (int) s.n_head_kv, (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
        }
    } else
#endif
#if !defined(__HIPCC__)
    if (kv_mode == 1 && (kernel < 0 ? staged_attn() : kernel == 2)) {
        static bool carve[64] = {};
        int dev = 0;
        if (cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < 64 && !carve[dev]) {
            // a hint (the largest shared-memory carveout): two 40 KB blocks per SM
            cudaFuncSetAttribute(attn_chunk_staged_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
            cudaGetLastError();
            carve[dev] = true;
        }
        attn_chunk_staged_kernel<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                           scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    } else
    if (kernel < 0 && kv_mode == 1 && attn_v3_on())   // (the parity harness names its kernel: never the environment)
        attn_chunk_q8_v3_kernel<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size,
                                                          scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride);
    else
#endif
    if (kv_mode == 3)
        STRATA_ATTN_CHUNK_SEL(3, pre)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv,
                                                                (int) s.page_size, scale, part_acc, part_m, part_l,
                                                                n_chunks, (int) cap, stride);
    else if (kv_mode == 2)
        STRATA_ATTN_CHUNK_SEL(2, pre)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv,
                                                                (int) s.page_size, scale, part_acc, part_m, part_l,
                                                                n_chunks, (int) cap, stride);
    else if (kv_mode == 1)
        STRATA_ATTN_CHUNK_SEL(1, pre)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv,
                                                                (int) s.page_size, scale, part_acc, part_m, part_l,
                                                                n_chunks, (int) cap, stride);
    else
        STRATA_ATTN_CHUNK_SEL(0, pre)<<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv,
                                                                (int) s.page_size, scale, part_acc, part_m, part_l,
                                                                n_chunks, (int) cap, stride);
}

// qsa_decode_attn_batch's merge pass over the partials launch_chunks_batch wrote; `v2` names the merge kernel
void launch_merge_batch(const int32_t* steps, int64_t cap, const QsaShapes& s, const float* scratch, float* attn,
                        int64_t n_q, int v2, cudaStream_t st) {
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    const float* part_acc = scratch;
    const float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    const float* part_l = part_m + (size_t) n_chunks * s.n_head;
    launch_merge(v2, part_acc, part_m, part_l, n_chunks, steps, attn, stride, (unsigned) s.n_head, (unsigned) n_q, st);
}

}  // namespace

void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table || n_q > 65535) {
        std::fprintf(stderr, "qsa_decode_attn_batch: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    cudaStream_t st = (cudaStream_t) stream;
    launch_chunks_batch(q, pools, ids, steps, cap, s, scratch, n_q, st, -1);
    launch_merge_batch(steps, cap, s, scratch, attn, n_q, attn_merge_v2(), st);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn_batch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

bool qsa_decode_attn_batch_unrot(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                                 int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (attn_merge_v2() != 1) return false;   // attn_merge_kernel: the caller rotates after it
    if (n_q <= 0) return true;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table || n_q > 65535) {
        std::fprintf(stderr, "qsa_decode_attn_batch_unrot: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    cudaStream_t st = (cudaStream_t) stream;
    launch_chunks_batch(q, pools, ids, steps, cap, s, scratch, n_q, st, -1);
    launch_merge_batch(steps, cap, s, scratch, attn, n_q, 2, st);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn_batch_unrot: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

bool qsa_decode_attn_draft(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                           const QsaShapes& s, float* scratch, float* attn, void* stream) {
    return qsa_decode_attn_draft_gq(q, pools, ids, steps, cap, s, scratch, attn, nullptr, nullptr, nullptr, false, stream);
}

bool qsa_decode_attn_draft_gq(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                              const QsaShapes& s, float* scratch, float* attn, const float* q_full, float* gated, void* xq,
                              bool rot, void* stream) {
#if defined(__HIPCC__)
    (void) q; (void) pools; (void) ids; (void) steps; (void) cap; (void) s; (void) scratch; (void) attn; (void) stream;
    (void) q_full; (void) gated; (void) xq; (void) rot;
    return false;
#else
    // STRATA_MTP_ATTN_STAGED=0 keeps qsa_decode_attn_batch; STRATA_MTP_ATTN_GQ=0 keeps the gate and the quantizer apart
    static const bool on = [] { const char* v = std::getenv("STRATA_MTP_ATTN_STAGED"); return v == nullptr || std::atoi(v) != 0; }();
    static const bool gq_on = [] { const char* v = std::getenv("STRATA_MTP_ATTN_GQ"); return v == nullptr || std::atoi(v) != 0; }();
    const bool gq = q_full != nullptr;
    if (gq && (!gq_on || gated == nullptr || xq == nullptr || (rot && HD != 256))) return false;
    if (!on || attn_kv_mode(pools) != 1 || pools.k_scale == nullptr || pools.v_scale == nullptr) return false;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps ||
        !pools.page_table)
        return false;
    constexpr int smem = (int) sizeof(DraftAttnSmem);
    static bool done[64] = {};
    static int n_sm[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { (void) cudaGetLastError(); return false; }
    if (!done[dev]) {
        if (cudaFuncSetAttribute(attn_chunk_draft_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
        int sms = 0;
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) { (void) cudaGetLastError(); sms = 0; }
        n_sm[dev] = sms > 0 ? sms : 1;
        done[dev] = true;
    }
    cudaStream_t st = (cudaStream_t) stream;
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    // one block a chunk (the plain kernel's grid); STRATA_MTP_ATTN_GRID=N caps the blocks a KV head (each then loops over
    // chunks: one wave of 2 x 170 measured slower on struct, 13 vs 9 us, the chunks' latency is serial)
    static const int grid_env = [] { const char* v = std::getenv("STRATA_MTP_ATTN_GRID"); return v != nullptr ? std::atoi(v) : 0; }();
    int gx = grid_env > 0 ? grid_env : n_chunks;   // one block a chunk (a loop over chunks measured slower: 13 vs 9 us)
    gx = gx < 1 ? 1 : (gx > n_chunks ? n_chunks : gx);
    const int v2 = attn_merge_v2();
    attn_chunk_draft_kernel<<<dim3((unsigned) s.n_head_kv, (unsigned) gx), DT_THREADS, smem, st>>>(
        q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, v2 ? 1 : 0);
    if (gq)
        attn_merge_draft_gq_kernel<kMergeU><<<(unsigned) s.n_head, HD, 0, st>>>(part_acc, part_m, part_l, n_chunks, steps, attn,
                                                                              q_full, gated, (DgqQ81*) xq, rot ? 1 : 0);
    else
        launch_merge_batch(steps, cap, s, scratch, attn, 1, v2, st);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn_draft: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
#endif
}

bool qsa_decode_attn_slots(const float* q, const QsaAttnSlotPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return true;
    if (n_q > kQsaAttnMaxSlots) return false;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !steps) {
        std::fprintf(stderr, "qsa_decode_attn_slots: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = attn_kv_mode(pools.p[0]);
    for (int64_t z = 0; z < n_q; ++z) {
        if (!pools.p[z].page_table) {
            std::fprintf(stderr, "qsa_decode_attn_slots: missing page table\n");
            std::exit(1);
        }
        if (attn_kv_mode(pools.p[z]) != kv_mode) return false;
    }
    const bool lane_cell = lane_cell_attn();
    // only where qsa_decode_attn_batch runs attn_chunk_kernel; its other variants keep the per-query calls
#if defined(STRATA_ATTN_PRE75_BUILT)
    if (!lane_cell && pre75_attn()) return false;
#endif
#if defined(STRATA_HIP_GFX906)
    static const bool query_swizzle = [] {
        const char* v = std::getenv("STRATA_GFX906_ATTN_QUERY_SWIZZLE"); return v && v[0] == '1';
    }();
    if (query_swizzle && !lane_cell && kv_mode == 1 && query_swizzle_gfx906_device()) return false;
#endif
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q);
    cudaStream_t st = (cudaStream_t) stream;
#define STRATA_ATTN_SLOTS(M, LC) attn_chunk_slots_kernel<M, LC><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride)
    if (lane_cell) {
        if (kv_mode == 3) STRATA_ATTN_SLOTS(3, true);
        else if (kv_mode == 2) STRATA_ATTN_SLOTS(2, true);
        else if (kv_mode == 1) STRATA_ATTN_SLOTS(1, true);
        else STRATA_ATTN_SLOTS(0, true);
    } else {
        if (kv_mode == 3) STRATA_ATTN_SLOTS(3, false);
        else if (kv_mode == 2) STRATA_ATTN_SLOTS(2, false);
        else if (kv_mode == 1) STRATA_ATTN_SLOTS(1, false);
        else STRATA_ATTN_SLOTS(0, false);
    }
#undef STRATA_ATTN_SLOTS
    launch_merge(attn_merge_v2(), part_acc, part_m, part_l, n_chunks, steps, attn, stride, (unsigned) s.n_head,
                 (unsigned) n_q, st);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn_slots: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

void qsa_decode_attn_batch_rows(const float* q, const QsaAttnPoolsRows& rows, const int32_t* ids, const int32_t* steps,
                                int64_t cap, const QsaShapes& s, float* scratch, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    const QsaAttnPools& pools = rows.p[0];
    if (rows.n != n_q || n_q > kAttnRowsMax || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 ||
        !scratch || !ids || !steps) {
        std::fprintf(stderr, "qsa_decode_attn_batch_rows: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q);
    cudaStream_t st = (cudaStream_t) stream;
    // The plain kernel is what qsa_decode_attn_batch runs unless one of its variants is on (lane-cell scoring, the
    // pre-sm_75 / gfx103x kernel, the gfx906 swizzle): there each row goes through that call, one launch per row.
    static const bool lane_cell_rows = [] { const char* v = std::getenv("STRATA_ATTN_LANECELL"); return v && v[0] == '1'; }();
    bool per_row = lane_cell_rows;
#if defined(STRATA_ATTN_PRE75_BUILT)
    per_row = per_row || pre75_attn();
#endif
#if defined(STRATA_HIP_GFX906)
    per_row = true;
#endif
    if (per_row) {
        for (int64_t r = 0; r < n_q; ++r)
            qsa_decode_attn_batch(q + (size_t) r * s.n_head * HD, rows.p[r], ids + (size_t) r * cap,
                                  steps + (size_t) r * kStepCount, cap, s, scratch + (size_t) r * stride,
                                  attn + (size_t) r * s.n_head * HD, 1, stream);
        return;
    }
#define STRATA_ATTN_ROWS(M) attn_chunk_rows_kernel<M><<<grid, THREADS, 0, st>>>(q, rows, ids, steps, (int) s.n_head_kv, \
        (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride)
    if (kv_mode == 3) STRATA_ATTN_ROWS(3);
    else if (kv_mode == 2) STRATA_ATTN_ROWS(2);
    else if (kv_mode == 1) STRATA_ATTN_ROWS(1);
    else STRATA_ATTN_ROWS(0);
#undef STRATA_ATTN_ROWS
    launch_merge(attn_merge_v2(), part_acc, part_m, part_l, n_chunks, steps, attn, stride, (unsigned) s.n_head,
                 (unsigned) n_q, st);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn_batch_rows: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) s.n_head * (HD + 2) + 64;
}

void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream) {
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !ids || !step ||
        !pools.page_table) {
        std::fprintf(stderr, "qsa_decode_attn: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    if (kv_mode == 3 ? (!pools.k_scale || !pools.v_q4)
                     : (kv_mode == 2 ? (!pools.v_q4) : (kv_mode == 1 ? (!pools.v_q || !pools.k_scale || !pools.v_scale)
                                                                     : (!pools.k_pool || !pools.v_pool)))) {
        std::fprintf(stderr, "qsa_decode_attn: incomplete KV pools\n");
        std::exit(1);
    }
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv);
    cudaStream_t st = (cudaStream_t) stream;
#if !defined(__HIPCC__)
    if (kv_mode == 1 && attn_v3_on())
        attn_chunk_q8_v3_kernel<<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                          scale, part_acc, part_m, part_l, n_chunks, 0, 0);
    else
#endif
    if (kv_mode == 3)
        STRATA_ATTN_CHUNK(3)<<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, 0, 0);
    else if (kv_mode == 2)
        STRATA_ATTN_CHUNK(2)<<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, 0, 0);
    else if (kv_mode == 1)
        STRATA_ATTN_CHUNK(1)<<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, 0, 0);
    else
        STRATA_ATTN_CHUNK(0)<<<grid, THREADS, 0, st>>>(q, pools, ids, step, (int) s.n_head_kv, (int) s.page_size,
                                                        scale, part_acc, part_m, part_l, n_chunks, 0, 0);
    launch_merge(attn_merge_v2(), part_acc, part_m, part_l, n_chunks, step, attn, 0, (unsigned) s.n_head, 1u, st);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

// qsa_merge_daily_parity: qsa_decode_attn_batch's two passes apart, the chunk kernel and the merge named by the
// arguments (never the environment), so the Daily's kernels and upstream's can be run on the same inputs and the
// outputs compared bit for bit.  The engine never calls these.
void qsa_decode_attn_chunks_only(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                                 int64_t cap, const QsaShapes& s, float* scratch, int64_t n_q, int daily_chunk,
                                 void* stream) {
    if (n_q <= 0 || n_q > 65535 || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch ||
        !q || !ids || !steps || !pools.page_table || (daily_chunk < 0 || daily_chunk > 2)) {   // dec/int: 2 = attn_chunk_staged_kernel (INT8)
        std::fprintf(stderr, "qsa_decode_attn_chunks_only: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    launch_chunks_batch(q, pools, ids, steps, cap, s, scratch, n_q, (cudaStream_t) stream, daily_chunk);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn_chunks_only: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

void qsa_decode_attn_merge_only(const int32_t* steps, int64_t cap, const QsaShapes& s, const float* scratch,
                                float* attn, int64_t n_q, int v2, void* stream) {
    if (n_q <= 0 || n_q > 65535 || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch ||
        (v2 != 0 && v2 != 1 && v2 != 2) || !steps || !attn) {   // dec/int: 2 = v2 + the inverse rotation
        std::fprintf(stderr, "qsa_decode_attn_merge_only: unsupported geometry or missing buffers\n");
        std::exit(1);
    }
    launch_merge_batch(steps, cap, s, scratch, attn, n_q, v2, (cudaStream_t) stream);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn_merge_only: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
