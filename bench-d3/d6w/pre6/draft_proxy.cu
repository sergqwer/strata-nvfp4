// src/kernels/cuda/draft_proxy.cu - see include/strata/kernels/draft_proxy.hpp.
//
// The 2-bit copy's layout, its fit, the approximate scan, the candidate collection and the pick are Basalt's
// (src/models/qwen4exp/kernels/head_proxy.cuh, MIT, (c) 2026 Jesús Gulfo); here the scan's row count comes from device
// memory, and the exact rescoring reads either plain ggml Q8_0 rows (34-byte blocks) or the main head's bit planes
// (q80_vhead.hpp, STRATA_HEAD_PLANES, the default where those kernels run): the same codes and scales either way.
#include "strata/kernels/draft_proxy.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int kNw4 = 2560 / 64;   // uint4 of weight words per row
constexpr int kPer = 4;           // candidates kept per tile of 32 rows
constexpr int kMaxCand = 64;      // rows rescored exactly
constexpr float kDelta = 5.0f;    // a row is rescored when its approximate score is within this of the best
constexpr int BPR = 80;           // Q8_0 / q8_1 blocks of 32 in a row of 2560
constexpr int RB = BPR * 34;      // a Q8_0 row's bytes

// The main head's bit-plane rows (q80_vhead.cu's layout, Basalt q80p.cuh): 2720 bytes a row = 80 fp16 scales, then per
// block 16 bytes of bits 2..5 of u = q + 128, 8 bytes of bits 6..7 and 8 bytes of bits 0..1.  pl_words() rebuilds the
// two Q8_0 words of quarter p (values 8p..8p+3, 8p+4..8p+7) bit for bit, so either layout gives the same dot product.
constexpr int kPlNib = 160, kPlTop = 1440, kPlLow = 2080;
static_assert(RB == 2720 && kPlLow + BPR * 8 == RB, "plane layout");
__device__ __forceinline__ void pl_words(const uint8_t* __restrict__ row, int b, int p, int& v0, int& v1) {
    const uint32_t nib = __ldg(reinterpret_cast<const uint32_t*>(row + kPlNib + b * 16) + p);
    const uint32_t top = __ldg(reinterpret_cast<const uint32_t*>(row + kPlTop + b * 8) + (p >> 1));
    const uint32_t low = __ldg(reinterpret_cast<const uint32_t*>(row + kPlLow + b * 8) + (p >> 1));
    const int s = 4 * (p & 1);
    const uint32_t h0 = (nib & 0x0F0F0F0Fu) | (((top >> s) & 0x03030303u) << 4);
    const uint32_t h1 = ((nib >> 4) & 0x0F0F0F0Fu) | (((top >> (s + 2)) & 0x03030303u) << 4);
    v0 = (int) (((h0 << 2) | ((low >> s) & 0x03030303u)) ^ 0x80808080u);
    v1 = (int) (((h1 << 2) | ((low >> (s + 2)) & 0x03030303u)) ^ 0x80808080u);
}
__device__ __forceinline__ float pl_scale(const uint8_t* __restrict__ row, int b) {
    return __half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(row) + b)));
}

__host__ __device__ inline int64_t tiles(int64_t n_cap) { return (n_cap + 31) / 32; }
inline uint16_t* scales(void* proxy, int64_t n_cap) {
    return (uint16_t*) ((char*) proxy + (size_t) tiles(n_cap) * kNw4 * 32 * 16);
}

struct X81 {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(X81) == 36, "q8_1 block");

__device__ __forceinline__ int dp4a(int a, int b, int c) { return __dp4a(a, b, c); }

// The level (0..3) of weight w at quantization step sh.
__device__ __forceinline__ int level(float w, float sh) { return min(3, max(0, (int) floorf(w / sh + 2.0f))); }

// One 32-weight group's scale: four levels with the scale (of a few candidates) that leaves the least squared error,
// then multiplied by sqrt(|w|^2 / <w, w_hat>) (the least-squares copy scores the rows near the best low, which inflates
// the draft probability).
__device__ __forceinline__ __half fit_group(const float* w) {
    float amax = 1e-9f;
    for (int l = 0; l < 32; ++l) amax = fmaxf(amax, fabsf(w[l]));
    float best_s = amax / 1.5f, best_e = INFINITY;
    for (int t = 0; t < 6; ++t) {
        const float s = amax / 1.5f * (1.0f - 0.1f * (float) t);
        const float sh = __half2float(__float2half(s));
        float e = 0.0f;
        for (int l = 0; l < 32; ++l) {
            const float r = ((float) level(w[l], sh) - 1.5f) * sh - w[l];
            e += r * r;
        }
        if (e < best_e) { best_e = e; best_s = s; }
    }
    __half hs = __float2half(best_s);
    const float sh = __half2float(hs);
    float ww = 0.0f, wr = 0.0f;
    for (int l = 0; l < 32; ++l) {
        const float r = ((float) level(w[l], sh) - 1.5f) * sh;
        ww += w[l] * w[l];
        wr += w[l] * r;
    }
    if (wr > 0.0f) hs = __float2half(sh * sqrtf(ww / wr));
    return hs;
}

// One thread per (copy row, 32-weight group): the head row's Q8_0 block, fitted and packed.
__global__ void pack_kernel(const uint8_t* __restrict__ head, const int32_t* __restrict__ map, int64_t row0, int64_t n,
                            uint4* __restrict__ wq, uint16_t* __restrict__ sc, bool planes) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * BPR) return;
    const int64_t row = row0 + i / BPR;
    const int g = (int) (i % BPR);
    const int32_t hr = map[row];
    if (hr < 0) return;
    float w[32];
    if (planes) {
        const uint8_t* row = head + (size_t) hr * RB;
        const float d = pl_scale(row, g);
        for (int p = 0; p < 4; ++p) {
            int v0, v1;
            pl_words(row, g, p, v0, v1);
            for (int j = 0; j < 4; ++j) {
                w[8 * p + j] = d * (float) (int8_t) (v0 >> (8 * j));
                w[8 * p + 4 + j] = d * (float) (int8_t) (v1 >> (8 * j));
            }
        }
    } else {
        const uint8_t* b = head + (size_t) hr * RB + (size_t) g * 34;
        const float d = __half2float(__ushort_as_half((unsigned short) (b[0] | (b[1] << 8))));
        for (int l = 0; l < 32; ++l) w[l] = d * (float) (int8_t) b[2 + l];
    }
    // the copy's row is fitted at the level step the stored scale was chosen from (before the unshrink)
    float amax = 1e-9f;
    for (int l = 0; l < 32; ++l) amax = fmaxf(amax, fabsf(w[l]));
    float best_s = amax / 1.5f, best_e = INFINITY;
    for (int t = 0; t < 6; ++t) {
        const float s = amax / 1.5f * (1.0f - 0.1f * (float) t);
        const float sh = __half2float(__float2half(s));
        float e = 0.0f;
        for (int l = 0; l < 32; ++l) {
            const float r = ((float) level(w[l], sh) - 1.5f) * sh - w[l];
            e += r * r;
        }
        if (e < best_e) { best_e = e; best_s = s; }
    }
    const float sh = __half2float(__float2half(best_s));
    const __half hs = fit_group(w);
    const int tile = (int) (row >> 5), lane = (int) (row & 31), w4 = g / 2, h = g & 1;
    uint32_t words[2] = {0u, 0u};
    for (int c = 0; c < 2; ++c)
        for (int ii = 0; ii < 4; ++ii)
            for (int bb = 0; bb < 4; ++bb) words[c] |= (uint32_t) level(w[c * 16 + 4 * ii + bb], sh) << (8 * bb + 2 * ii);
    uint32_t* wd = reinterpret_cast<uint32_t*>(wq) + (((size_t) tile * kNw4 + w4) * 32 + lane) * 4 + 2 * h;
    wd[0] = words[0];
    wd[1] = words[1];
    sc[(((size_t) tile * kNw4 + w4) * 32 + lane) * 2 + h] = __half_as_ushort(hs);
}

// One tile per block, four warps splitting the row length; lane = row.  Writes the tile's best kPer approximate scores and
// the tile's softmax partial (max, sum of exp(score - max)); a tile past *n_rows writes empty ones.
// The 2-bit copy's loads: streamed past L2 (Basalt's __ldcs), or KEEP (STRATA_PROXY_L2=1): marked evict-last, so the copy can
// stay in the 96 MB L2 between the chain's steps.  The same values either way.
template <bool KEEP> __device__ __forceinline__ uint4 px_ld4(const uint4* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __ldcs(p);                             // no L2 cache policy before sm_80: the streamed load
#endif
    if (!KEEP) return __ldcs(p);
    uint4 v;
    uint64_t pol;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
#endif
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(pol));
#endif
    return v;
}
template <bool KEEP> __device__ __forceinline__ uint32_t px_ld1(const uint32_t* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    return __ldcs(p);
#else
    if (!KEEP) return __ldcs(p);
    uint32_t v;
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
    asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.u32 %0, [%1], %2;" : "=r"(v) : "l"(p), "l"(pol));
    return v;
#endif
}
template <bool KEEP>
__global__ void __launch_bounds__(128) score_kernel(const uint4* __restrict__ wq, const uint32_t* __restrict__ sc,
                                                   const X81* __restrict__ x, const int32_t* __restrict__ n_rows_dev,
                                                   const int32_t* __restrict__ map, float2* __restrict__ cand,
                                                   float2* __restrict__ part) {
    __shared__ __align__(16) int xi[640];
    __shared__ float xs[80], xdd[80];
    __shared__ float red[4][32];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, tile = blockIdx.x;
    const int n_rows = *n_rows_dev;
    const float ninf = __int_as_float(0xff800000);
    if (tile * 32 >= n_rows) {
        if (tid == 0) part[tile] = make_float2(ninf, 0.0f);
        if (tid < kPer) cand[tile * kPer + tid] = make_float2(ninf, __int_as_float(tile * 32));
        return;
    }
    const size_t base = ((size_t) tile * kNw4 + warp * 10) * 32 + lane;
    uint4 w[10];
    uint32_t s[10];
#pragma unroll
    for (int j = 0; j < 10; ++j) {
        w[j] = px_ld4<KEEP>(wq + base + (size_t) j * 32);
        s[j] = px_ld1<KEEP>(sc + base + (size_t) j * 32);
    }
    const int row = tile * 32 + lane;
    const int32_t id = (warp == 0 && row < n_rows) ? __ldg(map + row) : 0;
    for (int i = tid; i < 640; i += 128) xi[i] = reinterpret_cast<const int*>(x[i / 8].qs)[i % 8];
    if (tid < 80) {
        const int* q = reinterpret_cast<const int*>(x[tid].qs);
        int sum = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) sum = dp4a(0x01010101, q[i], sum);
        const float d8 = __low2float(x[tid].ds);
        xdd[tid] = d8;
        xs[tid] = d8 * (float) sum;
    }
    __syncthreads();
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < 10; ++j) {
        const int w4 = warp * 10 + j;
        const uint32_t ww[4] = {w[j].x, w[j].y, w[j].z, w[j].w};
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int g = 2 * w4 + h;
            int P = 0;
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const uint32_t v = ww[2 * h + c];
                const int4 x0 = *reinterpret_cast<const int4*>(&xi[g * 8 + c * 4]);
                P = dp4a((int) (v & 0x03030303u), x0.x, P);
                P = dp4a((int) ((v >> 2) & 0x03030303u), x0.y, P);
                P = dp4a((int) ((v >> 4) & 0x03030303u), x0.z, P);
                P = dp4a((int) ((v >> 6) & 0x03030303u), x0.w, P);
            }
            const float sg = __half2float(__ushort_as_half((unsigned short) (h ? (s[j] >> 16) : (s[j] & 0xffff))));
            acc += sg * (xdd[g] * (float) P - 1.5f * xs[g]);
        }
    }
    red[warp][lane] = acc;
    __syncthreads();
    if (warp != 0) return;
    float v = red[0][lane] + red[1][lane] + red[2][lane] + red[3][lane];
    if (row >= n_rows || id < 0) v = ninf;
    float m = v;
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float e = v > ninf ? __expf(v - m) : 0.0f;
#pragma unroll
    for (int o = 16; o; o >>= 1) e += __shfl_xor_sync(0xffffffffu, e, o);
    if (lane == 0) part[tile] = make_float2(m, e);
    for (int k = 0; k < kPer; ++k) {
        float bv = v;
        int bi = lane;
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) cand[tile * kPer + k] = make_float2(bv, __int_as_float(tile * 32 + bi));
        if (lane == bi) v = ninf;
    }
}

struct Final {
    float wmax[16];
    int list[kMaxCand];
    float lp[kMaxCand], ex[kMaxCand];
    int cnt;
    int hist[256];
    unsigned long long want;
    int need;
};

__device__ __forceinline__ unsigned long long cand_key(float2 c) {
    const unsigned u = __float_as_uint(c.x);
    const unsigned o = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((unsigned long long) o << 32) | (unsigned) (0xffffffffu - (unsigned) __float_as_int(c.y));
}

// The kMaxCand-th largest key among the candidates within kDelta of m: a radix select over the key's bytes.
__device__ __forceinline__ unsigned long long kth_key(Final& f, int nc, const float2* __restrict__ cand, float m) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float ninf = __int_as_float(0xff800000);
    if (tid == 0) { f.want = 0ull; f.need = kMaxCand; }
    unsigned long long mask = 0ull;
    for (int shift = 56; shift >= 0; shift -= 8) {
        if (tid < 256) f.hist[tid] = 0;
        __syncthreads();
        const unsigned long long want = f.want;
        for (int i = tid; i < nc; i += 512) {
            const float2 c = __ldcg(&cand[i]);
            if (!(c.x > ninf && c.x >= m - kDelta)) continue;
            const unsigned long long k = cand_key(c);
            if ((k & mask) == want) atomicAdd(&f.hist[(int) ((k >> shift) & 255u)], 1);
        }
        __syncthreads();
        if (warp == 0) {
            int b[8], s = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) { b[j] = f.hist[255 - 8 * lane - j]; s += b[j]; }
            int incl = s;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += v;
            }
            const int need = f.need, excl = incl - s;
            if (excl < need && incl >= need) {
                int acc = excl, j = 0;
                while (acc + b[j] < need) acc += b[j++];
                f.want = want | ((unsigned long long) (255 - 8 * lane - j) << shift);
                f.need = need - acc;
            }
        }
        mask |= 255ull << shift;
        __syncthreads();
    }
    return f.want;
}

// The rows within kDelta of the best approximate score into f.list (ascending rows; the kMaxCand best when more qualify).
__device__ __forceinline__ int collect(Final& f, int n_tiles, const int32_t* __restrict__ map,
                                       const float2* __restrict__ cand, int32_t* __restrict__ out_id,
                                       float* __restrict__ out_prob) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int nc = n_tiles * kPer;
    float m = __int_as_float(0xff800000);
    for (int i = tid; i < nc; i += 512) m = fmaxf(m, __ldcg(&cand[i].x));
#pragma unroll
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) f.wmax[warp] = m;
    __syncthreads();
    m = f.wmax[0];
    for (int i = 1; i < 16; ++i) m = fmaxf(m, f.wmax[i]);
    const float ninf = __int_as_float(0xff800000);
    if (m == ninf) {
        if (tid == 0) { *out_id = map[0] < 0 ? 0 : map[0]; *out_prob = 0.0f; }
        return 0;
    }
    for (int i = tid; i < nc; i += 512) {
        const float2 c = __ldcg(&cand[i]);
        if (c.x > ninf && c.x >= m - kDelta) {
            const int p = atomicAdd(&f.cnt, 1);
            if (p < kMaxCand) { f.list[p] = __float_as_int(c.y); f.lp[p] = c.x; }
        }
    }
    __syncthreads();
    const int total = f.cnt;
    if (total > kMaxCand) {
        const unsigned long long kth = kth_key(f, nc, cand, m);
        if (tid == 0) f.cnt = 0;
        __syncthreads();
        for (int i = tid; i < nc; i += 512) {
            const float2 c = __ldcg(&cand[i]);
            if (c.x > ninf && c.x >= m - kDelta && cand_key(c) >= kth) {
                const int p = atomicAdd(&f.cnt, 1);
                if (p < kMaxCand) { f.list[p] = __float_as_int(c.y); f.lp[p] = c.x; }
            }
        }
        __syncthreads();
    }
    const int n = total < kMaxCand ? total : kMaxCand;
    int row = 0, rank = 0;
    float lp = 0.0f;
    if (tid < n) {
        row = f.list[tid];
        lp = f.lp[tid];
        for (int j = 0; j < n; ++j) rank += f.list[j] < row;
    }
    __syncthreads();
    if (tid < n) { f.list[rank] = row; f.lp[rank] = lp; }
    __syncthreads();
    return n;
}

// A plain Q8_0 row (34-byte blocks) . the q8_1 column, by one warp: lane L takes quarter blocks L, L + 32, ... (8 weights each).
// PLANES: the same row held as bit planes (the same codes, scales and arithmetic order: the same bits).
template <bool PLANES>
__device__ __forceinline__ float q80_row_dot(const uint8_t* __restrict__ w, const X81* __restrict__ x, int lane) {
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < 10; ++c) {
        const int q = lane + 32 * c, b = q >> 2, p = q & 3;
        float d;
        int v0, v1;
        if constexpr (PLANES) {
            d = pl_scale(w, b);
            pl_words(w, b, p, v0, v1);
        } else {
            const uint8_t* blk = w + (size_t) b * 34;
            const unsigned short* h = reinterpret_cast<const unsigned short*>(blk);   // 2-byte aligned
            d = __half2float(__ushort_as_half(__ldg(h)));
            const unsigned short* qs = h + 1 + p * 4;
            v0 = (int) ((unsigned) __ldg(qs) | ((unsigned) __ldg(qs + 1) << 16));
            v1 = (int) ((unsigned) __ldg(qs + 2) | ((unsigned) __ldg(qs + 3) << 16));
        }
        const int* xq = reinterpret_cast<const int*>(x[b].qs) + 2 * p;
        int sumi = dp4a(v0, xq[0], 0);
        sumi = dp4a(v1, xq[1], sumi);
        s += d * __low2float(x[b].ds) * (float) sumi;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    return s;
}

template <bool PLANES>
__global__ void __launch_bounds__(512) final_kernel(const uint8_t* __restrict__ head, const X81* __restrict__ x,
                                                    int n_tiles, const int32_t* __restrict__ map,
                                                    const float2* __restrict__ cand, const float2* __restrict__ part,
                                                    int32_t* __restrict__ out_id, float* __restrict__ out_prob) {
    __shared__ X81 xs[BPR];
    __shared__ Final f;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid == 0) f.cnt = 0;
    for (int i = tid; i < BPR * 9; i += 512) reinterpret_cast<int*>(xs)[i] = reinterpret_cast<const int*>(x)[i];
    __syncthreads();
    const int n = collect(f, n_tiles, map, cand, out_id, out_prob);
    if (n == 0) return;
    for (int c = warp; c < n; c += 16) {
        const float y = q80_row_dot<PLANES>(head + (size_t) __ldg(map + f.list[c]) * RB, xs, lane);
        if (lane == 0) f.ex[c] = y;
    }
    __syncthreads();
    if (warp != 0) return;
    const float ninf = __int_as_float(0xff800000);
    float bv = ninf;
    int bi = 0x7fffffff;
    for (int c = lane; c < n; c += 32)
        if (f.ex[c] > bv || (f.ex[c] == bv && f.list[c] < bi)) { bv = f.ex[c]; bi = f.list[c]; }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    float sum = 0.0f;
    for (int t = lane; t < n_tiles; t += 32) {
        const float2 p = __ldcg(&part[t]);
        if (p.y > 0.0f) sum += p.y * __expf(p.x - bv);
    }
    for (int c = lane; c < n; c += 32) sum += __expf(f.ex[c] - bv) - __expf(f.lp[c] - bv);
#pragma unroll
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) {
        float one = 0.0f;
        for (int c = 0; c < n; ++c) if (f.list[c] == bi) one = 1.0f;
        sum = fmaxf(sum, one);
        *out_id = map[bi < 0x7fffffff ? bi : 0];
        *out_prob = 1.0f / sum;
    }
}

// final2_kernel (STRATA_PROXY_FINAL2, on by default): final_kernel's result bit for bit, cheaper.  Three changes, none to
// an operand or to the order of an operation:
//   - the tiles past the rows scanned this step (*n_rows_dev: the base tier's 41,037 rows of 59,475) are skipped: the
//     score kernel wrote them (-inf, 0) and empty candidates, which neither the best score, the list, the radix select nor
//     the softmax sum ever take (fmaxf with -inf, `c.x > ninf`, `p.y > 0`);
//   - the tiles' softmax partials are staged into shared memory by every thread at the start (warp 0 then sums them in
//     the same lane-strided order: one L2 round trip instead of ~40 dependent ones);
//   - a warp's rescored rows have their weight words loaded before any of them is reduced (q80_row_dot's per-row
//     arithmetic unchanged: the same chunk order, the same fma / add sequence, the same butterfly).
constexpr int kF2MaxTiles = 1920;   // 61,440 rows: the subset's 59,475 (+ the 512 context rows) fit; larger falls back
template <bool PLANES>
struct RowW {
    float d[10];
    int v0[10], v1[10];
};
template <bool PLANES>
__device__ __forceinline__ void row_load(const uint8_t* __restrict__ w, int lane, RowW<PLANES>& r) {
#pragma unroll
    for (int c = 0; c < 10; ++c) {
        const int q = lane + 32 * c, b = q >> 2, p = q & 3;
        if constexpr (PLANES) {
            r.d[c] = pl_scale(w, b);
            pl_words(w, b, p, r.v0[c], r.v1[c]);
        } else {
            const uint8_t* blk = w + (size_t) b * 34;
            const unsigned short* h = reinterpret_cast<const unsigned short*>(blk);
            r.d[c] = __half2float(__ushort_as_half(__ldg(h)));
            const unsigned short* qs = h + 1 + p * 4;
            r.v0[c] = (int) ((unsigned) __ldg(qs) | ((unsigned) __ldg(qs + 1) << 16));
            r.v1[c] = (int) ((unsigned) __ldg(qs + 2) | ((unsigned) __ldg(qs + 3) << 16));
        }
    }
}
template <bool PLANES>
__device__ __forceinline__ float row_dot(const RowW<PLANES>& r, const X81* __restrict__ x, int lane) {
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < 10; ++c) {
        const int q = lane + 32 * c, b = q >> 2, p = q & 3;
        const int* xq = reinterpret_cast<const int*>(x[b].qs) + 2 * p;
        int sumi = dp4a(r.v0[c], xq[0], 0);
        sumi = dp4a(r.v1[c], xq[1], sumi);
        s += r.d[c] * __low2float(x[b].ds) * (float) sumi;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    return s;
}

template <bool PLANES>
__global__ void __launch_bounds__(512) final2_kernel(const uint8_t* __restrict__ head, const X81* __restrict__ x,
                                                     int n_tiles_cap, const int32_t* __restrict__ n_rows_dev,
                                                     const int32_t* __restrict__ map, const float2* __restrict__ cand,
                                                     const float2* __restrict__ part, int32_t* __restrict__ out_id,
                                                     float* __restrict__ out_prob) {
    __shared__ X81 xs[BPR];
    __shared__ Final f;
    __shared__ float2 sp[kF2MaxTiles];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int nr = *n_rows_dev;
    const int nt_rows = (int) ((nr + 31) / 32);
    const int n_tiles = nt_rows < n_tiles_cap ? (nt_rows > 0 ? nt_rows : 1) : n_tiles_cap;
    if (tid == 0) f.cnt = 0;
    for (int i = tid; i < BPR * 9; i += 512) reinterpret_cast<int*>(xs)[i] = reinterpret_cast<const int*>(x)[i];
    for (int i = tid; i < n_tiles; i += 512) sp[i] = __ldcg(&part[i]);
    __syncthreads();
    const int n = collect(f, n_tiles, map, cand, out_id, out_prob);
    if (n == 0) return;
    {
        RowW<PLANES> ra, rb;
        int c = warp;
        for (; c + 16 < n; c += 32) {   // two rows in flight
            row_load<PLANES>(head + (size_t) __ldg(map + f.list[c]) * RB, lane, ra);
            row_load<PLANES>(head + (size_t) __ldg(map + f.list[c + 16]) * RB, lane, rb);
            const float ya = row_dot<PLANES>(ra, xs, lane);
            const float yb = row_dot<PLANES>(rb, xs, lane);
            if (lane == 0) { f.ex[c] = ya; f.ex[c + 16] = yb; }
        }
        if (c < n) {
            row_load<PLANES>(head + (size_t) __ldg(map + f.list[c]) * RB, lane, ra);
            const float ya = row_dot<PLANES>(ra, xs, lane);
            if (lane == 0) f.ex[c] = ya;
        }
    }
    __syncthreads();
    if (warp != 0) return;
    const float ninf = __int_as_float(0xff800000);
    float bv = ninf;
    int bi = 0x7fffffff;
    for (int c = lane; c < n; c += 32)
        if (f.ex[c] > bv || (f.ex[c] == bv && f.list[c] < bi)) { bv = f.ex[c]; bi = f.list[c]; }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    float sum = 0.0f;
    for (int t = lane; t < n_tiles; t += 32) {
        const float2 p = sp[t];
        if (p.y > 0.0f) sum += p.y * __expf(p.x - bv);
    }
    for (int c = lane; c < n; c += 32) sum += __expf(f.ex[c] - bv) - __expf(f.lp[c] - bv);
#pragma unroll
    for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) {
        float one = 0.0f;
        for (int c = 0; c < n; ++c) if (f.list[c] == bi) one = 1.0f;
        sum = fmaxf(sum, one);
        *out_id = map[bi < 0x7fffffff ? bi : 0];
        *out_prob = 1.0f / sum;
    }
}

int& final2_flag() {
    static int v = [] { const char* e = std::getenv("STRATA_PROXY_FINAL2"); return e == nullptr || std::atoi(e) != 0 ? 1 : 0; }();
    return v;
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "strata: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

void draft_proxy_final2(bool on) { final2_flag() = on ? 1 : 0; }

size_t draft_proxy_bytes(int64_t n_cap) { return (size_t) tiles(n_cap) * kNw4 * 32 * (16 + 4); }
size_t draft_proxy_scratch_bytes(int64_t n_cap) { return (size_t) tiles(n_cap) * (kPer + 1) * sizeof(float2); }

void draft_proxy_pack(const void* q80_head, bool planes, const int32_t* map, int64_t row0, int64_t n, void* proxy,
                      int64_t n_cap, cudaStream_t stream) {
    if (n <= 0) return;
    const int64_t total = n * BPR;
    pack_kernel<<<(unsigned) ((total + 127) / 128), 128, 0, stream>>>((const uint8_t*) q80_head, map, row0, n, (uint4*) proxy,
                                                                       scales(proxy, n_cap), planes);
    check("draft_proxy_pack");
}

void draft_proxy_top1(const void* proxy, const void* q80_head, bool planes, const void* x_q8_1, const int32_t* n_rows_dev, int64_t n_cap,
                      const int32_t* map, int32_t* out_id, float* out_prob, void* scratch, cudaStream_t stream) {
    const int nt = (int) tiles(n_cap);
    float2* cand = (float2*) scratch;
    float2* part = cand + (size_t) nt * kPer;
    static const bool keep = [] { const char* v = std::getenv("STRATA_PROXY_L2"); return v != nullptr && std::atoi(v) != 0; }();
    if (keep)
        score_kernel<true><<<nt, 128, 0, stream>>>((const uint4*) proxy, (const uint32_t*) scales(const_cast<void*>(proxy), n_cap),
                                                   (const X81*) x_q8_1, n_rows_dev, map, cand, part);
    else
        score_kernel<false><<<nt, 128, 0, stream>>>((const uint4*) proxy, (const uint32_t*) scales(const_cast<void*>(proxy), n_cap),
                                                    (const X81*) x_q8_1, n_rows_dev, map, cand, part);
    // STRATA_PROXY_FINAL2=0 keeps final_kernel (final2_kernel gives its bits; draft_proxy_parity checks both)
    if (final2_flag() && nt <= kF2MaxTiles) {
        if (planes)
            final2_kernel<true><<<1, 512, 0, stream>>>((const uint8_t*) q80_head, (const X81*) x_q8_1, nt, n_rows_dev, map, cand,
                                                       part, out_id, out_prob);
        else
            final2_kernel<false><<<1, 512, 0, stream>>>((const uint8_t*) q80_head, (const X81*) x_q8_1, nt, n_rows_dev, map, cand,
                                                        part, out_id, out_prob);
    } else if (planes)
        final_kernel<true><<<1, 512, 0, stream>>>((const uint8_t*) q80_head, (const X81*) x_q8_1, nt, map, cand, part, out_id,
                                                  out_prob);
    else
        final_kernel<false><<<1, 512, 0, stream>>>((const uint8_t*) q80_head, (const X81*) x_q8_1, nt, map, cand, part, out_id,
                                                   out_prob);
    check("draft_proxy_top1");
}

}  // namespace strata::kernels
