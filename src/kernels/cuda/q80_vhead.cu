// src/kernels/cuda/q80_vhead.cu - see include/strata/kernels/q80_vhead.hpp.
//
// Ported from Basalt (github fork of Strata, MIT licence): src/models/qwen4exp/kernels/q80_vhead.cu and q80p.cuh, the
// bit-plane layout, the exact pass and the screen; the two-card split (link, post, fetch, list, merge) is left out.
// Built with --use_fast_math like native_mmvq.cu, whose Q8_0 x q8_1 dot the exact rows repeat.
#include "strata/kernels/q80_vhead.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

#if !defined(__HIPCC__)
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/pdl.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <mutex>

namespace strata::kernels {
namespace {

// ---- the layout (Basalt q80p.cuh)
// A row of 80 blocks (n_in 2560) keeps its 2720 bytes: the 80 fp16 scales, then per block 16 bytes of bits 2..5 of
// u = q + 128 (word p, quarter p of the block: byte j holds the nibbles of values 8p + j and 8p + 4 + j), 8 bytes of
// bits 6..7 and 8 bytes of bits 0..1 (word h, quarters 2h and 2h + 1: byte j holds the 2-bit fields of values
// 16h + j, 16h + 4 + j, 16h + 8 + j, 16h + 12 + j from its low bits up).
namespace q80p {
constexpr int kBlocks = 80;
constexpr int kRowBytes = 2720;
constexpr int kScales = 0, kNib = 160, kTop = 1440, kLow = 2080;

// The screen's codes of quarter p: u >> 2 for values 8p..8p+3 (h0) and 8p+4..8p+7 (h1), one per byte (0..63).
__device__ __forceinline__ void hi6(uint32_t nib, uint32_t top, int p, uint32_t& h0, uint32_t& h1) {
    const int s = 4 * (p & 1);
    h0 = (nib & 0x0F0F0F0Fu) | (((top >> s) & 0x03030303u) << 4);
    h1 = ((nib >> 4) & 0x0F0F0F0Fu) | (((top >> (s + 2)) & 0x03030303u) << 4);
}

// The Q8_0 words of quarter p (values 8p..8p+3 and 8p+4..8p+7), as the Q8_0 row holds them.
__device__ __forceinline__ void codes(uint32_t nib, uint32_t top, uint32_t low, int p, int& v0, int& v1) {
    uint32_t h0, h1;
    hi6(nib, top, p, h0, h1);
    const int s = 4 * (p & 1);
    v0 = (int) (((h0 << 2) | ((low >> s) & 0x03030303u)) ^ 0x80808080u);
    v1 = (int) (((h1 << 2) | ((low >> (s + 2)) & 0x03030303u)) ^ 0x80808080u);
}

struct Quarter {
    uint32_t nib, top, low;
    float d;
};
__device__ __forceinline__ Quarter load_quarter(const uint8_t* __restrict__ row, int kb, int p) {
    const uint32_t* n = reinterpret_cast<const uint32_t*>(row + kNib + kb * 16) + p;
    const uint32_t* t = reinterpret_cast<const uint32_t*>(row + kTop + kb * 8) + (p >> 1);
    const uint32_t* l = reinterpret_cast<const uint32_t*>(row + kLow + kb * 8) + (p >> 1);
    const half* d = reinterpret_cast<const half*>(row + kScales) + kb;
    Quarter q;
    q.nib = __ldg(n); q.top = __ldg(t); q.low = __ldg(l); q.d = __half2float(__ldg(d));
    return q;
}

// The 32 codes of block b of a plane row, as int8.
__device__ __forceinline__ void block_codes(const uint8_t* __restrict__ row, int b, int8_t (&q)[32], float& d) {
    d = __half2float(reinterpret_cast<const half*>(row + kScales)[b]);
    for (int p = 0; p < 4; ++p) {
        const uint32_t nib = reinterpret_cast<const uint32_t*>(row + kNib + b * 16)[p];
        const uint32_t top = reinterpret_cast<const uint32_t*>(row + kTop + b * 8)[p >> 1];
        const uint32_t low = reinterpret_cast<const uint32_t*>(row + kLow + b * 8)[p >> 1];
        int v0, v1;
        codes(nib, top, low, p, v0, v1);
        for (int j = 0; j < 4; ++j) {
            q[8 * p + j] = (int8_t) (v0 >> (8 * j));
            q[8 * p + 4 + j] = (int8_t) (v1 >> (8 * j));
        }
    }
}
}  // namespace q80p

using q80p::kBlocks;
using q80p::kRowBytes;
constexpr int WARP = 32, WPB = 8;
constexpr int VHIST = 1024;        // bins per column of the block-maximum histogram
constexpr float VBIN = 8.0f;       // bins per logit unit, centred on zero
constexpr int kListCap = 1024;     // rows per block of the exact pass
constexpr int kMaxCols = 8;

struct Q80Block {
    half d;
    int8_t qs[32];
};
struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q80Block) == 34 && sizeof(Q81Block) == 36, "the GGML block layouts");

struct VScratch {   // the screen's last block clears the histogram and the counter for the next call
    unsigned hist[kMaxCols][VHIST];
    float thr[kMaxCols];
    unsigned done;
    unsigned pad_;
    unsigned long long stat_calls, stat_rows;   // never cleared: q80p_vhead_stats
};

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = WARP / 2; offset > 0; offset >>= 1) x += __shfl_xor_sync(0xffffffff, x, offset, WARP);
    return x;
}

// The warp's sums of v[0..M) (M a power of two, 2..16), scattered: lane L ends with the sum of v[(L / (32 / M)) % M].
template <int M>
__device__ __forceinline__ float sum_scatter(float (&v)[M], int lane) {
#pragma unroll
    for (int h = M / 2; h >= 1; h /= 2) {
        const int mask = 32 * h / M;
        const bool up = (lane & mask) != 0;
#pragma unroll
        for (int i = 0; i < h; ++i) {
            const float send = up ? v[i] : v[i + h];
            v[i] = (up ? v[i + h] : v[i]) + __shfl_xor_sync(0xffffffff, send, mask);
        }
    }
    float r = v[0];
#pragma unroll
    for (int mask = 16 / M; mask >= 1; mask /= 2) r += __shfl_xor_sync(0xffffffff, r, mask);
    return r;
}

// ---- the layout: one block of 80 threads per row, the row read whole, then written back in the other layout
__global__ void __launch_bounds__(kBlocks) q80p_pack_kernel(uint8_t* __restrict__ w) {
    uint8_t* row = w + (size_t) blockIdx.x * kRowBytes;
    const int b = threadIdx.x;
    const Q80Block* src = reinterpret_cast<const Q80Block*>(row) + b;
    const uint16_t d = *reinterpret_cast<const uint16_t*>(&src->d);
    uint8_t u[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) u[i] = (uint8_t) src->qs[i] ^ 0x80u;
    __syncthreads();
    uint32_t nib[4], top[2], low[2];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        uint32_t x = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            x |= ((uint32_t) ((u[8 * p + j] >> 2) & 15u) | ((uint32_t) ((u[8 * p + 4 + j] >> 2) & 15u) << 4)) << (8 * j);
        nib[p] = x;
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        uint32_t t = 0, l = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t v = u[16 * h + 4 * k + j];
                t |= (v >> 6) << (8 * j + 2 * k);
                l |= (v & 3u) << (8 * j + 2 * k);
            }
        top[h] = t;
        low[h] = l;
    }
    reinterpret_cast<uint16_t*>(row + q80p::kScales)[b] = d;
    reinterpret_cast<uint4*>(row + q80p::kNib)[b] = make_uint4(nib[0], nib[1], nib[2], nib[3]);
    reinterpret_cast<uint2*>(row + q80p::kTop)[b] = make_uint2(top[0], top[1]);
    reinterpret_cast<uint2*>(row + q80p::kLow)[b] = make_uint2(low[0], low[1]);
}

__global__ void __launch_bounds__(kBlocks) q80p_unpack_kernel(uint8_t* __restrict__ w) {
    uint8_t* row = w + (size_t) blockIdx.x * kRowBytes;
    const int b = threadIdx.x;
    int8_t q[32];
    float d;
    q80p::block_codes(row, b, q, d);
    const uint16_t dh = reinterpret_cast<const uint16_t*>(row + q80p::kScales)[b];
    __syncthreads();
    Q80Block* dst = reinterpret_cast<Q80Block*>(row) + b;
    *reinterpret_cast<uint16_t*>(&dst->d) = dh;
#pragma unroll
    for (int i = 0; i < 32; ++i) dst->qs[i] = q[i];
}

// ---- exact rows, native_mmvq's arithmetic
// native_mmvq's Q8_0 kernels run a row on four warps: thread tid takes blocks kbx = tid / 4, += 32, words
// kqs = 2 (tid % 4), kqs + 1; each thread's terms d_w * d_x * sumi are added in kbx order, warps 1..3 are added to
// warp 0 in that order, then the xor tree 16..1.  Lane L stands for its threads L, L + 32, L + 64, L + 96 (virtual
// threads 0..3: three blocks for 0 and 1, two for 2 and 3).
constexpr int kNch = 10;
__device__ __forceinline__ int kb_of(int lane, int c) {
    constexpr int off[kNch] = {0, 32, 64, 8, 40, 72, 16, 48, 24, 56};
    return lane / 4 + off[c];
}
__device__ __forceinline__ void row_load(const uint8_t* __restrict__ row, int lane, q80p::Quarter (&r)[kNch]) {
#pragma unroll
    for (int c = 0; c < kNch; ++c) r[c] = q80p::load_quarter(row, kb_of(lane, c), lane % 4);
}
// xe: per column j, q8_1 block b and word pair p, {q[2p], q[2p + 1], the block's scale as float, 0}
template <int NC>
__device__ __forceinline__ void row_dot(const q80p::Quarter (&r)[kNch], const int4* __restrict__ xe, int lane,
                                        float (&out)[NC]) {
    constexpr int vt[kNch] = {0, 0, 0, 1, 1, 1, 2, 2, 3, 3};
    const int p = lane % 4;
    float s[4][NC];
#pragma unroll
    for (int v = 0; v < 4; ++v)
#pragma unroll
        for (int j = 0; j < NC; ++j) s[v][j] = 0.0f;
#pragma unroll
    for (int c = 0; c < kNch; ++c) {
        int v0, v1;
        q80p::codes(r[c].nib, r[c].top, r[c].low, p, v0, v1);
        const int kb = kb_of(lane, c);
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            const int4 e = xe[((size_t) j * kBlocks + kb) * 4 + p];
            int sumi = 0;
            sumi = STRATA_DP4A(v0, e.x, sumi);
            sumi = STRATA_DP4A(v1, e.y, sumi);
            s[vt[c]][j] += r[c].d * __int_as_float(e.z) * float(sumi);
        }
    }
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        float t = s[0][j];
#pragma unroll
        for (int v = 1; v < 4; ++v) t += s[v][j];
        out[j] = warp_sum(t);
    }
}

__device__ __forceinline__ void load_xe(const Q81Block* x, int ncols, int4* __restrict__ xe) {
    for (int i = threadIdx.x; i < ncols * kBlocks * 4; i += blockDim.x) {
        const Q81Block* blk = x + i / 4;
        const int* q8 = reinterpret_cast<const int*>(blk->qs) + 2 * (i % 4);
        const unsigned dsb = *reinterpret_cast<const unsigned*>(&blk->ds);
        xe[i] = make_int4(q8[0], q8[1], __float_as_int(__low2float(*reinterpret_cast<const half2*>(&dsb))), 0);
    }
}

// The exact pass.  Every row and column (k null or 0): one resident wave walks the rows, a warp per row, the next row
// loaded while this one is computed.  Otherwise a slice of rows per block: the (row, column) pairs whose upper bound in
// y reaches the column's threshold, a warp per listed row.
template <int NC>
__global__ void __launch_bounds__(WPB * WARP) q80p_rows_kernel(const uint8_t* __restrict__ w, const Q81Block* x,
                                                               float* y, int n_out, int per_block, const int* k,
                                                               VScratch* vs) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750   // multiarch: Turing too (the one sm_80 instruction, redux, has a shuffle twin below)
    extern __shared__ __align__(16) int4 xe[];
    __shared__ uint32_t list[kListCap];
    __shared__ int cnt;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int r0 = blockIdx.x * per_block, r1 = min(n_out, r0 + per_block);
    q80p::Quarter cur[kNch];
    // the list length is written ahead of the window (a memset on the stream before the graph launch), never by the
    // kernel this one may overlap, so it is read before the wait
    const int keep = k != nullptr ? *k : 0;
    if (keep == 0) {
        const int row0 = blockIdx.x * WPB + warp, stride = gridDim.x * WPB;
        if (row0 < n_out) row_load(w + (size_t) row0 * kRowBytes, lane, cur);   // weights: before the wait
        pdl_wait();
        load_xe(x, NC, xe);
        __syncthreads();
        for (int row = row0; row < n_out; row += stride) {
            q80p::Quarter nxt[kNch];
            if (row + stride < n_out) row_load(w + (size_t) (row + stride) * kRowBytes, lane, nxt);
            float v[NC];
            row_dot<NC>(cur, xe, lane, v);
#pragma unroll
            for (int j = 0; j < NC; ++j)
                if (lane == j) y[(size_t) j * n_out + row] = v[j];
#pragma unroll
            for (int c = 0; c < kNch; ++c) cur[c] = nxt[c];
        }
        return;
    }
    pdl_wait();
    load_xe(x, NC, xe);
    if (tid == 0) cnt = 0;
    __syncthreads();
    {
        float thr[NC];
#pragma unroll
        for (int j = 0; j < NC; ++j) thr[j] = __ldcg(&vs->thr[j]);
        for (int i = tid; i < r1 - r0; i += blockDim.x) {
            uint32_t m = 0u;
#pragma unroll
            for (int j = 0; j < NC; ++j)
                if (!(__ldcg(y + (size_t) j * n_out + r0 + i) < thr[j])) m |= 1u << j;   // NaN: scored
            if (m != 0u) list[atomicAdd(&cnt, 1)] = (uint32_t) i | m << 16;
        }
    }
    __syncthreads();
    const int n = cnt;
    if (tid == 0 && n > 0) atomicAdd(&vs->stat_rows, (unsigned long long) n);
    if (warp < n) row_load(w + (size_t) (r0 + (int) (list[warp] & 0xffffu)) * kRowBytes, lane, cur);
    for (int i = warp; i < n; i += WPB) {
        q80p::Quarter nxt[kNch];
        if (i + WPB < n) row_load(w + (size_t) (r0 + (int) (list[i + WPB] & 0xffffu)) * kRowBytes, lane, nxt);
        float v[NC];
        row_dot<NC>(cur, xe, lane, v);
        const uint32_t e = list[i];
        const int r = r0 + (int) (e & 0xffffu);
#pragma unroll
        for (int j = 0; j < NC; ++j)
            if (lane == j && ((e >> (16 + j)) & 1u)) y[(size_t) j * n_out + r] = v[j];
#pragma unroll
        for (int c = 0; c < kNch; ++c) cur[c] = nxt[c];
    }
#else
    __trap();
#endif
}

// ---- the screen.  Lane L takes blocks L, L + 32 and (L < 16) L + 64 of a row: their scales and bits 2..7.
struct ScreenRow {
    uint4 nib[3];
    uint2 top[3];
    float d[3];
};
__device__ __forceinline__ void screen_load(const uint8_t* __restrict__ row, int lane, ScreenRow& r) {
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int b = lane + 32 * i;
        if (b < kBlocks) {
            r.nib[i] = __ldcs(reinterpret_cast<const uint4*>(row + q80p::kNib) + b);
            r.top[i] = __ldcs(reinterpret_cast<const uint2*>(row + q80p::kTop) + b);
            r.d[i] = __half2float(__ldcs(reinterpret_cast<const half*>(row + q80p::kScales) + b));
        }
    }
}

// Every row's upper bound into y; each block adds its per-column maximum lower bound to the histogram, and the last
// block turns the histogram into the columns' thresholds: the lower edge of the bin holding the K-th largest block
// maximum, at or below the K-th largest lower bound and so at or below the K-th largest exact logit.  Blocks hold
// disjoint rows, so their maxima are K distinct rows' bounds.
template <int NC>
__global__ void __launch_bounds__(WPB * WARP) q80p_screen_kernel(const uint8_t* __restrict__ w, const Q81Block* x,
                                                                 float* y, int n_out, const int* k, VScratch* vs) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750   // multiarch: Turing too (the one sm_80 instruction, redux, has a shuffle twin below)
    __shared__ __align__(16) int4 sy[NC][2][kBlocks];   // a column's codes per block, in two halves
    __shared__ __align__(16) float4 sc[NC][kBlocks];    // {4 d, -31.625 sum q, 1.5 d sum |q|, 0}
    __shared__ float wmax[WPB][NC];
    __shared__ int bad[NC];
    __shared__ bool last;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int row0 = blockIdx.x * WPB + warp, stride = gridDim.x * WPB;
    ScreenRow cur;
    const int keep = *k;   // written ahead of the window (see q80p_rows_kernel)
    if (keep != 0 && row0 < n_out) screen_load(w + (size_t) row0 * kRowBytes, lane, cur);   // weights: before the wait
    pdl_wait();
    if (keep == 0) return;   // the exact pass takes every row
    if (tid < NC) bad[tid] = 0;
    __syncthreads();
    for (int i = tid; i < NC * kBlocks; i += blockDim.x) {
        const int j = i / kBlocks, b = i % kBlocks;
        const Q81Block* blk = x + (size_t) j * kBlocks + b;
        const int* q = reinterpret_cast<const int*>(blk->qs);
        int v[8], sum = 0;
        unsigned sabs = 0u;
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            v[t] = q[t];
            sum = STRATA_DP4A(v[t], 0x01010101, sum);
            sabs = __dp4a((unsigned) __vabs4((unsigned) v[t]), 0x01010101u, sabs);   // |-128| counts 128
        }
        const unsigned dsb = *reinterpret_cast<const unsigned*>(&blk->ds);
        const float d = __low2float(*reinterpret_cast<const half2*>(&dsb));
        sy[j][0][b] = make_int4(v[0], v[1], v[2], v[3]);
        sy[j][1][b] = make_int4(v[4], v[5], v[6], v[7]);
        sc[j][b] = make_float4(4.0f * d, -31.625f * (float) sum, 1.5f * d * (float) sabs, 0.0f);
        if ((__float_as_uint(d) & 0x7f800000u) == 0x7f800000u) bad[j] = 1;   // not finite (a bit test: fast math)
    }
    __syncthreads();
    float lmax[NC];
#pragma unroll
    for (int j = 0; j < NC; ++j) lmax[j] = __int_as_float(0xff800000);
    for (int row = row0; row < n_out; row += stride) {
        const int nrow = row + stride;
        ScreenRow nxt;
        if (nrow < n_out) screen_load(w + (size_t) nrow * kRowBytes, lane, nxt);
        constexpr int P = NC <= 1 ? 1 : NC <= 2 ? 2 : NC <= 4 ? 4 : 8;   // the columns, padded to a power of two
        float v[2 * P];   // c of each column, then its half-width
#pragma unroll
        for (int j = 0; j < 2 * P; ++j) v[j] = 0.0f;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const int b = lane + 32 * i;
            if (b < kBlocks) {
                const uint32_t nb[4] = {cur.nib[i].x, cur.nib[i].y, cur.nib[i].z, cur.nib[i].w};
                const uint32_t tp[2] = {cur.top[i].x, cur.top[i].y};
                uint32_t h[8];
#pragma unroll
                for (int p = 0; p < 4; ++p) q80p::hi6(nb[p], tp[p >> 1], p, h[2 * p], h[2 * p + 1]);
#pragma unroll
                for (int j = 0; j < NC; ++j) {
                    const int4 a = sy[j][0][b], e = sy[j][1][b];
                    int u = 0;
                    u = STRATA_DP4A((int) h[0], a.x, u);
                    u = STRATA_DP4A((int) h[1], a.y, u);
                    u = STRATA_DP4A((int) h[2], a.z, u);
                    u = STRATA_DP4A((int) h[3], a.w, u);
                    u = STRATA_DP4A((int) h[4], e.x, u);
                    u = STRATA_DP4A((int) h[5], e.y, u);
                    u = STRATA_DP4A((int) h[6], e.z, u);
                    u = STRATA_DP4A((int) h[7], e.w, u);
                    const float4 f = sc[j][b];
                    v[j] = fmaf(cur.d[i] * f.x, (float) u + f.y, v[j]);   // (float) u + f.y is exact
                    v[P + j] = fmaf(cur.d[i], f.z, v[P + j]);
                }
            }
        }
        // lane L holds c of column (L / (16 / P)) % P when L < 16, and lane L + 16 its half-width
        const float cs = sum_scatter<2 * P>(v, lane);
        const float ws = __shfl_xor_sync(0xffffffff, cs, 16);
        const int col = (lane / (16 / P)) % P;
        const float m = fmaf(ws, 1.0625f, 1e-6f);   // the rounding of both sides is far inside the 1/16 margin
        if (lane < 16 && lane % (16 / P) == 0 && col < NC) y[(size_t) col * n_out + row] = cs + m;
#pragma unroll
        for (int j = 0; j < NC; ++j)
            if (col == j && lane < 16) lmax[j] = fmaxf(lmax[j], cs - m);
        cur = nxt;
    }
#pragma unroll
    for (int j = 0; j < NC; ++j) {   // the lanes holding column j share its maximum
        float m = lmax[j];
#pragma unroll
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        if (lane == 0) wmax[warp][j] = m;
    }
    __syncthreads();
    if (tid < NC && !bad[tid]) {
        float m = wmax[0][tid];
#pragma unroll
        for (int v = 1; v < WPB; ++v) m = fmaxf(m, wmax[v][tid]);
        if (m > __int_as_float(0xff800000)) {
            m = fminf(fmaxf(m, -1000.0f), 1000.0f);
            const int bin = min(VHIST - 1, max(0, (int) floorf(m * VBIN) + VHIST / 2));
            atomicAdd(&vs->hist[tid][bin], 1u);
        }
    }
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        last = atomicAdd(&vs->done, 1u) == gridDim.x - 1;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    const unsigned want = (unsigned) min(max(keep, 1), kQ80pKeepMax);
    for (int c = warp; c < NC; c += WPB) {
        unsigned v[VHIST / 32], tot[VHIST / 32];   // lane l of row i: bin 32 i + l
#pragma unroll
        for (int i = 0; i < VHIST / 32; ++i) v[i] = __ldcg(&vs->hist[c][i * 32 + lane]);
#pragma unroll
        for (int i = 0; i < VHIST / 32; ++i) {
#if __CUDA_ARCH__ >= 800
            tot[i] = __reduce_add_sync(0xffffffffu, v[i]);
#else   // Turing: the same integer sum by shuffles
            unsigned s = v[i];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
            tot[i] = s;
#endif
        }
        unsigned run = 0, row_v = 0;
        int rw = -1;
#pragma unroll
        for (int i = VHIST / 32 - 1; i >= 0; --i)
            if (rw < 0) {
                if (run + tot[i] >= want) { rw = i; row_v = v[i]; }
                else run += tot[i];
            }
        int bin = -1;
        if (rw >= 0) {
            unsigned suf = row_v;   // this bin's count and those of the bins above it in the row
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const unsigned u = __shfl_down_sync(0xffffffffu, suf, o);
                if (lane + o < 32) suf += u;
            }
            const unsigned mm = __ballot_sync(0xffffffffu, run + suf >= want);
            bin = rw * 32 + 31 - __clz((int) mm);
        }
        if (lane == 0)   // fewer than K maxima, the lowest bin, or a column without finite scales: every row
            vs->thr[c] = bin > 0 && !bad[c] ? (float) (bin - VHIST / 2) / VBIN : __int_as_float(0xff800000);
        for (int i = lane; i < VHIST; i += 32) vs->hist[c][i] = 0u;
    }
    if (tid == 0) {
        vs->done = 0u;
        vs->stat_calls += 1ull;
    }
#else
    __trap();
#endif
}

// ---- launches: one resident wave of each kernel (the screen's for eight columns at every column count, so its
// blocks and thresholds do not depend on the window's width; the exact pass's for its own column count)
template <int NC>
void grids(int& screen, int& rows) {
    static std::atomic<bool> set[64];
    static int g[64][2];
    static std::mutex mu;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { (void) cudaGetLastError(); dev = 0; }
    if (!set[dev].load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lk(mu);
        if (!set[dev].load(std::memory_order_relaxed)) {
            int sms = 0, a = 0, b = 0;
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&a, q80p_screen_kernel<8>, WPB * WARP, 0);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, q80p_rows_kernel<NC>, WPB * WARP,
                                                          (size_t) NC * kBlocks * 4 * sizeof(int4));
            g[dev][0] = (sms > 0 ? sms : 1) * (a > 0 ? a : 1);
            g[dev][1] = (sms > 0 ? sms : 1) * (b > 0 ? b : 1);
            set[dev].store(true, std::memory_order_release);
        }
    }
    screen = g[dev][0];
    rows = g[dev][1];
}

template <int NC>
void launch(const void* w, const void* x, float* y, int n_out, const int* k, VScratch* vs, cudaStream_t s) {
    int screen = 0, grid = 0;
    grids<NC>(screen, grid);
    if (k != nullptr)
        launch_pdl(q80p_screen_kernel<NC>, dim3((unsigned) screen), dim3(WPB * WARP), 0, s, (const uint8_t*) w,
                   (const Q81Block*) x, y, n_out, k, vs);
    if ((long long) grid * kListCap < n_out) grid = (n_out + kListCap - 1) / kListCap;
    const int per_block = (n_out + grid - 1) / grid;
    grid = (n_out + per_block - 1) / per_block;
    launch_pdl(q80p_rows_kernel<NC>, dim3((unsigned) grid), dim3(WPB * WARP), (size_t) NC * kBlocks * 4 * sizeof(int4), s,
               (const uint8_t*) w, (const Q81Block*) x, y, n_out, per_block, k, vs);
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

bool arch_ok() {
    static std::atomic<int> ok[64];   // 0 unknown, 1 yes, 2 no
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { (void) cudaGetLastError(); return false; }
    int v = ok[dev].load(std::memory_order_relaxed);
    if (v == 0) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) { (void) cudaGetLastError(); major = minor = 0; }
        cudaFuncAttributes fa{};
        // ptxVersion too (as the PDL and cluster probes check): a 75-virtual build JITs to the card's binary version, and only the
        // PTX version tells that the kernels are trap stubs (a pre-sm_75 build; the STRATA_EMULATE_CC tests JIT older PTX)
        const bool built = cudaFuncGetAttributes(&fa, q80p_rows_kernel<1>) == cudaSuccess && fa.ptxVersion >= 75 && fa.binaryVersion >= 75;
        if (!built) (void) cudaGetLastError();
        v = major * 10 + minor >= 75 && built ? 1 : 2;
        ok[dev].store(v, std::memory_order_relaxed);
    }
    return v == 1;
}

void dispatch(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, const int* k, void* scratch,
              void* stream, const char* who) {
    if (!q80p_supported(n_in, ncols) || n_out <= 0 || !w || !x || !y || (k != nullptr && scratch == nullptr))
        throw std::invalid_argument(std::string(who) + ": unsupported shape (n_in " + std::to_string(n_in) + ", n_out " +
                                    std::to_string(n_out) + ", ncols " + std::to_string(ncols) + ")");
    const auto s = (cudaStream_t) stream;
    VScratch* vs = (VScratch*) scratch;
    switch (ncols) {
        case 1: launch<1>(w, x, y, n_out, k, vs, s); break;
        case 2: launch<2>(w, x, y, n_out, k, vs, s); break;
        case 3: launch<3>(w, x, y, n_out, k, vs, s); break;
        case 4: launch<4>(w, x, y, n_out, k, vs, s); break;
        case 5: launch<5>(w, x, y, n_out, k, vs, s); break;
        case 6: launch<6>(w, x, y, n_out, k, vs, s); break;
        case 7: launch<7>(w, x, y, n_out, k, vs, s); break;
        default: launch<8>(w, x, y, n_out, k, vs, s); break;
    }
    check(who);
}

}  // namespace

bool q80p_supported(int n_in, int ncols) {
    return n_in == kBlocks * 32 && ncols >= 1 && ncols <= kMaxCols && arch_ok();
}

void q80p_pack(void* rows, int64_t n, void* stream) {
    if (n <= 0) return;
    q80p_pack_kernel<<<(unsigned) n, kBlocks, 0, (cudaStream_t) stream>>>((uint8_t*) rows);
    check("q80p_pack");
}

void q80p_unpack(void* rows, int64_t n, void* stream) {
    if (n <= 0) return;
    q80p_unpack_kernel<<<(unsigned) n, kBlocks, 0, (cudaStream_t) stream>>>((uint8_t*) rows);
    check("q80p_unpack");
}

void q80p_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    // more columns than a launch takes: passes of 8 (each column's bits do not depend on its neighbours)
    for (int j = 0; j < ncols; j += kMaxCols) {
        const int nj = ncols - j < kMaxCols ? ncols - j : kMaxCols;
        dispatch(weights, (const uint8_t*) x_q8_1 + (size_t) j * (n_in / 32) * sizeof(Q81Block), y + (size_t) j * n_out,
                 n_in, n_out, nj, nullptr, nullptr, stream, "q80p_mmvq");
    }
}

size_t q80p_vhead_scratch_bytes() { return sizeof(VScratch); }

void q80p_vhead(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, const int* k,
                void* scratch, void* stream) {
    if (k == nullptr) throw std::invalid_argument("q80p_vhead: no list length");
    dispatch(weights, x_q8_1, y, n_in, n_out, ncols, k, scratch, stream, "q80p_vhead");
}

bool q80p_vhead_stats(const void* scratch, uint64_t& calls, uint64_t& exact_rows) {
    calls = exact_rows = 0;
    if (scratch == nullptr) return false;
    unsigned long long h[2] = {0, 0};
    if (cudaMemcpy(h, (const uint8_t*) scratch + offsetof(VScratch, stat_calls), sizeof h, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    calls = h[0];
    exact_rows = h[1];
    return true;
}

}  // namespace strata::kernels

#else  // HIP: no planes (the head stays Q8_0)

namespace strata::kernels {
bool q80p_supported(int, int) { return false; }
void q80p_pack(void*, int64_t, void*) { throw std::logic_error("q80p_pack: not on HIP"); }
void q80p_unpack(void*, int64_t, void*) { throw std::logic_error("q80p_unpack: not on HIP"); }
void q80p_mmvq(const void*, const void*, float*, int, int, int, void*) { throw std::logic_error("q80p_mmvq: not on HIP"); }
size_t q80p_vhead_scratch_bytes() { return 0; }
void q80p_vhead(const void*, const void*, float*, int, int, int, const int*, void*, void*) {
    throw std::logic_error("q80p_vhead: not on HIP");
}
bool q80p_vhead_stats(const void*, uint64_t& calls, uint64_t& exact_rows) { calls = exact_rows = 0; return false; }
}  // namespace strata::kernels

#endif
