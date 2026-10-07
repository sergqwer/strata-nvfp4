// src/prefill/mmq_nvfp4_a44.cu - NVFP4 experts x two FP4 activation terms on Blackwell's block-scaled FP4 MMA
// (opt-in: STRATA_PREFILL_NVFP4=w4a4x2, the products with K >= 2048 - the gate/up projections).
//
// W4A8 runs the NVFP4 experts on int8 tensor cores; its per-16 weight scales are applied in FP32 after each MMA, and
// that epilogue, not the tensor cores, bounds it (gate/up at 32K: 241 TOPS of the card's 732 INT8). W4A4 lets the
// FP4 MMA apply both sides' scales in hardware (649 TOPS) but rounds the activations to E2M1: 8.5% error per element.
// Here an activation row is two FP4 terms with the same row scale: q1 as W4A4 quantizes it, q2 the residual x - q1
// quantized the same way (its own E4M3 scale per 16 values). q1 + q2 is as close to x as W4A8's q8_1 (0.72% against
// 0.72% per element on heavy-tailed rows), and both terms go through the FP4 MMA with each weight tile in one kernel
// (src/prefill/mmq_vendor: llama.cpp's mmq.cuh with STRATA_MMQ_Y2). Two FP4 terms take exactly q8_1's bytes, so the
// prompt path's activation buffers are unchanged. Built for 12xa only, as mmq_nvfp4_w4a4.cu.
#include "common.cuh"

#include <cuda_fp4.h>

#define STRATA_MMQ_Y2 1
#define mul_mat_q_switch_J strata_a44_mul_mat_q_switch_J
#define mul_mat_q_case strata_a44_mul_mat_q_case
#include "mmq_vendor/mmq.cuh"

DECL_MMQ_CASE(GGML_TYPE_NVFP4);

namespace {

constexpr int kQuantThreads = 128;

#if defined(BLACKWELL_MMA_AVAILABLE)
// one 16-value sub-block (row units: x / row_scale) as llama.cpp's quantize_mmq_nvfp4 does it - the E4M3 code from
// amax / 6 and the best of it and +-1, +-2 by squared error - packed in its order (q0: 0 8 1 9 2 10 3 11; q1: 4 12
// 5 13 6 14 7 15); deq gets what the term represents (fp4 x 2 x scale, the convention of its error function)
__device__ __forceinline__ void fp4_term(const float v[QK_NVFP4_SUB], uint32_t& q0, uint32_t& q1, uint8_t& code,
                                         float deq[QK_NVFP4_SUB]) {
    float amax = 0.0f;
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB; ++k) amax = fmaxf(amax, fabsf(v[k]));
    const int first = (int) ggml_cuda_fp32_to_ue4m3(amax / 6.0f);
    float best_err = 0.0f;
    int best = first;
#pragma unroll
    for (int i = 0; i < 5; ++i) {
        const int off = i == 0 ? 0 : (i == 1 ? -1 : (i == 2 ? 1 : (i == 3 ? -2 : 2)));
        const int c = first + off;
        if (c < 0 || c > 0x7e) continue;
        const float sc = ggml_cuda_ue4m3_to_fp32((uint8_t) c);
        const float inv = sc > 0.0f ? 0.5f / sc : 0.0f;
        float err = 0.0f;
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB; ++k) {
            const float q = __half2float(__nv_cvt_fp4_to_halfraw(__nv_cvt_float_to_fp4(v[k] * inv, __NV_E2M1, cudaRoundNearest), __NV_E2M1));
            const float e = fabsf(v[k]) - fabsf(q) * 2.0f * sc;
            err = fmaf(e, e, err);
        }
        if (i == 0 || err < best_err) { best_err = err; best = c; }
    }
    code = (uint8_t) best;
    const float sc = ggml_cuda_ue4m3_to_fp32(code);
    const float inv = sc > 0.0f ? 0.5f / sc : 0.0f;
    __nv_fp4x4_e2m1 q0_lo(make_float4(v[0] * inv, v[8]  * inv, v[1] * inv, v[9]  * inv));
    __nv_fp4x4_e2m1 q0_hi(make_float4(v[2] * inv, v[10] * inv, v[3] * inv, v[11] * inv));
    __nv_fp4x4_e2m1 q1_lo(make_float4(v[4] * inv, v[12] * inv, v[5] * inv, v[13] * inv));
    __nv_fp4x4_e2m1 q1_hi(make_float4(v[6] * inv, v[14] * inv, v[7] * inv, v[15] * inv));
    const char2 a = *reinterpret_cast<char2 *>(&q0_lo), b = *reinterpret_cast<char2 *>(&q0_hi);
    const char2 c = *reinterpret_cast<char2 *>(&q1_lo), d = *reinterpret_cast<char2 *>(&q1_hi);
    q0 = uint32_t(uint8_t(a.x)) | (uint32_t(uint8_t(a.y)) << 8) | (uint32_t(uint8_t(b.x)) << 16) | (uint32_t(uint8_t(b.y)) << 24);
    q1 = uint32_t(uint8_t(c.x)) | (uint32_t(uint8_t(c.y)) << 8) | (uint32_t(uint8_t(d.x)) << 16) | (uint32_t(uint8_t(d.y)) << 24);
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB; ++k)
        deq[k] = __half2float(__nv_cvt_fp4_to_halfraw(__nv_cvt_float_to_fp4(v[k] * inv, __NV_E2M1, cudaRoundNearest), __NV_E2M1)) * 2.0f * sc;
}
#endif

// rows of x (through ids, ld s01) to two FP4 terms in MMQ's layout: term 1 at y, term 2 right after it (blocks_per_col
// x ne1 blocks on), one row scale (amax / (6 x 448), as quantize_mmq_nvfp4) for both; ne0 is the padded row length
__global__ void __launch_bounds__(kQuantThreads) quantize_x2_kernel(const float* __restrict__ x, const int32_t* __restrict__ ids,
                                                                     block_fp4_mmq* __restrict__ y, float* __restrict__ scale,
                                                                     const int64_t ne00, const int64_t s01, const int64_t ne0,
                                                                     const int64_t ne1) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t row = blockIdx.x;
    const float* __restrict__ xr = x + (ids ? ids[row] : row) * s01;
    float amax = 0.0f;
    for (int64_t i = threadIdx.x; i < ne00; i += blockDim.x) amax = fmaxf(amax, fabsf(xr[i]));
    amax = warp_reduce_max<WARP_SIZE>(amax);
    __shared__ float wmax[kQuantThreads / WARP_SIZE];
    if (threadIdx.x % WARP_SIZE == 0) wmax[threadIdx.x / WARP_SIZE] = amax;
    __syncthreads();
    float rs = 0.0f;
#pragma unroll
    for (int w = 0; w < kQuantThreads / WARP_SIZE; ++w) rs = fmaxf(rs, wmax[w]);
    rs /= 6.0f * 448.0f;
    if (threadIdx.x == 0) scale[row] = rs;
    const float inv_rs = rs > 0.0f ? 1.0f / rs : 0.0f;
    const int64_t blocks_per_col = (ne0 + QK_FP4_MMQ - 1) / QK_FP4_MMQ;
    block_fp4_mmq* __restrict__ y2 = y + blocks_per_col * ne1;
    const int64_t nsub = (ne0 + QK_NVFP4_SUB - 1) / QK_NVFP4_SUB;
    for (int64_t isb = threadIdx.x; isb < nsub; isb += blockDim.x) {
        const int64_t i0 = isb * QK_NVFP4_SUB, kb = i0 / QK_FP4_MMQ;
        const int sub = (int) ((i0 % QK_FP4_MMQ) / QK_NVFP4_SUB);
        float v[QK_NVFP4_SUB], deq[QK_NVFP4_SUB];
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB; ++k) v[k] = i0 + k < ne00 ? xr[i0 + k] * inv_rs : 0.0f;
        uint32_t q0, q1;
        uint8_t code;
        fp4_term(v, q0, q1, code, deq);
        block_fp4_mmq* yb = y + kb * ne1 + row;
        reinterpret_cast<uint32_t*>(yb->qs)[2 * sub + 0] = q0;
        reinterpret_cast<uint32_t*>(yb->qs)[2 * sub + 1] = q1;
        reinterpret_cast<uint8_t*>(yb->d4)[sub] = code;
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB; ++k) v[k] -= deq[k];   // the residual, in the same row units
        fp4_term(v, q0, q1, code, deq);
        yb = y2 + kb * ne1 + row;
        reinterpret_cast<uint32_t*>(yb->qs)[2 * sub + 0] = q0;
        reinterpret_cast<uint32_t*>(yb->qs)[2 * sub + 1] = q1;
        reinterpret_cast<uint8_t*>(yb->d4)[sub] = code;
    }
#else
    GGML_UNUSED_VARS(x, ids, y, scale, ne00, s01, ne0, ne1);
    NO_DEVICE_CODE;
#endif
}

// the same per token: block t quantizes x row t once and writes both terms and the scale to its k_used rows
// slot[t * k_used + k] (ne1 = tokens x k_used rows in all)
__global__ void __launch_bounds__(kQuantThreads) quantize_x2_scatter_kernel(const float* __restrict__ x,
                                                                             const int32_t* __restrict__ slot,
                                                                             block_fp4_mmq* __restrict__ y,
                                                                             float* __restrict__ scale, const int64_t ne00,
                                                                             const int64_t s_tok, const int64_t ne0,
                                                                             const int64_t ne1, const int k_used) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t tok = blockIdx.x;
    const float* __restrict__ xr = x + tok * s_tok;
    const int32_t* __restrict__ rows = slot + tok * k_used;
    float amax = 0.0f;
    for (int64_t i = threadIdx.x; i < ne00; i += blockDim.x) amax = fmaxf(amax, fabsf(xr[i]));
    amax = warp_reduce_max<WARP_SIZE>(amax);
    __shared__ float wmax[kQuantThreads / WARP_SIZE];
    if (threadIdx.x % WARP_SIZE == 0) wmax[threadIdx.x / WARP_SIZE] = amax;
    __syncthreads();
    float rs = 0.0f;
#pragma unroll
    for (int w = 0; w < kQuantThreads / WARP_SIZE; ++w) rs = fmaxf(rs, wmax[w]);
    rs /= 6.0f * 448.0f;
    for (int k = threadIdx.x; k < k_used; k += blockDim.x) scale[rows[k]] = rs;
    const float inv_rs = rs > 0.0f ? 1.0f / rs : 0.0f;
    const int64_t blocks_per_col = (ne0 + QK_FP4_MMQ - 1) / QK_FP4_MMQ;
    block_fp4_mmq* __restrict__ y2 = y + blocks_per_col * ne1;
    const int64_t nsub = (ne0 + QK_NVFP4_SUB - 1) / QK_NVFP4_SUB;
    for (int64_t isb = threadIdx.x; isb < nsub; isb += blockDim.x) {
        const int64_t i0 = isb * QK_NVFP4_SUB, kb = i0 / QK_FP4_MMQ;
        const int sub = (int) ((i0 % QK_FP4_MMQ) / QK_NVFP4_SUB);
        float v[QK_NVFP4_SUB], deq[QK_NVFP4_SUB];
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB; ++k) v[k] = i0 + k < ne00 ? xr[i0 + k] * inv_rs : 0.0f;
        uint32_t a0, a1, b0, b1;
        uint8_t ca, cb;
        fp4_term(v, a0, a1, ca, deq);
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB; ++k) v[k] -= deq[k];   // the residual, in the same row units
        fp4_term(v, b0, b1, cb, deq);
        for (int k = 0; k < k_used; ++k) {
            const int64_t p = rows[k];
            block_fp4_mmq* yb = y + kb * ne1 + p;
            reinterpret_cast<uint32_t*>(yb->qs)[2 * sub + 0] = a0;
            reinterpret_cast<uint32_t*>(yb->qs)[2 * sub + 1] = a1;
            reinterpret_cast<uint8_t*>(yb->d4)[sub] = ca;
            yb = y2 + kb * ne1 + p;
            reinterpret_cast<uint32_t*>(yb->qs)[2 * sub + 0] = b0;
            reinterpret_cast<uint32_t*>(yb->qs)[2 * sub + 1] = b1;
            reinterpret_cast<uint8_t*>(yb->d4)[sub] = cb;
        }
    }
#else
    GGML_UNUSED_VARS(x, slot, y, scale, ne00, s_tok, ne0, ne1, k_used);
    NO_DEVICE_CODE;
#endif
}

__global__ void a44_probe_kernel(int* out) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    *out = 1;
#else
    *out = 0;
#endif
}

}  // namespace

namespace strata::prefill::mmq {
bool a44_available() {
    int* d = nullptr;
    int h = 0;
    if (cudaMalloc(&d, sizeof(int)) != cudaSuccess) { (void) cudaGetLastError(); return false; }
    a44_probe_kernel<<<1, 1>>>(d);
    const bool ran = cudaGetLastError() == cudaSuccess && cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess;
    (void) cudaGetLastError();
    cudaFree(d);
    return ran && h == 1;
}
void run_nvfp4_a44(ggml_backend_cuda_context& ctx, const mmq_args& a, cudaStream_t s) {
    strata_a44_mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, a, s);
}
void quantize_nvfp4_x2(const float* x, const int32_t* ids, void* xq, float* yscale, int64_t cols, int64_t ld,
                       int64_t rows, int64_t padded, cudaStream_t s) {
    quantize_x2_kernel<<<(unsigned) rows, kQuantThreads, 0, s>>>(x, ids, (block_fp4_mmq*) xq, yscale, cols, ld, padded, rows);
}
void quantize_nvfp4_x2_scatter(const float* x, const int32_t* slot, void* xq, float* yscale, int64_t cols, int64_t ld,
                               int64_t tokens, int k_used, int64_t padded, cudaStream_t s) {
    quantize_x2_scatter_kernel<<<(unsigned) tokens, kQuantThreads, 0, s>>>(x, slot, (block_fp4_mmq*) xq, yscale, cols, ld,
                                                                            padded, tokens * k_used, k_used);
}
}  // namespace strata::prefill::mmq
