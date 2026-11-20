// src/kernels/cuda/mtp_cond.cu - see include/strata/kernels/mtp_cond.hpp.
#include "strata/kernels/mtp_cond.hpp"

#include "strata/spec/draft_cum.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

namespace {
__global__ void mtp_select_wide_kernel(const float4* __restrict__ R_src, long long n4, const int32_t* __restrict__ ids,
                                       const int32_t* __restrict__ row_dev, float4* __restrict__ R_dst,
                                       int32_t* __restrict__ tok_dst, int32_t* out, int j, const float* probs, float* out_p) {
    const int row = *row_dev;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (long long) gridDim.x * blockDim.x)
        R_dst[i] = R_src[(size_t) row * n4 + i];
    if (blockIdx.x == 0 && threadIdx.x == 0) {   // (the next kernels in the stream see R_dst complete either way)
        const int32_t tok = ids[row];
        *tok_dst = tok;
        if (probs != nullptr && out_p != nullptr) ((volatile float*) out_p)[j] = probs[row];
        __threadfence_system();
        if (out != nullptr) ((volatile int32_t*) out)[j] = tok;
    }
}
}  // namespace

void mtp_select_wide(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                     int32_t* tok_dst, int32_t* out, int j, cudaStream_t stream, const float* probs, float* out_p) {
    if (R_stride % 4 != 0 || ((uintptr_t) R_src | (uintptr_t) R_dst) % 16 != 0) {
        std::fprintf(stderr, "mtp_select_wide: unaligned rows\n");
        std::exit(1);
    }
    const long long n4 = R_stride / 4;
    const int blocks = (int) ((n4 + 255) / 256 < 16 ? (n4 + 255) / 256 : 16);
    mtp_select_wide_kernel<<<blocks, 256, 0, stream>>>((const float4*) R_src, n4, ids, row_dev, (float4*) R_dst, tok_dst, out, j,
                                                       probs, out_p);
}

#if !defined(__HIPCC__) && defined(CUDART_VERSION) && CUDART_VERSION >= 12040
namespace {
// the acceptance table of draft_cum_acc, as the device sees it (the same comparisons and constants)
__device__ __forceinline__ float cum_acc(float p) {
    return p < 0.30f ? 0.12f : p < 0.50f ? 0.26f : p < 0.70f ? 0.43f : p < 0.80f ? 0.51f
         : p < 0.90f ? 0.59f : p < 0.95f ? 0.66f : p < 0.99f ? 0.79f : 0.96f;
}
__global__ void mtp_cond_keep_kernel(unsigned long long h, const float* __restrict__ probs, const int32_t* __restrict__ row_dev,
                                     const int32_t* __restrict__ th_bits, float* hist, int j) {
    const float p = probs[*row_dev];
    hist[j] = p;
    // draft_cum_keep's products in its order: c = 1, c = c * acc(p_i) for i = 0..j (multiplications only, no contraction)
    float c = 1.0f;
    for (int i = 0; i < j; ++i) c = __fmul_rn(c, cum_acc(hist[i]));
    c = __fmul_rn(c, cum_acc(p));
    cudaGraphSetConditional((cudaGraphConditionalHandle) h, c >= __int_as_float(*th_bits) ? 1u : 0u);
}
}  // namespace

bool mtp_cond_supported() { return true; }

bool mtp_cond_keep(unsigned long long h, const float* probs, const int32_t* row_dev, const int32_t* th_bits, float* hist,
                   int j, cudaStream_t stream) {
    if (j < 0 || j >= 16) { std::fprintf(stderr, "mtp_cond_keep: j out of range\n"); std::exit(1); }
    mtp_cond_keep_kernel<<<1, 1, 0, stream>>>(h, probs, row_dev, th_bits, hist, j);
    return cudaPeekAtLastError() == cudaSuccess;
}
#else
bool mtp_cond_supported() { return false; }
bool mtp_cond_keep(unsigned long long, const float*, const int32_t*, const int32_t*, float*, int, cudaStream_t) { return false; }
#endif

}  // namespace strata::kernels
