// src/kernels/q81_div_parity.cu - dec/int: the fused q8_1 quantizers' division (q8_1_finite.hpp q81_div with
// STRATA_Q81_FASTMATH_DIV, as fused_gr.cu / verify_kernels.cu compile it for STRATA_QFUSE) against the separate
// quantizer every q8_1 consumer was written for (native_quantize_q8_1, native_mmvq.cu, --use_fast_math), byte for byte,
// on blocks built to put x / d next to the rounding boundaries (k + 0.5) and on random activations.
// Prints the share of blocks that differ; exit 0 either way unless a CUDA call fails (it is a measurement).
#define STRATA_Q81_FASTMATH_DIV 1
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/q8_1_finite.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

struct Q81 { half2 ds; int8_t qs[32]; };

// gdn_q8_1_store's quantizer (verify_kernels.cu): one warp per block, contraction off
__global__ void fused_q8_1_kernel(const float* __restrict__ x, Q81* __restrict__ y, int nblk) {
#pragma clang fp contract(off)
    const int b = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (b >= nblk) return;
    const float xi = x[(size_t) b * 32 + lane];
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float d = k::q8_1_finite(k::q81_div(amax, 127.0f));
    const int8_t q = k::q8_1_quant(xi, d, amax);
    y[b].qs[lane] = q;
    if (lane == 0) y[b].ds = k::q8_1_ds(d, sum);
}

int main() {
    const int nblk = 1 << 20;   // 32M values
    std::vector<float> h((size_t) nblk * 32);
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f), um(0.001f, 50.0f);
    for (int b = 0; b < nblk; ++b) {
        float* v = h.data() + (size_t) b * 32;
        const float amax = um(rng);
        if (b % 2 == 0) {   // near the boundaries: x = (k + 0.5 + tiny) * d for the d the block's amax gives
            const float d = amax / 127.0f;
            v[0] = amax;
            for (int j = 1; j < 32; ++j) {
                const int kk = (int) (rng() % 254) - 127;
                const float e = ud(rng) * 4e-7f;
                v[j] = ((float) kk + 0.5f + e) * d;
                if (std::fabs(v[j]) > amax) v[j] = 0.0f;
            }
        } else {
            for (int j = 0; j < 32; ++j) v[j] = ud(rng) * amax;
        }
    }
    float* dx;
    Q81 *y0, *y1;
    if (cudaMalloc(&dx, h.size() * 4) != cudaSuccess || cudaMalloc(&y0, (size_t) nblk * 36) != cudaSuccess ||
        cudaMalloc(&y1, (size_t) nblk * 36) != cudaSuccess) { std::printf("malloc failed\n"); return 2; }
    cudaStream_t st;
    cudaStreamCreate(&st);
    cudaMemcpy(dx, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    k::native_quantize_q8_1(dx, y0, nblk * 32, 1, st);   // one column: blocks in order
    fused_q8_1_kernel<<<nblk / 8, 256, 0, st>>>(dx, y1, nblk);
    if (cudaStreamSynchronize(st) != cudaSuccess) { std::printf("kernel failed\n"); return 2; }
    std::vector<uint8_t> a((size_t) nblk * 36), b((size_t) nblk * 36);
    cudaMemcpy(a.data(), y0, a.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), y1, b.size(), cudaMemcpyDeviceToHost);
    long long db[2] = {0, 0}, dc[2] = {0, 0}, ds[2] = {0, 0};
    for (int i = 0; i < nblk; ++i) {
        const uint8_t* p = a.data() + (size_t) i * 36;
        const uint8_t* q = b.data() + (size_t) i * 36;
        if (std::memcmp(p, q, 36) == 0) continue;
        ++db[i & 1];
        if (std::memcmp(p, q, 4) != 0) ++ds[i & 1];
        for (int j = 4; j < 36; ++j) dc[i & 1] += p[j] != q[j];
    }
    std::printf("q81_div_parity: near-boundary blocks: %lld of %d differ (%lld codes, %lld scale/sum words); random blocks: %lld of %d (%lld codes, %lld)\n",
                db[0], nblk / 2, dc[0], ds[0], db[1], nblk / 2, dc[1], ds[1]);
    return 0;
}
