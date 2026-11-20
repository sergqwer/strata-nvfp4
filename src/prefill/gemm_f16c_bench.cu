// src/prefill/gemm_f16c_bench.cu - STRATA_DENSE_F16C's kernel (gemm_f16c.hpp) against the cuBLAS call Gemm::f16 makes
// (f16 x f16 -> f32, COMPUTE_32F), at the prompt path's dense shapes: time (median of the reps, back to back, so the
// clock is the power-limited one of a long run) and error against an FP64 reference on 64 rows (X ~ N(0, 1) with 1% of
// the channels x 20, W ~ N(0, 0.02) through Q8_0 rounding, as gpu-probe/poc/gemm_poc.cu).  Not a ctest (GPU, timing).
// Usage: gemm_f16c_bench [T=16675] [reps=20] [N:K ...]  (default: the seven Qwen3.8-Flash-Next shapes)
#include "strata/prefill/gemm_f16c.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
__global__ void ref_kernel(const __half* X, const __half* W, double* R, int rows, int N, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (n >= N || t >= rows) return;
    double s = 0;
    for (int k = 0; k < K; ++k) s += (double) __half2float(X[(size_t) t * K + k]) * (double) __half2float(W[(size_t) n * K + k]);
    R[(size_t) t * N + n] = s;
}
}  // namespace

int main(int argc, char** argv) {
    const int64_t T = argc > 1 ? std::atoll(argv[1]) : 16675;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
    std::vector<std::pair<int64_t, int64_t>> shapes;
    for (int i = 3; i < argc; ++i) {
        long long n = 0, k = 0;
        if (std::sscanf(argv[i], "%lld:%lld", &n, &k) == 2) shapes.emplace_back(n, k);
    }
    if (shapes.empty()) shapes = {{10240, 2560}, {12288, 2560}, {6144, 2560}, {2560, 6144}, {512, 2560}, {640, 2560}, {2560, 640}};
    int64_t maxn = 0, maxk = 0;
    for (auto& s : shapes) { maxn = std::max(maxn, s.first); maxk = std::max(maxk, s.second); }
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> chs((size_t) maxk);
    for (auto& c : chs) c = (rng() % 100 == 0) ? 20.f : 1.f;
    std::vector<__half> hx((size_t) (T * maxk)), hw((size_t) (maxn * maxk));
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = __float2half(nd(rng) * chs[i % (size_t) maxk]);
    for (size_t i = 0; i < hw.size(); i += 32) {
        float w[32], mx = 0;
        for (int j = 0; j < 32; ++j) { w[j] = nd(rng) * 0.02f; mx = std::max(mx, std::fabs(w[j])); }
        const float d = __half2float(__float2half(mx / 127.f));
        for (int j = 0; j < 32; ++j) hw[i + j] = __float2half(d * std::max(-127.f, std::min(127.f, std::nearbyint(w[j] / d))));
    }
    __half *dX = nullptr, *dW = nullptr;
    float *dA = nullptr, *dB = nullptr;
    double* dR = nullptr;
    ck(cudaMalloc(&dX, hx.size() * 2), "malloc");
    ck(cudaMalloc(&dW, hw.size() * 2), "malloc");
    ck(cudaMalloc(&dA, (size_t) (T * maxn) * 4), "malloc");
    ck(cudaMalloc(&dB, (size_t) (T * maxn) * 4), "malloc");
    ck(cudaMalloc(&dR, (size_t) (64 * maxn) * 8), "malloc");
    ck(cudaMemcpy(dX, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice), "up");
    ck(cudaMemcpy(dW, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice), "up");
    cublasHandle_t h;
    cublasCreate(&h);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    double tot_c = 0, tot_f = 0;
    std::printf("T %lld, %d reps\n", (long long) T, reps);
    for (auto& s : shapes) {
        const int64_t N = s.first, K = s.second;
        const float one = 1.f, zero = 0.f;
        auto run_c = [&] {
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &one, dW, CUDA_R_16F, (int) K, dX, CUDA_R_16F,
                         (int) K, &zero, dB, CUDA_R_32F, (int) N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        };
        auto run_f = [&] {
            if (!strata::prefill::gemm_f16c(reinterpret_cast<const uint16_t*>(dX), reinterpret_cast<const uint16_t*>(dW), dA, T, N,
                                            K, N, 0.f, nullptr)) {
                std::fprintf(stderr, "gemm_f16c refused %lld x %lld x %lld\n", (long long) T, (long long) N, (long long) K);
                std::exit(2);
            }
        };
        auto timed = [&](auto f) {
            f();
            ck(cudaDeviceSynchronize(), "warm");
            std::vector<float> v;
            for (int r = 0; r < reps; ++r) {
                cudaEventRecord(e0);
                f();
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                v.push_back(ms);
            }
            std::sort(v.begin(), v.end());
            return (double) v[v.size() / 2];
        };
        // alternate the two so both see the same clock
        double tc = 0, tf = 0;
        for (int a = 0; a < 2; ++a) { tc += timed(run_c); tf += timed(run_f); }
        tc /= 2;
        tf /= 2;
        ref_kernel<<<dim3((unsigned) ((N + 127) / 128), 64), 128>>>(dX, dW, dR, 64, (int) N, (int) K);
        ck(cudaDeviceSynchronize(), "ref");
        std::vector<double> R((size_t) (64 * N));
        std::vector<float> A((size_t) (64 * N)), B((size_t) (64 * N));
        ck(cudaMemcpy(R.data(), dR, R.size() * 8, cudaMemcpyDeviceToHost), "down");
        ck(cudaMemcpy(A.data(), dA, A.size() * 4, cudaMemcpyDeviceToHost), "down");
        ck(cudaMemcpy(B.data(), dB, B.size() * 4, cudaMemcpyDeviceToHost), "down");
        double sr = 0, sa = 0, sb = 0, dd = 0;
        for (size_t i = 0; i < R.size(); ++i) {
            sr += R[i] * R[i];
            sa += (A[i] - R[i]) * (A[i] - R[i]);
            sb += (B[i] - R[i]) * (B[i] - R[i]);
            dd += ((double) A[i] - B[i]) * ((double) A[i] - B[i]);
        }
        // the whole output: identical row count and no tile recomputed
        const double fl = 2.0 * (double) T * (double) N * (double) K;
        tot_c += tc;
        tot_f += tf;
        std::printf("N %6lld K %5lld: cuBLAS %7.3f ms %6.1f TFLOPS | f16c %7.3f ms %6.1f TFLOPS (%.3fx) | rel RMS vs FP64 cuBLAS %.2e "
                    "f16c %.2e, f16c vs cuBLAS %.2e\n",
                    (long long) N, (long long) K, tc, fl / tc / 1e9, tf, fl / tf / 1e9, tc / tf, std::sqrt(sb / sr), std::sqrt(sa / sr),
                    std::sqrt(dd / sr));
    }
    std::printf("sum of the shapes: cuBLAS %.3f ms, f16c %.3f ms (%.3fx); f32 redo tiles %u\n", tot_c, tot_f, tot_c / tot_f,
                strata::prefill::gemm_f16c_redo_count(false));
    return 0;
}
