// dequant_q8_0_f16_test - strata::kernels::dequant_f16 / dequant_f16_ld of Q8_0 (the prompt path's dense weights
// before cuBLAS) against the exact value per element: FP16(d) as float times the int8, rounded to nearest-even FP16.
// Random blocks with scales from zero and subnormal to large, row slices (row0), padded rows (ld), and an output
// pointer that is not 16-byte aligned (the generic kernel).  Exit 77 without a CUDA device.
#include "strata/kernels/dequant_bf16.hpp"

#include <cuda_runtime.h>
#include <immintrin.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }
uint16_t f2h(float f) { return (uint16_t) _mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT), 0); }

int run_case(int64_t rows_total, int64_t cols, int64_t row0, int64_t rows, int64_t ld, int64_t out_off, uint32_t seed) {
    std::mt19937 rng(seed);
    const int64_t bpr = cols / 32, row_bytes = bpr * 34;
    std::vector<uint8_t> w((size_t) (rows_total * row_bytes));
    std::uniform_int_distribution<int> byte(0, 255), pick(0, 9);
    for (auto& b : w) b = (uint8_t) byte(rng);
    for (int64_t i = 0; i < rows_total * bpr; ++i) {   // scales: mostly normal, some zero / subnormal / large
        uint16_t d;
        switch (pick(rng)) {
            case 0: d = 0; break;
            case 1: d = (uint16_t) (byte(rng) | 0x100); break;   // subnormal
            case 2: d = (uint16_t) (0x7000 + byte(rng)); break;  // ~8K: products past FP16's range become inf
            default: d = (uint16_t) (0x1c00 + byte(rng) * 16); break;
        }
        std::memcpy(w.data() + i * 34, &d, 2);
    }
    const size_t n_out = (size_t) (rows * ld + out_off + 64);
    uint8_t* dw = nullptr;
    uint16_t* dout = nullptr;
    cudaMalloc(&dw, w.size());
    cudaMalloc(&dout, n_out * 2);
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemset(dout, 0xee, n_out * 2);
    if (ld == cols) strata::kernels::dequant_f16(8, dw, row0, rows, cols, dout + out_off, nullptr);
    else if (!strata::kernels::dequant_f16_ld(8, dw, row0, rows, cols, ld, dout + out_off, nullptr)) {
        std::printf("  dequant_f16_ld refused ld %lld\n", (long long) ld);
        return 1;
    }
    std::vector<uint16_t> got(n_out);
    if (cudaMemcpy(got.data(), dout, n_out * 2, cudaMemcpyDeviceToHost) != cudaSuccess) { std::printf("  CUDA error\n"); return 1; }
    cudaFree(dw);
    cudaFree(dout);
    size_t bad = 0;
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t k = 0; k < cols; ++k) {
            const uint8_t* b = w.data() + (row0 + r) * row_bytes + (k / 32) * 34;
            uint16_t dh;
            std::memcpy(&dh, b, 2);
            const uint16_t want = f2h((float) (int8_t) b[2 + k % 32] * h2f(dh));
            if (got[(size_t) (out_off + r * ld + k)] != want) ++bad;
        }
    for (int64_t r = 0; r < rows; ++r)   // the padding between rows untouched
        for (int64_t k = cols; k < ld; ++k)
            if (got[(size_t) (out_off + r * ld + k)] != 0xeeee) ++bad;
    std::printf("  Q8_0 %5lld x %5lld from row %4lld (ld %5lld, out +%lld): %zu of %lld differ\n", (long long) rows,
                (long long) cols, (long long) row0, (long long) ld, (long long) out_off, bad, (long long) (rows * cols));
    return bad ? 1 : 0;
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("dequant_q8_0_f16_test: no CUDA device (skipped)\n");
        return 77;
    }
    int fails = 0;
    fails += run_case(64, 2560, 0, 64, 2560, 0, 1);
    fails += run_case(300, 10240, 37, 200, 10240, 0, 2);
    fails += run_case(10, 32, 3, 7, 32, 0, 3);
    fails += run_case(40, 640, 5, 33, 704, 0, 4);       // padded rows (the strided path)
    fails += run_case(16, 2560, 1, 15, 2560, 1, 5);     // output not 16-byte aligned: the generic kernel
    std::printf("dequant_q8_0_f16_test: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
