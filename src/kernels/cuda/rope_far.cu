// src/kernels/cuda/rope_far.cu - see include/strata/kernels/rope_far.hpp.
#include "strata/kernels/rope_far.hpp"
#include "strata/kernels/rope_dynamic.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <thread>
#include <vector>

namespace strata::kernels {

namespace {
RopeFarDev g_far;
float* g_cs = nullptr;
int32_t* g_ctl = nullptr;
}  // namespace

const RopeFarDev& rope_far() { return g_far; }
bool rope_far_on() { return g_far.cs != nullptr; }

bool rope_far_init(int max_cells) {
    if (g_cs != nullptr) return true;
    if (max_cells <= 0) return false;
    if (cudaMalloc((void**) &g_cs, (size_t) max_cells * 64 * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**) &g_ctl, sizeof(int32_t)) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    const int32_t off = INT32_MAX;
    if (cudaMemcpy(g_ctl, &off, sizeof off, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    g_far = RopeFarDev{g_cs, g_ctl, max_cells};
    return true;
}

bool rope_far_update(int w, double k, const RopeScaling& s, bool yarn) {
    if (g_cs == nullptr) return false;
    if (cudaDeviceSynchronize() != cudaSuccess) return false;
    const int32_t ctl = k > 1.0 ? std::max(0, w) : INT32_MAX;
    if (k > 1.0) {
        const RopeRates r = rope_rates(64, s);
        // linear: every pair's excess distance shrinks by k; yarn: by yarn's per-pair rates of factor k (the high
        // frequencies keep theirs)
        double dw[32];
        RopeScaling y = s;
        y.type = RopeScalingType::YaRN;
        y.factor = k;
        y.freq_scale_in = 0;
        y.ext_factor = 1.0;
        const RopeRates ry = rope_rates(64, y);
        for (int i = 0; i < 32; ++i) dw[i] = yarn ? r.w[i] - ry.w[i] : r.w[i] * (1.0 - 1.0 / k);
        const int n = g_far.max_m;
        std::vector<float> h((size_t) n * 64);
        auto rows = [&](int a, int b) {
            for (int m = a; m < b; ++m)
                for (int i = 0; i < 32; ++i) {
                    const double ang = dw[i] * (double) m;
                    h[(size_t) m * 64 + 2 * i] = (float) std::cos(ang);
                    h[(size_t) m * 64 + 2 * i + 1] = (float) std::sin(ang);
                }
        };
        const int th = (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
        std::vector<std::thread> pool;
        const int step = (n + th - 1) / th;
        for (int t = 0; t < th; ++t) {
            const int a = t * step, b = std::min(n, a + step);
            if (a < b) pool.emplace_back(rows, a, b);
        }
        for (auto& x : pool) x.join();
        if (cudaMemcpy(g_cs, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return false;
    }
    return cudaMemcpy(g_ctl, &ctl, sizeof ctl, cudaMemcpyHostToDevice) == cudaSuccess;
}

}  // namespace strata::kernels
