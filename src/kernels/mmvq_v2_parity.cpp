// src/kernels/mmvq_v2_parity.cpp - STRATA_MMVQ_V2's Q8_0 decode GEMV against the old path, bitwise and timed.
//
//     build/mmvq_v2_parity            the bit checks (ctest)
//     build/mmvq_v2_parity --bench    and the timings at the decode shapes
//
// WHAT IT COMPARES.  The same weights and the same Q8_1 input through native_q8_0_mmvq with V2 off (the EXACT layout:
// native_small_mmvq_kernel for one column, native_mmvq_multi_kernel for 2-8) and with V2 on, single and grouped
// (native_q8_0_mmvq_group: attention k + v + q, the drafter's k + v, a 640-row pair): every output must be
// the same bits.  The single shapes are the model's decode matrices (Q8_0, 2560 wide but the out projections and the
// shared expert's down), the head included, all through native_q8_0_mmvq with V2 on (it leaves single calls alone, so
// they check that V2 changes nothing there); the groups are the engine's (attention k + v + q, the drafter's k + v)
// and a 640-row pair.  Weights as mmvq_multi_parity makes them (random bytes, sane fp16 scales).
//
// THE TIMINGS.  The engine reads each matrix once a round, cold: here every launch reads its own copy of the weights,
// the copies rotated past the 96 MB L2, and a CUDA graph holds the launches (as the verify window's graph does), so
// the time is the kernel's and not the launch's.  Medians of alternating replays.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int Q8 = 8;   // GGML Q8_0

struct Mat { const char* name; int n_in, n_out; };
const Mat MATS[] = {
    {"GDN qkv", 2560, 10240}, {"GDN gate", 2560, 6144}, {"attn q", 2560, 12288}, {"attn k/v", 2560, 512},
    {"out proj", 6144, 2560}, {"shexp gate/up", 2560, 640}, {"shexp down", 640, 2560}, {"head", 2560, 248320},
};
struct Group { const char* name; int n_in; int n; int n_out[3]; };
const Group GROUPS[] = {
    {"attn k + v + q", 2560, 3, {512, 512, 12288}},   // verify.cpp
    {"mtp k + v", 2560, 2, {512, 512, 0}},            // mtp.cpp
    {"640-row pair", 2560, 2, {640, 640, 0}},         // the shared expert's gate + up shape (native_mmvq_pair runs those)
};

bool ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::printf("CUDA: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

uint16_t sane_half(std::mt19937& rng) {
    const uint32_t r = rng();
    return (uint16_t) (((r >> 31) << 15) | ((5u + (r >> 10) % 5u) << 10) | (r & 0x3ffu));
}

// a Q8_0 matrix on the device: random int8, sane scales
void* make_weights(int n_in, int n_out, unsigned seed) {
    const std::size_t bytes = strata::kernels::native_mmvq_weight_bytes(Q8, n_in, n_out);
    std::vector<uint8_t> w(bytes);
    std::mt19937 rng(seed);
    for (auto& b : w) b = (uint8_t) (rng() & 0xff);
    const std::size_t blocks = bytes / 34;
    for (std::size_t k = 0; k < blocks; ++k) {
        const uint16_t h = sane_half(rng);
        std::memcpy(&w[k * 34], &h, 2);
    }
    void* d = nullptr;
    if (!ck(cudaMalloc(&d, bytes), "malloc w") || !ck(cudaMemcpy(d, w.data(), bytes, cudaMemcpyHostToDevice), "copy w"))
        return nullptr;
    return d;
}

void* make_input(int n_in, int T, unsigned seed, cudaStream_t s) {
    std::vector<float> x((std::size_t) T * n_in);
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);
    float* dx = nullptr;
    void* xq = nullptr;
    if (!ck(cudaMalloc(&dx, x.size() * 4), "malloc x") ||
        !ck(cudaMalloc(&xq, strata::kernels::native_q8_1_bytes(n_in, T)), "malloc xq") ||
        !ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "copy x"))
        return nullptr;
    strata::kernels::quantize_q8_1_rows(dx, T, n_in, xq, s);
    ck(cudaStreamSynchronize(s), "quantize");
    cudaFree(dx);
    return xq;
}

int bits_single(const Mat& m, int T, cudaStream_t s) {
    void* w = make_weights(m.n_in, m.n_out, 77u + (unsigned) m.n_out);
    void* xq = make_input(m.n_in, T, 5u + (unsigned) T, s);
    const std::size_t yb = (std::size_t) T * m.n_out * 4;
    float *a = nullptr, *b = nullptr;
    ck(cudaMalloc(&a, yb), "a");
    ck(cudaMalloc(&b, yb), "b");
    cudaMemset(a, 0xff, yb);
    cudaMemset(b, 0x7f, yb);
    strata::kernels::native_mmvq_set_v2(false);
    strata::kernels::native_q8_0_mmvq(w, xq, a, m.n_in, m.n_out, T, s);
    strata::kernels::native_mmvq_set_v2(true);
    strata::kernels::native_q8_0_mmvq(w, xq, b, m.n_in, m.n_out, T, s);
    ck(cudaStreamSynchronize(s), "run");
    std::vector<uint32_t> ha(yb / 4), hb(yb / 4);
    cudaMemcpy(ha.data(), a, yb, cudaMemcpyDeviceToHost);
    cudaMemcpy(hb.data(), b, yb, cudaMemcpyDeviceToHost);
    long long diff = 0;
    for (std::size_t i = 0; i < ha.size(); ++i) diff += ha[i] != hb[i];
    cudaFree(w); cudaFree(xq); cudaFree(a); cudaFree(b);
    if (diff) std::printf("FAIL %-14s T=%d: %lld of %zu outputs differ\n", m.name, T, diff, ha.size());
    return diff ? 1 : 0;
}

int bits_group(const Group& g, int T, cudaStream_t s) {
    void* w[3] = {};
    float *a[3] = {}, *b[3] = {};
    void* xq = make_input(g.n_in, T, 9u + (unsigned) T, s);
    for (int k = 0; k < g.n; ++k) {
        w[k] = make_weights(g.n_in, g.n_out[k], 300u + 7u * (unsigned) k + (unsigned) g.n_out[k]);
        const std::size_t yb = (std::size_t) T * g.n_out[k] * 4;
        ck(cudaMalloc(&a[k], yb), "a");
        ck(cudaMalloc(&b[k], yb), "b");
        cudaMemset(a[k], 0xff, yb);
        cudaMemset(b[k], 0x7f, yb);
    }
    strata::kernels::native_mmvq_set_v2(false);
    for (int k = 0; k < g.n; ++k) strata::kernels::native_q8_0_mmvq(w[k], xq, a[k], g.n_in, g.n_out[k], T, s);
    strata::kernels::native_mmvq_set_v2(true);
    const bool ran = strata::kernels::native_q8_0_mmvq_group(g.n, w, b, g.n_out, xq, g.n_in, T, s);
    ck(cudaStreamSynchronize(s), "run");
    int fails = 0;
    if (!ran) { std::printf("FAIL %-16s T=%d: the group did not run\n", g.name, T); fails = 1; }
    for (int k = 0; k < g.n && ran; ++k) {
        const std::size_t yb = (std::size_t) T * g.n_out[k] * 4;
        std::vector<uint32_t> ha(yb / 4), hb(yb / 4);
        cudaMemcpy(ha.data(), a[k], yb, cudaMemcpyDeviceToHost);
        cudaMemcpy(hb.data(), b[k], yb, cudaMemcpyDeviceToHost);
        long long diff = 0;
        for (std::size_t i = 0; i < ha.size(); ++i) diff += ha[i] != hb[i];
        if (diff) { std::printf("FAIL %-16s T=%d matrix %d: %lld of %zu outputs differ\n", g.name, T, k, diff, ha.size()); fails = 1; }
    }
    for (int k = 0; k < g.n; ++k) { cudaFree(w[k]); cudaFree(a[k]); cudaFree(b[k]); }
    cudaFree(xq);
    return fails;
}

// launches of `n` matrices (one group or n singles) on rotating cold copies, in a graph: microseconds per launch set
struct Timed { float old_us, v2_us; };
Timed time_set(int n, const int* n_out, int n_in, int T, cudaStream_t s) {
    std::size_t set_bytes = 0;
    for (int k = 0; k < n; ++k) set_bytes += strata::kernels::native_mmvq_weight_bytes(Q8, n_in, n_out[k]);
    const int copies = (int) std::min<std::size_t>(256, std::max<std::size_t>(1, (400u << 20) / set_bytes + 1));
    std::vector<void*> w((std::size_t) copies * n);
    void* proto[3] = {};
    for (int k = 0; k < n; ++k) proto[k] = make_weights(n_in, n_out[k], 900u + (unsigned) k);
    for (int c = 0; c < copies; ++c)
        for (int k = 0; k < n; ++k) {
            const std::size_t b = strata::kernels::native_mmvq_weight_bytes(Q8, n_in, n_out[k]);
            if (c == 0) { w[k] = proto[k]; continue; }
            ck(cudaMalloc(&w[(std::size_t) c * n + k], b), "copy malloc");
            ck(cudaMemcpy(w[(std::size_t) c * n + k], proto[k], b, cudaMemcpyDeviceToDevice), "copy");
        }
    void* xq = make_input(n_in, T, 11u, s);
    float* y[3] = {};
    for (int k = 0; k < n; ++k) ck(cudaMalloc(&y[k], (std::size_t) T * n_out[k] * 4), "y");
    cudaGraphExec_t ex[2] = {};
    for (int v = 0; v < 2; ++v) {
        strata::kernels::native_mmvq_set_v2(v == 1);
        cudaGraph_t gr;
        ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
        for (int c = 0; c < copies; ++c) {
            void* const* wc = &w[(std::size_t) c * n];
            if (v == 1 && n > 1) strata::kernels::native_q8_0_mmvq_group(n, wc, y, n_out, xq, n_in, T, s);
            else
                for (int k = 0; k < n; ++k) strata::kernels::native_q8_0_mmvq(wc[k], xq, y[k], n_in, n_out[k], T, s);
        }
        ck(cudaStreamEndCapture(s, &gr), "end capture");
        ck(cudaGraphInstantiate(&ex[v], gr, 0), "instantiate");
        cudaGraphDestroy(gr);
    }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<float> t[2];
    for (int r = 0; r < 12; ++r)
        for (int j = 0; j < 2; ++j) {
            const int v = (r + j) % 2;
            cudaEventRecord(e0, s);
            cudaGraphLaunch(ex[v], s);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            if (r >= 2) t[v].push_back(1000.f * ms / (float) copies);
        }
    for (auto& v : t) std::sort(v.begin(), v.end());
    for (int v = 0; v < 2; ++v) cudaGraphExecDestroy(ex[v]);
    for (auto p : w) cudaFree(p);
    for (int k = 0; k < n; ++k) cudaFree(y[k]);
    cudaFree(xq);
    return {t[0][t[0].size() / 2], t[1][t[1].size() / 2]};
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    cudaStream_t s;
    if (!ck(cudaStreamCreate(&s), "stream")) return 2;
    if (!strata::kernels::native_mmvq_multi_exact()) { std::printf("the multi-column layout is not EXACT\n"); return 2; }
    int fails = 0, checks = 0;
    for (const Mat& m : MATS)
        for (int T = 1; T <= 8; ++T) {
            if (m.n_out > 100000 && T != 1 && T != 4 && T != 5) continue;   // the head: fewer widths (memory, time)
            fails += bits_single(m, T, s);
            ++checks;
        }
    for (const Group& g : GROUPS)
        for (int T = 1; T <= 8; ++T) { fails += bits_group(g, T, s); ++checks; }
    std::printf("mmvq_v2_parity: %d of %d checks bitwise equal (single and grouped, T = 1..8)\n", checks - fails, checks);
    if (bench) {
        const double gbs = 1e-3;
        std::printf("--bench: cold weights, launches in a graph, medians (us per call or group; GB/s of the V2 path)\n");
        std::printf("  %-16s %2s %9s %9s %6s %8s\n", "matrix", "T", "old us", "V2 us", "x", "V2 GB/s");
        for (const Mat& m : MATS)
            for (int T : {1, 2, 4, 5, 8}) {
                const int no[1] = {m.n_out};
                const Timed r = time_set(1, no, m.n_in, T, s);
                const double b = (double) strata::kernels::native_mmvq_weight_bytes(Q8, m.n_in, m.n_out);
                std::printf("  %-16s %2d %9.2f %9.2f %6.2f %8.0f\n", m.name, T, r.old_us, r.v2_us, r.old_us / r.v2_us,
                            b / r.v2_us * gbs);
            }
        for (const Group& g : GROUPS)
            for (int T : {1, 2, 4, 5, 8}) {
                const Timed r = time_set(g.n, g.n_out, g.n_in, T, s);
                double b = 0;
                for (int k = 0; k < g.n; ++k) b += (double) strata::kernels::native_mmvq_weight_bytes(Q8, g.n_in, g.n_out[k]);
                std::printf("  %-16s %2d %9.2f %9.2f %6.2f %8.0f  (separate calls / one launch)\n", g.name, T, r.old_us,
                            r.v2_us, r.old_us / r.v2_us, b / r.v2_us * gbs);
            }
    }
    return fails ? 1 : 0;
}
