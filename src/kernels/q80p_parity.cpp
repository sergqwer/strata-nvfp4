// src/kernels/q80p_parity.cpp - the Q8_0 head's bit planes and two-stage verify head (q80_vhead.hpp) against native_mmvq.
//
//   q80p_parity --selftest                         synthetic head (65,536 rows) and inputs
//   q80p_parity --gguf <dense.gguf> --hidden <f>  the model's own output.weight and real head inputs (the engine's
//                                                  STRATA_HEAD_IN_DUMP records), [--windows N] (default all)
//
// Per window (its T columns) and per K in {1, 20, 64}:
//   1. exact: q80p_mmvq on the planes is BITWISE native_mmvq on the Q8_0 rows (every row, every column);
//   2. two-stage: every entry at or above the column's K-th largest exact logit is bitwise exact; every other entry is
//      its upper bound: at or above its exact value and strictly below the K-th largest;
//   3. consumers: the greedy pick and the sampled pick (temperature 1, top_p 0.95, top_k K - 0 for K = 64 - eight seeds)
//      of the screened logits are the exact logits' picks;
// then the same on an adversarial head: rows overwritten with copies of the column's top row (exact ties, above and
// below its index) and of its top and K-th rows with one code moved by +-1 (near ties the screen cannot separate).
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/q80_vhead.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int N = 2560, RB = 2720;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

struct Dev {
    void* q8 = nullptr;    // the Q8_0 head
    void* pl = nullptr;    // the same rows as planes
    float* x = nullptr;    // inputs
    void* xq = nullptr;
    float* yr = nullptr;   // native_mmvq
    float* ye = nullptr;   // q80p_mmvq
    float* ys = nullptr;   // q80p_vhead
    int* k = nullptr;
    void* scr = nullptr;
    int* tok = nullptr;
    cudaStream_t s = nullptr;
};

long long g_fail = 0, g_checks = 0, g_exact_rows = 0, g_screens = 0;

void fail(const char* what, int w, int K, int col, long long row, float a, float b) {
    if (g_fail < 20)
        std::fprintf(stderr, "FAIL %s: window %d K %d column %d row %lld: %.9g vs %.9g\n", what, w, K, col, row, a, b);
    ++g_fail;
}

std::vector<int> picks(Dev& d, const float* logits, int T, int V, const strata::kernels::SamplerParams& sp) {
    strata::kernels::sample_tokens(logits, T, V, nullptr, 0, sp, d.tok, d.s);
    std::vector<int> h((size_t) T);
    ck(cudaMemcpyAsync(h.data(), d.tok, (size_t) T * 4, cudaMemcpyDeviceToHost, d.s), "picks copy");
    ck(cudaStreamSynchronize(d.s), "picks");
    return h;
}

// One window on the head now in d.q8 / d.pl (both hold the same rows).
void check_window(Dev& d, int V, int T, int w, const char* tag) {
    using namespace strata::kernels;
    native_mmvq(8, d.q8, d.xq, d.yr, N, V, T, d.s);
    q80p_mmvq(d.pl, d.xq, d.ye, N, V, T, d.s);
    std::vector<float> yr((size_t) T * V), ye((size_t) T * V), ys((size_t) T * V);
    ck(cudaMemcpyAsync(yr.data(), d.yr, yr.size() * 4, cudaMemcpyDeviceToHost, d.s), "yr");
    ck(cudaMemcpyAsync(ye.data(), d.ye, ye.size() * 4, cudaMemcpyDeviceToHost, d.s), "ye");
    ck(cudaStreamSynchronize(d.s), "exact");
    ++g_checks;
    if (std::memcmp(yr.data(), ye.data(), yr.size() * 4) != 0) {
        for (size_t i = 0; i < yr.size(); ++i)
            if (std::memcmp(&yr[i], &ye[i], 4) != 0) { fail("exact bits", w, 0, (int) (i / V), (long long) (i % V), yr[i], ye[i]); break; }
    }
    for (int K : {1, 20, 64}) {
        ck(cudaMemcpyAsync(d.k, &K, 4, cudaMemcpyHostToDevice, d.s), "k");
        uint64_t c0 = 0, r0 = 0, c1 = 0, r1 = 0;
        ck(cudaStreamSynchronize(d.s), "k sync");
        q80p_vhead_stats(d.scr, c0, r0);
        q80p_vhead(d.pl, d.xq, d.ys, N, V, T, d.k, d.scr, d.s);
        ck(cudaMemcpyAsync(ys.data(), d.ys, ys.size() * 4, cudaMemcpyDeviceToHost, d.s), "ys");
        ck(cudaStreamSynchronize(d.s), "vhead");
        q80p_vhead_stats(d.scr, c1, r1);
        g_screens += (long long) (c1 - c0);
        g_exact_rows += (long long) (r1 - r0);
        if (c1 - c0 != 1) fail("the screen did not run", w, K, 0, 0, (float) c0, (float) c1);
        ++g_checks;
        for (int j = 0; j < T; ++j) {
            const float* r = yr.data() + (size_t) j * V;
            const float* sc = ys.data() + (size_t) j * V;
            std::vector<float> sorted(r, r + V);
            std::nth_element(sorted.begin(), sorted.begin() + (K - 1), sorted.end(), std::greater<float>());
            const float vk = sorted[(size_t) K - 1];
            for (int i = 0; i < V; ++i) {
                if (std::memcmp(&r[i], &sc[i], 4) == 0) continue;
                if (r[i] >= vk) fail("top-K entry not exact", w, K, j, i, r[i], sc[i]);
                else if (!(sc[i] >= r[i])) fail("bound below the exact logit", w, K, j, i, r[i], sc[i]);
                else if (!(sc[i] < vk)) fail("bound reaches the K-th logit", w, K, j, i, vk, sc[i]);
            }
        }
        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        if (K == 1 && picks(d, d.yr, T, V, sp) != picks(d, d.ys, T, V, sp)) fail("greedy pick", w, K, -1, -1, 0, 0);
        if (K > 1) {
            sp.greedy = false;
            sp.temperature = 1.0f;
            sp.top_p = 0.95f;
            sp.top_k = K == 64 ? 0 : K;
            for (uint64_t seed = 1; seed <= 8; ++seed) {
                sp.seed = seed * 7919u;
                sp.counter = (uint64_t) w * 16u + seed;
                if (picks(d, d.yr, T, V, sp) != picks(d, d.ys, T, V, sp)) fail("sampled pick", w, K, -1, (long long) seed, 0, 0);
            }
        }
    }
    (void) tag;
}

// Rows of column 0's top and K-th logit, copied over other rows (exact ties) and with one code moved (near ties).
void adversarial(Dev& d, std::vector<uint8_t>& host_q8, int V, int T, int w, std::mt19937& rng) {
    std::vector<float> yr((size_t) T * V);
    strata::kernels::native_mmvq(8, d.q8, d.xq, d.yr, N, V, T, d.s);
    ck(cudaMemcpyAsync(yr.data(), d.yr, yr.size() * 4, cudaMemcpyDeviceToHost, d.s), "adv yr");
    ck(cudaStreamSynchronize(d.s), "adv");
    std::vector<int> idx((size_t) V);
    for (int i = 0; i < V; ++i) idx[(size_t) i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 20, idx.end(), [&](int a, int b) { return yr[(size_t) a] > yr[(size_t) b]; });
    const int top = idx[0], k20 = idx[19];
    std::vector<uint8_t> mod = host_q8;
    std::uniform_int_distribution<int> any(0, V - 1), pos(0, 31), blk(0, N / 32 - 1);
    auto copy_row = [&](int src, int dst, bool nudge) {
        if (dst == top || dst == k20) return;
        std::memcpy(mod.data() + (size_t) dst * RB, host_q8.data() + (size_t) src * RB, RB);
        if (nudge) {
            int8_t* q = (int8_t*) (mod.data() + (size_t) dst * RB + (size_t) blk(rng) * 34 + 2) + pos(rng);
            const int v = *q + ((rng() & 1) ? 1 : -1);
            *q = (int8_t) std::max(-127, std::min(127, v));
        }
    };
    for (int i = 0; i < 8; ++i) copy_row(top, std::uniform_int_distribution<int>(0, std::max(0, top - 1))(rng), false);   // below its index
    for (int i = 0; i < 8; ++i) copy_row(top, any(rng), false);
    for (int i = 0; i < 16; ++i) copy_row(top, any(rng), true);
    for (int i = 0; i < 8; ++i) copy_row(k20, any(rng), false);
    for (int i = 0; i < 16; ++i) copy_row(k20, any(rng), true);
    ck(cudaMemcpy(d.q8, mod.data(), mod.size(), cudaMemcpyHostToDevice), "adv up");
    ck(cudaMemcpy(d.pl, mod.data(), mod.size(), cudaMemcpyHostToDevice), "adv up2");
    strata::kernels::q80p_pack(d.pl, V, d.s);
    check_window(d, V, T, w, "adversarial");
    ck(cudaMemcpy(d.q8, host_q8.data(), host_q8.size(), cudaMemcpyHostToDevice), "restore");
    ck(cudaMemcpy(d.pl, host_q8.data(), host_q8.size(), cudaMemcpyHostToDevice), "restore2");
    strata::kernels::q80p_pack(d.pl, V, d.s);
    ck(cudaStreamSynchronize(d.s), "restore pack");
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf, hidden;
    int max_windows = -1;
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") selftest = true;
        else if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--hidden" && i + 1 < argc) hidden = argv[++i];
        else if (a == "--windows" && i + 1 < argc) max_windows = std::atoi(argv[++i]);
    }
    if (!strata::kernels::q80p_supported(N, 1)) {
        std::printf("q80p_parity: SKIP (the planes need CUDA sm_80+)\n");
        return 0;
    }
    std::mt19937 rng(1234);
    std::vector<uint8_t> head;
    int V = 0;
    if (!gguf.empty()) {
        const strata::GgufFile f(gguf);
        const strata::TensorInfo* t = nullptr;
        for (const auto& ti : f.tensors())
            if (ti.name == "output.weight") t = &ti;
        if (t == nullptr || t->type != 8 || t->shape[0] != (uint64_t) N) { std::fprintf(stderr, "no Q8_0 output.weight of width %d\n", N); return 1; }
        V = (int) t->shape[1];
        head.assign(f.tensor_data(*t), f.tensor_data(*t) + (size_t) V * RB);
    } else {
        V = 65536;
        head.resize((size_t) V * RB);
        std::normal_distribution<float> nq(0.0f, 30.0f);
        std::uniform_real_distribution<float> nd(0.001f, 0.004f);
        for (int r = 0; r < V; ++r)
            for (int b = 0; b < N / 32; ++b) {
                uint8_t* blk = head.data() + (size_t) r * RB + (size_t) b * 34;
                const __half h = __float2half(nd(rng));
                std::memcpy(blk, &h, 2);
                for (int i = 0; i < 32; ++i)
                    blk[2 + i] = (uint8_t) (int8_t) std::max(-127, std::min(127, (int) std::lround(nq(rng))));
            }
    }
    // the windows' inputs
    std::vector<std::vector<float>> wins;
    if (!hidden.empty()) {
        std::FILE* f = std::fopen(hidden.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", hidden.c_str()); return 1; }
        int32_t hdr[4];
        while (std::fread(hdr, sizeof hdr, 1, f) == 1 && (max_windows < 0 || (int) wins.size() < max_windows)) {
            if (hdr[0] != 0x4E494448 || hdr[2] != N || hdr[1] < 1 || hdr[1] > 16) { std::fprintf(stderr, "bad record\n"); return 1; }
            std::vector<float> x((size_t) hdr[1] * N);
            if (std::fread(x.data(), 4, x.size(), f) != x.size()) break;
            for (int c = 0; c < hdr[1]; c += 8) {   // at most 8 columns a launch
                const int n = std::min(8, hdr[1] - c);
                wins.emplace_back(x.begin() + (size_t) c * N, x.begin() + (size_t) (c + n) * N);
            }
        }
        std::fclose(f);
    } else {
        std::normal_distribution<float> nx(0.0f, 1.0f);
        for (int w = 0; w < 24; ++w) {
            const int T = 1 + w % 8;
            std::vector<float> x((size_t) T * N);
            for (float& v : x) v = nx(rng);
            for (int i = 0; i < 8; ++i) x[(size_t) (rng() % x.size())] *= 12.0f;   // a few outliers, as real activations have
            wins.push_back(std::move(x));
        }
    }
    if (wins.empty()) { std::fprintf(stderr, "no windows\n"); return 1; }
    Dev d;
    ck(cudaStreamCreateWithFlags(&d.s, cudaStreamNonBlocking), "stream");
    ck(cudaMalloc(&d.q8, head.size()), "q8");
    ck(cudaMalloc(&d.pl, head.size()), "pl");
    ck(cudaMalloc((void**) &d.x, (size_t) 8 * N * 4), "x");
    ck(cudaMalloc(&d.xq, strata::kernels::native_q8_1_bytes(N, 8)), "xq");
    for (float** p : {&d.yr, &d.ye, &d.ys}) ck(cudaMalloc((void**) p, (size_t) 8 * V * 4), "y");
    ck(cudaMalloc((void**) &d.k, 16), "k");
    ck(cudaMalloc(&d.scr, strata::kernels::q80p_vhead_scratch_bytes()), "scr");
    ck(cudaMemset(d.scr, 0, strata::kernels::q80p_vhead_scratch_bytes()), "scr0");
    ck(cudaMalloc((void**) &d.tok, 64), "tok");
    ck(cudaMemcpy(d.q8, head.data(), head.size(), cudaMemcpyHostToDevice), "up");
    ck(cudaMemcpy(d.pl, head.data(), head.size(), cudaMemcpyHostToDevice), "up2");
    strata::kernels::q80p_pack(d.pl, V, d.s);
    {   // the layout round-trips
        void* rt = nullptr;
        ck(cudaMalloc(&rt, head.size()), "rt");
        ck(cudaMemcpyAsync(rt, d.pl, head.size(), cudaMemcpyDeviceToDevice, d.s), "rt copy");
        strata::kernels::q80p_unpack(rt, V, d.s);
        std::vector<uint8_t> back(head.size());
        ck(cudaMemcpyAsync(back.data(), rt, back.size(), cudaMemcpyDeviceToHost, d.s), "rt down");
        ck(cudaStreamSynchronize(d.s), "rt");
        cudaFree(rt);
        ++g_checks;
        if (back != head) fail("pack/unpack round trip", -1, 0, 0, 0, 0, 0);
    }
    int w = 0;
    for (const auto& x : wins) {
        const int T = (int) (x.size() / N);
        ck(cudaMemcpyAsync(d.x, x.data(), x.size() * 4, cudaMemcpyHostToDevice, d.s), "x up");
        strata::kernels::native_quantize_q8_1(d.x, d.xq, N, T, d.s);
        check_window(d, V, T, w, "plain");
        if (w % 4 == 0) adversarial(d, head, V, T, w, rng);
        ++w;
    }
    std::printf("q80p_parity: %s head, %d rows, %d windows, %lld checks, %lld screens, %.1f exact rows a screen: %s (%lld failures)\n",
                gguf.empty() ? "synthetic" : "real", V, w, g_checks, g_screens,
                g_screens ? (double) g_exact_rows / (double) g_screens : 0.0, g_fail ? "FAIL" : "PASS", g_fail);
    (void) selftest;
    return g_fail ? 1 : 0;
}
