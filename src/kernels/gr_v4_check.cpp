// src/kernels/gr_v4_check.cpp - the hyper-connection read v4 (STRATA_GR_V4=1: the weights first; =2: and launched as
// programmatic dependents) against v3: the same bits for T = 1..8 tokens, with and without the pending write and the
// inject rows, called directly and through a captured graph replayed twice.  --bench: the time per read (down + up)
// at those T, the weights from DRAM as in decode (24 weight sets in rotation, 316 MB, more than the L2 holds), the
// variants alternating.
#include "strata/kernels/fused_gr.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::kernels;

namespace {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, MAXT = kFusedGrMaxT, NSETS = 24;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "gr_v4_check: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "alloc");
    return p;
}
uint16_t bf16(float x) { uint32_t u; std::memcpy(&u, &x, 4); return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }

struct WSet { float* norm; uint16_t *down, *up, *inject; };
struct Tok { float *R, *R_out, *bo, *inj, *lo, *rs, *inj_out, *mixed; };

std::mt19937 rng(1234);
std::vector<float> gauss(size_t n, float s, float mean = 0.0f) {
    std::normal_distribution<float> g(mean, s);
    std::vector<float> v(n);
    for (auto& x : v) x = g(rng);
    return v;
}
template <typename T> void up(T* d, const std::vector<T>& h) { ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload"); }
std::vector<uint16_t> bfv(size_t n, float s) {
    const auto f = gauss(n, s);
    std::vector<uint16_t> b(n);
    for (size_t i = 0; i < n; ++i) b[i] = bf16(f[i]);
    return b;
}

WSet make_set() {
    WSet w{dalloc<float>(D), dalloc<uint16_t>((size_t) LR * D), dalloc<uint16_t>((size_t) D * LR), dalloc<uint16_t>((size_t) HC * D)};
    up(w.norm, gauss(D, 0.1f, 1.0f));
    up(w.down, bfv((size_t) LR * D, 0.2f));
    up(w.up, bfv((size_t) D * LR, 0.2f));
    up(w.inject, bfv((size_t) HC * D, 0.05f));
    return w;
}

std::vector<FusedGrArgs> args_for(const WSet& w, const std::vector<Tok>& tok, int T, bool apply, bool inject) {
    std::vector<FusedGrArgs> a((size_t) T);
    for (int t = 0; t < T; ++t) {
        FusedGrArgs& x = a[(size_t) t];
        x.R = tok[(size_t) t].R;
        x.R_out = tok[(size_t) t].R_out;
        x.apply = apply;
        x.bo_prev = tok[(size_t) t].bo;
        x.inj_prev = tok[(size_t) t].inj;
        x.w_norm = w.norm;
        x.w_down = w.down;
        x.w_up = w.up;
        x.w_inject = inject ? w.inject : nullptr;
        x.lo = tok[(size_t) t].lo;
        x.rs = tok[(size_t) t].rs;
        x.inject_out = tok[(size_t) t].inj_out;
        x.mixed = tok[(size_t) t].mixed;
    }
    return a;
}

struct Snap { std::vector<float> R_out, rs, inj, mixed; };
Snap snap(const std::vector<Tok>& tok, int T) {
    Snap s;
    auto get = [](std::vector<float>& v, const float* d, size_t n) {
        const size_t o = v.size();
        v.resize(o + n);
        ck(cudaMemcpy(v.data() + o, d, n * 4, cudaMemcpyDeviceToHost), "download");
    };
    for (int t = 0; t < T; ++t) {
        get(s.R_out, tok[(size_t) t].R_out, D);
        get(s.rs, tok[(size_t) t].rs, HC);
        get(s.inj, tok[(size_t) t].inj_out, HC);
        get(s.mixed, tok[(size_t) t].mixed, N);
    }
    return s;
}
bool eq(const Snap& a, const Snap& b) {
    auto cmp = [](const std::vector<float>& x, const std::vector<float>& y, const char* what) {
        if (std::memcmp(x.data(), y.data(), x.size() * 4) == 0) return true;
        if (std::getenv("GRV4_DEBUG")) {
            size_t n = 0, first = 0;
            double worst = 0;
            for (size_t i = 0; i < x.size(); ++i)
                if (std::memcmp(&x[i], &y[i], 4) != 0) {
                    if (n++ == 0) first = i;
                    worst = std::max(worst, (double) std::fabs(x[i] - y[i]));
                }
            std::printf("    %s: %zu of %zu differ, first %zu, worst |diff| %.3e\n", what, n, x.size(), first, worst);
        }
        return false;
    };
    const bool r = cmp(a.R_out, b.R_out, "R_out"), s = cmp(a.rs, b.rs, "rs"), i = cmp(a.inj, b.inj, "inject"),
               m = cmp(a.mixed, b.mixed, "mixed");
    return r && s && i && m;
}
// on the read's own stream (a non-blocking one does not wait for the legacy stream's memsets)
void clear(const std::vector<Tok>& tok, int T, cudaStream_t st) {
    for (int t = 0; t < T; ++t) {
        ck(cudaMemsetAsync(tok[(size_t) t].R_out, 0, D * 4, st), "clear");
        ck(cudaMemsetAsync(tok[(size_t) t].rs, 0, HC * 4, st), "clear");
        ck(cudaMemsetAsync(tok[(size_t) t].inj_out, 0, HC * 4, st), "clear");
        ck(cudaMemsetAsync(tok[(size_t) t].mixed, 0, N * 4, st), "clear");
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--bench") bench = true;
        else if (std::string(argv[i]) != "--selftest") { std::fprintf(stderr, "usage: gr_v4_check [--selftest] [--bench]\n"); return 2; }
    }
    cudaDeviceProp p{};
    ck(cudaGetDeviceProperties(&p, 0), "device");
    std::printf("gr_v4_check on %s (%d SMs)\n", p.name, p.multiProcessorCount);
    std::vector<Tok> tok((size_t) MAXT);
    for (auto& t : tok) {
        t = {dalloc<float>(D), dalloc<float>(D), dalloc<float>(N), dalloc<float>(HC), dalloc<float>(LR), dalloc<float>(HC),
             dalloc<float>(HC), dalloc<float>(N)};
        up(t.R, gauss(D, 1.0f));
        up(t.bo, gauss(N, 0.5f));
        up(t.inj, gauss(HC, 1.0f));
    }
    float* xn = dalloc<float>((size_t) MAXT * D);
    cudaStream_t st = nullptr;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    const WSet w0 = make_set();
    int bad = 0, cases = 0;
    for (int T = 1; T <= MAXT; ++T)
        for (int apply = 0; apply < 2; ++apply)
            for (int inject = 0; inject < 2; ++inject) {
                const auto a = args_for(w0, tok, T, apply != 0, inject != 0);
                fused_gr_set_v4(0);
                clear(tok, T, st);
                fused_gr_read_multi(a.data(), T, xn, st);
                ck(cudaStreamSynchronize(st), "v3");
                const Snap ref = snap(tok, T);
                for (int mode = std::getenv("GRV4_SELF") ? 0 : 1; mode <= 2; ++mode) {
                    fused_gr_set_v4(mode);
                    clear(tok, T, st);
                    fused_gr_read_multi(a.data(), T, xn, st);
                    ck(cudaStreamSynchronize(st), "v4 direct");
                    const bool d_ok = eq(ref, snap(tok, T));
                    cudaGraph_t g = nullptr;
                    cudaGraphExec_t gx = nullptr;
                    ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
                    fused_gr_read_multi(a.data(), T, xn, st);
                    ck(cudaStreamEndCapture(st, &g), "capture end");
                    ck(cudaGraphInstantiate(&gx, g, nullptr, nullptr, 0), "instantiate");
                    clear(tok, T, st);
                    ck(cudaGraphLaunch(gx, st), "replay 1");
                    ck(cudaGraphLaunch(gx, st), "replay 2");
                    ck(cudaStreamSynchronize(st), "replay sync");
                    const bool g_ok = eq(ref, snap(tok, T));
                    cudaGraphExecDestroy(gx);
                    cudaGraphDestroy(g);
                    ++cases;
                    if (!d_ok || !g_ok) {
                        std::printf("  FAIL T=%d apply=%d inject=%d STRATA_GR_V4=%d: %s%s\n", T, apply, inject, mode,
                                    d_ok ? "" : "direct ", g_ok ? "" : "graph");
                        ++bad;
                    }
                }
            }
    std::printf("  v4 (=1 and =2) against v3, T 1..8, pending write on/off, inject on/off, direct and graph: %d of %d "
                "cases the same bits\n", cases - bad, cases);
    if (bench) {
        std::vector<WSet> sets;
        sets.push_back(w0);
        for (int i = 1; i < NSETS; ++i) sets.push_back(make_set());
        const double wbytes = (double) LR * D * 2 * 2 + (double) HC * D * 2;   // down + up + inject, BF16
        std::printf("  time per read (down + up), %d weight sets in rotation (%.0f MB), medians of 15 replays of %d reads:\n",
                    NSETS, NSETS * wbytes / 1e6, NSETS);
        std::printf("    T   v3 us  (GB/s)    v4=1 us (GB/s)  speedup   v4=2 us (GB/s)  speedup\n");
        for (int T = 1; T <= MAXT; ++T) {
            cudaGraphExec_t gx[3] = {};
            for (int v = 0; v < 3; ++v) {
                fused_gr_set_v4(v);
                cudaGraph_t g = nullptr;
                ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "bench capture");
                for (int s = 0; s < NSETS; ++s) {
                    const auto a = args_for(sets[(size_t) s], tok, T, true, true);
                    fused_gr_read_multi(a.data(), T, xn, st);
                }
                ck(cudaStreamEndCapture(st, &g), "bench capture end");
                ck(cudaGraphInstantiate(&gx[v], g, nullptr, nullptr, 0), "bench instantiate");
                cudaGraphDestroy(g);
                ck(cudaGraphLaunch(gx[v], st), "warm");
            }
            ck(cudaStreamSynchronize(st), "warm sync");
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            std::vector<float> ms[3];
            for (int r = 0; r < 15; ++r)
                for (int j = 0; j < 3; ++j) {
                    const int v = (r + j) % 3;
                    cudaEventRecord(e0, st);
                    cudaGraphLaunch(gx[v], st);
                    cudaEventRecord(e1, st);
                    cudaEventSynchronize(e1);
                    float x = 0;
                    cudaEventElapsedTime(&x, e0, e1);
                    ms[v].push_back(x);
                }
            double us[3];
            for (int v = 0; v < 3; ++v) {
                std::sort(ms[v].begin(), ms[v].end());
                us[v] = 1000.0 * ms[v][ms[v].size() / 2] / NSETS;
                cudaGraphExecDestroy(gx[v]);
            }
            std::printf("    %d  %6.2f (%5.0f)   %6.2f (%5.0f)  %5.2fx   %6.2f (%5.0f)  %5.2fx\n", T, us[0], wbytes / us[0] / 1e3,
                        us[1], wbytes / us[1] / 1e3, us[0] / us[1], us[2], wbytes / us[2] / 1e3, us[0] / us[2]);
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
        }
    }
    std::printf("gr_v4_check: %d failures\n", bad);
    return bad ? 1 : 0;
}
