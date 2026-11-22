// src/kernels/gr_hc8_check.cpp - the hc8 read (STRATA_HC_Q8=1 on CUDA: int8 rows + fp16 scales, fp32 activations and
// sums) checked three ways, T = 1..8 tokens, with and without the pending write and the inject rows:
//   - window invariance: every token's outputs (R_out, rs, lo, inject, mixed) bitwise its outputs read alone (T = 1,
//     column 0), whatever T and its column;
//   - graphs: a captured read replayed twice gives the direct call's bits;
//   - accuracy: every output against an fp64 CPU reference on the same Q8_0 weights (and, for scale, the bf16 v4 read
//     against the fp64 reference on its bf16 weights); the q8_1 image of mixed (STRATA_QFUSE) against the host's.
// --bench: the time per read (down + up) of hc8 and the bf16 v4 read (the default), the weights from DRAM (24 sets in
// rotation), medians of 15 replays.
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace strata::kernels;

namespace {

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, MAXT = kFusedGrMaxT, NSETS = 24;
constexpr float EPS = 1e-6f;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "gr_hc8_check: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "alloc");
    ck(cudaMemset(p, 0, n * sizeof(T)), "memset");
    return p;
}
uint16_t bf16(float x) { uint32_t u; std::memcpy(&u, &x, 4); return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
float bf2f(uint16_t b) { uint32_t u = (uint32_t) b << 16; float f; std::memcpy(&f, &u, 4); return f; }
float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }

std::mt19937 rng(4321);
std::vector<float> gauss(size_t n, float s, float mean = 0.0f) {
    std::normal_distribution<float> g(mean, s);
    std::vector<float> v(n);
    for (auto& x : v) x = g(rng);
    return v;
}
template <typename T> void up(T* d, const std::vector<T>& h) { ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload"); }
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}

struct WSet {
    float* norm;
    uint16_t *down, *up, *inject;   // bf16
    uint8_t *qd, *qu;               // hc8 copies of down / up
    std::vector<float> hn, hd, hu, hi;   // host: norm, the Q8_0 values of down / up, the bf16 inject
    std::vector<float> hdb, hub;         // host: the bf16 values of down / up
};

std::vector<float> q8_values(const uint8_t* q8dev, size_t nb) {
    const auto b = down(q8dev, nb * 34);   // Q8_0 blocks (before the repack)
    std::vector<float> v(nb * 32);
    for (size_t i = 0; i < nb; ++i) {
        const float d = h2f((uint16_t) (b[i * 34] | (b[i * 34 + 1] << 8)));
        for (int j = 0; j < 32; ++j) v[i * 32 + j] = d * (float) (int8_t) b[i * 34 + 2 + j];
    }
    return v;
}

WSet make_set(bool host) {
    WSet w{};
    w.norm = dalloc<float>(D);
    w.down = dalloc<uint16_t>((size_t) LR * D);
    w.up = dalloc<uint16_t>((size_t) D * LR);
    w.inject = dalloc<uint16_t>((size_t) HC * D);
    const size_t nb = (size_t) LR * D / 32;
    w.qd = dalloc<uint8_t>(nb * 34);
    w.qu = dalloc<uint8_t>(nb * 34);
    auto bfv = [&](size_t n, float s, std::vector<float>* keep) {
        const auto f = gauss(n, s);
        std::vector<uint16_t> b(n);
        for (size_t i = 0; i < n; ++i) b[i] = bf16(f[i]);
        if (keep) { keep->resize(n); for (size_t i = 0; i < n; ++i) (*keep)[i] = bf2f(b[i]); }
        return b;
    };
    const auto n = gauss(D, 0.1f, 1.0f);
    up(w.norm, n);
    up(w.down, bfv((size_t) LR * D, 0.02f, host ? &w.hdb : nullptr));
    up(w.up, bfv((size_t) D * LR, 0.05f, host ? &w.hub : nullptr));
    up(w.inject, bfv((size_t) HC * D, 0.01f, host ? &w.hi : nullptr));
    fused_gr_quantize_q8_0(w.down, w.qd, (int64_t) nb, nullptr);
    fused_gr_quantize_q8_0(w.up, w.qu, (int64_t) nb, nullptr);
    ck(cudaDeviceSynchronize(), "quantize");
    if (host) {
        w.hn = n;
        w.hd = q8_values(w.qd, nb);
        w.hu = q8_values(w.qu, nb);
    }
    if (!fused_gr_hc8_repack(w.qd, (int64_t) nb) || !fused_gr_hc8_repack(w.qu, (int64_t) nb)) {
        std::fprintf(stderr, "gr_hc8_check: repack failed\n");
        std::exit(1);
    }
    return w;
}

struct Tok { float *R, *R_out, *bo, *inj, *lo, *rs, *inj_out, *mixed; uint8_t* xq; std::vector<float> hR, hbo, hinj; };

std::vector<FusedGrArgs> args_for(const WSet& w, std::vector<Tok*> tok, bool apply, bool inject, bool q8w, unsigned* qcnt) {
    std::vector<FusedGrArgs> a(tok.size());
    for (size_t t = 0; t < tok.size(); ++t) {
        FusedGrArgs& x = a[t];
        x.R = tok[t]->R; x.R_out = tok[t]->R_out; x.apply = apply;
        x.bo_prev = tok[t]->bo; x.inj_prev = tok[t]->inj;
        x.w_norm = w.norm; x.w_down = w.down; x.w_up = w.up;
        x.w_inject = inject ? w.inject : nullptr;
        if (q8w) { x.q8_down = w.qd; x.q8_up = w.qu; }
        x.eps = EPS;
        x.lo = tok[t]->lo; x.rs = tok[t]->rs; x.inject_out = tok[t]->inj_out; x.mixed = tok[t]->mixed;
        if (qcnt) { x.q8_cnt = qcnt; x.q8_mixed = tok[t]->xq; }
    }
    return a;
}

struct Out { std::vector<float> R_out, rs, lo, inj, mixed; std::vector<uint8_t> xq; };
Out snap(const Tok& t, bool q) {
    Out o{down(t.R_out, D), down(t.rs, HC), down(t.lo, LR), down(t.inj_out, HC), down(t.mixed, N), {}};
    if (q) o.xq = down(t.xq, (size_t) N / 32 * 36);
    return o;
}
bool same(const Out& a, const Out& b, bool apply, bool inject) {
    auto eqv = [](const std::vector<float>& x, const std::vector<float>& y) { return std::memcmp(x.data(), y.data(), x.size() * 4) == 0; };
    return (!apply || eqv(a.R_out, b.R_out)) && eqv(a.rs, b.rs) && eqv(a.lo, b.lo) && (!inject || eqv(a.inj, b.inj)) &&
           eqv(a.mixed, b.mixed) && a.xq == b.xq;
}
void clear(Tok& t, cudaStream_t st) {
    ck(cudaMemsetAsync(t.R_out, 0, D * 4, st), "clear");
    ck(cudaMemsetAsync(t.rs, 0, HC * 4, st), "clear");
    ck(cudaMemsetAsync(t.lo, 0, LR * 4, st), "clear");
    ck(cudaMemsetAsync(t.inj_out, 0, HC * 4, st), "clear");
    ck(cudaMemsetAsync(t.mixed, 0, N * 4, st), "clear");
    ck(cudaMemsetAsync(t.xq, 0, (size_t) N / 32 * 36, st), "clear");
}

// the fp64 reference of one token on host weights (wd / wu: the values the read multiplies)
struct Ref { std::vector<double> R_out, rs, lo, inj, mixed; };
Ref reference(const WSet& w, const std::vector<float>& wd, const std::vector<float>& wu, const Tok& t, bool apply, bool inject) {
    Ref r;
    r.R_out.resize(D);
    for (int c = 0; c < HC; ++c) {
        const double gw = apply ? 2.0 / (1.0 + std::exp(-(double) t.hinj[c] / HC)) : 0.0;
        for (int d = 0; d < N; ++d) r.R_out[(size_t) c * N + d] = (double) t.hR[(size_t) c * N + d] + (apply ? (double) t.hbo[d] * gw : 0.0);
    }
    r.rs.resize(HC);
    for (int c = 0; c < HC; ++c) {
        double ss = 0;
        for (int d = 0; d < N; ++d) ss += r.R_out[(size_t) c * N + d] * r.R_out[(size_t) c * N + d];
        r.rs[c] = 1.0 / std::sqrt(ss / N + EPS);
    }
    std::vector<double> xn(D);
    for (int i = 0; i < D; ++i) xn[i] = r.R_out[i] * w.hn[i] * r.rs[i / N];
    r.lo.resize(LR);
    for (int k = 0; k < LR; ++k) {
        double s = 0;
        const float* row = wd.data() + (size_t) k * D;
        for (int i = 0; i < D; ++i) s += (double) row[i] * xn[i];
        const double x = s / HC;
        r.lo[k] = x / (1.0 + std::exp(-x));
    }
    r.inj.assign(HC, 0.0);
    if (inject)
        for (int c = 0; c < HC; ++c) {
            double s = 0;
            for (int i = 0; i < D; ++i) s += (double) w.hi[(size_t) c * D + i] * xn[i];
            r.inj[c] = s;
        }
    r.mixed.resize(N);
    for (int d = 0; d < N; ++d) {
        double m = 0;
        for (int c = 0; c < HC; ++c) {
            const size_t i = (size_t) c * N + d;
            double s = 0;
            for (int k = 0; k < LR; ++k) s += (double) wu[i * LR + k] * r.lo[k];
            m += xn[i] / (1.0 + std::exp(-s));
        }
        r.mixed[d] = m / HC;
    }
    return r;
}
// the largest |gpu - ref| over the vector's RMS
double err(const std::vector<float>& g, const std::vector<double>& r) {
    double rms = 0, worst = 0;
    for (size_t i = 0; i < r.size(); ++i) rms += r[i] * r[i];
    rms = std::sqrt(rms / (double) r.size());
    for (size_t i = 0; i < r.size(); ++i) worst = std::max(worst, std::fabs((double) g[i] - r[i]));
    return rms > 0 ? worst / rms : worst;
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--bench") bench = true;
        else if (std::string(argv[i]) != "--selftest") { std::fprintf(stderr, "usage: gr_hc8_check [--selftest] [--bench]\n"); return 2; }
    }
    cudaDeviceProp p{};
    ck(cudaGetDeviceProperties(&p, 0), "device");
    std::printf("gr_hc8_check on %s (%d SMs)\n", p.name, p.multiProcessorCount);
    std::vector<Tok> tok((size_t) MAXT);
    for (auto& t : tok) {
        t.R = dalloc<float>(D); t.R_out = dalloc<float>(D); t.bo = dalloc<float>(N); t.inj = dalloc<float>(HC);
        t.lo = dalloc<float>(LR); t.rs = dalloc<float>(HC); t.inj_out = dalloc<float>(HC); t.mixed = dalloc<float>(N);
        t.xq = dalloc<uint8_t>((size_t) N / 32 * 36);
        // a residual stream with a few large dimensions, as the model's
        t.hR = gauss(D, 1.0f);
        for (int j = 0; j < 16; ++j) t.hR[(size_t) (rng() % D)] *= 40.0f;
        t.hbo = gauss(N, 0.5f);
        t.hinj = gauss(HC, 1.0f);
        up(t.R, t.hR); up(t.bo, t.hbo); up(t.inj, t.hinj);
    }
    float* xn = dalloc<float>((size_t) MAXT * D);
    unsigned* qcnt = dalloc<unsigned>(N / 32);
    cudaStream_t st = nullptr;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    const WSet w0 = make_set(true);
    int bad = 0, cases = 0;
    // 1. window invariance, graphs, the q8_1 image
    for (int apply = 0; apply < 2; ++apply)
        for (int inject = 0; inject < 2; ++inject)
            for (int qf = 0; qf < 2; ++qf) {
                std::vector<Out> solo((size_t) MAXT);
                for (int t = 0; t < MAXT; ++t) {
                    const auto a = args_for(w0, {&tok[(size_t) t]}, apply, inject, true, qf ? qcnt : nullptr);
                    clear(tok[(size_t) t], st);
                    fused_gr_read_multi(a.data(), 1, xn, st);
                    ck(cudaStreamSynchronize(st), "solo");
                    solo[(size_t) t] = snap(tok[(size_t) t], qf != 0);
                }
                for (int T = 1; T <= MAXT; ++T)
                    for (int rot = 0; rot < (T < MAXT ? 2 : 1); ++rot) {   // rot 1: the tokens from column 8 - T on
                        std::vector<Tok*> tp;
                        for (int k = 0; k < T; ++k) tp.push_back(&tok[(size_t) (rot ? MAXT - T + k : k)]);
                        const auto a = args_for(w0, tp, apply, inject, true, qf ? qcnt : nullptr);
                        for (auto* x : tp) clear(*x, st);
                        fused_gr_read_multi(a.data(), T, xn, st);
                        ck(cudaStreamSynchronize(st), "window");
                        bool ok = true;
                        for (int k = 0; k < T; ++k) ok = ok && same(snap(*tp[(size_t) k], qf != 0), solo[(size_t) (tp[(size_t) k] - tok.data())], apply, inject);
                        cudaGraph_t g = nullptr;
                        cudaGraphExec_t gx = nullptr;
                        ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
                        fused_gr_read_multi(a.data(), T, xn, st);
                        ck(cudaStreamEndCapture(st, &g), "capture end");
                        ck(cudaGraphInstantiate(&gx, g, nullptr, nullptr, 0), "instantiate");
                        for (auto* x : tp) clear(*x, st);
                        ck(cudaGraphLaunch(gx, st), "replay 1");
                        ck(cudaGraphLaunch(gx, st), "replay 2");
                        ck(cudaStreamSynchronize(st), "replay");
                        bool gok = true;
                        for (int k = 0; k < T; ++k) gok = gok && same(snap(*tp[(size_t) k], qf != 0), solo[(size_t) (tp[(size_t) k] - tok.data())], apply, inject);
                        cudaGraphExecDestroy(gx);
                        cudaGraphDestroy(g);
                        ++cases;
                        if (!ok || !gok) {
                            ++bad;
                            std::printf("  FAIL T=%d cols %s apply=%d inject=%d qfuse=%d: %s%s\n", T, rot ? "late" : "0..", apply, inject, qf,
                                        ok ? "" : "direct ", gok ? "" : "graph");
                        }
                    }
                // the q8_1 image against the host's quantization of the read's own mixed
                if (qf) {
                    int qbad = 0;
                    for (int t = 0; t < MAXT; ++t) {
                        const Out& o = solo[(size_t) t];
                        for (int b = 0; b < N / 32; ++b) {
                            float amax = 0;
                            for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(o.mixed[(size_t) b * 32 + j]));
                            const float d = amax / 127.0f;
                            const float dg = h2f((uint16_t) (o.xq[(size_t) b * 36] | (o.xq[(size_t) b * 36 + 1] << 8)));
                            if (std::fabs(dg - d) > 1e-3f * d + 1e-30f) ++qbad;
                            for (int j = 0; j < 32; ++j) {
                                const int q = (int) (int8_t) o.xq[(size_t) b * 36 + 4 + j];
                                const int qr = d > 0 ? (int) std::lround(o.mixed[(size_t) b * 32 + j] / d) : 0;
                                if (std::abs(q - qr) > 1) ++qbad;
                            }
                        }
                    }
                    ++cases;
                    if (qbad) { ++bad; std::printf("  FAIL q8_1 image (apply=%d inject=%d): %d values off\n", apply, inject, qbad); }
                    // dec/int: the image byte for byte against the separate quantizer (native_quantize_q8_1, the one every q8_1
                    // consumer was written for) on the read's own mixed - informational: STRATA_QFUSE is opt-in on CUDA
                    {
                        float* dm = nullptr;
                        void* dq = nullptr;
                        ck(cudaMalloc(&dm, (size_t) N * 4), "malloc");
                        ck(cudaMalloc(&dq, (size_t) N / 32 * 36), "malloc");
                        size_t diffb = 0, diffc = 0, blocks = 0;
                        for (int t = 0; t < MAXT; ++t) {
                            const Out& o = solo[(size_t) t];
                            ck(cudaMemcpy(dm, o.mixed.data(), (size_t) N * 4, cudaMemcpyHostToDevice), "up");
                            native_quantize_q8_1(dm, dq, N, 1, st);
                            ck(cudaStreamSynchronize(st), "quantize");
                            std::vector<uint8_t> ref((size_t) N / 32 * 36);
                            ck(cudaMemcpy(ref.data(), dq, ref.size(), cudaMemcpyDeviceToHost), "down");
                            for (int b = 0; b < N / 32; ++b, ++blocks) {
                                if (std::memcmp(ref.data() + b * 36, o.xq.data() + b * 36, 36) == 0) continue;
                                ++diffb;
                                for (int j = 0; j < 32; ++j) diffc += ref[(size_t) b * 36 + 4 + j] != o.xq[(size_t) b * 36 + 4 + j];
                            }
                        }
                        std::printf("  q8_1 image vs native_quantize_q8_1 (apply=%d inject=%d): %zu of %zu blocks differ, %zu codes\n",
                                    apply, inject, diffb, blocks, diffc);
                        cudaFree(dm); cudaFree(dq);
                    }
                }
            }
    std::printf("  window invariance (T 1..8, early and late columns), graph replays, q8_1 image: %d of %d cases pass\n",
                cases - bad, cases);
    // 2. accuracy against fp64 (and the bf16 v4 read against fp64 on its bf16 weights, for scale)
    std::printf("  accuracy, max |err| / RMS over 8 tokens (pending write + inject):\n");
    std::printf("    read             R_out      rs         lo         inject     mixed\n");
    double worst_hc8 = 0;
    for (int arm = 0; arm < 2; ++arm) {
        std::vector<Tok*> tp;
        for (auto& t : tok) tp.push_back(&t);
        const auto a = args_for(w0, tp, true, true, arm == 0, nullptr);
        for (auto* x : tp) clear(*x, st);
        fused_gr_read_multi(a.data(), MAXT, xn, st);
        ck(cudaStreamSynchronize(st), "accuracy");
        double e[5] = {0, 0, 0, 0, 0};
        for (int t = 0; t < MAXT; ++t) {
            const Ref r = reference(w0, arm == 0 ? w0.hd : w0.hdb, arm == 0 ? w0.hu : w0.hub, tok[(size_t) t], true, true);
            const Out o = snap(tok[(size_t) t], false);
            e[0] = std::max(e[0], err(o.R_out, r.R_out));
            e[1] = std::max(e[1], err(o.rs, r.rs));
            e[2] = arm == 0 ? std::max(e[2], err(o.lo, r.lo)) : -1;   // the bf16 read keeps lo on chip
            e[3] = std::max(e[3], err(o.inj, r.inj));
            e[4] = std::max(e[4], err(o.mixed, r.mixed));
        }
        char lo_s[16] = "        -";
        if (e[2] >= 0) std::snprintf(lo_s, sizeof lo_s, "%9.2e", e[2]);
        std::printf("    %-15s %9.2e  %9.2e  %s  %9.2e  %9.2e\n", arm == 0 ? "hc8 (Q8_0)" : "v4 (bf16)", e[0], e[1], lo_s, e[3], e[4]);
        if (arm == 0) for (double x : e) worst_hc8 = std::max(worst_hc8, x);
    }
    ++cases;
    if (!(worst_hc8 < 1e-4)) { ++bad; std::printf("  FAIL hc8 vs fp64: %.2e\n", worst_hc8); }
    if (bench) {
        std::vector<WSet> sets;
        sets.push_back(w0);
        for (int i = 1; i < NSETS; ++i) sets.push_back(make_set(false));
        std::printf("  time per read (down + up), %d weight sets in rotation, medians of 15 replays of %d reads:\n", NSETS, NSETS);
        std::printf("    T   v4 bf16 us   hc8 us   speedup\n");
        for (int T = 1; T <= MAXT; ++T) {
            cudaGraphExec_t gx[2] = {};
            for (int v = 0; v < 2; ++v) {
                cudaGraph_t g = nullptr;
                ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "bench capture");
                for (int s = 0; s < NSETS; ++s) {
                    std::vector<Tok*> tp;
                    for (int k = 0; k < T; ++k) tp.push_back(&tok[(size_t) k]);
                    const auto a = args_for(sets[(size_t) s], tp, true, true, v == 1, nullptr);
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
            std::vector<float> ms[2];
            for (int r = 0; r < 15; ++r)
                for (int j = 0; j < 2; ++j) {
                    const int v = (r + j) % 2;
                    cudaEventRecord(e0, st);
                    cudaGraphLaunch(gx[v], st);
                    cudaEventRecord(e1, st);
                    cudaEventSynchronize(e1);
                    float x = 0;
                    cudaEventElapsedTime(&x, e0, e1);
                    ms[v].push_back(x);
                }
            double us[2];
            for (int v = 0; v < 2; ++v) {
                std::sort(ms[v].begin(), ms[v].end());
                us[v] = 1000.0 * ms[v][ms[v].size() / 2] / NSETS;
                cudaGraphExecDestroy(gx[v]);
            }
            std::printf("    %d   %8.2f   %8.2f   %5.2fx\n", T, us[0], us[1], us[0] / us[1]);
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
        }
    }
    std::printf("gr_hc8_check: %d failures\n", bad);
    return bad ? 1 : 0;
}
