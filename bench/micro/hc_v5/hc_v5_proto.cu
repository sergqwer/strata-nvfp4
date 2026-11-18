// bench/micro/hc_v5/hc_v5_proto.cu - prototype bench for the hyper-connection read v5 (decode verify windows T = 1..8).
// v4's kernels copied verbatim from src/kernels/cuda/fused_gr.cu (b08a70078) and the v5 candidates beside them; every
// candidate is compared bit for bit with v4 (T 1..8, pending write on/off, inject on/off) and timed like gr_v4_check
// --bench (24 weight sets in rotation, graph replays, medians).  Standalone: nvcc -O3 -arch=sm_120 -std=c++17 (MSVC: -Xcompiler /Zc:preprocessor);
// variant ids and names in hc_v5_kernels.cuh (variant_name); --part 1 / 2 times down / up alone, --l2 one weight set,
// --stamps per-phase clock64 stamps of the stamp variants.
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

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, PR = LR + HC, THREADS = 256, WARPS = 8, MAXT = 8, NSETS = 24;
constexpr int UPM_COLS = 16, UPM_BLOCKS = N / UPM_COLS;

struct FusedGrArgs {
    const float* R; float* R_out; bool apply; const float* bo_prev; const float* inj_prev; const float* w_norm;
    const uint16_t* w_down; const uint16_t* w_up; const uint16_t* w_inject; float eps; float* lo; float* rs;
    float* inject_out; float* mixed;
};
struct GrMulti { FusedGrArgs a[MAXT]; float* xn; int T; };

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }
struct Bf16x8 { float4 a, b; };
__device__ __forceinline__ Bf16x8 unpack8(const uint4 w) {
    return {make_float4(__uint_as_float(w.x << 16), __uint_as_float(w.x & 0xffff0000u), __uint_as_float(w.y << 16),
                        __uint_as_float(w.y & 0xffff0000u)),
            make_float4(__uint_as_float(w.z << 16), __uint_as_float(w.z & 0xffff0000u), __uint_as_float(w.w << 16),
                        __uint_as_float(w.w & 0xffff0000u))};
}
__device__ __forceinline__ float dot8u(const Bf16x8& w, const float4 x0, const float4 x1) {
    float acc = 0.0f;
    acc = fmaf(w.a.x, x0.x, acc); acc = fmaf(w.a.y, x0.y, acc); acc = fmaf(w.a.z, x0.z, acc); acc = fmaf(w.a.w, x0.w, acc);
    acc = fmaf(w.b.x, x1.x, acc); acc = fmaf(w.b.y, x1.y, acc); acc = fmaf(w.b.z, x1.z, acc); acc = fmaf(w.b.w, x1.w, acc);
    return acc;
}
__device__ __forceinline__ void gr_pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void gr_pdl_launch_dependents() { asm volatile("griddepcontrol.launch_dependents;" :::); }

#include "hc_v4_kernels.cuh"   // v4, verbatim
#include "hc_v5_kernels.cuh"   // the candidates

// ------------------------------------------------------------------------------------------------ host
static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
template <typename T> T* dalloc(size_t n) { T* p = nullptr; ck(cudaMalloc(&p, n * sizeof(T)), "alloc"); return p; }
static uint16_t bf16(float x) { uint32_t u; std::memcpy(&u, &x, 4); return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
struct WSet { float* norm; uint16_t *down, *up, *inject; };
struct Tok { float *R, *R_out, *bo, *inj, *lo, *rs, *inj_out, *mixed; };
static std::mt19937 rng(1234);
static std::vector<float> gauss(size_t n, float s, float mean = 0.0f) {
    std::normal_distribution<float> g(mean, s); std::vector<float> v(n); for (auto& x : v) x = g(rng); return v;
}
template <typename T> void up(T* d, const std::vector<T>& h) { ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload"); }
static std::vector<uint16_t> bfv(size_t n, float s) { const auto f = gauss(n, s); std::vector<uint16_t> b(n); for (size_t i = 0; i < n; ++i) b[i] = bf16(f[i]); return b; }
static WSet make_set() {
    WSet w{dalloc<float>(D), dalloc<uint16_t>((size_t) LR * D), dalloc<uint16_t>((size_t) D * LR), dalloc<uint16_t>((size_t) HC * D)};
    up(w.norm, gauss(D, 0.1f, 1.0f)); up(w.down, bfv((size_t) LR * D, 0.2f)); up(w.up, bfv((size_t) D * LR, 0.2f));
    up(w.inject, bfv((size_t) HC * D, 0.05f));
    return w;
}
static GrMulti make_m(const WSet& w, const std::vector<Tok>& tok, int T, bool apply, bool inject) {
    GrMulti m{};
    for (int t = 0; t < T; ++t) {
        FusedGrArgs& x = m.a[t];
        x.R = tok[t].R; x.R_out = tok[t].R_out; x.apply = apply; x.bo_prev = tok[t].bo; x.inj_prev = tok[t].inj;
        x.w_norm = w.norm; x.w_down = w.down; x.w_up = w.up; x.w_inject = inject ? w.inject : nullptr; x.eps = 1e-6f;
        x.lo = tok[t].lo; x.rs = tok[t].rs; x.inject_out = tok[t].inj_out; x.mixed = tok[t].mixed;
    }
    m.T = T;
    return m;
}
struct Snap { std::vector<float> R_out, rs, inj, mixed; };
static Snap snap(const std::vector<Tok>& tok, int T) {
    Snap s;
    auto get = [](std::vector<float>& v, const float* d, size_t n) { const size_t o = v.size(); v.resize(o + n); ck(cudaMemcpy(v.data() + o, d, n * 4, cudaMemcpyDeviceToHost), "download"); };
    for (int t = 0; t < T; ++t) { get(s.R_out, tok[t].R_out, D); get(s.rs, tok[t].rs, HC); get(s.inj, tok[t].inj_out, HC); get(s.mixed, tok[t].mixed, N); }
    return s;
}
static double maxdiff(const std::vector<float>& x, const std::vector<float>& y) {
    double w = 0; for (size_t i = 0; i < x.size(); ++i) w = std::max(w, (double) std::fabs(x[i] - y[i])); return w;
}
static bool eq(const Snap& a, const Snap& b, bool verbose) {
    const bool r = !std::memcmp(a.R_out.data(), b.R_out.data(), a.R_out.size() * 4), s = !std::memcmp(a.rs.data(), b.rs.data(), a.rs.size() * 4),
               i = !std::memcmp(a.inj.data(), b.inj.data(), a.inj.size() * 4), m = !std::memcmp(a.mixed.data(), b.mixed.data(), a.mixed.size() * 4);
    if (verbose && !(r && s && i && m))
        std::printf("      max|diff| R_out %.3e rs %.3e inject %.3e mixed %.3e\n", maxdiff(a.R_out, b.R_out), maxdiff(a.rs, b.rs),
                    maxdiff(a.inj, b.inj), maxdiff(a.mixed, b.mixed));
    return r && s && i && m;
}
static void clear(const std::vector<Tok>& tok, int T, cudaStream_t st) {
    for (int t = 0; t < T; ++t) {
        ck(cudaMemsetAsync(tok[t].R_out, 0, D * 4, st), "clear"); ck(cudaMemsetAsync(tok[t].rs, 0, HC * 4, st), "clear");
        ck(cudaMemsetAsync(tok[t].inj_out, 0, HC * 4, st), "clear"); ck(cudaMemsetAsync(tok[t].mixed, 0, N * 4, st), "clear");
    }
}

int main(int argc, char** argv) {
    std::vector<int> vars;          // variants to compare with v4 (ids in hc_v5_kernels.cuh)
    int part = 0;                   // 0 = the read, 1 = down only, 2 = up only (timing)
    int reps = 15;
    bool check = true, bench = true, l2 = false, stamps = false;   // --l2: one weight set (L2-resident)
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "--part") part = std::atoi(argv[++i]);
        else if (s == "--reps") reps = std::atoi(argv[++i]);
        else if (s == "--nocheck") check = false;
        else if (s == "--nobench") bench = false;
        else if (s == "--l2") l2 = true;
        else if (s == "--stamps") stamps = true;
        else vars.push_back(std::atoi(s.c_str()));
    }
    if (vars.empty()) vars.push_back(1);
    cudaDeviceProp p{};
    ck(cudaGetDeviceProperties(&p, 0), "device");
    std::printf("hc_v5_proto on %s (%d SMs)\n", p.name, p.multiProcessorCount);
    set_attrs();
    std::vector<Tok> tok(MAXT);
    for (auto& t : tok) {
        t = {dalloc<float>(D), dalloc<float>(D), dalloc<float>(N), dalloc<float>(HC), dalloc<float>(LR), dalloc<float>(HC), dalloc<float>(HC), dalloc<float>(N)};
        up(t.R, gauss(D, 1.0f)); up(t.bo, gauss(N, 0.5f)); up(t.inj, gauss(HC, 1.0f));
    }
    float* scratch = dalloc<float>((size_t) MAXT * D);
    cudaStream_t st = nullptr;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    std::vector<WSet> sets;
    sets.push_back(make_set());
    int bad = 0, cases = 0;
    if (check)
        for (int v : vars) {
            int vb = 0, vc = 0;
            for (int T = 1; T <= MAXT; ++T)
                for (int apply = 0; apply < 2; ++apply)
                    for (int inject = 0; inject < 2; ++inject) {
                        const GrMulti m = make_m(sets[0], tok, T, apply, inject);
                        clear(tok, T, st);
                        ck(launch_variant(0, m, scratch, T, st, 0), "v4");
                        ck(cudaStreamSynchronize(st), "v4 sync");
                        const Snap ref = snap(tok, T);
                        clear(tok, T, st);
                        ck(launch_variant(v, m, scratch, T, st, 0), "variant");
                        ck(cudaStreamSynchronize(st), "variant sync");
                        const bool d_ok = eq(ref, snap(tok, T), true);
                        cudaGraph_t g = nullptr; cudaGraphExec_t gx = nullptr;
                        ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "capture");
                        ck(launch_variant(v, m, scratch, T, st, 0), "variant capture");
                        ck(cudaStreamEndCapture(st, &g), "capture end");
                        ck(cudaGraphInstantiate(&gx, g, nullptr, nullptr, 0), "instantiate");
                        clear(tok, T, st);
                        ck(cudaGraphLaunch(gx, st), "replay 1"); ck(cudaGraphLaunch(gx, st), "replay 2");
                        ck(cudaStreamSynchronize(st), "replay sync");
                        const bool g_ok = eq(ref, snap(tok, T), true);
                        cudaGraphExecDestroy(gx); cudaGraphDestroy(g);
                        ++vc;
                        if (!d_ok || !g_ok) { ++vb; std::printf("    FAIL variant %d T=%d apply=%d inject=%d %s%s\n", v, T, apply, inject, d_ok ? "" : "direct ", g_ok ? "" : "graph"); }
                    }
            std::printf("  variant %d (%s) against v4: %d of %d cases the same bits\n", v, variant_name(v), vc - vb, vc);
            bad += vb; cases += vc;
        }
    if (stamps) {
        for (int i = (int) sets.size(); i < NSETS; ++i) sets.push_back(make_set());
        for (int v : vars)
            for (int T : {1, 4, 8}) {
                {
                    static long long z[256 * 10] = {};
                    ck(cudaMemcpyToSymbol(g_stamp, z, sizeof(z)), "zero stamps");
                }
                for (int s = 0; s < NSETS; ++s) ck(launch_variant(v, make_m(sets[l2 ? 0 : s], tok, T, true, true), scratch, T, st, 1), "stamp launch");
                ck(cudaStreamSynchronize(st), "stamp sync");
                long long h[256 * 10];
                ck(cudaMemcpyFromSymbol(h, g_stamp, sizeof(h)), "stamps");
                int nb = 0;
                for (int b = 0; b < 256; ++b) if (h[b * 10 + 1] != 0) nb = b + 1;
                std::printf("  %s T=%d phase clocks since start (mean / max over %d blocks):", variant_name(v), T, nb);
                for (int ph = 1; ph <= 8; ++ph) {
                    if (h[ph] == 0) continue;
                    double mean = 0; long long mx = 0;
                    for (int b = 0; b < nb; ++b) { mean += h[b * 10 + ph]; mx = std::max(mx, h[b * 10 + ph]); }
                    std::printf(" %d:%.0f/%lld", ph, mean / nb, mx);
                }
                std::printf("\n");
            }
    }
    if (bench) {
        for (int i = 1; i < NSETS; ++i) sets.push_back(make_set());
        const double wbytes = (double) LR * D * 2 * 2 + (double) HC * D * 2;
        std::vector<int> all = {0};
        for (int v : vars) all.push_back(v);
        std::printf("  time per read (%s), %d weight sets in rotation, medians of %d replays; us (GB/s of 13.2 MB)\n",
                    part == 0 ? "down + up" : part == 1 ? "down only" : "up only", NSETS, reps);
        std::printf("    T");
        for (int v : all) std::printf("  %14s", variant_name(v));
        std::printf("\n");
        for (int T = 1; T <= MAXT; ++T) {
            std::vector<cudaGraphExec_t> gx(all.size());
            for (size_t j = 0; j < all.size(); ++j) {
                cudaGraph_t g = nullptr;
                ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal), "bench capture");
                for (int s = 0; s < NSETS; ++s) ck(launch_variant(all[j], make_m(sets[l2 ? 0 : s], tok, T, true, true), scratch, T, st, part), "bench launch");
                ck(cudaStreamEndCapture(st, &g), "bench capture end");
                ck(cudaGraphInstantiate(&gx[j], g, nullptr, nullptr, 0), "bench instantiate");
                cudaGraphDestroy(g);
                ck(cudaGraphLaunch(gx[j], st), "warm");
            }
            ck(cudaStreamSynchronize(st), "warm sync");
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            std::vector<std::vector<float>> ms(all.size());
            for (int r = 0; r < reps; ++r)
                for (size_t j = 0; j < all.size(); ++j) {
                    const size_t v = (r + j) % all.size();
                    cudaEventRecord(e0, st); cudaGraphLaunch(gx[v], st); cudaEventRecord(e1, st); cudaEventSynchronize(e1);
                    float x = 0; cudaEventElapsedTime(&x, e0, e1); ms[v].push_back(x);
                }
            std::printf("    %d", T);
            for (size_t j = 0; j < all.size(); ++j) {
                std::sort(ms[j].begin(), ms[j].end());
                const double us = 1000.0 * ms[j][ms[j].size() / 2] / NSETS;
                std::printf("  %6.2f (%5.0f)", us, wbytes / us / 1e3);
                cudaGraphExecDestroy(gx[j]);
            }
            std::printf("\n");
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        }
    }
    std::printf("hc_v5_proto: %d of %d check cases failed\n", bad, cases);
    return bad ? 1 : 0;
}
