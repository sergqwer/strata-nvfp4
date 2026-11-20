// src/prefill/fp4g_parity.cpp - the grouped FP4 kernel (mmq_nvfp4_fp4g.cu, STRATA_PF_FP4G) against the MMQ path it
// replaces, on one synthetic in-place group as the streamed walk runs it (16 experts, R rows each, gate/up read per
// token through a row table):
//     STRATA_PREFILL_NVFP4=w4a4x2 (the default) fp4g_parity [R ...]
// MMQ: gate/up (w4a4x2) -> swiglu_quant (H's q8_1 + s_down) -> down (w4a8).  Mode 1: the FP4 kernel's gate/up with
// SwiGLU + q8_1 in its epilogue, MMQ's down.  Mode 2: the FP4 kernel for both, H as two FP4 terms.  Prints H's q8_1
// bytes that differ (mode 1), the down output's relative RMS difference against MMQ's, and the medians of 15 runs.
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace mmq = strata::prefill::mmq;
namespace strata::prefill::fp4g {
bool available();
int run_group(int n, const void* const* gu, const void* const* dn, const float* const* tail, const int32_t* bounds_gu,
              const int32_t* bounds_d, int64_t r0, int64_t nr, int64_t max_rows, const void* xq, const float* xscale,
              const int32_t* yrows, int64_t x_rows, float* h, void* hq, float* hscale, float* sd, float* dm, int mode,
              void* stream);
}
namespace strata::prefill::mmq {
void quantize_nvfp4_x2(const float* x, const int32_t* ids, void* xq, float* yscale, int64_t cols, int64_t ld,
                       int64_t rows, int64_t padded, cudaStream_t s);
}

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
template <class T> T* dalloc(size_t n) { T* p = nullptr; ck(cudaMalloc(&p, n * sizeof(T) + 4096), "alloc"); return p; }

// NVFP4 rows with scales in the model's range (e4m3 0.5..1 x a per-expert spread) and random codes
std::vector<uint8_t> nvfp4(int64_t rows, int64_t cols, std::mt19937& rng) {
    std::vector<uint8_t> w((size_t) rows * cols / 64 * 36);
    std::uniform_int_distribution<int> byte(0, 255);
    for (size_t b = 0; b < w.size() / 36; ++b) {
        uint8_t* blk = w.data() + b * 36;
        for (int s = 0; s < 4; ++s) blk[s] = (uint8_t) (0x30 + byte(rng) % 16);
        for (int i = 4; i < 36; ++i) blk[i] = (uint8_t) byte(rng);
    }
    return w;
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<int> Rs;
    for (int i = 1; i < argc; ++i) Rs.push_back(std::atoi(argv[i]));
    if (Rs.empty()) Rs = {12, 160, 626};
    if (!strata::prefill::fp4g::available() || !mmq::fp4_activations(GGML_TYPE_NVFP4)) {
        std::printf("fp4g_parity: needs sm_120a and STRATA_PREFILL_NVFP4=w4a4x2 (the default)\n");
        return 77;
    }
    constexpr int E = 16, N = 2560, FF = 640, K_USED = 10;
    std::mt19937 rng(11);
    std::vector<const void*> gu(E), dn(E);
    std::vector<std::vector<uint8_t>> dn_host(E);   // the down weights, for the FP64 reference
    std::vector<const float*> tails(E);
    float* dtail = dalloc<float>(4 * E);
    std::vector<float> th(4 * E);
    std::uniform_real_distribution<float> ts(0.5f, 2.0f);
    for (int e = 0; e < E; ++e) {
        th[4 * e] = ts(rng); th[4 * e + 1] = ts(rng); th[4 * e + 2] = ts(rng); th[4 * e + 3] = 0.0f;
        std::vector<uint8_t> a = nvfp4(2 * FF, N, rng), b = nvfp4(N, FF, rng);
        dn_host[e] = b;
        uint8_t* p = dalloc<uint8_t>(a.size() + b.size() + 4096);
        ck(cudaMemcpy(p, a.data(), a.size(), cudaMemcpyHostToDevice), "gu");
        const size_t doff = (a.size() + 255) / 256 * 256;
        ck(cudaMemcpy(p + doff, b.data(), b.size(), cudaMemcpyHostToDevice), "d");
        gu[e] = p; dn[e] = p + doff;
        tails[e] = dtail + 4 * e;
    }
    ck(cudaMemcpy(dtail, th.data(), th.size() * 4, cudaMemcpyHostToDevice), "tails");
    std::printf("fp4g_parity: 16 experts, gate/up read per token; rel RMS against MMQ; medians of 15 (ms)\n");
    std::printf("%6s %12s %12s %12s %10s %10s %10s %10s %11s %11s\n", "R", "Hq diff", "m1 relRMS", "m2 relRMS", "mmq ms", "m1 gu ms",
                "m1 all ms", "m2 ms", "w4a8:FP64", "x2:FP64");
    for (int R : Rs) {
        const int rows = E * R, T = (rows + K_USED - 1) / K_USED;
        std::vector<float> x((size_t) T * N);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::uniform_real_distribution<float> mag(0.1f, 3.f);
        for (int t = 0; t < T; ++t) { const float m = mag(rng); for (int k = 0; k < N; ++k) x[(size_t) t * N + k] = nd(rng) * m; }
        std::vector<int32_t> bounds(E + 1), src(rows);
        for (int e = 0; e <= E; ++e) bounds[e] = e * R;
        for (int r = 0; r < rows; ++r) src[r] = (int32_t) (((int64_t) r * 7919 + 17) % T);
        float* dx = dalloc<float>(x.size());
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
        void* xq = dalloc<uint8_t>(mmq::q8_bytes(T, N));
        float* xs = dalloc<float>(T);
        int32_t* db = dalloc<int32_t>(E + 1);
        int32_t* dsrc = dalloc<int32_t>(rows);
        int32_t* dids = dalloc<int32_t>((size_t) rows + 4096);
        ck(cudaMemcpy(db, bounds.data(), (E + 1) * 4, cudaMemcpyHostToDevice), "b");
        ck(cudaMemcpy(dsrc, src.data(), rows * 4, cudaMemcpyHostToDevice), "src");
        mmq::iota(dids, (int64_t) rows + 4096, nullptr);
        mmq::quantize(dx, nullptr, xq, GGML_TYPE_NVFP4, N, N, T, nullptr, xs);
        float* GU = dalloc<float>((size_t) rows * 2 * FF);
        float* H = dalloc<float>((size_t) rows * FF);
        void* hq_ref = dalloc<uint8_t>(mmq::q8_bytes(rows, FF));
        void* hq = dalloc<uint8_t>(mmq::q8_bytes(rows, FF));
        float* hs = dalloc<float>(rows);
        float* sd_ref = dalloc<float>(rows);
        float* sd = dalloc<float>(rows);
        float* dm_ref = dalloc<float>((size_t) rows * N);
        float* dm = dalloc<float>((size_t) rows * N);
        mmq::Context ctx;
        auto mmq_gu = [&]() {
            mmq::Product p;
            p.w = gu[0]; p.type = GGML_TYPE_NVFP4; p.w_rows = 2 * FF; p.w_cols = N; p.expert_bytes = 0; p.n = E;
            p.xq = xq; p.bounds = db; p.ids = dids; p.total_rows = rows; p.max_rows = R; p.dst = GU; p.ld_dst = 2 * FF;
            p.y_scale = xs; p.w_ptrs = gu.data(); p.y_rows = dsrc; p.y_count = T;
            ctx.run(p, nullptr);
            mmq::swiglu_quant(GU, hq_ref, rows, db, E, tails.data(), 0, sd_ref, nullptr);
        };
        auto mmq_down = [&](void* hqx, float* out) {
            mmq::Product p;
            p.w = dn[0]; p.type = GGML_TYPE_NVFP4; p.w_rows = N; p.w_cols = FF; p.expert_bytes = 0; p.n = E;
            p.xq = hqx; p.bounds = db; p.ids = dids; p.total_rows = rows; p.max_rows = R; p.dst = out; p.ld_dst = N;
            p.y_scale = hs; p.w_ptrs = dn.data();
            ctx.run(p, nullptr);
        };
        auto fp4g = [&](int mode, float* out) {
            return strata::prefill::fp4g::run_group(E, gu.data(), dn.data(), tails.data(), db, db, 0, rows, R, xq, xs, dsrc, T,
                                                    H, hq, hs, sd, out, mode, nullptr);
        };
        // references and results
        mmq_gu();
        mmq_down(hq_ref, dm_ref);
        if (fp4g(1, nullptr) != 1) { std::printf("mode 1 not taken\n"); return 1; }
        mmq_down(hq, dm);
        ck(cudaDeviceSynchronize(), "run");
        const size_t hqb = mmq::q8_bytes(rows, FF);
        std::vector<uint8_t> a(hqb), b(hqb);
        ck(cudaMemcpy(a.data(), hq_ref, hqb, cudaMemcpyDeviceToHost), "a");
        ck(cudaMemcpy(b.data(), hq, hqb, cudaMemcpyDeviceToHost), "b");
        // only the five 128-value blocks of the 640 values are compared (the padding blocks are never read)
        size_t diff = 0, tot = 0;
        for (int64_t kb = 0; kb < 5; ++kb)
            for (int64_t r = 0; r < rows; ++r) {
                const size_t o = ((size_t) kb * rows + r) * 144;
                for (int i = 0; i < 144; ++i) { diff += a[o + i] != b[o + i]; ++tot; }
            }
        std::vector<float> ref((size_t) rows * N), got((size_t) rows * N);
        std::vector<float> sr(rows), sg(rows);
        ck(cudaMemcpy(ref.data(), dm_ref, ref.size() * 4, cudaMemcpyDeviceToHost), "r");
        ck(cudaMemcpy(sr.data(), sd_ref, rows * 4, cudaMemcpyDeviceToHost), "sr");
        auto rel = [&](float* d) {
            ck(cudaMemcpy(got.data(), d, got.size() * 4, cudaMemcpyDeviceToHost), "g");
            double num = 0, den = 0;
            for (size_t i = 0; i < ref.size(); ++i) { const double e = (double) got[i] - ref[i]; num += e * e; den += (double) ref[i] * ref[i]; }
            return std::sqrt(num / den);
        };
        const double r1 = rel(dm);
        std::vector<float> d_w4a8(got);   // mode 1's down (MMQ w4a8 on the q8_1 H of the kernel's epilogue)
        ck(cudaMemcpy(sg.data(), sd, rows * 4, cudaMemcpyDeviceToHost), "sg");
        const bool sd_ok = std::memcmp(sr.data(), sg.data(), rows * 4) == 0;
        if (fp4g(2, dm) != 2) { std::printf("mode 2 not taken\n"); return 1; }
        ck(cudaDeviceSynchronize(), "run2");
        const double r2 = rel(dm);
        std::vector<float> d_x2(got);     // mode 2's down (two FP4 terms of the same H values)
        // one down product against FP64 on H's FP32 values (mode 2's gate/up epilogue: the values mode 1 rounds to
        // q8_1): the first 32 rows of each expert, the weights decoded as the FP4 MMA reads them (E2M1 x E4M3)
        double e8 = 0, e2 = 0, den = 0;
        {
            std::vector<float> hh((size_t) rows * FF);
            ck(cudaMemcpy(hh.data(), H, hh.size() * 4, cudaMemcpyDeviceToHost), "H");
            static const float e2m1[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -0.5f, -1, -1.5f, -2, -3, -4, -6};
            auto e4m3 = [](uint8_t b) {
                const int e = (b >> 3) & 15, m = b & 7;
                return e == 0 ? std::ldexp((double) m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, e - 7);
            };
            std::vector<double> wrow(FF);
            for (int e = 0; e < E; ++e)
                for (int n = 0; n < N; ++n) {
                    const uint8_t* rowp = dn_host[e].data() + (size_t) n * (FF / 64) * 36;
                    for (int b = 0; b < FF / 64; ++b)
                        for (int sb = 0; sb < 4; ++sb) {
                            const double sc = e4m3(rowp[b * 36 + sb]);
                            for (int j = 0; j < 8; ++j) {
                                const uint8_t c = rowp[b * 36 + 4 + sb * 8 + j];
                                wrow[b * 64 + sb * 16 + j] = e2m1[c & 15] * sc;
                                wrow[b * 64 + sb * 16 + j + 8] = e2m1[c >> 4] * sc;
                            }
                        }
                    for (int r = e * R; r < e * R + std::min(R, 32); ++r) {
                        double ref = 0;
                        for (int k = 0; k < FF; ++k) ref += (double) hh[(size_t) r * FF + k] * wrow[k];
                        const size_t i = (size_t) r * N + n;
                        e8 += (d_w4a8[i] - ref) * (d_w4a8[i] - ref);
                        e2 += (d_x2[i] - ref) * (d_x2[i] - ref);
                        den += ref * ref;
                    }
                }
        }
        // mode 2's H quantizer (a warp per row) against quantize_nvfp4_x2 on the same H: the bytes of the 640 values
        bool q_same = true;
        {
            std::vector<uint8_t> qa(hqb), qb(hqb);
            ck(cudaMemcpy(qa.data(), hq, hqb, cudaMemcpyDeviceToHost), "qa");
            std::vector<float> sa(rows), sb(rows);
            ck(cudaMemcpy(sa.data(), hs, rows * 4, cudaMemcpyDeviceToHost), "sa");
            mmq::quantize_nvfp4_x2(H, nullptr, hq, hs, FF, FF, rows, 1024, nullptr);
            ck(cudaDeviceSynchronize(), "q");
            ck(cudaMemcpy(qb.data(), hq, hqb, cudaMemcpyDeviceToHost), "qb");
            ck(cudaMemcpy(sb.data(), hs, rows * 4, cudaMemcpyDeviceToHost), "sb");
            q_same = std::memcmp(sa.data(), sb.data(), rows * 4) == 0;
            // mode 2 writes H in MMQ's layout, or with STRATA_PF_FP4G_R144=1 in the 144-byte K-step layout:
            // each sub-block's 8 code bytes and its scale byte of each term, wherever the layout puts them
            const bool r144 = std::getenv("STRATA_PF_FP4G_R144") != nullptr && std::getenv("STRATA_PF_FP4G_QOLD") == nullptr;
            for (int term = 0; term < 2 && q_same; ++term)
                for (int64_t r = 0; r < rows && q_same; ++r)
                    for (int isb = 0; isb < FF / 16 && q_same; ++isb) {
                        const int kb = isb / 16, sub = isb % 16;
                        const size_t ob = ((size_t) (term * 4 + kb) * rows + r) * 144;   // MMQ's block_fp4_mmq
                        const uint8_t* cb = &qb[ob + 16 + 8 * sub];
                        const uint8_t sb = qb[ob + sub];
                        const uint8_t* ca;
                        uint8_t sa;
                        if (r144) {
                            const size_t oa = ((size_t) (isb / 8) * rows + r) * 144;
                            ca = &qa[oa + 64 * term + 8 * (isb % 8)];
                            sa = qa[oa + 128 + 8 * term + isb % 8];
                        } else {
                            ca = &qa[ob + 16 + 8 * sub];
                            sa = qa[ob + sub];
                        }
                        q_same = std::memcmp(ca, cb, 8) == 0 && sa == sb;
                    }
        }
        // timing
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        auto time = [&](auto&& f) {
            std::vector<float> t;
            for (int i = 0; i < 18; ++i) {
                cudaEventRecord(e0); f(); cudaEventRecord(e1);
                ck(cudaEventSynchronize(e1), "t");
                float ms; cudaEventElapsedTime(&ms, e0, e1);
                if (i >= 3) t.push_back(ms);
            }
            std::sort(t.begin(), t.end());
            return t[t.size() / 2];
        };
        const float t_mmq = time([&] { mmq_gu(); mmq_down(hq_ref, dm_ref); });
        const float t_m1gu = time([&] { fp4g(1, nullptr); });
        const float t_m1 = time([&] { fp4g(1, nullptr); mmq_down(hq, dm_ref); });
        const float t_m2 = time([&] { fp4g(2, dm); });
        std::printf("%6d %11.4f%% %12.3e %12.3e %10.4f %10.4f %10.4f %10.4f %10.3f%% %10.3f%%%s%s\n", R, 100.0 * diff / tot, r1, r2, t_mmq, t_m1gu,
                    t_m1, t_m2, 100.0 * std::sqrt(e8 / den), 100.0 * std::sqrt(e2 / den), sd_ok ? "" : "  (s_down rows differ!)", q_same ? "  H quantizer: same bytes" : "  H quantizer: BYTES DIFFER");
        cudaFree(dx); cudaFree(xq); cudaFree(xs); cudaFree(db); cudaFree(dsrc); cudaFree(dids); cudaFree(GU); cudaFree(H);
        cudaFree(hq_ref); cudaFree(hq); cudaFree(hs); cudaFree(sd_ref); cudaFree(sd); cudaFree(dm_ref); cudaFree(dm);
    }
    return 0;
}
