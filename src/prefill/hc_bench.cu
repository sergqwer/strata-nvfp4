// src/prefill/hc_bench.cu - pf-hc: the prompt path's hyper-connection writes with their two byte cuts, on synthetic
// inputs of the real shapes (N 2560, 4 streams, 10 routed rows a token), against the default kernels and FP64:
//   inject fold (STRATA_PF_HC_INJ): gr_write_norm_rs / moe_combine_write_norm_rs also write the next half's inject sums
//     (then hc_inj_reduce) instead of the cuBLAS GEMM re-reading xn16.  R, rs and xn16 must stay the same bits;
//     inj is compared with cuBLAS's and with FP64 of the same BF16 inputs.
//   FP16 Dm rows (STRATA_PF_DM16H): the combine reads rows marked row_sd < 0 as FP16 (s_down applied); bo's error
//     against FP64 of the FP32 rows, for all rows FP16, every other row, and the two-kernel combine (moe_combine).
// Time: median of the reps (back to back).  Not a ctest (GPU, timing).
// Usage: hc_bench [T=22272] [reps=20]
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"

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

using namespace strata::prefill;

namespace {
constexpr int N = 2560, HC = 4, D = N * HC, KU = 10, XN_PAD = 64;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
uint16_t to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
double from_bf16(uint16_t h) {
    const uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
template <class T> T* dnew(size_t n) { T* p = nullptr; ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc"); return p; }
template <class T> T* dup(const std::vector<T>& v) {
    T* p = dnew<T>(v.size());
    ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "up");
    return p;
}
template <class T> std::vector<T> down(const T* p, size_t n) {
    std::vector<T> v(n);
    ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
    return v;
}
double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }
}  // namespace

int main(int argc, char** argv) {
    const int64_t T = argc > 1 ? std::atoll(argv[1]) : 22272;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
    const int64_t ldx = D + XN_PAD;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    // inputs: R with a few loud channels (x20, as the residual's), the rest N(0, 1)
    std::vector<float> chs(D);
    for (auto& c : chs) c = rng() % 200 == 0 ? 20.f : 1.f;
    std::vector<float> hR((size_t) T * D), hbo((size_t) T * N), hinj((size_t) T * HC), hwn(D), hwn2(D);
    for (size_t i = 0; i < hR.size(); ++i) hR[i] = nd(rng) * chs[i % D];
    for (auto& v : hbo) v = nd(rng) * 0.5f;
    for (auto& v : hinj) v = nd(rng);
    for (auto& v : hwn) v = 1.f + 0.1f * nd(rng);
    for (auto& v : hwn2) v = 1.f + 0.1f * nd(rng);
    std::vector<uint16_t> hwi((size_t) HC * D);
    for (auto& v : hwi) v = to_bf16(nd(rng) * 0.02f);
    // MoE: T * KU rows, the slots a random permutation, weights summing to ~1, s_down per row
    const int64_t rows = T * KU;
    std::vector<float> hDm((size_t) rows * N), hsh((size_t) T * N), hsg(T), hw(rows), hsd(rows);
    for (auto& v : hDm) v = nd(rng) * 0.3f;
    for (auto& v : hsh) v = nd(rng) * 0.3f;
    for (auto& v : hsg) v = nd(rng);
    for (auto& v : hsd) v = 0.5f + 1.5f * ud(rng);
    std::vector<int32_t> hslot(rows);
    for (int64_t i = 0; i < rows; ++i) hslot[i] = (int32_t) i;
    std::shuffle(hslot.begin(), hslot.end(), rng);
    for (int64_t t = 0; t < T; ++t) {
        float s = 0;
        for (int k = 0; k < KU; ++k) s += (hw[t * KU + k] = ud(rng) + 0.05f);
        for (int k = 0; k < KU; ++k) hw[t * KU + k] /= s;
    }
    // FP16 rows: s_down applied, saturated, in the first half of each row slot; row_sd -1 (all rows / every other row)
    std::vector<float> hDm16((size_t) rows * N, 0.f), hDmMix(hDm), hsd16(rows, -1.f), hsdMix(hsd);
    for (int64_t r = 0; r < rows; ++r) {
        __half* d16 = reinterpret_cast<__half*>(hDm16.data() + r * N);
        __half* m16 = reinterpret_cast<__half*>(hDmMix.data() + r * N);
        for (int d = 0; d < N; ++d) {
            const float v = std::fmin(std::fmax(hDm[r * N + d] * hsd[r], -65504.f), 65504.f);
            d16[d] = __float2half_rn(v);
        }
        if (r % 2 == 0) {
            std::vector<__half> tmp(N);
            for (int d = 0; d < N; ++d) tmp[d] = d16[d];
            std::memcpy(m16, tmp.data(), N * 2);
            hsdMix[r] = -1.f;
        }
    }
    float *R0 = dup(hR), *R = dnew<float>(hR.size()), *bo = dup(hbo), *inj = dup(hinj), *wn = dup(hwn);
    uint16_t* wi16 = dup(hwi);
    uint16_t* wi = dnew<uint16_t>(hwi.size());   // the fused writes read the interleaved copy
    hc_inj_interleave(wi16, wi, nullptr);
    ck(cudaDeviceSynchronize(), "interleave");
    float *rs = dnew<float>((size_t) T * HC), *injA = dnew<float>((size_t) T * HC), *injB = dnew<float>((size_t) T * HC);
    float* injp = dnew<float>((size_t) T * 16);
    uint16_t* xn = dnew<uint16_t>((size_t) T * ldx);
    float *Dm = dup(hDm), *Dm16 = dup(hDm16), *DmMix = dup(hDmMix), *sh = dup(hsh), *sg = dup(hsg), *w = dup(hw);
    float *sd = dup(hsd), *sd16 = dup(hsd16), *sdMix = dup(hsdMix), *bob = dnew<float>((size_t) T * N);
    int32_t* slot = dup(hslot);
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    Gemm gm;
    std::string err;
    if (!gm.init(s, 1 << 20, err)) { std::fprintf(stderr, "gemm init: %s\n", err.c_str()); return 2; }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    auto timeit = [&](auto&& f) {
        std::vector<float> t;
        for (int i = 0; i < reps + 3; ++i) {
            cudaEventRecord(e0, s); f(); cudaEventRecord(e1, s);
            ck(cudaEventSynchronize(e1), "time");
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (i >= 3) t.push_back(ms);
        }
        std::sort(t.begin(), t.end());
        return t[t.size() / 2];
    };
    auto resetR = [&] { ck(cudaMemcpyAsync(R, R0, hR.size() * 4, cudaMemcpyDeviceToDevice, s), "reset"); };
    const size_t nsamp = 256;   // FP64 references on sampled tokens
    std::vector<int64_t> samp(nsamp);
    for (auto& t : samp) t = (int64_t) (rng() % (uint32_t) T);
    auto inj_err = [&](const std::vector<uint16_t>& x16, const std::vector<float>& got, double& rel, double& mx) {
        double num = 0, den = 0;
        mx = 0;
        for (int64_t t : samp)
            for (int q = 0; q < HC; ++q) {
                double ref = 0;
                for (int j = 0; j < D; ++j) ref += from_bf16(x16[(size_t) t * ldx + j]) * from_bf16(hwi[(size_t) q * D + j]);
                const double e = got[(size_t) t * HC + q] - ref;
                num += e * e; den += ref * ref; mx = std::max(mx, std::fabs(e));
            }
        rel = std::sqrt(num / den);
    };
    std::printf("T %lld, %d reps (median); xn16 token stride %lld\n", (long long) T, reps, (long long) ldx);

    // ---- 1. gr_write_norm_rs: the default write + cuBLAS inject vs the fold
    resetR();
    gr_write_norm_rs(R, bo, inj, HC, wn, 1e-6f, rs, xn, T, s, nullptr, ldx, false);
    gm.bf16(xn, wi16, injA, T, HC, D, 0, 0.0f, ldx);
    ck(cudaStreamSynchronize(s), "a");
    const auto Ra = down(R, hR.size()), rsa = down(rs, (size_t) T * HC);
    const auto xa = down(xn, (size_t) T * ldx);
    const auto ia = down(injA, (size_t) T * HC);
    resetR();
    gr_write_norm_rs(R, bo, inj, HC, wn, 1e-6f, rs, xn, T, s, nullptr, ldx, false, wi16, injp);
    hc_inj_reduce(injp, 4, injB, T, s);
    ck(cudaStreamSynchronize(s), "b");
    const auto Rb = down(R, hR.size()), rsb = down(rs, (size_t) T * HC);
    const auto xb = down(xn, (size_t) T * ldx);
    const auto ib = down(injB, (size_t) T * HC);
    bool same = std::memcmp(Ra.data(), Rb.data(), Ra.size() * 4) == 0 && std::memcmp(rsa.data(), rsb.data(), rsa.size() * 4) == 0;
    for (int64_t t = 0; t < T && same; ++t) same = std::memcmp(&xa[(size_t) t * ldx], &xb[(size_t) t * ldx], D * 2) == 0;
    double rA, mA, rB, mB;
    inj_err(xa, ia, rA, mA);
    inj_err(xb, ib, rB, mB);
    double dab = 0;
    for (size_t i = 0; i < ia.size(); ++i) dab = std::max(dab, (double) std::fabs(ia[i] - ib[i]));
    const float t_w = timeit([&] { gr_write_norm_rs(R, bo, inj, HC, wn, 1e-6f, rs, xn, T, s, nullptr, ldx, false); });
    const float t_g = timeit([&] { gm.bf16(xn, wi16, injA, T, HC, D, 0, 0.0f, ldx); });
    const float t_wf = timeit([&] { gr_write_norm_rs(R, bo, inj, HC, wn, 1e-6f, rs, xn, T, s, nullptr, ldx, false, wi16, injp); });
    const float t_r = timeit([&] { hc_inj_reduce(injp, 4, injB, T, s); });
    std::printf("\n[1] gr_write_norm_rs + inject: R / rs / xn16 %s with the fold\n", same ? "IDENTICAL" : "DIFFER");
    std::printf("    inj vs FP64 (256 tokens): cuBLAS rel %.3e max %.3e | fold rel %.3e max %.3e | max |cuBLAS - fold| %.3e\n",
                rA, mA, rB, mB, dab);
    std::printf("    ms: write %.4f + cuBLAS inject %.4f = %.4f | fold write %.4f + reduce %.4f = %.4f | saved %.4f ms\n", t_w,
                t_g, t_w + t_g, t_wf, t_r, t_wf + t_r, t_w + t_g - t_wf - t_r);

    // ---- 2. moe_combine_write_norm_rs: FP32 rows, + fold, FP16 rows, FP16 + fold, every other row FP16
    auto combine = [&](const float* dm, const float* sdp, bool fold) {
        return moe_combine_write_norm_rs(dm, slot, w, sh, sg, sdp, R, inj, HC, wn, 1e-6f, rs, xn, T, s, nullptr, ldx, false,
                                         fold ? wi : nullptr, fold ? injp : nullptr);
    };
    resetR();
    if (!combine(Dm, sd, false)) { std::printf("combine refused\n"); return 1; }
    gm.bf16(xn, wi16, injA, T, HC, D, 0, 0.0f, ldx);
    ck(cudaStreamSynchronize(s), "c");
    const auto Rc = down(R, hR.size()), rsc = down(rs, (size_t) T * HC);
    const auto xc = down(xn, (size_t) T * ldx);
    const auto ic = down(injA, (size_t) T * HC);
    resetR();
    combine(Dm, sd, true);
    hc_inj_reduce(injp, 1, injB, T, s);
    ck(cudaStreamSynchronize(s), "d");
    const auto Rd = down(R, hR.size()), rsd = down(rs, (size_t) T * HC);
    const auto xd = down(xn, (size_t) T * ldx);
    const auto id = down(injB, (size_t) T * HC);
    bool same2 = std::memcmp(Rc.data(), Rd.data(), Rc.size() * 4) == 0 && std::memcmp(rsc.data(), rsd.data(), rsc.size() * 4) == 0;
    for (int64_t t = 0; t < T && same2; ++t) same2 = std::memcmp(&xc[(size_t) t * ldx], &xd[(size_t) t * ldx], D * 2) == 0;
    inj_err(xc, ic, rA, mA);
    inj_err(xd, id, rB, mB);
    // bo's error against FP64 of the FP32 rows: through R's change (R - R0 = bo * sc, sc from inj), sampled tokens
    auto bo_err = [&](const std::vector<float>& Rn, double& rel, double& mx) {
        double num = 0, den = 0;
        mx = 0;
        for (int64_t t : samp)
            for (int c = 0; c < HC; ++c) {
                const double sc = 2.0 * sigm(hinj[(size_t) t * HC + c] / (double) HC);
                for (int d = 0; d < N; ++d) {
                    double b = hsh[(size_t) t * N + d] * sigm(hsg[t]);
                    for (int k = 0; k < KU; ++k) {
                        const int64_t r = hslot[(size_t) t * KU + k];
                        b += (double) hw[(size_t) t * KU + k] * (double) hDm[(size_t) r * N + d] * (double) hsd[r];
                    }
                    const size_t i = (size_t) t * D + (size_t) c * N + d;
                    const double ref = hR[i] + b * sc;
                    const double e = (double) Rn[i] - ref;
                    num += e * e; den += (b * sc) * (b * sc); mx = std::max(mx, std::fabs(e));
                }
            }
        rel = std::sqrt(num / den);
    };
    double r32, m32, r16, m16, rmx, mmx;
    bo_err(Rc, r32, m32);
    resetR();
    combine(Dm16, sd16, false);
    ck(cudaStreamSynchronize(s), "e");
    const auto Re = down(R, hR.size());
    bo_err(Re, r16, m16);
    resetR();
    combine(DmMix, sdMix, false);
    ck(cudaStreamSynchronize(s), "f");
    bo_err(down(R, hR.size()), rmx, mmx);
    // the two-kernel combine (moe_combine, float4 path) with FP16 rows: bo against FP64
    auto bo_err2 = [&](const std::vector<float>& b, double& rel) {
        double num = 0, den = 0;
        for (int64_t t : samp)
            for (int d = 0; d < N; ++d) {
                double ref = hsh[(size_t) t * N + d] * sigm(hsg[t]);
                for (int k = 0; k < KU; ++k) {
                    const int64_t r = hslot[(size_t) t * KU + k];
                    ref += (double) hw[(size_t) t * KU + k] * (double) hDm[(size_t) r * N + d] * (double) hsd[r];
                }
                const double e = b[(size_t) t * N + d] - ref;
                num += e * e; den += ref * ref;
            }
        rel = std::sqrt(num / den);
    };
    double rc32, rc16, rcmx;
    moe_combine(Dm, slot, w, sh, sg, bob, T, s, sd);
    ck(cudaStreamSynchronize(s), "g");
    bo_err2(down(bob, (size_t) T * N), rc32);
    moe_combine(Dm16, slot, w, sh, sg, bob, T, s, sd16);
    ck(cudaStreamSynchronize(s), "h");
    bo_err2(down(bob, (size_t) T * N), rc16);
    moe_combine(DmMix, slot, w, sh, sg, bob, T, s, sdMix);
    ck(cudaStreamSynchronize(s), "i");
    bo_err2(down(bob, (size_t) T * N), rcmx);
    const float t_c = timeit([&] { combine(Dm, sd, false); });
    const float t_cf = timeit([&] { combine(Dm, sd, true); });
    const float t_c16 = timeit([&] { combine(Dm16, sd16, false); });
    const float t_c16f = timeit([&] { combine(Dm16, sd16, true); });
    const float t_r1 = timeit([&] { hc_inj_reduce(injp, 1, injB, T, s); });
    const float t_m32 = timeit([&] { moe_combine(Dm, slot, w, sh, sg, bob, T, s, sd); });
    const float t_m16 = timeit([&] { moe_combine(Dm16, slot, w, sh, sg, bob, T, s, sd16); });
    std::printf("\n[2] moe_combine_write_norm_rs: R / rs / xn16 %s with the fold\n", same2 ? "IDENTICAL" : "DIFFER");
    std::printf("    inj vs FP64: cuBLAS rel %.3e max %.3e | fold rel %.3e max %.3e\n", rA, mA, rB, mB);
    std::printf("    bo (through R) vs FP64 of the FP32 rows, rel RMS / max: FP32 rows %.3e / %.3e | FP16 rows %.3e / %.3e | "
                "half FP16 %.3e / %.3e\n", r32, m32, r16, m16, rmx, mmx);
    std::printf("    moe_combine (two-kernel path) bo vs FP64: FP32 %.3e | FP16 %.3e | half FP16 %.3e\n", rc32, rc16, rcmx);
    std::printf("    ms: FP32 %.4f (+cuBLAS inject %.4f) | FP32 + fold %.4f (+reduce %.4f) | FP16 %.4f | FP16 + fold %.4f\n", t_c, t_g,
                t_cf, t_r1, t_c16, t_c16f);
    std::printf("    ms moe_combine: FP32 %.4f | FP16 %.4f\n", t_m32, t_m16);
    std::printf("    implied KB/token at 1.78 TB/s: FP32 %.0f, FP16 %.0f\n", t_c * 1.78e6 / T, t_c16 * 1.78e6 / T);
    return 0;
}
