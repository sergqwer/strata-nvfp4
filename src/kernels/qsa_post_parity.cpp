// src/kernels/qsa_post_parity.cpp - dec/int: the QSA decode window's fused kernels against the kernels they replace, byte
// for byte: kv_rot_append_q8 (kv_q8.hpp) = fwht256 on K and V + kv_append_q8_steps (after native_qsa_rms_norm_rope on K);
// qsa_gate_q8_1 (native_mmvq.hpp) = native_qsa_gate_apply + native_quantize_q8_1; dense_tc_rows = dense_tc's rows.  K / V:
// the K / V rows left in kcur / vcur, the INT8 codes and scales in VRAM and in the host copy.  Random rows at several
// magnitudes, 1..8 tokens, positions near 0 and far out, a page table with a non-resident page.
#include "strata/kernels/dense_tc.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("cuda: %s at %d\n", cudaGetErrorString(e_), __LINE__); return 2; } } while (0)

int main() {
    k::native_rope_set_enabled(true);
    k::QsaShapes s;
    s.n_head = 24; s.n_head_kv = 2; s.head_dim = 256; s.n_rot = 64; s.page_size = 16;
    const int HD = 256, NKV = 2, PAGES = 64, CELLS = PAGES * 16;
    k::RopeScaling sc;   // the model's: none, base 1e7
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    const size_t codes = (size_t) PAGES * NKV * 16 * HD, scales = codes / 64;
    int8_t *kq[2], *vq[2], *hkq[2], *hvq[2];
    uint16_t *ks[2], *vs[2], *hks[2], *hvs[2];
    for (int a = 0; a < 2; ++a) {
        CK(cudaMalloc(&kq[a], codes)); CK(cudaMalloc(&vq[a], codes)); CK(cudaMalloc(&hkq[a], codes)); CK(cudaMalloc(&hvq[a], codes));
        CK(cudaMalloc(&ks[a], scales * 2)); CK(cudaMalloc(&vs[a], scales * 2)); CK(cudaMalloc(&hks[a], scales * 2)); CK(cudaMalloc(&hvs[a], scales * 2));
    }
    std::vector<int32_t> table(PAGES);
    for (int p = 0; p < PAGES; ++p) table[(size_t) p] = p == 5 ? -1 : PAGES - 1 - p;   // page 5 not resident
    int32_t* d_table;
    CK(cudaMalloc(&d_table, PAGES * 4));
    CK(cudaMemcpy(d_table, table.data(), PAGES * 4, cudaMemcpyHostToDevice));
    float *d_gamma, *k0, *v0, *k1, *v1;
    int32_t *d_pos, *d_step;
    CK(cudaMalloc(&d_gamma, HD * 4));
    CK(cudaMalloc(&k0, 8 * NKV * HD * 4)); CK(cudaMalloc(&v0, 8 * NKV * HD * 4));
    CK(cudaMalloc(&k1, 8 * NKV * HD * 4)); CK(cudaMalloc(&v1, 8 * NKV * HD * 4));
    CK(cudaMalloc(&d_pos, 8 * NKV * 4)); CK(cudaMalloc(&d_step, 8 * k::kStepCount * 4));
    std::mt19937 rng(7);
    int fails = 0, checks = 0;
    const float mags[3] = {0.02f, 1.0f, 40.0f};
    for (int round = 0; round < 240; ++round) {
        const int n = 1 + round % 8;
        const float mag = mags[round % 3];
        const bool with_host = (round & 1) == 1;
        std::normal_distribution<float> nd(0.0f, mag);
        std::vector<float> kh((size_t) n * NKV * HD), vh(kh.size()), gh(HD);
        for (auto& x : kh) x = nd(rng);
        for (auto& x : vh) x = nd(rng);
        std::uniform_real_distribution<float> ud(0.5f, 1.5f);
        for (auto& x : gh) x = ud(rng);
        const int base = round < 120 ? (int) (rng() % 200) : 900 + (int) (rng() % 100);   // incl. the page that is not resident
        std::vector<int32_t> pos((size_t) n * NKV), step((size_t) n * k::kStepCount, 0);
        for (int t = 0; t < n; ++t) {
            for (int h = 0; h < NKV; ++h) pos[(size_t) t * NKV + h] = (base + t) % CELLS;
            step[(size_t) t * k::kStepCount + k::kStepPos] = (base + t) % CELLS;
            step[(size_t) t * k::kStepCount + k::kStepNKv] = (base + t) % CELLS + 1;
        }
        if (round % 5 == 2) for (int t = 0; t < n; ++t) step[(size_t) t * k::kStepCount] = 5 * 16 + t;   // page 5 (host only)
        CK(cudaMemcpy(d_gamma, gh.data(), HD * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_pos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_step, step.data(), step.size() * 4, cudaMemcpyHostToDevice));
        for (float* d : {k0, k1}) CK(cudaMemcpy(d, kh.data(), kh.size() * 4, cudaMemcpyHostToDevice));
        for (float* d : {v0, v1}) CK(cudaMemcpy(d, vh.data(), vh.size() * 4, cudaMemcpyHostToDevice));
        for (int a = 0; a < 2; ++a) {
            for (int8_t* d : {kq[a], vq[a], hkq[a], hvq[a]}) CK(cudaMemset(d, 0x55, codes));
            for (uint16_t* d : {ks[a], vs[a], hks[a], hvs[a]}) CK(cudaMemset(d, 0x55, scales * 2));
        }
        k::KvHostPools hp[2];
        for (int a = 0; a < 2; ++a)
            if (with_host) { hp[a].k_q = hkq[a]; hp[a].v_q = hvq[a]; hp[a].k_scale = hks[a]; hp[a].v_scale = hvs[a]; }
        // reference: the three kernels as verify.cpp records them
        k::native_qsa_rms_norm_rope(k0, HD, d_gamma, k0, n * NKV, HD, 64, 1e-6f, sc, d_pos, st);
        k::fwht256_inplace_cuda(k0, (int64_t) n * NKV, st);
        k::fwht256_inplace_cuda(v0, (int64_t) n * NKV, st);
        k::kv_append_q8_steps(kq[0], vq[0], ks[0], vs[0], d_table, d_step, k::kStepCount, k0, v0, NKV * HD, n, s, st, &hp[0]);
        // the fused kernel
        k::native_qsa_rms_norm_rope(k1, HD, d_gamma, k1, n * NKV, HD, 64, 1e-6f, sc, d_pos, st);
        k::kv_rot_append_q8(k1, v1, kq[1], vq[1], ks[1], vs[1], d_table, d_step, k::kStepCount, n, s, &hp[1], st);
        CK(cudaStreamSynchronize(st));
        auto same = [&](const void* a, const void* b, size_t bytes, const char* what) {
            std::vector<uint8_t> x(bytes), y(bytes);
            cudaMemcpy(x.data(), a, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(y.data(), b, bytes, cudaMemcpyDeviceToHost);
            ++checks;
            if (std::memcmp(x.data(), y.data(), bytes) != 0) {
                size_t i = 0;
                while (x[i] == y[i]) ++i;
                std::printf("round %d (n %d, mag %g, host %d): %s differs at byte %zu\n", round, n, mag, (int) with_host, what, i);
                ++fails;
            }
        };
        same(k0, k1, kh.size() * 4, "K rows");
        same(v0, v1, vh.size() * 4, "V rows");
        same(kq[0], kq[1], codes, "K codes"); same(vq[0], vq[1], codes, "V codes");
        same(ks[0], ks[1], scales * 2, "K scales"); same(vs[0], vs[1], scales * 2, "V scales");
        same(hkq[0], hkq[1], codes, "host K codes"); same(hvq[0], hvq[1], codes, "host V codes");
        same(hks[0], hks[1], scales * 2, "host K scales"); same(hvs[0], hvs[1], scales * 2, "host V scales");
    }
    // qsa_gate_q8_1 (after fwht256) against native_qsa_gate_apply + native_quantize_q8_1
    k::native_qsa_set_enabled(true);
    for (int round = 0; round < 64; ++round) {
        const int rows = (1 + round % 8) * 24;
        std::normal_distribution<float> nd(0.0f, round % 2 ? 0.3f : 6.0f);
        std::vector<float> ah((size_t) rows * HD), qh((size_t) rows * 2 * HD);
        for (auto& x : ah) x = nd(rng);
        for (auto& x : qh) x = nd(rng);
        float *a0, *qf, *o0, *o1;
        void *y0, *y1;
        const size_t qb = (size_t) rows * 8 * 36;
        CK(cudaMalloc(&a0, ah.size() * 4)); CK(cudaMalloc(&qf, qh.size() * 4));
        CK(cudaMalloc(&o0, ah.size() * 4)); CK(cudaMalloc(&o1, ah.size() * 4));
        CK(cudaMalloc(&y0, qb)); CK(cudaMalloc(&y1, qb));
        CK(cudaMemcpy(a0, ah.data(), ah.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(qf, qh.data(), qh.size() * 4, cudaMemcpyHostToDevice));
        k::fwht256_inplace_cuda(a0, rows, st);
        k::native_qsa_gate_apply(a0, qf, o0, rows, HD, st);
        k::native_quantize_q8_1(o0, y0, 24 * HD, rows / 24, st);
        k::qsa_gate_q8_1(a0, qf, o1, y1, rows, HD, st);
        CK(cudaStreamSynchronize(st));
        std::vector<float> x(ah.size()), y(ah.size());
        CK(cudaMemcpy(x.data(), o0, x.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(y.data(), o1, y.size() * 4, cudaMemcpyDeviceToHost));
        ++checks;
        if (std::memcmp(x.data(), y.data(), x.size() * 4) != 0) { std::printf("gate_q8_1 round %d: output differs\n", round); ++fails; }
        std::vector<uint8_t> p0(qb), p1(qb);
        CK(cudaMemcpy(p0.data(), y0, qb, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(p1.data(), y1, qb, cudaMemcpyDeviceToHost));
        ++checks;
        if (std::memcmp(p0.data(), p1.data(), qb) != 0) { std::printf("gate_q8_1 round %d: q8_1 differs\n", round); ++fails; }
        cudaFree(a0); cudaFree(qf); cudaFree(o0); cudaFree(o1); cudaFree(y0); cudaFree(y1);
    }
    // the merge with the inverse rotation (merge_only v2 = 2) against attn_merge_v2_kernel + fwht256
    {
        k::QsaShapes q = s;
        q.n_head = 24; q.idx_top_k = 2048;
        for (int round = 0; round < 24; ++round) {
            const int64_t cap = round % 2 ? 2048 : 640;
            const int nq = 1 + round % 8;
            const size_t stride = (size_t) k::qsa_decode_attn_scratch_floats(cap, q);
            std::vector<float> sh(stride * nq);
            std::normal_distribution<float> nd(0.0f, 2.0f);
            std::uniform_real_distribution<float> ul(0.5f, 40.0f);
            for (auto& x : sh) x = nd(rng);
            // part_l (after part_acc and part_m) positive: the chunk pass's sums of exponentials
            const int nch = (int) ((cap + 63) / 64);
            for (int z = 0; z < nq; ++z) {
                float* base = sh.data() + stride * z;
                float* pl = base + (size_t) nch * 24 * 256 + (size_t) nch * 24;
                for (int i = 0; i < nch * 24; ++i) pl[i] = ul(rng);
            }
            std::vector<int32_t> steps((size_t) nq * k::kStepCount, 0);
            for (int z = 0; z < nq; ++z) steps[(size_t) z * k::kStepCount + k::kStepWidth] = 1 + (int) (rng() % cap);
            float *dsc, *o0, *o1;
            int32_t* dst;
            CK(cudaMalloc(&dsc, sh.size() * 4)); CK(cudaMalloc(&o0, (size_t) nq * 24 * 256 * 4)); CK(cudaMalloc(&o1, (size_t) nq * 24 * 256 * 4));
            CK(cudaMalloc(&dst, steps.size() * 4));
            CK(cudaMemcpy(dsc, sh.data(), sh.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dst, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice));
            k::qsa_decode_attn_merge_only(dst, cap, q, dsc, o0, nq, 1, st);
            k::fwht256_inplace_cuda(o0, (int64_t) nq * 24, st);
            k::qsa_decode_attn_merge_only(dst, cap, q, dsc, o1, nq, 2, st);
            CK(cudaStreamSynchronize(st));
            std::vector<float> x((size_t) nq * 24 * 256), y(x.size());
            CK(cudaMemcpy(x.data(), o0, x.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(y.data(), o1, y.size() * 4, cudaMemcpyDeviceToHost));
            ++checks;
            if (std::memcmp(x.data(), y.data(), x.size() * 4) != 0) { std::printf("merge+unrot round %d differs\n", round); ++fails; }
            cudaFree(dsc); cudaFree(o0); cudaFree(o1); cudaFree(dst);
        }
    }
    // dense_tc_rows: the query rows and the gate rows of a 2560 x 12288 Q8_0 q|gate projection against the whole call
    {
        const int NI = 2560, NO = 12288, nb = NI / 32;
        std::vector<uint8_t> wh((size_t) NO * nb * 34);
        std::uniform_int_distribution<int> bi(-127, 127);
        for (size_t b = 0; b < (size_t) NO * nb; ++b) {
            uint8_t* blk = wh.data() + b * 34;
            const uint16_t d = (uint16_t) (0x2000 + (rng() % 0x0800));   // fp16 ~ 2^-7 .. 2^-6
            std::memcpy(blk, &d, 2);
            for (int j = 0; j < 32; ++j) blk[2 + j] = (uint8_t) (int8_t) bi(rng);
        }
        void* dw;
        CK(cudaMalloc(&dw, wh.size()));
        CK(cudaMemcpy(dw, wh.data(), wh.size(), cudaMemcpyHostToDevice));
        float *xf, *y0, *y1;
        void* xq;
        CK(cudaMalloc(&xf, 8 * NI * 4)); CK(cudaMalloc(&xq, 8 * nb * 36)); CK(cudaMalloc(&y0, 8 * NO * 4)); CK(cudaMalloc(&y1, 8 * NO * 4));
        for (int nc = 1; nc <= 8; ++nc) {
            std::normal_distribution<float> nd(0.0f, 1.0f);
            std::vector<float> xh((size_t) nc * NI);
            for (auto& v : xh) v = nd(rng);
            CK(cudaMemcpy(xf, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice));
            k::native_quantize_q8_1(xf, xq, NI, nc, st);
            k::dense_tc(8, dw, xq, y0, NI, NO, nc, st);
            CK(cudaMemsetAsync(y1, 0xff, (size_t) nc * NO * 4, st));
            const bool a1 = k::dense_tc_rows(dw, xq, y1, NI, NO, 256, 512, 0, nc, st);
            const bool a2 = k::dense_tc_rows(dw, xq, y1, NI, NO, 256, 512, 256, nc, st, 13 * nc + 20);   // a capped grid too
            CK(cudaStreamSynchronize(st));
            std::vector<float> r0((size_t) nc * NO), r1(r0.size());
            CK(cudaMemcpy(r0.data(), y0, r0.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(r1.data(), y1, r1.size() * 4, cudaMemcpyDeviceToHost));
            ++checks;
            if (!a1 || !a2 || std::memcmp(r0.data(), r1.data(), r0.size() * 4) != 0) {
                std::printf("dense_tc_rows ncols %d: %s\n", nc, (!a1 || !a2) ? "declined" : "differs"); ++fails;
            }
        }
        cudaFree(dw); cudaFree(xf); cudaFree(xq); cudaFree(y0); cudaFree(y1);
    }
    std::printf("qsa_post_parity: %s, %d checks, %d failures\n", fails == 0 ? "PASS" : "FAIL", checks, fails);
    return fails == 0 ? 0 : 1;
}
