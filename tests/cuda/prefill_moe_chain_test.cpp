// prefill_moe_chain_test - the prompt path's fused NVFP4 expert steps against the kernels they replace, byte for byte:
// - mmq::swiglu_quant against swiglu_scaled + quantize + down_row_scales (the down product's q8_1 rows of H and each
//   row's s_down), for the down types an NVFP4 pack has (Q8_0, NVFP4), with absolute bounds that start past row 0 (a
//   second group) and experts of 0 rows;
// - an MMQ group read in place (Product::w_ptrs: each expert its own allocation, NaN bytes after it) against the same
//   experts gathered (gate/up NVFP4 K 2560, down NVFP4 and Q8_0 K 640, whose last K iteration runs past the row);
// - gate/up in place on each token's activations quantized once, read through the row -> token table
//   (Product::y_rows), against the same rows quantized per MMQ row.
// Exit 77 without a CUDA device.
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mmq = strata::prefill::mmq;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    explicit Dev(size_t n) { ck(cudaMalloc(&p, n), "cudaMalloc"); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    template <class T> T* as() const { return (T*) p; }
};

template <class T> void up(const Dev& d, const std::vector<T>& h) {
    ck(cudaMemcpy(d.p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}
template <class T> std::vector<T> down(const Dev& d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d.p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}

// one group: n experts with `counts` rows, its rows at [row0, row0 + rows) of the layer
int swiglu_quant_case(ggml_type t, const std::vector<int>& counts, int row0, uint32_t seed) {
    constexpr int64_t NFF = 640;
    const int n = (int) counts.size();
    std::vector<int32_t> abs_b((size_t) n + 1), rel_b((size_t) n + 1);
    abs_b[0] = row0;
    for (int e = 0; e < n; ++e) abs_b[(size_t) e + 1] = abs_b[(size_t) e] + counts[(size_t) e];
    for (int e = 0; e <= n; ++e) rel_b[(size_t) e] = abs_b[(size_t) e] - row0;
    const int64_t rows = abs_b[(size_t) n] - row0;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> mag(0.05f, 40.f), sc(0.0005f, 0.02f);
    std::vector<float> gu((size_t) rows * 2 * NFF);
    for (int64_t r = 0; r < rows; ++r) {
        const float m = mag(rng);
        for (int64_t k = 0; k < 2 * NFF; ++k) gu[(size_t) (r * 2 * NFF + k)] = nd(rng) * m;
    }
    if (rows > 3) {   // a zero row, a row with one huge value
        std::memset(gu.data() + 2 * NFF, 0, 2 * NFF * sizeof(float));
        gu[(size_t) (3 * 2 * NFF + 5)] = 3e4f;
    }
    std::vector<float> tails((size_t) n * 4);
    for (int e = 0; e < n; ++e) {
        tails[(size_t) e * 4] = sc(rng); tails[(size_t) e * 4 + 1] = sc(rng);
        tails[(size_t) e * 4 + 2] = sc(rng); tails[(size_t) e * 4 + 3] = 0.f;
    }
    const size_t qb = mmq::q8_bytes(rows, NFF);
    Dev dgu(gu.size() * 4), dh((size_t) rows * NFF * 4 + 16), dq0(qb), dq1(qb), dab(abs_b.size() * 4),
        drb(rel_b.size() * 4), dt(tails.size() * 4), dsd0((size_t) rows * 4 + 4), dsd1((size_t) rows * 4 + 4),
        dys((size_t) rows * 4 + 4);
    up(dgu, gu); up(dab, abs_b); up(drb, rel_b); up(dt, tails);
    ck(cudaMemset(dq0.p, 0x5a, qb), "memset"); ck(cudaMemset(dq1.p, 0x5a, qb), "memset");
    ck(cudaMemset(dsd0.p, 0, (size_t) rows * 4 + 4), "memset"); ck(cudaMemset(dsd1.p, 0, (size_t) rows * 4 + 4), "memset");
    // the kernels it replaces, as prefill.cpp launches them (swiglu_scaled on absolute bounds from row0, the
    // quantizer, down_row_scales on the group's relative bounds)
    mmq::swiglu_scaled(dgu.as<float>(), dh.as<float>(), rows, NFF, dab.as<int32_t>(), n, dt.as<float>(), row0, nullptr);
    mmq::quantize(dh.as<float>(), nullptr, dq0.p, (int) t, NFF, NFF, rows, nullptr, dys.as<float>());
    mmq::down_row_scales(dsd0.as<float>(), drb.as<int32_t>(), n, dt.as<float>(), rows, nullptr);
    std::vector<const float*> tp((size_t) n);
    for (int e = 0; e < n; ++e) tp[(size_t) e] = dt.as<float>() + 4 * e;
    mmq::swiglu_quant(dgu.as<float>(), dq1.p, rows, dab.as<int32_t>(), n, tp.data(), row0, dsd1.as<float>(), nullptr);
    ck(cudaDeviceSynchronize(), "run");
    const size_t used = (size_t) rows * (1024 / 128) * 144;   // the rows' 8 blocks of 128 values (144 bytes each)
    const auto a = down<uint8_t>(dq0, used), b = down<uint8_t>(dq1, used);
    const auto s0 = down<float>(dsd0, (size_t) rows), s1 = down<float>(dsd1, (size_t) rows);
    size_t bad = 0, first = used;
    for (size_t i = 0; i < used; ++i)
        if (a[i] != b[i]) { ++bad; if (first == used) first = i; }
    const bool sd_ok = std::memcmp(s0.data(), s1.data(), (size_t) rows * 4) == 0;
    std::printf("  swiglu_quant %-5s %2d experts %5lld rows from %4d: %zu of %zu bytes differ%s, s_down %s\n",
                ggml_type_name(t), n, (long long) rows, row0, bad, used,
                bad ? (" (first at " + std::to_string(first) + ")").c_str() : "", sd_ok ? "same" : "DIFFERS");
    return bad == 0 && sd_ok ? 0 : 1;
}

// valid synthetic weights (mmq_expert_bench's): random bytes, the scale fields small positive values
std::vector<uint8_t> weights(ggml_type t, int64_t rows, int64_t cols, std::mt19937& rng) {
    const size_t per = mmq::matrix_bytes(t, rows, cols);
    std::vector<uint8_t> w(per);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : w) b = (uint8_t) byte(rng);
    const size_t bs = ggml_type_size(t), nblk = per / bs;
    for (size_t i = 0; i < nblk; ++i) {
        uint8_t* blk = w.data() + i * bs;
        if (t == GGML_TYPE_Q8_0) {
            const uint16_t d = 0x2000 + (uint16_t) byte(rng);
            std::memcpy(blk, &d, 2);
        } else {
            const int nsc = (int) (bs - (size_t) ggml_blck_size(t) / 2);
            for (int s = 0; s < nsc; ++s) blk[s] = (uint8_t) (0x30 + byte(rng) % 16);
        }
    }
    return w;
}

// one grouped product read in place (each expert its own allocation, NaN bytes right after it) against the same
// experts gathered side by side with a zeroed tail (what prefill.cpp's gather builds): dst bit for bit
int in_place_case(ggml_type t, int64_t out_rows, int64_t cols, const std::vector<int>& counts, uint32_t seed) {
    const int n = (int) counts.size();
    std::vector<int32_t> b((size_t) n + 1, 0);
    for (int e = 0; e < n; ++e) b[(size_t) e + 1] = b[(size_t) e] + counts[(size_t) e];
    const int rows = b[(size_t) n];
    int maxr = 0;
    for (int c : counts) maxr = std::max(maxr, c);
    std::mt19937 rng(seed);
    const size_t per = mmq::matrix_bytes(t, out_rows, cols);
    constexpr size_t TAIL = 4096;
    std::vector<uint8_t> all((size_t) n * per + TAIL, 0);
    std::vector<std::unique_ptr<Dev>> own;
    std::vector<const void*> ptr((size_t) n);
    for (int e = 0; e < n; ++e) {
        auto w = weights(t, out_rows, cols, rng);
        std::memcpy(all.data() + (size_t) e * per, w.data(), per);
        w.resize(per + TAIL, 0xff);   // NaN as f16 and as e4m3 past the expert's end
        own.push_back(std::make_unique<Dev>(w.size()));
        up(*own.back(), w);
        ptr[(size_t) e] = own.back()->p;
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> mag(0.1f, 10.f);
    std::vector<float> x((size_t) rows * cols);
    for (int r = 0; r < rows; ++r) { const float m = mag(rng); for (int64_t k = 0; k < cols; ++k) x[(size_t) (r * cols + k)] = nd(rng) * m; }
    std::vector<int32_t> ids((size_t) rows);
    for (int r = 0; r < rows; ++r) ids[(size_t) r] = r;
    Dev dall(all.size()), dx(x.size() * 4), dq(mmq::q8_bytes(rows, cols)), dys((size_t) rows * 4 + 4), dids((size_t) rows * 4),
        db(b.size() * 4), d0((size_t) rows * out_rows * 4), d1((size_t) rows * out_rows * 4);
    up(dall, all); up(dx, x); up(dids, ids); up(db, b);
    mmq::quantize(dx.as<float>(), nullptr, dq.p, (int) t, cols, cols, rows, nullptr, dys.as<float>());
    mmq::Context ctx;
    mmq::Product p;
    p.w = dall.p; p.type = (int) t; p.w_rows = out_rows; p.w_cols = cols; p.expert_bytes = per; p.n = n;
    p.xq = dq.p; p.bounds = db.as<int32_t>(); p.ids = dids.as<int32_t>(); p.total_rows = rows; p.max_rows = maxr;
    p.ld_dst = out_rows;
    // the activation row scales of FP4 rows (gate/up in w4a4x2); q8_1 rows (K 640) ignore them
    p.y_scale = mmq::fp4_activations((int) t) ? dys.as<float>() : nullptr;
    ck(cudaMemset(d0.p, 0, (size_t) rows * out_rows * 4), "memset");
    ck(cudaMemset(d1.p, 0, (size_t) rows * out_rows * 4), "memset");
    p.dst = d0.as<float>();
    ctx.run(p, nullptr);
    p.dst = d1.as<float>();
    p.w_ptrs = ptr.data();
    ctx.run(p, nullptr);
    ck(cudaDeviceSynchronize(), "run");
    const auto a = down<float>(d0, (size_t) rows * out_rows), c = down<float>(d1, (size_t) rows * out_rows);
    size_t bad = 0, nonfinite = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &c[i], 4) != 0) ++bad;
        if (!std::isfinite(c[i])) ++nonfinite;
    }
    std::printf("  in place %-5s [%lld x %lld] %2d experts %5d rows: %zu of %zu outputs differ, %zu non-finite\n",
                ggml_type_name(t), (long long) out_rows, (long long) cols, n, rows, bad, a.size(), nonfinite);
    return bad == 0 && nonfinite == 0 ? 0 : 1;
}

// gate/up in place with each token quantized once (Product::y_rows: MMQ row r reads token src[r]) against the rows
// quantized per MMQ row from the same tokens, in place: dst bit for bit
int token_rows_case(const std::vector<int>& counts, int tokens, uint32_t seed) {
    constexpr int64_t OUT = 1280, COLS = 2560;
    const ggml_type t = GGML_TYPE_NVFP4;
    const int n = (int) counts.size();
    std::vector<int32_t> b((size_t) n + 1, 0);
    for (int e = 0; e < n; ++e) b[(size_t) e + 1] = b[(size_t) e] + counts[(size_t) e];
    const int rows = b[(size_t) n];
    int maxr = 0;
    for (int c : counts) maxr = std::max(maxr, c);
    std::mt19937 rng(seed);
    const size_t per = mmq::matrix_bytes(t, OUT, COLS);
    std::vector<std::unique_ptr<Dev>> own;
    std::vector<const void*> ptr((size_t) n);
    for (int e = 0; e < n; ++e) {
        auto w = weights(t, OUT, COLS, rng);
        w.resize(per + 4096, 0xff);
        own.push_back(std::make_unique<Dev>(w.size()));
        up(*own.back(), w);
        ptr[(size_t) e] = own.back()->p;
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> mag(0.1f, 10.f);
    std::vector<float> x((size_t) tokens * COLS);
    for (int r = 0; r < tokens; ++r) { const float m = mag(rng); for (int64_t k = 0; k < COLS; ++k) x[(size_t) (r * COLS + k)] = nd(rng) * m; }
    std::uniform_int_distribution<int> tok(0, tokens - 1);
    std::vector<int32_t> src((size_t) rows), ids((size_t) rows);
    for (int r = 0; r < rows; ++r) { src[(size_t) r] = tok(rng); ids[(size_t) r] = r; }
    Dev dx(x.size() * 4), dsrc(src.size() * 4 + 4), dids(ids.size() * 4 + 4), db(b.size() * 4),
        dq_rows(mmq::q8_bytes(rows, COLS)), dys_rows((size_t) rows * 4 + 4), dq_tok(mmq::q8_bytes(tokens, COLS)),
        dys_tok((size_t) tokens * 4 + 4), d0((size_t) rows * OUT * 4), d1((size_t) rows * OUT * 4);
    up(dx, x); up(dsrc, src); up(dids, ids); up(db, b);
    mmq::quantize(dx.as<float>(), dsrc.as<int32_t>(), dq_rows.p, (int) t, COLS, COLS, rows, nullptr, dys_rows.as<float>());
    mmq::quantize(dx.as<float>(), nullptr, dq_tok.p, (int) t, COLS, COLS, tokens, nullptr, dys_tok.as<float>());
    mmq::Context ctx;
    mmq::Product p;
    p.w = ptr[0]; p.type = (int) t; p.w_rows = OUT; p.w_cols = COLS; p.expert_bytes = per; p.n = n;
    p.bounds = db.as<int32_t>(); p.ids = dids.as<int32_t>(); p.total_rows = rows; p.max_rows = maxr; p.ld_dst = OUT;
    p.w_ptrs = ptr.data();
    p.xq = dq_rows.p; p.y_scale = dys_rows.as<float>(); p.dst = d0.as<float>();
    ctx.run(p, nullptr);
    p.xq = dq_tok.p; p.y_scale = dys_tok.as<float>(); p.dst = d1.as<float>();
    p.y_rows = dsrc.as<int32_t>(); p.y_count = tokens;
    ctx.run(p, nullptr);
    ck(cudaDeviceSynchronize(), "run");
    const auto a = down<float>(d0, (size_t) rows * OUT), c = down<float>(d1, (size_t) rows * OUT);
    size_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i) bad += std::memcmp(&a[i], &c[i], 4) != 0;
    std::printf("  token rows NVFP4 gate/up %2d experts %5d rows of %4d tokens: %zu of %zu outputs differ\n", n, rows,
                tokens, bad, a.size());
    return bad == 0 ? 0 : 1;
}

}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("prefill_moe_chain_test: no CUDA device (skipped)\n");
        return 77;
    }
    int fails = 0;
    try {
        for (ggml_type t : {GGML_TYPE_Q8_0, GGML_TYPE_NVFP4}) {
            if (!mmq::swiglu_quant_ok((int) t)) { std::printf("  %s: not on the fused path here\n", ggml_type_name(t)); continue; }
            fails += swiglu_quant_case(t, {37, 0, 1, 200, 5, 0, 64, 129}, 0, 1);
            fails += swiglu_quant_case(t, {700, 3, 0, 1251, 90, 626, 1, 2, 0, 400, 333, 17, 8, 1000, 64, 9}, 4113, 2);
            fails += swiglu_quant_case(t, {1}, 77, 3);
        }
        const std::vector<int> group = {130, 0, 7, 626, 1, 255, 128, 40, 900, 3, 64, 200, 129, 17, 2, 500};
        if (mmq::direct_ok((int) GGML_TYPE_NVFP4)) {
            fails += in_place_case(GGML_TYPE_NVFP4, 1280, 2560, group, 11);   // gate/up
            fails += in_place_case(GGML_TYPE_NVFP4, 2560, 640, group, 12);    // down (w4a8)
            fails += in_place_case(GGML_TYPE_NVFP4, 2560, 640, {1, 33}, 13);
        }
        if (mmq::token_rows_ok((int) GGML_TYPE_NVFP4, 2560)) {
            fails += token_rows_case(group, 400, 21);
            fails += token_rows_case({3, 0, 129}, 7, 22);
        }
        if (mmq::direct_ok((int) GGML_TYPE_Q8_0)) {
            fails += in_place_case(GGML_TYPE_Q8_0, 2560, 640, group, 14);     // the Q8_0-down pack's down
            fails += in_place_case(GGML_TYPE_Q8_0, 2560, 640, {5}, 15);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill_moe_chain_test: %s\n", e.what());
        return 1;
    }
    std::printf("prefill_moe_chain_test: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
