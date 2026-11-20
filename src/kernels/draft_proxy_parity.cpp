// src/kernels/draft_proxy_parity.cpp - the MTP proxy draft head (draft_proxy.hpp) on a head held as bit planes
// (q80_vhead.hpp, the default since the two-stage verify head) against the same head as plain ggml Q8_0 rows.
//
//   draft_proxy_parity --selftest
//
// A synthetic head (65,536 rows; some rows are near-copies of others so many rows land within the rescoring window),
// a row map with empty (-1) entries, the 2-bit copy packed from either layout (must be byte-identical), then the
// two-stage scan for 96 columns at both tiers (n_rows = base rows, all rows): the draft id and its probability must be
// bitwise equal between the layouts.
#include "strata/kernels/draft_proxy.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/q80_vhead.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
constexpr int N = 2560, RB = 2720;
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
}  // namespace

int main(int argc, char** argv) {
    (void) argc;
    (void) argv;
    if (!strata::kernels::q80p_supported(N, 1)) {
        std::printf("draft_proxy_parity: SKIP (the planes need CUDA sm_80+)\n");
        return 0;
    }
    std::mt19937 rng(4321);
    const int V = 65536;
    std::vector<uint8_t> head((size_t) V * RB);
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
    // near-copies: rows 1000..1999 = row 7 with one code moved, so the scan sends many rows to the exact pass
    for (int r = 1000; r < 2000; ++r) {
        std::memcpy(head.data() + (size_t) r * RB, head.data() + (size_t) 7 * RB, RB);
        uint8_t* q = head.data() + (size_t) r * RB + (size_t) (r % 80) * 34 + 2 + (r % 32);
        const int v = (int) (int8_t) *q + ((r & 1) ? 1 : -1);
        *q = (uint8_t) (int8_t) std::max(-127, std::min(127, v));
    }
    // the map: 45,000 draft rows (base 30,000 + 512 empty context rows + the rest), token ids shuffled
    const int n_base = 30000, n_ctx = 512, n_cap = 45000;
    std::vector<int32_t> ids(V);
    for (int i = 0; i < V; ++i) ids[i] = i;
    std::shuffle(ids.begin() + 2000, ids.end(), rng);
    std::vector<int32_t> map(n_cap);
    int at = 0;
    for (int r = 0; r < n_cap; ++r) map[r] = (r >= n_base && r < n_base + n_ctx) ? -1 : ids[at++];
    for (int r = 0; r < n_ctx / 2; ++r) map[n_base + r] = ids[at++];   // half the context rows filled

    void *q8 = nullptr, *pl = nullptr, *pa = nullptr, *pb = nullptr, *sa = nullptr, *sb = nullptr, *xq = nullptr;
    int32_t *dmap = nullptr, *nrows = nullptr, *ida = nullptr, *idb = nullptr;
    float *x = nullptr, *pra = nullptr, *prb = nullptr;
    const size_t PB = strata::kernels::draft_proxy_bytes(n_cap), SB = strata::kernels::draft_proxy_scratch_bytes(n_cap);
    ck(cudaMalloc(&q8, head.size()), "malloc");
    ck(cudaMalloc(&pl, head.size()), "malloc");
    ck(cudaMalloc(&pa, PB), "malloc");
    ck(cudaMalloc(&pb, PB), "malloc");
    ck(cudaMalloc(&sa, SB), "malloc");
    ck(cudaMalloc(&sb, SB), "malloc");
    ck(cudaMalloc(&xq, (size_t) N / 32 * 36), "malloc");
    ck(cudaMalloc(&x, (size_t) N * 4), "malloc");
    ck(cudaMalloc((void**) &dmap, (size_t) n_cap * 4), "malloc");
    ck(cudaMalloc((void**) &nrows, 4), "malloc");
    ck(cudaMalloc((void**) &ida, 4), "malloc");
    ck(cudaMalloc((void**) &idb, 4), "malloc");
    ck(cudaMalloc((void**) &pra, 4), "malloc");
    ck(cudaMalloc((void**) &prb, 4), "malloc");
    ck(cudaMemcpy(q8, head.data(), head.size(), cudaMemcpyHostToDevice), "h2d");
    ck(cudaMemcpy(pl, head.data(), head.size(), cudaMemcpyHostToDevice), "h2d");
    strata::kernels::q80p_pack(pl, V, nullptr);
    ck(cudaMemcpy(dmap, map.data(), (size_t) n_cap * 4, cudaMemcpyHostToDevice), "h2d");
    ck(cudaMemset(pa, 0, PB), "memset");
    ck(cudaMemset(pb, 0, PB), "memset");
    cudaStream_t s = nullptr;
    ck(cudaStreamCreate(&s), "stream");
    // pack in two pieces (the context rows' path packs at an offset)
    strata::kernels::draft_proxy_pack(q8, false, dmap, 0, n_base, pa, n_cap, s);
    strata::kernels::draft_proxy_pack(q8, false, dmap, n_base, n_cap - n_base, pa, n_cap, s);
    strata::kernels::draft_proxy_pack(pl, true, dmap, 0, n_base, pb, n_cap, s);
    strata::kernels::draft_proxy_pack(pl, true, dmap, n_base, n_cap - n_base, pb, n_cap, s);
    ck(cudaStreamSynchronize(s), "pack");
    long long checks = 0, fails = 0;
    {
        std::vector<uint8_t> a(PB), b(PB);
        ck(cudaMemcpy(a.data(), pa, PB, cudaMemcpyDeviceToHost), "d2h");
        ck(cudaMemcpy(b.data(), pb, PB, cudaMemcpyDeviceToHost), "d2h");
        ++checks;
        if (std::memcmp(a.data(), b.data(), PB) != 0) {
            ++fails;
            std::printf("FAIL: the 2-bit copy differs between the layouts\n");
        }
    }
    // the columns: random, and row 7 itself (its near-copies then crowd the exact pass)
    std::normal_distribution<float> nx(0.0f, 1.0f);
    int multi = 0;
    for (int c = 0; c < 96; ++c) {
        std::vector<float> hx(N);
        if (c % 3 == 0) {   // aligned with row 7: many candidates within kDelta
            for (int i = 0; i < N; ++i) {
                const uint8_t* blk = head.data() + (size_t) 7 * RB + (size_t) (i / 32) * 34;
                hx[i] = 0.02f * (float) (int8_t) blk[2 + i % 32] + 0.3f * nx(rng);
            }
        } else
            for (int i = 0; i < N; ++i) hx[i] = nx(rng) * (1.0f + (float) (c % 5));
        ck(cudaMemcpy(x, hx.data(), (size_t) N * 4, cudaMemcpyHostToDevice), "h2d");
        strata::kernels::native_quantize_q8_1(x, xq, N, 1, s);
        for (int tier = 0; tier < 2; ++tier) {
            const int nr = tier == 0 ? n_base + n_ctx : n_cap;
            ck(cudaMemcpyAsync(nrows, &nr, 4, cudaMemcpyHostToDevice, s), "h2d");
            // the first exact-pass kernel on plain rows is the reference; final2_kernel on both layouts must give its bits
            int32_t ir = 0;
            float fr = 0;
            strata::kernels::draft_proxy_final2(false);
            strata::kernels::draft_proxy_top1(pa, q8, false, xq, nrows, n_cap, dmap, ida, pra, sa, s);
            ck(cudaMemcpyAsync(&ir, ida, 4, cudaMemcpyDeviceToHost, s), "d2h");
            ck(cudaMemcpyAsync(&fr, pra, 4, cudaMemcpyDeviceToHost, s), "d2h");
            ck(cudaStreamSynchronize(s), "top1 ref");
            for (int f2 = 0; f2 < 2; ++f2) {
                strata::kernels::draft_proxy_final2(f2 != 0);
                strata::kernels::draft_proxy_top1(pa, q8, false, xq, nrows, n_cap, dmap, ida, pra, sa, s);
                strata::kernels::draft_proxy_top1(pb, pl, true, xq, nrows, n_cap, dmap, idb, prb, sb, s);
                int32_t ia = 0, ib = 0;
                float fa = 0, fb = 0;
                ck(cudaMemcpyAsync(&ia, ida, 4, cudaMemcpyDeviceToHost, s), "d2h");
                ck(cudaMemcpyAsync(&ib, idb, 4, cudaMemcpyDeviceToHost, s), "d2h");
                ck(cudaMemcpyAsync(&fa, pra, 4, cudaMemcpyDeviceToHost, s), "d2h");
                ck(cudaMemcpyAsync(&fb, prb, 4, cudaMemcpyDeviceToHost, s), "d2h");
                ck(cudaStreamSynchronize(s), "top1");
                checks += 2;
                if (f2 && ia >= 1000 && ia < 2000) ++multi;
                if (ia != ir || std::memcmp(&fa, &fr, 4) != 0 || ib != ir || std::memcmp(&fb, &fr, 4) != 0) {
                    ++fails;
                    if (fails < 20)
                        std::printf("FAIL col %d tier %d final2 %d: reference id %d p %.9g, plain id %d p %.9g, planes id %d p %.9g\n",
                                    c, tier, f2, ir, fr, ia, fa, ib, fb);
                }
            }
        }
    }
    std::printf("draft_proxy_parity: %s, %lld checks, %lld failures (%d picks among the near-copy rows)\n",
                fails == 0 ? "PASS" : "FAIL", checks, fails, multi);
    return fails == 0 ? 0 : 1;
}
