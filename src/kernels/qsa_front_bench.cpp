// src/kernels/qsa_front_bench.cpp - dec3 D5: the QSA indexer chain of a verify window on the engine's kernels and shapes.
// Usage: qsa_front_bench [ctx=1400] [reps=60] [mode=all|parity|time]
//   parity: the window append (STRATA_IDX_APPEND_PAR) and the KV-stream resolve (STRATA_KV_RESOLVE_ILP) against their
//           old kernels, byte for byte on every buffer, T 1..8 at four cell offsets, a cell-0 window, a skipped step, and
//           resolves with all hits, all misses and a mix.
//   time:   graphs of 12 layers (own buffers each, L2 scrubbed before each replay), us per layer: each kernel alone and
//           the chain wik -> append -> wiq -> norm+rope -> scores -> top-k -> resolve -> copy, old and new kernels,
//           and with the append forked beside the query (FORK).
//   dx:     r5 F2: the long-context select's DX block scorer (qsa_block_scores_long, STRATA_SCORES_DX=1) against the
//           default scores kernel on the same window, T 1..8 at four cell offsets: every score bitwise and the top-k's
//           ids identical (the cluster top-k on both rows).  QFB_MAX_CELLS: the capacity (the engine's 524288).
//   f5:     r5 F5 resolve with the resident-prefix fast path vs F2 (all runs of every resident block, 5 map states);
//           f5t: the resolve's timing, all hits, with M blocks evicted inside the graph, and on the fast path.
#include "strata/core/geometry.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/rope_scaling.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset");
    return p;
}
template <typename T> void up(T* d, const std::vector<T>& h) { ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "up"); }
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "down");
    return h;
}
uint16_t bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16); }

constexpr int L = 12, TMAX = 8;
const k::QsaShapes S = k::qsa_real_shapes();
const int64_t N = strata::core::kCompiledGeometry.n_embd, ID = 128, IQ = 4;
const float EPS = 1e-6f;
const k::RopeScaling RS{};

struct Layer {
    float *tail, *dead, *pooled, *gamma, *gq, *raw, *qidx, *scores, *x;
    int32_t *block_pos, *sel;
    uint16_t *wik, *wiq;
    k::KvStreamMap m;
    k::QsaAttnPools slots;
};
int64_t g_max_cells, g_max_blocks, g_cap, g_n_blocks, g_n_slots;
k::KvHostPools g_host;
int32_t *g_steps, *g_posi;
cudaStream_t g_s = nullptr;   // the indexer launchers want an explicit stream   // [TMAX][kStepCount]; per indexer-query-row positions [TMAX * IQ]

uint64_t block_bytes() { return k::kv_block_bytes(S, k::kKvInt8); }

Layer make_layer(std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    Layer y{};
    y.tail = dalloc<float>(3 * ID); y.dead = dalloc<float>(ID); y.pooled = dalloc<float>((g_max_cells / 4 + 1) * ID);
    y.block_pos = dalloc<int32_t>(1);
    std::vector<float> g(ID);
    for (auto& v : g) v = 1.0f + 0.1f * nd(rng);
    y.gamma = dalloc<float>(ID); up(y.gamma, g);
    for (auto& v : g) v = 1.0f + 0.1f * nd(rng);
    y.gq = dalloc<float>(ID); up(y.gq, g);
    y.raw = dalloc<float>(TMAX * ID); y.qidx = dalloc<float>(TMAX * IQ * ID);
    y.scores = dalloc<float>(TMAX * g_max_blocks); y.sel = dalloc<int32_t>(TMAX * g_cap);
    std::vector<float> x(TMAX * N);
    for (auto& v : x) v = nd(rng);
    y.x = dalloc<float>(TMAX * N); up(y.x, x);
    std::vector<uint16_t> w(ID * N);
    for (auto& v : w) v = bf16(0.02f * nd(rng));
    y.wik = dalloc<uint16_t>(ID * N); up(y.wik, w);
    w.resize(IQ * ID * N);
    for (auto& v : w) v = bf16(0.02f * nd(rng));
    y.wiq = dalloc<uint16_t>(IQ * ID * N); up(y.wiq, w);
    // the KV map (int8 K/V): slots in VRAM, the host copy shared by the layers
    y.m.n_blocks = g_n_blocks; y.m.n_slots = g_n_slots;
    y.m.page_table = dalloc<int32_t>(g_n_blocks); y.m.slot_block = dalloc<int32_t>(g_n_slots);
    y.m.slot_stamp = dalloc<int32_t>(g_n_slots); y.m.slot_ref = dalloc<int32_t>(g_n_slots);
    y.m.ctl = dalloc<int32_t>(k::kKvCtlInts); y.m.miss_block = dalloc<int32_t>(g_n_slots); y.m.miss_slot = dalloc<int32_t>(g_n_slots);
    const int64_t rows = S.n_head_kv * S.page_size, codes = rows * S.head_dim, scales = rows * (S.head_dim / 64);
    y.slots.k_q = dalloc<int8_t>(g_n_slots * codes); y.slots.v_q = dalloc<int8_t>(g_n_slots * codes);
    y.slots.k_scale = dalloc<uint16_t>(g_n_slots * scales); y.slots.v_scale = dalloc<uint16_t>(g_n_slots * scales);
    y.slots.page_table = y.m.page_table;
    k::kv_stream_reset(y.m, g_s);
    return y;
}

// the window's steps: T cells from p0 (a step at p0 + skip is pushed past max_cells: skipped)
void set_steps(int64_t p0, int T, int skip = -1) {
    std::vector<int32_t> st(TMAX * k::kStepCount, 0), pi(TMAX * IQ, 0);
    for (int t = 0; t < TMAX; ++t) {
        const int64_t pos = p0 + std::min(t, T - 1);
        int32_t* s = st.data() + t * k::kStepCount;
        s[k::kStepPos] = t == skip ? (int32_t) g_max_cells + 5 : (int32_t) pos;
        s[k::kStepNKv] = (int32_t) (pos + 1);
        s[k::kStepNBid] = (int32_t) ((pos + 1) / S.idx_block);
        s[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, S);
        for (int h = 0; h < IQ; ++h) pi[t * IQ + h] = (int32_t) pos;
    }
    up(g_steps, st);
    up(g_posi, pi);
}
k::QsaIndexerBuffers ib(const Layer& y) { return {y.tail, y.dead, y.pooled, y.block_pos}; }

// the indexer state of a prompt of `ctx` cells (the prompt path's batched append, random raw keys)
void prefill(Layer& y, int64_t ctx, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    ck(cudaMemset(y.tail, 0, 3 * ID * 4), "m"); ck(cudaMemset(y.dead, 0, ID * 4), "m");
    ck(cudaMemset(y.pooled, 0, (g_max_cells / 4 + 1) * ID * 4), "m");
    if (ctx <= 0) return;
    std::vector<float> raw(ctx * ID);
    for (auto& v : raw) v = nd(rng);
    float* d = dalloc<float>(raw.size()); up(d, raw);
    k::native_qsa_indexer_append_batch(d, ctx, 0, 0, y.gamma, EPS, ib(y), S, g_max_cells, RS, g_s);
    ck(cudaStreamSynchronize(g_s), "prefill");
    cudaFree(d);
}
void new_raw(Layer& y, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> r(TMAX * ID);
    for (auto& v : r) v = nd(rng);
    up(y.raw, r);
}

// ---- the chain's pieces on stream s
void k_wik(const Layer& y, int T, cudaStream_t s) { k::bf16_gemv_fp32_mmvf_multi(y.x, N, y.wik, y.raw, ID, N, ID, T, s); }
void k_wiq(const Layer& y, int T, cudaStream_t s) {
    k::bf16_gemv_fp32_mmvf_multi(y.x, N, y.wiq, y.qidx, IQ * ID, N, IQ * ID, T, s);
}
void k_nr(const Layer& y, int T, cudaStream_t s) {
    k::native_qsa_rms_norm_rope(y.qidx, (int) ID, y.gq, y.qidx, (int) (T * IQ), (int) ID, (int) S.n_rot, EPS, RS, g_posi, s);
}
void k_app(const Layer& y, int T, cudaStream_t s) {
    k::native_qsa_indexer_append_steps(y.raw, g_steps + k::kStepPos, k::kStepCount, T, 0, y.gamma, EPS, ib(y), S, g_max_cells, RS, s);
}
void k_sc(const Layer& y, int T, cudaStream_t s) {
    k::qsa_block_scores(y.pooled, y.dead, y.qidx, g_steps, T, g_max_blocks, S, y.scores, s);
}
void k_scx(const Layer& y, int T, cudaStream_t s) {   // r5 F2: the long-context select's scorer (DX with STRATA_SCORES_DX=1)
    k::qsa_block_scores_long(y.pooled, y.dead, y.qidx, g_steps, T, g_max_blocks, S, y.scores, s, -1);
}
void k_tk(const Layer& y, int T, cudaStream_t s) { k::qsa_block_topk(y.scores, g_steps, T, g_max_blocks, g_cap, S, y.sel, s); }
void k_res(const Layer& y, int T, cudaStream_t s) {
    k::kv_stream_resolve(y.m, y.slots, g_host, k::kKvInt8, y.sel, g_steps, T, g_cap, S, s);
}

// ---- parity
struct Snap { std::vector<float> tail, dead, pooled; std::vector<int32_t> bp; };
Snap snap_idx(const Layer& y) {
    return {down(y.tail, 3 * ID), down(y.dead, ID), down(y.pooled, (g_max_cells / 4 + 1) * ID), down(y.block_pos, 1)};
}
void restore_idx(Layer& y, const Snap& z) { up(y.tail, z.tail); up(y.dead, z.dead); up(y.pooled, z.pooled); up(y.block_pos, z.bp); }
bool same(const Snap& a, const Snap& b) {
    return std::memcmp(a.tail.data(), b.tail.data(), a.tail.size() * 4) == 0 &&
           std::memcmp(a.dead.data(), b.dead.data(), a.dead.size() * 4) == 0 &&
           std::memcmp(a.pooled.data(), b.pooled.data(), a.pooled.size() * 4) == 0 && a.bp == b.bp;
}
struct MapSnap { std::vector<int32_t> pt, sb, ss, sr, ctl; };
MapSnap snap_map(const Layer& y) {
    return {down(y.m.page_table, g_n_blocks), down(y.m.slot_block, g_n_slots), down(y.m.slot_stamp, g_n_slots),
            down(y.m.slot_ref, g_n_slots), down(y.m.ctl, k::kKvCtlInts)};
}
void restore_map(Layer& y, const MapSnap& z) {
    up(y.m.page_table, z.pt); up(y.m.slot_block, z.sb); up(y.m.slot_stamp, z.ss); up(y.m.slot_ref, z.sr); up(y.m.ctl, z.ctl);
}
// every resident block's slot holds the host copy's bytes
bool slots_ok(const Layer& y, const MapSnap& z) {
    const int64_t rows = S.n_head_kv * S.page_size, codes = rows * S.head_dim;
    auto kq = down((const int8_t*) y.slots.k_q, g_n_slots * codes);
    for (int64_t b = 0; b < g_n_blocks; ++b) {
        const int sl = z.pt[b];
        if (sl < -1) return false;
        if (sl >= 0 && std::memcmp(kq.data() + sl * codes, g_host.k_q + b * codes, codes) != 0) return false;
    }
    return true;
}

int parity(std::vector<Layer>& Ls, int64_t ctx) {
    std::mt19937 rng(11);
    Layer& y = Ls[0];
    int bad = 0, n = 0;
    auto app_case = [&](int64_t c0, int T, int skip, const char* what) {
        if (std::getenv("QFB_TRACE")) std::printf("case %s c0 %lld T %d\n", what, (long long) c0, T);
        prefill(y, c0, rng);
        new_raw(y, rng);
        set_steps(c0, T, skip);
        const Snap z = snap_idx(y);
        k::native_qsa_indexer_set_append_par(0);
        k_app(y, T, g_s);
        ck(cudaStreamSynchronize(g_s), "old");
        const Snap a = snap_idx(y);
        restore_idx(y, z);
        k::native_qsa_indexer_set_append_par(1);
        k_app(y, T, g_s);
        ck(cudaStreamSynchronize(g_s), "new");
        const Snap b = snap_idx(y);
        const bool ok = same(a, b) && !same(a, z);
        ++n; bad += !ok;
        if (!ok) std::printf("*** append %s c0 %lld T %d skip %d: differs\n", what, (long long) c0, T, skip);
    };
    for (int T = 1; T <= TMAX; ++T)
        for (int off = 0; off < 4; ++off) app_case(ctx - (ctx % 4) + off, T, -1, "window");
    for (int T = 1; T <= TMAX; ++T) app_case(0, T, -1, "cell0");
    for (int T = 3; T <= TMAX; ++T) app_case(ctx + 1, T, 1, "skip");
    for (int T = 2; T <= TMAX; ++T) app_case(4 * 37 + 2, T, -1, "short");
    std::printf("append parity: %d / %d cases bitwise\n", n - bad, n);
    // resolve: select on a real window, then old and new from the same map state
    int rbad = 0, rn = 0;
    prefill(y, ctx + TMAX, rng);
    for (int T = 1; T <= TMAX; ++T) {
        set_steps(ctx, T);
        new_raw(y, rng);
        k_wiq(y, T, g_s); k_nr(y, T, g_s); k_sc(y, T, g_s); k_tk(y, T, g_s);
        for (int kind = 0; kind < 3; ++kind) {   // 0 all misses (reset), 1 all hits (after one resolve), 2 a mix
            k::kv_stream_reset(y.m, g_s);
            if (kind == 1) k_res(y, T, g_s);
            if (kind == 2) {   // the blocks of a short earlier selection resident, the rest missing
                set_steps(ctx / 3, 1);
                k_sc(y, 1, g_s); k_tk(y, 1, g_s); k_res(y, 1, g_s);
                set_steps(ctx, T);
                k_sc(y, T, g_s); k_tk(y, T, g_s);
            }
            ck(cudaStreamSynchronize(g_s), "pre");
            const MapSnap z = snap_map(y);
            k::kv_stream_set_resolve_ilp(0);
            k_res(y, T, g_s);
            ck(cudaStreamSynchronize(g_s), "old");
            const MapSnap a = snap_map(y);
            const bool aok = slots_ok(y, a);
            restore_map(y, z);
            k::kv_stream_set_resolve_ilp(1);
            k_res(y, T, g_s);
            ck(cudaStreamSynchronize(g_s), "new");
            const MapSnap b = snap_map(y);
            const bool bok = slots_ok(y, b);
            // all hits: every array equal.  With misses the slot a block lands in follows the miss list's order (an
            // atomic, never fixed): the same resident set, counters and data
            bool ok = aok && bok && a.ctl == b.ctl;
            if (kind == 1) ok = ok && a.pt == b.pt && a.sb == b.sb && a.ss == b.ss && a.sr == b.sr;
            else
                for (int64_t i = 0; i < g_n_blocks && ok; ++i) ok = (a.pt[i] >= 0) == (b.pt[i] >= 0);
            ++rn; rbad += !ok;
            if (!ok) std::printf("*** resolve T %d kind %d differs (old data %d new data %d)\n", T, kind, aok, bok);
            if (T == TMAX || T == 1)
                std::printf("  resolve T %d kind %d: misses %d (old) %d (new)\n", T, kind, a.ctl[2], b.ctl[2]);
        }
    }
    k::kv_stream_set_resolve_ilp(-1);
    // scores: the pair kernel against the multi kernel, every row of the scores buffer, T 1..8 at four offsets
    int sbad = 0, sn = 0;
    for (int T = 1; T <= TMAX; ++T)
        for (int off = 0; off < 4; ++off) {
            set_steps(ctx - (ctx % 4) + off, T);
            new_raw(y, rng);
            k_wiq(y, T, g_s); k_nr(y, T, g_s);
            std::vector<float> r[2];
            for (int v = 0; v < 2; ++v) {
                ck(cudaMemsetAsync(y.scores, 0x7f, TMAX * g_max_blocks * 4, g_s), "m");
                k::qsa_set_scores_pairs(v);
                k_sc(y, T, g_s);
                ck(cudaStreamSynchronize(g_s), "sc");
                r[v] = down(y.scores, TMAX * g_max_blocks);
            }
            const bool ok = std::memcmp(r[0].data(), r[1].data(), r[0].size() * 4) == 0;
            ++sn; sbad += !ok;
            if (!ok) std::printf("*** scores T %d off %d differ\n", T, off);
        }
    k::qsa_set_scores_pairs(-1);
    std::printf("scores parity: %d / %d cases bitwise\n", sn - sbad, sn);
    rbad += sbad;
    k::native_qsa_indexer_set_append_par(-1);
    std::printf("resolve parity: %d / %d cases\n", rn - rbad, rn);
    return bad + rbad;
}

// r5 F2: DX scores + top-k against the default kernels on a prompt of ctx cells
int dx_parity(std::vector<Layer>& Ls, int64_t ctx) {
    std::mt19937 rng(17);
    Layer& y = Ls[0];
    prefill(y, ctx + TMAX, rng);
    const long long dx0 = (long long) k::qsa_scores_dx_launches();
    int bad = 0, n = 0;
    for (int T = 1; T <= TMAX; ++T)
        for (int off = 0; off < 4; ++off) {
            set_steps(ctx - (ctx % 4) + off, T);
            new_raw(y, rng);
            k_wiq(y, T, g_s); k_nr(y, T, g_s);
            std::vector<float> sc[2];
            std::vector<int32_t> id[2];
            for (int v = 0; v < 2; ++v) {
                ck(cudaMemsetAsync(y.scores, 0x7f, TMAX * g_max_blocks * 4, g_s), "m");
                ck(cudaMemsetAsync(y.sel, 0xff, TMAX * g_cap * 4, g_s), "m");
                if (v == 0) k_sc(y, T, g_s); else k_scx(y, T, g_s);
                k_tk(y, T, g_s);
                ck(cudaStreamSynchronize(g_s), "dx");
                sc[v] = down(y.scores, TMAX * g_max_blocks);
                id[v] = down(y.sel, TMAX * g_cap);
            }
            const bool ok = std::memcmp(sc[0].data(), sc[1].data(), sc[0].size() * 4) == 0 && id[0] == id[1];
            ++n; bad += !ok;
            if (!ok) std::printf("*** dx T %d off %d differs\n", T, off);
        }
    const long long dx = (long long) k::qsa_scores_dx_launches() - dx0;
    std::printf("dx parity ctx %lld: %d / %d cases bitwise (scores + ids), DX launches %lld\n", (long long) ctx, n - bad, n, dx);
    if (dx < n) { std::printf("*** the DX kernel did not run for every case (set STRATA_SCORES_DX=1)\n"); ++bad; }
    return bad;
}

// ---- r5 F5: the resolve with the resident-prefix fast path (STRATA_KV_RESIDENT_FAST) against F2's resolve from the
// same map state; kinds 0 all misses, 1 all hits, 2 a mix, 3 clock evictions (the slots filled by earlier random
// windows first), 4 a new sequence's resident map.  Every run of every resident block checked against the host.
bool slots_ok_all(const Layer& y, const MapSnap& z) {
    const int64_t rows = S.n_head_kv * S.page_size, codes = rows * S.head_dim, scales = rows * (S.head_dim / 64);
    auto kq = down((const int8_t*) y.slots.k_q, g_n_slots * codes);
    auto vq = down((const int8_t*) y.slots.v_q, g_n_slots * codes);
    auto ks = down((const uint16_t*) y.slots.k_scale, g_n_slots * scales);
    auto vs = down((const uint16_t*) y.slots.v_scale, g_n_slots * scales);
    for (int64_t b = 0; b < g_n_blocks; ++b) {
        const int sl = z.pt[b];
        if (sl < -1) return false;
        if (sl < 0) continue;
        if (std::memcmp(kq.data() + sl * codes, g_host.k_q + b * codes, codes) != 0 ||
            std::memcmp(vq.data() + sl * codes, g_host.v_q + b * codes, codes) != 0 ||
            std::memcmp(ks.data() + sl * scales, g_host.k_scale + b * scales, scales * 2) != 0 ||
            std::memcmp(vs.data() + sl * scales, g_host.v_scale + b * scales, scales * 2) != 0)
            return false;
    }
    return true;
}
void select_at(Layer& y, int64_t p0, int T) {
    set_steps(p0, T);
    k_wiq(y, T, g_s); k_nr(y, T, g_s); k_sc(y, T, g_s); k_tk(y, T, g_s);
}
int f5_parity(std::vector<Layer>& Ls, int64_t ctx) {
    std::mt19937 rng(23);
    Layer& y = Ls[0];
    prefill(y, ctx + TMAX, rng);
    struct Arm { int fast; const char* name; };
    const Arm arms[] = {{1, "fast"}};
    int bad = 0, n = 0;
    long long misses_seen = 0, evict_cases = 0, fast_cases = 0;
    // kind 4: a new sequence's resident map (kv_stream_reset_resident) with the slots filled from the host copy, as
    // the appends would have written them; below the window the fast path must leave the same table and data
    for (int T = 1; T <= TMAX; ++T)
        for (int kind = 0; kind < 5; ++kind) {
            k::kv_stream_set_resident_fast(0);
            k::kv_stream_reset(y.m, g_s);
            if (kind == 4) {
                k::kv_stream_set_resident_fast(1);
                const int64_t w = k::kv_stream_reset_resident(y.m, g_n_slots, g_s);
                k::kv_stream_set_resident_fast(0);
                k::kv_ring_restore(y.slots, g_host, k::kKvInt8, 0, w, g_n_slots, S, g_s);
            }
            if (kind == 1) { select_at(y, ctx, T); k_res(y, T, g_s); }
            if (kind == 2) { select_at(y, ctx / 3, 1); k_res(y, 1, g_s); }
            if (kind == 3)
                for (int r = 0; r < 24; ++r) {
                    const int64_t p = 64 + (int64_t) (rng() % (uint32_t) (ctx - 64));
                    select_at(y, p, TMAX);
                    k_res(y, TMAX, g_s);
                }
            select_at(y, ctx, T);
            ck(cudaStreamSynchronize(g_s), "pre");
            const MapSnap z = snap_map(y);
            k_res(y, T, g_s);
            ck(cudaStreamSynchronize(g_s), "ref");
            const MapSnap a = snap_map(y);
            const bool aok = slots_ok_all(y, a);
            misses_seen += a.ctl[2];
            int evicted = 0;
            for (int64_t i = 0; i < g_n_slots; ++i) evicted += z.sb[i] >= 0 && a.sb[i] != z.sb[i];
            evict_cases += evicted > 0;
            for (const Arm& m : arms) {
                restore_map(y, z);
                k::kv_stream_set_resident_fast(m.fast);
                const auto f0 = k::kv_stream_counters(y.m).fast_calls;
                k_res(y, T, g_s);
                ck(cudaStreamSynchronize(g_s), "arm");
                const bool took_fast = k::kv_stream_counters(y.m).fast_calls != f0;
                const MapSnap b = snap_map(y);
                const bool bok = slots_ok_all(y, b);
                bool ok = aok && bok;
                if (took_fast)   // only the stamps, reference bits and lookup counter differ: nothing was looked up
                    ok = ok && a.ctl[2] == 0 && b.ctl[2] == 0 && a.pt == b.pt && a.sb == b.sb && kind == 4;
                else {
                    ok = ok && std::equal(a.ctl.begin(), a.ctl.begin() + 10, b.ctl.begin()) && a.ss == b.ss && a.sr == b.sr;
                    if (kind == 1) ok = ok && a.pt == b.pt && a.sb == b.sb;
                    else
                        for (int64_t i = 0; i < g_n_blocks && ok; ++i) ok = (a.pt[i] >= 0) == (b.pt[i] >= 0);
                    // the resident prefix only shrinks, and only to a block this call evicted
                    if (m.fast && kind == 4) ok = ok && b.ctl[k::kKvCtlResident] <= z.ctl[k::kKvCtlResident];
                }
                if (m.fast && kind == 4 && (int64_t) 4 * g_n_slots > ctx + T && !took_fast) ok = false;   // must have been fast
                fast_cases += took_fast;
                ++n; bad += !ok;
                if (!ok) std::printf("*** f5 %s T %d kind %d differs (ref data %d arm data %d fast %d)\n", m.name, T, kind, aok,
                                     bok, (int) took_fast);
            }
            if (T == 1 || T == TMAX)
                std::printf("  f5 T %d kind %d: misses %d, slots re-assigned %d\n", T, kind, a.ctl[2], evicted);
        }
    k::kv_stream_set_resident_fast(-1);
    std::printf("f5 parity ctx %lld: %d / %d cases (misses %lld, %lld cases with evictions, %lld on the fast path)\n",
                (long long) ctx, n - bad, n, misses_seen, evict_cases, fast_cases);
    return bad;
}

// ---- timing: a graph of the 12 layers, the L2 scrubbed before each replay; us per layer (median of reps)
uint8_t* g_scrub = nullptr;
const size_t kScrub = 256u << 20;
template <typename F> double time_graph(F&& rec, int reps, cudaStream_t s) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
    rec(s);
    ck(cudaStreamEndCapture(s, &g), "end capture");
    ck(cudaGraphInstantiate(&ge, g, 0), "instantiate");
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    std::vector<float> v;
    for (int r = 0; r < reps + 3; ++r) {
        ck(cudaMemsetAsync(g_scrub, r & 0xff, kScrub, s), "scrub");
        cudaEventRecord(a, s);
        ck(cudaGraphLaunch(ge, s), "launch");
        cudaEventRecord(b, s);
        ck(cudaEventSynchronize(b), "sync");
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        if (r >= 3) v.push_back(ms);
    }
    std::sort(v.begin(), v.end());
    cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return 1e3 * v[v.size() / 2] / L;
}

void timing(std::vector<Layer>& Ls, int64_t ctx, int reps) {
    std::mt19937 rng(5);
    cudaStream_t s, s2;
    ck(cudaStreamCreate(&s), "stream");
    ck(cudaStreamCreate(&s2), "stream");
    cudaEvent_t ef[L], ej[L];
    for (int i = 0; i < L; ++i) { cudaEventCreateWithFlags(&ef[i], cudaEventDisableTiming); cudaEventCreateWithFlags(&ej[i], cudaEventDisableTiming); }
    ck(cudaMalloc(&g_scrub, kScrub), "scrub");
    for (auto& y : Ls) prefill(y, ctx, rng), new_raw(y, rng);
    std::printf("ctx %lld, us per layer (median of %d replays of 12 layers, L2 scrubbed)\n", (long long) ctx, reps);
    std::printf("%-26s", "T");
    for (int T = 1; T <= TMAX; ++T) std::printf("%8d", T);
    std::printf("\n");
    using Fn = void (*)(const Layer&, int, cudaStream_t);
    struct Row { const char* name; Fn f; int par, ilp, pairs; };
    const Row rows[] = {
        {"wik gemv", k_wik, -1, -1, -1}, {"wiq gemv", k_wiq, -1, -1, -1}, {"norm+rope", k_nr, -1, -1, -1},
        {"append old", k_app, 0, -1, -1}, {"append new", k_app, 1, -1, -1}, {"scores old", k_sc, -1, -1, 0},
        {"scores new", k_sc, -1, -1, 1}, {"scores dx", k_scx, -1, -1, -1}, {"topk", k_tk, -1, -1, -1},
        {"resolve+copy old", k_res, -1, 0, -1}, {"resolve+copy new", k_res, -1, 1, -1},
    };
    auto setsw = [](int par, int ilp, int pairs) {
        k::native_qsa_indexer_set_append_par(par); k::kv_stream_set_resolve_ilp(ilp); k::qsa_set_scores_pairs(pairs);
    };
    // every timed resolve sees all hits (the engine's steady state): one resolve per layer and T first
    auto warm = [&](int T) {
        set_steps(ctx, T);
        for (auto& y : Ls) { k_wiq(y, T, s); k_nr(y, T, s); k_sc(y, T, s); k_tk(y, T, s); k_res(y, T, s); }
        ck(cudaStreamSynchronize(s), "warm");
    };
    std::vector<std::vector<double>> res(20);
    for (int T = 1; T <= TMAX; ++T) {
        warm(T);
        int ri = 0;
        for (const Row& r : rows) {
            setsw(r.par, r.ilp, r.pairs);
            res[ri++].push_back(time_graph([&](cudaStream_t st) { for (auto& y : Ls) r.f(y, T, st); }, reps, s));
        }
        // the chain: 0 old kernels, 1 new kernels, 2 new + the append forked beside wiq, 3 new + the key and its append
        // on a branch beside the query (both GEMVs start at once), 4 = 3 with the old kernels
        for (int v = 0; v < 5; ++v) {
            const int nw = v == 0 || v == 4 ? 0 : 1;
            setsw(nw, nw, nw);
            res[ri++].push_back(time_graph([&](cudaStream_t st) {
                for (int i = 0; i < L; ++i) {
                    const Layer& y = Ls[i];
                    if (v >= 3) {
                        cudaEventRecord(ef[i], st); cudaStreamWaitEvent(s2, ef[i], 0);
                        k_wik(y, T, s2); k_app(y, T, s2);
                        cudaEventRecord(ej[i], s2);
                        k_wiq(y, T, st); k_nr(y, T, st);
                        cudaStreamWaitEvent(st, ej[i], 0);
                        k_sc(y, T, st); k_tk(y, T, st); k_res(y, T, st);
                        continue;
                    }
                    k_wik(y, T, st);
                    if (v == 2) {
                        cudaEventRecord(ef[i], st); cudaStreamWaitEvent(s2, ef[i], 0);
                        k_app(y, T, s2);
                        cudaEventRecord(ej[i], s2);
                    } else k_app(y, T, st);
                    k_wiq(y, T, st); k_nr(y, T, st);
                    if (v == 2) cudaStreamWaitEvent(st, ej[i], 0);
                    k_sc(y, T, st); k_tk(y, T, st); k_res(y, T, st);
                }
            }, reps, s));
        }
    }
    setsw(-1, -1, -1);
    const char* names[] = {"wik gemv", "wiq gemv", "norm+rope", "append old", "append new", "scores old", "scores new",
                           "scores dx", "topk", "resolve+copy old", "resolve+copy new", "CHAIN old", "CHAIN new", "CHAIN new+fork",
                           "CHAIN new+par", "CHAIN old+par"};
    for (int i = 0; i < 16; ++i) {
        std::printf("%-26s", names[i]);
        for (double x : res[i]) std::printf("%8.2f", x);
        std::printf("\n");
    }
}

// r5 F5 timing: the resolve + copy alone, all hits and with M blocks of each layer's selection evicted before its
// resolve inside the graph (the evict kernel timed alone too), and on a new sequence's resident map (fast path)
void f5_timing(std::vector<Layer>& Ls, int64_t ctx, int reps) {
    std::mt19937 rng(5);
    cudaStream_t s;
    ck(cudaStreamCreate(&s), "stream");
    ck(cudaMalloc(&g_scrub, kScrub), "scrub");
    for (auto& y : Ls) prefill(y, ctx, rng), new_raw(y, rng);
    std::printf("f5 ctx %lld, us per layer (median of %d replays of 12 layers, L2 scrubbed)\n", (long long) ctx, reps);
    std::printf("%-30s", "T");
    for (int T = 1; T <= TMAX; ++T) std::printf("%8d", T);
    std::printf("\n");
    const int Ms[] = {2, 32, 96, 192};
    std::vector<int32_t*> ev(L * 4);
    for (auto& p : ev) p = dalloc<int32_t>(256);
    std::vector<std::string> names;
    std::vector<std::vector<double>> res;
    auto put = [&](size_t i, const std::string& nm, double v) {
        if (res.size() <= i) { res.resize(i + 1); names.resize(i + 1); }
        names[i] = nm; res[i].push_back(v);
    };
    for (int T = 1; T <= TMAX; ++T) {
        k::kv_stream_set_resident_fast(0);
        set_steps(ctx, T);
        for (auto& y : Ls) { k_wiq(y, T, s); k_nr(y, T, s); k_sc(y, T, s); k_tk(y, T, s); k_res(y, T, s); }
        ck(cudaStreamSynchronize(s), "warm");
        // the evict lists: M distinct blocks of the last query's selection, spread over it
        for (int i = 0; i < L; ++i) {
            const auto sel = down(Ls[i].sel + (size_t) (T - 1) * g_cap, g_cap);
            const int w = (int) k::qsa_selection_width(ctx + T, S);
            std::vector<int32_t> blk;
            for (int j = 0; j < w; ++j) if (blk.empty() || blk.back() != sel[j] / 4) blk.push_back(sel[j] / 4);
            for (int mi = 0; mi < 4; ++mi) {
                std::vector<int32_t> pick(256, 0);
                const int M = std::min<int>(Ms[mi], (int) blk.size());
                for (int j = 0; j < M; ++j) pick[j] = blk[(size_t) j * blk.size() / M];
                up(ev[i * 4 + mi], pick);
            }
        }
        size_t ri = 0;
        put(ri++, "hits", time_graph([&](cudaStream_t st) { for (auto& y : Ls) k_res(y, T, st); }, reps, s));
        for (int mi = 0; mi < 4; ++mi) {
            const int M = Ms[mi];
            char nm[64];
            std::snprintf(nm, sizeof nm, "evict %d only", M);
            put(ri++, nm, time_graph([&](cudaStream_t st) {
                for (int i = 0; i < L; ++i) k::kv_stream_test_evict(Ls[i].m, ev[i * 4 + mi], M, st);
            }, reps, s));
            std::snprintf(nm, sizeof nm, "evict %d + resolve", M);
            put(ri++, nm, time_graph([&](cudaStream_t st) {
                for (int i = 0; i < L; ++i) { k::kv_stream_test_evict(Ls[i].m, ev[i * 4 + mi], M, st); k_res(Ls[i], T, st); }
            }, reps, s));
        }
        // the resident fast path: a new sequence's map (slots filled from the host copy); with the selection past
        // the window (ctx > 4 x 5,120 here) it is the slow path plus the check
        k::kv_stream_set_resident_fast(1);
        for (auto& y : Ls) {
            const int64_t w = k::kv_stream_reset_resident(y.m, g_n_slots, s);
            k::kv_ring_restore(y.slots, g_host, k::kKvInt8, 0, w, g_n_slots, S, s);
            k_res(y, T, s);
        }
        ck(cudaStreamSynchronize(s), "fast warm");
        put(ri++, "resident fast", time_graph([&](cudaStream_t st) { for (auto& y : Ls) k_res(y, T, st); }, reps, s));
        k::kv_stream_set_resident_fast(0);
        put(ri++, "resident, fast off", time_graph([&](cudaStream_t st) { for (auto& y : Ls) k_res(y, T, st); }, reps, s));
        k::kv_stream_set_resident_fast(-1);
        for (auto& y : Ls) k::kv_stream_reset(y.m, s);
    }
    for (size_t i = 0; i < res.size(); ++i) {
        std::printf("%-30s", names[i].c_str());
        for (double x : res[i]) std::printf("%8.2f", x);
        std::printf("\n");
    }
}
}  // namespace

int main(int argc, char** argv) {
    const int64_t ctx = argc > 1 ? std::atoll(argv[1]) : 1400;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 60;
    const std::string mode = argc > 3 ? argv[3] : "all";
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    g_max_cells = std::getenv("QFB_MAX_CELLS") ? std::atoll(std::getenv("QFB_MAX_CELLS")) : 32768;   // r5: the engine capacity (524288)
    g_max_blocks = g_max_cells / 4 + 2;
    g_cap = k::qsa_selection_width(k::kTopkMaxCells, S);
    g_n_blocks = g_max_cells / S.page_size;
    g_n_slots = 5120;
    if (ctx + 64 > g_max_cells) { std::fprintf(stderr, "ctx must be below %lld\n", (long long) (g_max_cells - 64)); return 2; }
    // the host copy: pinned, device-mapped, random bytes
    const uint64_t bb = block_bytes();
    const int64_t rows = S.n_head_kv * S.page_size, codes = rows * S.head_dim, scales = rows * (S.head_dim / 64);
    if ((uint64_t) (2 * codes + 2 * scales * 2) != bb) { std::fprintf(stderr, "block bytes %llu\n", (unsigned long long) bb); return 2; }
    void* h = nullptr;
    ck(cudaHostAlloc(&h, (size_t) (g_n_blocks * bb), cudaHostAllocMapped), "host");
    {
        std::mt19937 r(3);
        uint32_t* p = (uint32_t*) h;
        for (size_t i = 0; i < g_n_blocks * bb / 4; ++i) p[i] = r();
    }
    uint8_t* hp = (uint8_t*) h;
    g_host.k_q = (int8_t*) hp; hp += g_n_blocks * codes;
    g_host.v_q = (int8_t*) hp; hp += g_n_blocks * codes;
    g_host.k_scale = (uint16_t*) hp; hp += g_n_blocks * scales * 2;
    g_host.v_scale = (uint16_t*) hp;
    ck(cudaStreamCreate(&g_s), "stream");
    g_steps = dalloc<int32_t>(TMAX * k::kStepCount);
    g_posi = dalloc<int32_t>(TMAX * IQ);
    std::mt19937 rng(1);
    std::vector<Layer> Ls;
    for (int i = 0; i < L; ++i) Ls.push_back(make_layer(rng));
    ck(cudaDeviceSynchronize(), "setup");
    int bad = 0;
    try {
        if (mode == "all" || mode == "parity") bad = parity(Ls, ctx);
        if (mode == "dx") bad = dx_parity(Ls, ctx);
        if (mode == "f5") bad = f5_parity(Ls, ctx);
        if (mode == "f5t") f5_timing(Ls, ctx, reps);
        if (mode == "all" || mode == "time") timing(Ls, ctx, reps);
    } catch (const std::exception& e) {
        std::printf("exception: %s\n", e.what());
        return 3;
    }
    std::printf("%s\n", bad ? "PARITY FAILED" : "done");
    return bad ? 1 : 0;
}
