// src/kernels/cuda/gdn_mid.cu - see include/strata/kernels/gdn_mid.hpp.
//
// dec/host round 6: the structure of Basalt's k_mid (mid_chain.cu, MIT, (c) 2026 Jesus Gulfo) - one persistent launch of
// one CTA per SM, the pair / rec / out phases handed over by release-add and acquire-poll counters, weights streamed by
// bulk copies ahead of their phase - with Strata's arithmetic in each phase:
//   pair: dense_tc_tile_block's (pair_terms_xr<8> over the same chunks, the same eight-warp sum), as Basalt's pair;
//   rec:  gdn_rec_window_kernel's expressions (verify_kernels.cu, bitwise gdn_conv_l2_multi + gdn_ab_multi +
//         gdn_step_split_kernel<true>), gdn_out_norm_kernel<true>'s norm and gdn_q8_1_store (Basalt quantizes differently);
//   out:  dense_tc_block<8, 3, 1, 2>'s (the route dense_tc takes for 6144 -> 2560 on 170 SMs), as Basalt's FO == 8 out.
// Built without fast math, like verify_kernels.cu and dense_tc.cu.
#define STRATA_Q81_FASTMATH_DIV 1   // the q8_1 division as native_quantize_q8_1 (see q8_1_finite.hpp, #3948)
#include "strata/kernels/q8_1_finite.hpp"

#define FZ_DEV_ONLY
#include "dense_tc.cu"
#undef FZ_DEV_ONLY

#include "strata/kernels/gdn_mid.hpp"
#include "strata/core/emulate.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {
namespace gm {

constexpr int CT = 512, MAXT = 8, S = 128, RG = 4, RPG = S / RG;
// pair: qkv 2560 -> 10240 and z 2560 -> 6144, 16-row Q8_0 tiles of 43520 bytes, two ring slots
constexpr int NIN = 2560, NQ = 10240, NZ = 6144, NTQ = NQ / 16, NT = NTQ + NZ / 16;
constexpr int SB8 = 272, PROW = (NIN / 256) * SB8, PTILE = 16 * PROW;
// rec: 48 value heads, 16 key heads
constexpr int HV = 48, HK = 16;
// out: 6144 -> 2560, 160 tiles; a row's first och 512-value chunks in shared memory (8, or 4 for T > 3, which leaves room
// for the T staged q8_1 rows), the rest read from L2
constexpr int ONIN = 6144, ONOUT = 2560, OTILES = ONOUT / 16, OROW = (ONIN / 256) * SB8, OCH = 8, OSMB = OCH * 2 * SB8;
constexpr int ORS = one_row_stride(OSMB), OCH4 = 4, ORS4 = one_row_stride(OCH4 * 2 * SB8);
// shared memory: [0, 2 PTILE) the pair ring, then its warp sums; after the pair: [0, 16 ORS) the out tile, the rec at
// OFF_REC, the out sums at OFF_ORED; the mbarriers at OFF_BAR
constexpr int OFF_PRED = 2 * PTILE, OFF_REC = 70144, OFF_ORED = 95232, OFF_BAR = 100352, SMEM = 101376;
constexpr int OFF_OX = 70016, OFF_OX4 = 35200, OXB = (ONIN / 32) * 36;   // the out phase's staged q8_1 rows (och 8 / 4), a row's bytes
static_assert(16 * ORS4 <= OFF_OX4 && OFF_OX4 + MAXT * OXB <= OFF_ORED && 3 * OXB <= OFF_ORED - OFF_OX, "out x layout");
enum { C_PAIR, C_PZ, C_AB, C_REC, C_DONE, NCTR = 8 };
// dec/round3 D2 (STRATA_GDN_MID_DEPTH): the pair ring's tiles copied in NP pieces, at most `depth` pieces in flight per CTA
// (the CTA's tiles in order); a piece's mbarrier at OFF_BAR + 8 (4 + slot * NPMAX + piece), then a slot's armed-piece count
constexpr int NPMAX = 8, OFF_ARM = OFF_BAR + 8 * (4 + 2 * NPMAX);
constexpr int kLayers = 64;
static_assert(16 * ORS <= OFF_REC && OFF_PRED + 2 * 8 * 16 * kMaxCols * 4 <= OFF_ORED, "layout");
static_assert(OFF_ORED + 8 * 16 * kMaxCols * 4 <= OFF_BAR && OFF_ARM + 8 <= SMEM, "layout");

struct RecSm {
    float sq[MAXT][S], sk[MAXT][S], sv[MAXT][S], oc[MAXT][S];
    float red[4][RG][32];
    float part[2][MAXT][RG];
    float sg[MAXT], sb[MAXT];
    float wsum[MAXT][4];
};
static_assert(OFF_REC + (int) sizeof(RecSm) <= OFF_ORED, "rec below the out sums");

struct GdnQ81M { half2 ds; int8_t qs[32]; };

struct MidP {
    const uint8_t *xq, *wq, *wz, *wo;
    const float *x, *state, *conv_w, *dt, *ssm_a, *gamma;
    const uint16_t *wa, *wb;
    float *qkv, *h, *gate, *beta, *z, *y, *bo;
    GdnQ81M* yq;
    int layer;   // the counter set: g_mid_cnt[layer]
    unsigned long long* stamps;   // bench: [CTA][16] globaltimer at the phase points, or null
    int ef;      // evict-first weight copies
    int head;    // dec/front STRATA_GDN_MID_HEAD: before the grid wait (else after it) bit 0 the first tile copy, bit 1 the L2 prefetches, bit 2 the second tile copy
    int split;   // 1: the rec CTAs take RQ qkv tiles each (the last ones), the others the rest of qkv, then z; 0: round robin
    int depth;   // D2: pieces in flight per CTA (0: the two-tile ring, each tile one copy)
    int np;      // D2: pieces per tile (1, 2, 4, 5 or 8)
    float eps;
    int T;
};

}  // namespace gm

__device__ unsigned g_mid_cnt[gm::kLayers][gm::NCTR];   // zero at load; the last CTA of a launch clears its set
__device__ unsigned g_mid_abort;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ uint32_t gm_s32(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void gm_mb_init(uint64_t* b) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(gm_s32(b))); }
__device__ __forceinline__ void gm_mb_expect(uint64_t* b, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(gm_s32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void gm_mb_wait(uint64_t* b, uint32_t par) {
    uint32_t done = 0;
    do {
        asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                     " selp.u32 %0, 1, 0, p;\n}" : "=r"(done) : "r"(gm_s32(b)), "r"(par) : "memory");
    } while (!done);
}
__device__ __forceinline__ void gm_bulk(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 ::"r"(gm_s32(dst)), "l"(src), "r"(bytes), "r"(gm_s32(b)) : "memory");
}
// a weight copy with the L2 evict-first policy (read once per layer, as Basalt's bulk_w2s): the activations and the other
// kernels' data stay in L2.  STRATA_GDN_MID_EF=0 (p.ef) uses the plain copy.
__device__ __forceinline__ void gm_bulk_w(void* dst, const void* src, uint32_t bytes, uint64_t* b, bool ef) {
    if (!ef) { gm_bulk(dst, src, bytes, b); return; }
    uint64_t pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
                 ::"r"(gm_s32(dst)), "l"(src), "r"(bytes), "r"(gm_s32(b)), "l"(pol) : "memory");
}
__device__ __forceinline__ void gm_pf_l2(const void* p, uint32_t bytes) {   // 16-byte aligned, a multiple of 16
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}
__device__ __forceinline__ void gm_pfence() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void gm_cbar() { asm volatile("bar.sync 1, 512;" ::: "memory"); }
__device__ __forceinline__ void gm_barn(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ unsigned gm_ldacq(const unsigned* p) {
    unsigned v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void gm_redrel(unsigned* p) { asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(p) : "memory"); }
__device__ __forceinline__ unsigned long long gm_now() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
// a hand-off wait, bounded: past 2 s (a CTA that never became resident) the sticky abort word is set and every wait falls
// through, so a lost hand-off ends the launch with wrong results and a message instead of a hang
__device__ __forceinline__ void gm_spin(const unsigned* c, unsigned target) {
    if (gm_ldacq(c) >= target) return;
    const unsigned long long t0 = gm_now();
    while (gm_ldacq(c) < target) {
        if (*(volatile unsigned*) &g_mid_abort != 0u) return;
        if (gm_now() - t0 > 2000000000ull) {
            if (atomicExch(&g_mid_abort, 1u) == 0u)
                printf("strata gdn_mid: a hand-off wait timed out (block %d, target %u): results invalid\n", (int) blockIdx.x, target);
            return;
        }
    }
}

// gdn_q8_1_store (verify_kernels.cu) verbatim
__device__ __forceinline__ void gm_q8_1_store(gm::GdnQ81M* __restrict__ xq, size_t idx, float xi) {
#pragma clang fp contract(off)
    float amax = fabsf(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    const float d = q8_1_finite(q81_div(amax, 127.0f));
    const int8_t q = q8_1_quant(xi, d, amax);
    xq[idx / 32].qs[idx % 32] = q;
    if (idx % 32 == 0) xq[idx / 32].ds = q8_1_ds(d, sum);
}
#endif

__global__ void __launch_bounds__(gm::CT, 1) gdn_mid_kernel(const gm::MidP p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using namespace gm;
    extern __shared__ __align__(128) uint8_t smem[];
    uint64_t* bar = reinterpret_cast<uint64_t*>(smem + OFF_BAR);   // [0] [1] the pair slots, [2] the out tile, [3] the out x
    const int tid = threadIdx.x, wp = tid >> 5, lane = tid & 31, G = (int) gridDim.x, b = (int) blockIdx.x;
    const int T = p.T;
    unsigned* c = g_mid_cnt[p.layer];
    const int och = T <= 3 ? OCH : OCH4, osmb = och * 2 * SB8, ors = T <= 3 ? ORS : ORS4, off_ox = T <= 3 ? OFF_OX : OFF_OX4;
#define STAMP(i) do { if (p.stamps != nullptr && tid == 0) p.stamps[(size_t) b * 16 + (i)] = gm_now(); } while (0)
    STAMP(0);
    const int rec0 = G - HV, head = b - rec0;
    const bool rec_role = b >= rec0, out_role = b < OTILES;
    // the CTA's tiles: round robin (split 0), or (split 1, Basalt's prt -3 idea) the rec CTAs only qkv - RQ tiles each from
    // qkv's end - so they reach the recurrence first, and the other CTAs qkv's first part, then all of z
    constexpr int RQ = 4, QS = NTQ - HV * RQ, NO = NT - HV * RQ;
    const int nn = G - HV;
    const int mine = p.split == 0 ? (b < NT ? (NT - 1 - b) / G + 1 : 0) : rec_role ? RQ : (b < NO ? (NO - 1 - b) / nn + 1 : 0);
    auto tile_of = [&](int k) {
        if (p.split == 0) return b + k * G;
        if (rec_role) return QS + head * RQ + k;
        const int i = b + k * nn;
        return i < QS ? i : i + HV * RQ;
    };
    auto tsrc = [&](int k) {
        const int tile = tile_of(k);
        return tile < NTQ ? p.wq + (size_t) tile * PTILE : p.wz + (size_t) (tile - NTQ) * PTILE;
    };
    auto fill = [&](int k) {   // tile k into slot k & 1
        gm_mb_expect(bar + (k & 1), (uint32_t) PTILE);
        gm_bulk_w(smem + (size_t) (k & 1) * PTILE, tsrc(k), (uint32_t) PTILE, bar + (k & 1), p.ef != 0);
    };
    // D2, the bounded stream: piece n = tile n / np, part n % np; piece n is copied once piece n - depth has landed
    const int dep = p.depth, np = p.np, pb = PTILE / np;
    const int p0 = dep > 0 ? (dep < mine * np ? dep : mine * np) : 0;   // the pieces copied up front (no wait)
    auto pbar = [&](int s, int j) { return bar + 4 + s * NPMAX + j; };
    volatile int* armed = reinterpret_cast<volatile int*>(smem + OFF_ARM);   // [slot]: pieces armed in it so far
    auto piece = [&](int n) {
        const int k = n / np, j = n % np, s = k & 1;
        gm_mb_expect(pbar(s, j), (uint32_t) pb);
        gm_bulk_w(smem + (size_t) s * PTILE + (size_t) j * pb, tsrc(k) + (size_t) j * pb, (uint32_t) pb, pbar(s, j), p.ef != 0);
        __threadfence_block();
        armed[s] = (k >> 1) * np + j + 1;
    };
    // piece n has landed; the other slot's piece is waited for only once armed (else its barrier may still be on the phase
    // before, two tiles back)
    auto landed = [&](int n) {
        const int k = n / np, j = n % np, s = k & 1;
        while (armed[s] <= (k >> 1) * np + j) {}
        gm_mb_wait(pbar(s, j), (uint32_t) ((k >> 1) & 1));
    };
    // the weights and the state do not depend on the kernel before: requested before the grid dependency wait
    // dec/front: each part before the grid wait (the head start) or after it (p.head; the tile copies flood the memory
    // system while the HC read before this kernel finishes)
    auto pf = [&]() {
        if (rec_role && tid >= 64 && tid < 64 + S) gm_pf_l2(p.state + ((size_t) (tid - 64) * HV + head) * S, S * 4);
        if (out_role && tid >= 32 && tid < 48)   // the out tile's row tails (read from L2)
            gm_pf_l2(p.wo + (size_t) (b * 16 + tid - 32) * OROW + osmb, (uint32_t) (OROW - osmb));
    };
    if (tid == 0) {
        gm_mb_init(bar); gm_mb_init(bar + 1); gm_mb_init(bar + 2); gm_mb_init(bar + 3);
        if (dep > 0) {
            for (int i = 0; i < 2 * NPMAX; ++i) gm_mb_init(bar + 4 + i);
            armed[0] = armed[1] = 0;
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        if (dep > 0) {
            if (p.head & 1)
                for (int n = 0; n < p0; ++n) piece(n);
        } else {
            for (int k = 0; k < 2 && k < mine; ++k)
                if (p.head & (k == 0 ? 1 : 4)) fill(k);
        }
    }
    if (p.head & 2) pf();
    __syncthreads();
    pdl_wait();
    if (tid == 0) {
        if (dep > 0) {
            if (!(p.head & 1))
                for (int n = 0; n < p0; ++n) piece(n);
        } else {
            for (int k = 0; k < 2 && k < mine; ++k)
                if (!(p.head & (k == 0 ? 1 : 4))) fill(k);
        }
    }
    if (!(p.head & 2)) pf();
    pdl_trigger();   // the next read's first kernel may take the SMs this launch's CTAs leave (it waits for the whole grid)
    STAMP(1);
    STAMP(2);
    // ======== pair: two 8-warp groups take the CTA's tiles alternately (group gi: slot gi)
    {
        const int gi = wp >> 3, lw = wp & 7, ltid = tid & 255, g = lane >> 2, t = lane & 3, hb = lw & 3, n = hb & 1;
        constexpr int nchunk = NIN / 512, nb32 = NIN / 32, WS = 16 * kMaxCols;
        float* gred = reinterpret_cast<float*>(smem + OFF_PRED) + gi * 8 * WS;
        XFrag xs[3];
#pragma unroll
        for (int j = 0; j < 3; ++j)
            if (gi < mine && (lw >> 2) + 2 * j < nchunk) pair_xfrag(xs[j], (lw >> 2) + 2 * j, hb, g, t, p.xq, nb32, T);
        bool qrel = false;
        if (gi >= mine || tile_of(gi) >= NTQ) {   // no qkv tile in this group
            if (ltid == 0) gm_redrel(&c[C_PAIR]);
            qrel = true;
        }
        for (int k = gi; k < mine; k += 2) {
            const int tile = tile_of(k), slot = k & 1;
            const bool isq = tile < NTQ;
            float* y = isq ? p.qkv : p.z;
            const int no = isq ? NQ : NZ, row0 = (isq ? tile : tile - NTQ) * 16;
            if (dep > 0) {
                // the group's own tile: its pieces not copied up front, each after piece n - depth (the slot is free: the
                // group read tile k - 2 out of it before its last barrier)
                if (ltid == 0) {
                    const int n0 = p0 > k * np ? p0 : k * np;
                    if (n0 < (k + 1) * np) gm_pfence();
                    for (int n = n0; n < (k + 1) * np; ++n) {
                        if (n >= dep) landed(n - dep);
                        piece(n);
                    }
                }
                for (int j = 0; j < np; ++j) gm_mb_wait(pbar(slot, j), (uint32_t) ((k >> 1) & 1));
            } else {
                gm_mb_wait(bar + slot, (uint32_t) ((k >> 1) & 1));
            }
            const uint8_t* sl = smem + (size_t) slot * PTILE;
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            pair_terms_xr<8>(acc, sl + g * PROW, sl + (g + 8) * PROW, lw, hb, n, t, nchunk, xs, nullptr);
            float* rw = gred + lw * WS;
            rw[g * kMaxCols + 2 * t] = acc[0];
            rw[g * kMaxCols + 2 * t + 1] = acc[1];
            rw[(g + 8) * kMaxCols + 2 * t] = acc[2];
            rw[(g + 8) * kMaxCols + 2 * t + 1] = acc[3];
            gm_barn(2 + gi, 256);   // the slot is read and the warp sums are in
            if (dep == 0 && ltid == 0 && k + 2 < mine) {
                gm_pfence();
                fill(k + 2);
            }
            if (ltid < 16 * kMaxCols) {
                const int row = ltid / kMaxCols, col = ltid % kMaxCols;
                if (col < T) {
                    const float* rr = gred + row * kMaxCols + col;
                    const float v = ((rr[0] + rr[4 * WS]) + (rr[WS] + rr[5 * WS])) +
                                    ((rr[2 * WS] + rr[6 * WS]) + (rr[3 * WS] + rr[7 * WS]));
                    y[(size_t) col * no + row0 + row] = v;
                }
            }
            gm_barn(2 + gi, 256);   // the sums are read
            if (!qrel && isq && (k + 2 >= mine || tile_of(k + 2) >= NTQ)) {   // this group's last qkv tile
                if (ltid == 0) gm_redrel(&c[C_PAIR]);
                if (p.stamps != nullptr && ltid == 0) p.stamps[(size_t) b * 16 + 11 + gi] = gm_now();
                qrel = true;
            }
        }
        if (ltid == 0) gm_redrel(&c[C_PZ]);
    }
    gm_cbar();   // the ring and the pair sums are free
    STAMP(3);
    if (out_role && tid == 32) {   // the out tile's first OCH chunks of its 16 rows, while the recurrence runs
        gm_pfence();
        gm_mb_expect(bar + 2, (uint32_t) (16 * osmb));
        for (int r = 0; r < 16; ++r) gm_bulk_w(smem + r * ors, p.wo + (size_t) (b * 16 + r) * OROW, (uint32_t) osmb, bar + 2, p.ef != 0);
    }
    // ======== rec: value head `head` (gdn_rec_window_kernel's code; rank = its blockIdx.y, a named barrier per rank)
    if (rec_role) {
        RecSm& R = *reinterpret_cast<RecSm*>(smem + OFF_REC);
        const int rank = tid >> 7, ltid = tid & 127, rg = ltid >> 5, lc = lane, col = rank * 32 + lc;
        const int qh = head % HK, qk = S * HK, value_dim = S * HV, C = NQ;
        const float* hist = p.state + (size_t) S * HV * S;
        const float* w = p.conv_w;
        // the head's gates (2T dots, warp r: token r >> 1, beta when r is odd; gdn_ab_body's chain, every load first),
        // computed while the other CTAs finish qkv
        if (wp < 2 * T) {
            const int hd = head, tt = wp >> 1, ib = wp & 1;
            const float* xt = p.x + (size_t) tt * NIN;
            const uint4* w4 = reinterpret_cast<const uint4*>((ib ? p.wb : p.wa) + (size_t) hd * NIN);
            constexpr int NJ = NIN / 256;
            uint4 wvs[NJ];
            float4 xas[NJ], xbs[NJ];
#pragma unroll
            for (int q = 0; q < NJ; ++q) {
                const int jj = lane + 32 * q;
                wvs[q] = __ldg(w4 + jj);
                xas[q] = __ldg(reinterpret_cast<const float4*>(xt + jj * 8));
                xbs[q] = __ldg(reinterpret_cast<const float4*>(xt + jj * 8 + 4));
            }
            float a = 0.0f;
#pragma unroll
            for (int q = 0; q < NJ; ++q) {
                const uint4 wv = wvs[q];
                const float w0 = __uint_as_float(wv.x << 16), w1 = __uint_as_float(wv.x & 0xffff0000u);
                const float w2 = __uint_as_float(wv.y << 16), w3 = __uint_as_float(wv.y & 0xffff0000u);
                const float w4f = __uint_as_float(wv.z << 16), w5 = __uint_as_float(wv.z & 0xffff0000u);
                const float w6 = __uint_as_float(wv.w << 16), w7 = __uint_as_float(wv.w & 0xffff0000u);
                const float4 xa = xas[q], xb = xbs[q];
                a = fmaf(w0, xa.x, a); a = fmaf(w1, xa.y, a);
                a = fmaf(w2, xa.z, a); a = fmaf(w3, xa.w, a);
                a = fmaf(w4f, xb.x, a); a = fmaf(w5, xb.y, a);
                a = fmaf(w6, xb.z, a); a = fmaf(w7, xb.w, a);
            }
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            if (lane == 0) {
                if (ib) {
                    const float bv = 1.0f / (1.0f + __expf(-a));
                    R.sb[tt] = bv;
                    p.beta[(size_t) tt * HV + hd] = bv;
                } else {
                    const float v = a + p.dt[hd];
                    const float sp = v > 20.0f ? v : log1pf(__expf(v));
                    const float gv = sp * p.ssm_a[hd];
                    R.sg[tt] = gv;
                    p.gate[(size_t) tt * HV + hd] = gv;
                }
            }
        }
        if (tid == 0) gm_spin(&c[C_PAIR], 2u * (unsigned) G);
        gm_cbar();
        STAMP(4);
        // conv + SiLU: rank 0 the head's q channels, rank 1 its k channels; every rank's row group 0 its 32 v columns
        for (int t = 0; t < T; ++t) {
            if (rank < 2) {
                const int pq = rank;
                const int cc = (pq ? qk : 0) + qh * S + ltid;
                float win[3];
#pragma unroll
                for (int j = 0; j < 3; ++j) {
                    const int src = t + j;
                    win[j] = src < 3 ? hist[cc * 3 + src] : __ldcg(p.qkv + (size_t) (src - 3) * C + cc);
                }
                const float v0 = win[0], v1 = win[1], v2 = win[2], xx = __ldcg(p.qkv + (size_t) t * C + cc);
                float sum = v0 * w[cc * 4] + v1 * w[cc * 4 + 1] + v2 * w[cc * 4 + 2] + xx * w[cc * 4 + 3];
                float yy = sum / (1.0f + __expf(-sum));
                (pq ? R.sk : R.sq)[t][ltid] = yy;
                float sq2 = yy * yy;
                for (int o = 16; o > 0; o >>= 1) sq2 += __shfl_xor_sync(0xffffffffu, sq2, o);
                if (lc == 0) R.part[pq][t][rg] = sq2;
            }
            if (rg == 0) {
                const int cc = 2 * qk + head * S + col;
                float win[3];
#pragma unroll
                for (int j = 0; j < 3; ++j) {
                    const int src = t + j;
                    win[j] = src < 3 ? hist[cc * 3 + src] : __ldcg(p.qkv + (size_t) (src - 3) * C + cc);
                }
                const float v0 = win[0], v1 = win[1], v2 = win[2], xx = __ldcg(p.qkv + (size_t) t * C + cc);
                float sum = v0 * w[cc * 4] + v1 * w[cc * 4 + 1] + v2 * w[cc * 4 + 2] + xx * w[cc * 4 + 3];
                float yy = sum / (1.0f + __expf(-sum));
                R.sv[t][col] = yy;
                p.h[(size_t) t * C + cc] = yy;
            }
        }
        gm_cbar();
        if (rank < 2) {
            const int pq = rank;
            for (int t = 0; t < T; ++t) {
                const float ss = R.part[pq][t][0] + R.part[pq][t][1] + R.part[pq][t][2] + R.part[pq][t][3];
                float yy = (pq ? R.sk : R.sq)[t][ltid];
                yy *= rsqrtf(ss + p.eps);
                (pq ? R.sk : R.sq)[t][ltid] = yy;
                p.h[(size_t) t * C + (pq ? qk : 0) + qh * S + ltid] = yy;
            }
        }
        // the recurrence: gdn_step_split_kernel<true>'s
        float s[RPG];
        const float* base = p.state + ((size_t) (rg * RPG) * HV + head) * S + col;
        const size_t row_stride = (size_t) HV * S;
#pragma unroll
        for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
        gm_cbar();   // the normed q / k and the gates
        const int rb = 8 + rank;
        for (int t = 0; t < T; ++t) {
            gm_barn(rb, 128);
            const float g = __expf(R.sg[t]);
            float kv = 0.0f;
#pragma unroll
            for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], R.sk[t][rg * RPG + r], kv);
            R.red[rank][rg][lc] = kv;
            gm_barn(rb, 128);
            const float kv_col = R.red[rank][0][lc] + R.red[rank][1][lc] + R.red[rank][2][lc] + R.red[rank][3][lc];
            const float delta = (R.sv[t][col] - g * kv_col) * R.sb[t];
            float o = 0.0f;
#pragma unroll
            for (int r = 0; r < RPG; ++r) {
                s[r] = fmaf(g, s[r], R.sk[t][rg * RPG + r] * delta);
                o = fmaf(s[r], R.sq[t][rg * RPG + r], o);
            }
            gm_barn(rb, 128);
            R.red[rank][rg][lc] = o;
            gm_barn(rb, 128);
            if (rg == 0)
                R.oc[t][col] = (R.red[rank][0][lc] + R.red[rank][1][lc] + R.red[rank][2][lc] + R.red[rank][3][lc]) * rsqrtf((float) S);
        }
        // the gated norm and its q8_1: gdn_out_norm_kernel<true>'s (the rank's row group 0 = its warp of 32 columns)
        STAMP(5);
        if (tid == 0) gm_spin(&c[C_PZ], 2u * (unsigned) G);
        gm_cbar();
        STAMP(6);
        for (int t = 0; t < T; ++t) {
            float oc = 0.0f, sq_part = 0.0f;
            if (rg == 0) {
                oc = R.oc[t][col];
                sq_part = oc * oc;
            }
            for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
            if (rg == 0 && lc == 0) R.wsum[t][rank] = sq_part;
        }
        gm_cbar();
        if (rg == 0) {
            for (int t = 0; t < T; ++t) {
                const float oc = R.oc[t][col];
                const float ss = R.wsum[t][0] + R.wsum[t][1] + R.wsum[t][2] + R.wsum[t][3];
                const float scale = rsqrtf(ss / (float) S + p.eps);
                const float zz = __ldcg(p.z + (size_t) t * value_dim + head * S + col);
                const float yv = oc * scale * p.gamma[col] * (1.0f / (1.0f + __expf(-zz)));
                p.y[(size_t) t * value_dim + head * S + col] = yv;
                gm_q8_1_store(p.yq, (size_t) t * value_dim + head * S + col, yv);
            }
        }
        gm_cbar();
        if (tid == 0) gm_redrel(&c[C_REC]);
        STAMP(7);
    }
    // ======== out: tile b, dense_tc_block<8, 3, 1, 2>'s arithmetic (warp 4 ks + hb: half super-block hb of the 512-value
    // chunks 2 j + ks in order; the row: (((hb0 + hb1) + hb2) + hb3) of ks 0, plus the same of ks 1)
    if (out_role) {
        constexpr int nb32 = ONIN / 32, WS = 16 * kMaxCols;
        if (tid == 0) gm_spin(&c[C_REC], (unsigned) HV);
        gm_cbar();
        STAMP(8);
        // the q8_1 rows staged in shared memory where they fit (T <= 3: above the out tile); else read through L2
        const bool oxs = true;   // every T fits (och 4 above 3 rows)
        if (oxs && tid == 0) {
            asm volatile("fence.proxy.async.global;" ::: "memory");   // the rec CTAs' generic writes, read by the async proxy
            gm_mb_expect(bar + 3, (uint32_t) (T * OXB));
            gm_bulk(smem + off_ox, p.yq, (uint32_t) (T * OXB), bar + 3);
        }
        gm_mb_wait(bar + 2, 0u);
        if (oxs) gm_mb_wait(bar + 3, 0u);
        STAMP(9);
        float* red = reinterpret_cast<float*>(smem + OFF_ORED);
        const int g = lane >> 2, t = lane & 3;
        if (wp < 8) {
            const int hb = wp & 3, ks = wp >> 2, sbi = hb >> 1, n = hb & 1;
            const uint8_t* rowA = smem + g * ors + sbi * SB8;
            const uint8_t* rowB = smem + (g + 8) * ors + sbi * SB8;
            const uint8_t* gA = p.wo + (size_t) (b * 16 + g) * OROW + sbi * SB8;
            const uint8_t* gB = p.wo + (size_t) (b * 16 + g + 8) * OROW + sbi * SB8;
            const uint8_t* yq = oxs ? smem + off_ox : reinterpret_cast<const uint8_t*>(p.yq);
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            auto ld_u32 = [&](const void* q) { return oxs ? *reinterpret_cast<const unsigned int*>(q) : __ldcg(reinterpret_cast<const unsigned int*>(q)); };
            auto ld_h2 = [&](const void* q) { return oxs ? *reinterpret_cast<const half2*>(q) : __ldcg(reinterpret_cast<const half2*>(q)); };
            auto load_x = [&](XFrag& xf, int cch) {
#pragma unroll
                for (int k4 = 0; k4 < 4; ++k4) {
                    const int k = 16 * cch + 4 * hb + k4;
                    xf.b0[k4] = xf.b1[k4] = 0;
                    if (g < T) {
                        const uint8_t* xb = yq + ((size_t) g * nb32 + k) * kQ81;
                        xf.b0[k4] = ld_u32(xb + 4 + 4 * t);
                        xf.b1[k4] = ld_u32(xb + 20 + 4 * t);
                    }
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const int cl = 2 * t + e;
                        xf.d8[k4][e] = cl < T ? __low2float(ld_h2(yq + ((size_t) cl * nb32 + k) * kQ81)) : 0.0f;
                    }
                }
            };
            XFrag xf;
            load_x(xf, ks);
#pragma unroll 1
            for (int j = 0; j < ONIN / 1024; ++j) {
                const int cch = 2 * j + ks;
                XFrag xn;
                if (j + 1 < ONIN / 1024) load_x(xn, cch + 2);
                const uint8_t* const pp[2] = {(cch < och ? rowA : gA) + cch * 2 * SB8, (cch < och ? rowB : gB) + cch * 2 * SB8};
                half_sb_terms<8>(acc, pp, n, t, xf, 0u, 0u, 0u, 0u);
                xf = xn;
            }
            float* rw = red + wp * WS;
            rw[g * kMaxCols + 2 * t] = acc[0];
            rw[g * kMaxCols + 2 * t + 1] = acc[1];
            rw[(g + 8) * kMaxCols + 2 * t] = acc[2];
            rw[(g + 8) * kMaxCols + 2 * t + 1] = acc[3];
        }
        gm_cbar();
        if (tid < 16 * kMaxCols) {
            const int row = tid / kMaxCols, col = tid % kMaxCols;
            if (col < T) {
                const float* rr = red + row * kMaxCols + col;
                const float v0 = ((rr[0] + rr[WS]) + rr[2 * WS]) + rr[3 * WS];
                const float v1 = ((rr[4 * WS] + rr[5 * WS]) + rr[6 * WS]) + rr[7 * WS];
                const float v = v0 + v1;
                p.bo[(size_t) col * ONOUT + b * 16 + row] = v;
            }
        }
        STAMP(10);
    }
#undef STAMP
    // the last CTA out clears the layer's counters (every wait has passed by then; the next launch of this set is a later
    // stream operation)
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        if (atomicAdd(&c[C_DONE], 1u) == (unsigned) G - 1) {
            for (int k = 0; k < NCTR; ++k) atomicExch(&c[k], 0u);
            __threadfence();
        }
    }
#elif defined(__CUDA_ARCH__)
    __trap();
#endif
}

int mid_grid() {
    static const int g = [] {
        int dev = 0, sms = 0, optin = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        sms = strata::sms_of(sms);   // STRATA_EMULATE_SMS (tests only): its grid waits for every CTA
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        if (major < 9 || optin < gm::SMEM || sms < gm::OTILES) return 0;
        if (cudaFuncSetAttribute(gdn_mid_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, gm::SMEM) != cudaSuccess) return 0;
        int occ = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, gdn_mid_kernel, gm::CT, gm::SMEM);
        if (occ < 1) return 0;
        return sms < 170 ? sms : 170;
    }();
    return g;
}

}  // namespace

bool gdn_mid_supported(int type_q, int type_z, int type_o, int n_embd, int h_k, int h_v, int n_tok, const void* wq,
                       const void* wz, const void* wo, const void* state) noexcept {
    return type_q == 8 && type_z == 8 && type_o == 8 && n_embd == gm::NIN && h_k == gm::HK && h_v == gm::HV && n_tok >= 1 &&
           n_tok <= gm::MAXT && ((uintptr_t) wq & 15) == 0 && ((uintptr_t) wz & 15) == 0 && ((uintptr_t) wo & 15) == 0 &&
           ((uintptr_t) state & 15) == 0 && mid_grid() > 0;
}

void gdn_mid(const GdnMidArgs& a, void* stream) {
    const int G = mid_grid();
    if (G <= 0 || a.layer < 0 || a.layer >= gm::kLayers || a.T < 1 || a.T > gm::MAXT) {
        std::fprintf(stderr, "strata gdn_mid: unsupported arguments or device\n");
        std::exit(1);
    }
    static std::atomic<bool> said{false};
    if (!said.exchange(true)) std::fprintf(stderr, "strata: STRATA_GDN_MID=1: GDN layers on gdn_mid (%d CTAs)\n", G);
    gm::MidP p{};
    p.xq = static_cast<const uint8_t*>(a.xq);
    p.wq = static_cast<const uint8_t*>(a.wq);
    p.wz = static_cast<const uint8_t*>(a.wz);
    p.wo = static_cast<const uint8_t*>(a.wo);
    p.x = a.x; p.state = a.state; p.conv_w = a.conv_w; p.dt = a.dt; p.ssm_a = a.ssm_a; p.gamma = a.gamma;
    p.wa = a.wa; p.wb = a.wb;
    p.qkv = a.qkv; p.h = a.h; p.gate = a.gate; p.beta = a.beta; p.z = a.z; p.y = a.y; p.bo = a.bo;
    p.yq = static_cast<gm::GdnQ81M*>(a.yq);
    p.layer = a.layer;
    p.stamps = a.stamps;
    static const int split = [] { const char* v = std::getenv("STRATA_GDN_MID_SPLIT"); return v != nullptr ? std::atoi(v) : 1; }();
    p.split = split;
    // dec/round3 D2: STRATA_GDN_MID_DEPTH pieces in flight per CTA (0: the two-tile ring), STRATA_GDN_MID_PIECES pieces a tile.
    // Default 1 x 10880 B (10.6 KiB in flight; gpu-probe lat1: 8-16 KiB/SM is full DRAM rate, more only slows every L2 hit):
    // bitwise, cold-L2 gdn_parity T 1/3/8 47.1/49.5/59.4 -> 46.3/47.3/55.3 us (bench-d3 NOTES-d3.md D2)
    static const int pieces = [] {
        const char* v = std::getenv("STRATA_GDN_MID_PIECES");
        const int n = v != nullptr ? std::atoi(v) : 4;
        return n == 1 || n == 2 || n == 4 || n == 5 || n == 8 ? n : 4;
    }();
    static const int depth = [] {
        const char* v = std::getenv("STRATA_GDN_MID_DEPTH");
        const int d = v != nullptr ? std::atoi(v) : 1;
        return d < 0 ? 0 : d > 2 * pieces ? 2 * pieces : d;
    }();
    p.depth = depth;
    p.np = pieces;
    static const int ef = [] { const char* v = std::getenv("STRATA_GDN_MID_EF"); return v != nullptr ? std::atoi(v) : 1; }();
    p.ef = ef;
    static const int head = [] { const char* v = std::getenv("STRATA_GDN_MID_HEAD"); return v != nullptr ? std::atoi(v) : 7; }();
    p.head = head;
    p.eps = a.eps;
    p.T = a.T;
    launch_pdl(gdn_mid_kernel, dim3((unsigned) G), dim3(gm::CT), (size_t) gm::SMEM, (cudaStream_t) stream, p);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "strata gdn_mid: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
