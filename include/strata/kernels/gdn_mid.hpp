// include/strata/kernels/gdn_mid.hpp - dec/host round 6 (STRATA_GDN_MID=1): a verify window's GDN layer middle in one
// persistent launch, after Basalt's k_mid (src/models/qwen4exp/kernels/mid_chain.cu, MIT, (c) 2026 Jesus Gulfo).
//
// One CTA per SM (512 threads).  Three phases hand over through per-layer global counters:
//   pair: the qkv and z projections (Q8_0) as dense_tc_tile computes them, 16-row tiles streamed into a two-slot ring by
//         bulk copies; the alpha / beta gates (gdn_ab_multi's chain) on the first warps while the first tiles land;
//   rec:  one CTA per value head - conv + SiLU + q/k L2 (gdn_conv_l2_multi), the recurrence (gdn_step_split_kernel), the
//         gated norm and its q8_1 (gdn_out_norm_kernel<true>);
//   out:  the output projection (Q8_0, 6144 -> 2560) as dense_tc_kernel<8, 3, 1, 2> computes it, one 16-row tile per CTA,
//         the tile's first 8 chunks prefetched into shared memory during the recurrence.
// Every output (qkv, z, h, gate, beta, y, the q8_1 of y, bo) is the chain's, bit for bit (gdn_parity section 8).  The
// state and the conv history are read, never written (the commit stays the caller's).  Each row's result does not
// depend on the window's row count.
#pragma once

#include <cstdint>

namespace strata::kernels {

struct GdnMidArgs {
    const void* xq = nullptr;          // the layer input's q8_1 rows [T][n_embd / 32]
    const float* x = nullptr;          // the layer input [T][n_embd] (the gates)
    const void* wq = nullptr;          // attn_qkv (Q8_0)
    const void* wz = nullptr;          // attn_gate (Q8_0)
    const void* wo = nullptr;          // ssm_out (Q8_0)
    const float* state = nullptr;      // [S][h_v][S] state, then the conv history [C][3]
    const float* conv_w = nullptr;     // [C][4]
    const uint16_t* wa = nullptr;      // ssm_alpha (bf16)
    const uint16_t* wb = nullptr;      // ssm_beta (bf16)
    const float* dt = nullptr;
    const float* ssm_a = nullptr;
    const float* gamma = nullptr;      // ssm_norm
    float eps = 0.0f;
    float* qkv = nullptr;              // [T][C]
    float* h = nullptr;                // [T][C] conv outputs (the commit's)
    float* gate = nullptr;             // [T][h_v]
    float* beta = nullptr;             // [T][h_v]
    float* z = nullptr;                // [T][h_v * S]
    float* y = nullptr;                // [T][h_v * S] the normed output
    void* yq = nullptr;                // its q8_1 (may be xq: written only after every tile has read xq)
    float* bo = nullptr;               // [T][n_embd]
    int T = 0;
    int layer = 0;                     // the counter set (0..63)
    unsigned long long* stamps = nullptr;   // bench: [170][16] globaltimer phase points per CTA (gdn_mid.cu STAMP)
};

/// The shapes and types this kernel is built for (the artifact's GDN layer: 2560, 16 / 48 heads of 128, Q8_0), T 1..8,
/// a device with at least 160 SMs and 99 KB of opt-in shared memory, 16-byte aligned weights and state.
bool gdn_mid_supported(int type_q, int type_z, int type_o, int n_embd, int h_k, int h_v, int n_tok, const void* wq,
                       const void* wz, const void* wo, const void* state) noexcept;
void gdn_mid(const GdnMidArgs& a, void* stream);

}  // namespace strata::kernels
