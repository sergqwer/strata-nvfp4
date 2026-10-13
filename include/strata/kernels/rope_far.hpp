// include/strata/kernels/rope_far.hpp - STRATA_ROPE_FAR (experimental): the attention sees far keys at compressed
// distances (Leaky ReRoPE): a key d > W cells behind the query scores as if it were W + (d - W) / k behind, so every
// distance of a context past the trained one stays inside it.  Near keys (d <= W) keep their exact rotation.
//
// K is cached post-RoPE at its own position, so the remap is one more rotation of the key's rotary pairs by
// w_i * (1 - 1/k) * (d - W) - per (query, key), applied where the attention reads the key (qsa_decode_attn.cu).  The
// angles come from a float64-built table [m = d - W][32] of cos/sin.  Needs the key in its own basis: fp16 K/V or
// int8 without the Hadamard rotation (STRATA_KV_ROT=0).  The indexer keeps the exact rotation (a prototype).
#pragma once

#include "strata/kernels/rope_scaling.hpp"

#include <cstdint>

namespace strata::kernels {

/// What the attention kernels take (a CUDA graph bakes these: the table and control word stay put, their contents
/// change between requests).
struct RopeFarDev {
    const float* cs = nullptr;    ///< [max_m][32][2] cos, sin
    const int32_t* ctl = nullptr; ///< ctl[0] = W while a compression is active, else INT32_MAX
    int max_m = 0;
};

/// The process's far rotation: none until rope_far_init.
const RopeFarDev& rope_far();
bool rope_far_on();
/// Allocates the table for `max_cells` distances (VRAM) and the control word; off until rope_far_update.
bool rope_far_init(int max_cells);
/// Fills the table for window `w` and compression `k` (k <= 1: off) on the angle rates of `s` (n_rot 64): every
/// pair's excess distance / k, or with `yarn` yarn's per-pair rates of factor k (STRATA_ROPE_FAR_MODE=yarn).
/// Synchronizes the device.
bool rope_far_update(int w, double k, const RopeScaling& s, bool yarn = false);

}  // namespace strata::kernels
