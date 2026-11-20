// include/strata/kernels/q80_vhead.hpp - a Q8_0 output head (n_in 2560) held as bit planes, its exact kernel and the
// two-stage verify head.  Ported from Basalt (src/models/qwen4exp/kernels/q80_vhead.cu, MIT), without its two-card split.
//
// The planes: a row keeps its 2720 bytes, laid out as the 80 block scales (fp16), then for every code u = q + 128 its
// bits 2..5 (a nibble plane), its bits 6..7 and its bits 0..1 (two 2-bit planes).  The exact kernel reads the planes
// back into the Q8_0 codes and runs native_mmvq's Q8_0 arithmetic (thread-to-block map, per-thread order, cross-warp
// order, warp tree): every logit is the same bits as native_mmvq on the Q8_0 rows (q80p_parity checks it).
//
// The two-stage head reads the scales and the top six bits of every code.  With the low two bits l in [0, 3] unknown,
// sum q y = 4 sum (u >> 2) y + sum l y - 128 sum y lies within 1.5 sum |y| of its value at l = 1.5, so every row's
// logit is inside [c - w, c + w] (w widened by 1/16 plus 1e-6 for the float rounding of both sides, which is below
// 1e-3 w).  A lower bound of the K-th largest c - w is the column's threshold; the rows whose c + w reaches it are
// scored exactly.  Every row of the exact top K (ties included) is among them, so the K largest entries of the column,
// their order and their values are the exact head's; every other entry holds its upper bound, strictly below all K.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// The planes' kernels run here: CUDA sm_80+ and n_in 2560 (80 blocks), 1..8 columns a launch.
bool q80p_supported(int n_in, int ncols);
/// Rows [0, n) of a Q8_0 head (2720 bytes each) rewritten as planes, in place.
void q80p_pack(void* rows, int64_t n, void* stream);
/// Planes back to Q8_0 rows, in place.
void q80p_unpack(void* rows, int64_t n, void* stream);
/// Every row exact: y[j * n_out + r] = row r . column j, the bits of native_mmvq on the Q8_0 rows (any ncols >= 1).
void q80p_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);

/// The two-stage head over 1..8 columns.  `k` (device int): the exact list's length, 1..64; 0 runs every row exactly.
/// `scratch`: q80p_vhead_scratch_bytes(), zeroed once before first use (each call leaves it zeroed but its counters).
size_t q80p_vhead_scratch_bytes();
void q80p_vhead(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, const int* k,
                void* scratch, void* stream);
/// The widest exact list the two-stage head keeps.
constexpr int kQ80pKeepMax = 64;
/// The scratch's running counters (a synchronous copy): screened calls, and rows scored exactly in them (a row counts
/// once however many of its columns it was scored for).
bool q80p_vhead_stats(const void* scratch, uint64_t& calls, uint64_t& exact_rows);

}  // namespace strata::kernels
