// include/strata/kernels/draft_proxy.hpp - the MTP draft head's two-stage scan (--mtp-head proxy), ported from Basalt
// (src/models/qwen4exp/kernels/head_proxy.cuh and q80_proxy.cu, MIT, (c) 2026 Jesús Gulfo, github.com/jesdga95/basalt).
//
// The draft head is the main head's Q8_0 rows (n_in 2560, 2720 bytes a row) of a token subset, read through a row map
// (`map[r]` = the head row, i.e. the token id, of draft row r; -1 = an empty row that never wins).  Every draft step:
//   1. a 2-bit copy of the subset's rows (~800 bytes a row, packed once from the Q8_0 rows) scores each row
//      approximately, keeping the best four of every tile of 32 rows and the tile's softmax partial;
//   2. the rows within kDelta logits of the best approximate score (at most 64) are rescored exactly from their Q8_0 rows
//      in the main head, and the exact best is the draft; its probability is the tiles' softmax with the rescored rows'
//      terms swapped for their exact ones.
// It reads ~30% of the bytes of the plain subset scan (which reads every row's 2720 bytes).  Drafts only: the verify
// window decides every token, so the copy's error changes which tokens are proposed, never the output.
// The number of rows scanned is read from device memory (`n_rows_dev`, <= n_cap), so a captured graph can switch between
// the base rows and the whole subset (the draft head's tiers) without a recapture.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// The 2-bit copy's bytes for up to n_cap rows, and the scan's scratch.
size_t draft_proxy_bytes(int64_t n_cap);
size_t draft_proxy_scratch_bytes(int64_t n_cap);

/// Pack copy rows [row0, row0 + n) from head rows map[r] (a negative entry leaves the copy's row as it was).
/// `planes`: the head is held as bit planes (NativeHead::planes(), q80_vhead.hpp), not as ggml Q8_0 rows.
void draft_proxy_pack(const void* q80_head, bool planes, const int32_t* map, int64_t row0, int64_t n, void* proxy,
                      int64_t n_cap, cudaStream_t stream);

/// The two-stage scan over copy rows [0, *n_rows_dev) for one q8_1 column `x_q8_1` (80 blocks of 36 bytes): `out_id` gets the
/// TOKEN id (map[best]), `out_prob` its probability.  Launches tiles(n_cap) + 1 blocks whatever *n_rows_dev is.
void draft_proxy_top1(const void* proxy, const void* q80_head, bool planes, const void* x_q8_1, const int32_t* n_rows_dev, int64_t n_cap,
                      const int32_t* map, int32_t* out_id, float* out_prob, void* scratch, cudaStream_t stream);
/// The exact pass's kernel for the launches that follow: true = final2_kernel (the default, STRATA_PROXY_FINAL2=0 turns
/// it off), false = the first kernel; the same bits either way (draft_proxy_parity --selftest compares them).
void draft_proxy_final2(bool on);

}  // namespace strata::kernels
