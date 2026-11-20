// include/strata/kernels/mtp_cond.hpp - the MTP draft chain as one CUDA graph with conditional nodes (STRATA_MTP_COND),
// after Basalt (src/models/qwen4exp/decode/mtp/conditional.cpp and verify_kernels.cu mtp_draft_cond, MIT, (c) 2026
// Jesus Gulfo, github.com/jesdga95/basalt): [round, keep(0)] -> IF h1 [step 1, keep(1)] -> IF h2 [...] ...
// keep(j) applies the --mtp-cum rule (draft_cum.hpp) to the chain's probabilities on the device, in the host's order,
// so the device stops where the host would: the drafts are the same kernels' outputs, only the host round trip between
// the steps is gone.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace strata::kernels {

/// Whether this build and device can run conditional graph nodes (CUDA 12.4+, not HIP).
bool mtp_cond_supported();

/// keep(j): hist[j] = probs[*row_dev]; prod_{i <= j} acc(hist[i]) >= th (th = the float in *th_bits); sets the handle
/// h (an unsigned long long cudaGraphConditionalHandle) to 1 when the chain goes on, 0 when it stops.
bool mtp_cond_keep(unsigned long long h, const float* probs, const int32_t* row_dev, const int32_t* th_bits, float* hist,
                   int j, cudaStream_t stream);

/// mtp_select (verify_kernels.hpp) over several blocks: R_dst = R_src row *row_dev (R_stride floats, a multiple of 4,
/// 16-byte aligned), *tok_dst = ids[row], out[j] / out_p[j] = the token / probs[row] (mapped); the same values.
void mtp_select_wide(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                     int32_t* tok_dst, int32_t* out, int j, cudaStream_t stream, const float* probs, float* out_p);

}  // namespace strata::kernels
