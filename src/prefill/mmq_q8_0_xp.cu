// src/prefill/mmq_q8_0_xp.cu - Q8_0 experts (the GPTQ + Q8_0-down pack's down projections) through mmq_vendor's
// mmq.cuh with STRATA_MMQ_XPTR: each expert's weights read where they lie (its blob in the staging ring or the cache),
// not gathered into a group buffer first.  The K tail past a row's end (K 640 in 256-value iterations) loads as zeros
// instead of being read: the same sums as llama.cpp's own Q8_0 instance, which the gathered products keep.
#include "common.cuh"

#define STRATA_MMQ_XPTR 1
#define STRATA_MMQ_YPIPE 1
#define mul_mat_q_switch_J strata_xp_mul_mat_q_switch_J
#define mul_mat_q_case strata_xp_mul_mat_q_case
#include "mmq_vendor/mmq.cuh"

DECL_MMQ_CASE(GGML_TYPE_Q8_0);

namespace strata::prefill::mmq {
void run_q8_0_xp(ggml_backend_cuda_context& ctx, const mmq_args& a, cudaStream_t s, const void* const* w, int n) {
    strata_mmq_xptr xp{};
    for (int i = 0; i < n; ++i) xp.p[i] = (const char*) w[i];
    strata_mmq_xp_host = xp;
    strata_xp_mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, a, s);
    strata_mmq_xp_host = strata_mmq_xptr{};
}
}  // namespace strata::prefill::mmq
