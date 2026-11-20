// include/strata/kernels/mtp_route.hpp - the MTP draft layer's one-token router and its expert groups in one launch
// (STRATA_MTP_ROUTE_GROUP, mtp.cpp): native_router_top10's route kernel (llama.cpp's topk-moe, MIT) followed in the same
// block by moe_group_resident's grouping (every expert resident at base + id * blob), bit for bit the two kernels.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace strata::kernels {

/// One token's 512 logits -> ids[10], weights[10] (native_router_top10), then moe_group_resident(ids, 10, 10, base, blob,
/// grp_ptr, grp_start, counts, ent_dst, ent_tok).
void mtp_route_group(const float* logits, int32_t* ids, float* weights, const uint8_t* base, int64_t blob,
                     unsigned long long* grp_ptr, int32_t* grp_start, int32_t* counts, int32_t* ent_dst, int32_t* ent_tok,
                     cudaStream_t stream);

}  // namespace strata::kernels
