// src/kernels/cuda/mtp_route.cu - see include/strata/kernels/mtp_route.hpp.  Built with --use_fast_math like
// native_router.cu, whose route kernel's body is repeated here unchanged (Adapted from topk-moe.cu/common.cuh in llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d, MIT License, Copyright (c) 2023-2026 The ggml authors; the full notice is in
// native_router.cu).  The grouping is s2_expert_grouped.cu's group_resident_kernel for n = k = 10, in warp 0.
#include "strata/kernels/mtp_route.hpp"

#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {
__device__ __forceinline__ float rg_warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value += __shfl_xor_sync(0xffffffffu, value, mask, 32);
    return value;
}
__device__ __forceinline__ float rg_warp_max(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask, 32));
    return value;
}
__launch_bounds__(256, 1)
__global__ void route_group_kernel(const float* __restrict__ logits, int32_t* __restrict__ ids,
                                   float* __restrict__ weights, const uint8_t* base, long long blob,
                                   unsigned long long* __restrict__ grp_ptr, int32_t* __restrict__ grp_start,
                                   int32_t* __restrict__ counts, int32_t* __restrict__ ent_dst, int32_t* __restrict__ ent_tok) {
    __shared__ int e_s[32], first_s[32], size_s[32], gidx_s[32], gstart_s[33];
    // ---- native_router.cu route (one token)
    if (threadIdx.y != 0) return;
    const int lane = threadIdx.x;
    float values[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + i * 32];
    __syncwarp();
    float maximum = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) maximum = max(maximum, values[i]);
    maximum = rg_warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float reciprocal = 1.0f / rg_warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] *= reciprocal;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    int my_id = -1;   // (the grouping: lane r keeps the rank-r expert)
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask, 32);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask, 32);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            ids[rank] = expert;
            selected_sum += best;
        }
        if (rank == lane) { selected = best; my_id = expert; }
    }
    selected_sum = max(rg_warp_sum(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = 1.0f / selected_sum;
    if (lane < 10) weights[lane] = selected * inverse_selected_sum;
    // ---- group_resident_kernel (n = 10, k_per_tok = 10): groups in first-appearance order, entries in routing order
    constexpr int n = 10, k_per_tok = 10;
    const int i = lane;
    const int e = i < n ? my_id : -1;
    e_s[i] = e;
    __syncwarp();
    int first = i, rk = 0, size = 0;
    if (i < n) {
        for (int j = 0; j < i; ++j)
            if (e_s[j] == e) { if (first == i) first = j; ++rk; }
        if (first == i)
            for (int j = i; j < n; ++j) size += e_s[j] == e;
    }
    first_s[i] = first;
    size_s[i] = (i < n && first == i) ? size : 0;
    __syncwarp();
    if (i == 0) {
        int gi = 0, acc = 0;
        for (int j = 0; j < n; ++j)
            if (first_s[j] == j) {
                gidx_s[j] = gi;
                gstart_s[gi] = acc;
                grp_ptr[gi] = (unsigned long long) (base + (size_t) e_s[j] * (size_t) blob);
                grp_start[gi] = acc;
                acc += size_s[j];
                ++gi;
            }
        grp_start[gi] = acc;
        counts[0] = gi;
        counts[1] = acc;
    }
    __syncwarp();
    if (i < n) {
        const int at = gstart_s[gidx_s[first]] + rk;
        ent_dst[at] = i;
        ent_tok[at] = i / k_per_tok;
    }
}
}  // namespace

void mtp_route_group(const float* logits, int32_t* ids, float* weights, const uint8_t* base, int64_t blob,
                     unsigned long long* grp_ptr, int32_t* grp_start, int32_t* counts, int32_t* ent_dst, int32_t* ent_tok,
                     cudaStream_t stream) {
    route_group_kernel<<<1, 32, 0, stream>>>(logits, ids, weights, base, (long long) blob, grp_ptr, grp_start, counts,
                                                     ent_dst, ent_tok);
    const cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "mtp_route_group: %s\n", cudaGetErrorString(err));
        std::exit(1);
    }
}

}  // namespace strata::kernels
