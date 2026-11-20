// include/strata/prefill/gemm_f16c.hpp - the prompt path's dense FP16 GEMM with f16-accumulate chains (CUDA sm_80+).
//
// Y[T, N] = X[T, K] . W[N, K]^T + beta * Y (row-major, Y row stride ldy, f32), with the K sum in f16 chains of 64
// promoted to f32 (twice cuBLAS's f32-accumulate tensor rate on GeForce cards; relative error ~1.5x cuBLAS's).  A tile
// that comes out non-finite is recomputed with f32 accumulation.  Gemm::f16 takes it with STRATA_DENSE_F16C=1.
#pragma once

#include <cstdint>

namespace strata::prefill {

/// The shapes the kernel takes: K a multiple of 32 (and >= 32), T and N any size.
bool gemm_f16c_fits(int64_t T, int64_t N, int64_t K);
/// False (nothing launched): a shape gemm_f16c_fits refuses, X or W not 16-byte aligned, no device.
bool gemm_f16c(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
               void* stream);
/// Tiles recomputed with f32 accumulation since the last reset (synchronizes the device).
unsigned gemm_f16c_redo_count(bool reset);
/// STRATA_DENSE_F16C_CHECK: sum (a - b)^2, sum b^2 and max |a - b| over n floats (synchronizes `stream`).
bool gemm_f16c_diff(const float* a, const float* b, int64_t n, double* host_out3, void* stream);

}  // namespace strata::prefill
