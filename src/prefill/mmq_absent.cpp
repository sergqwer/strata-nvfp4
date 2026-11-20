// src/prefill/mmq_absent.cpp - see include/strata/prefill/moe_mmq.hpp.  P4-SPLIT: an engine built without llama.cpp's MMQ prompt kernels
// (-DSTRATA_MMQ=OFF) keeps Strata's own (strata_pf_own: the fused experts and moe_helpers.cu); this file answers the MMQ half of the
// interface for it.  Every query says "no MMQ", so the prompt path never plans an MMQ product; the calls that would launch one cannot be
// reached from there and stop the process with a plain message instead of leaving garbage in a buffer.
#include "strata/prefill/moe_mmq.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::prefill::mmq {
namespace {
[[noreturn]] void absent(const char* what) {
    std::fprintf(stderr, "strata: internal error: %s was called, but this engine was built without llama.cpp's MMQ prompt kernels\n", what);
    std::fflush(stderr);
    std::abort();
}
}  // namespace

bool built() { return false; }
bool supported(int) { return false; }
bool fits(int, int64_t) { return false; }
Nvfp4Mode nvfp4_mode() { return Nvfp4Mode::FP16; }
bool fp4_activations(int) { return false; }
void quantize(const float*, const int32_t*, void*, int, int64_t, int64_t, int64_t, void*, float*, const int32_t*, int) { absent("mmq::quantize"); }
void quantize_scatter(const float*, const int32_t*, const int32_t*, void*, int, int64_t, int64_t, int64_t, int, void*) {
    absent("mmq::quantize_scatter");
}
void scale_gu_rows(float*, int64_t, int64_t, const int32_t*, int, const float*, int64_t, int64_t, void*) { absent("mmq::scale_gu_rows"); }
void scale_down_rows(float*, int64_t, const int32_t*, int, const float*, int64_t, void*) { absent("mmq::scale_down_rows"); }
// (a Context is made where the prompt path sets up; with no MMQ product planned it is never run)
Context::Context() {}
Context::~Context() {}
void Context::run(const Product&, void*) { absent("mmq::Context::run"); }
bool swiglu_quant_ok(int) { return false; }
void swiglu_quant(const float*, void*, int64_t, bool, void*) { absent("mmq::swiglu_quant"); }
void f16_to_f32(const uint16_t*, float*, int64_t, void*) { absent("mmq::f16_to_f32"); }
void set_bounds(int32_t*, int32_t, void*) { absent("mmq::set_bounds"); }
bool tab_built() { return false; }
bool tab_fits(int, int64_t) { return false; }
size_t overread_bytes(int, int64_t) { return 0; }
bool whole_tiles(int, int64_t, int) { return false; }
bool launch_splits_k(int, int64_t, int64_t, int) { return false; }
int direct_mode() { return 0; }
// strata-nvfp4: the in-place (pointer-table) and w4a4x2 queries and the fused swiglu_quant of the MMQ side: never taken here
bool direct_ok(int) { return false; }
bool token_rows_ok(int, int64_t) { return false; }
bool swiglu_quant_scaled_ok(int) { return false; }
void swiglu_quant(const float*, void*, int64_t, const int32_t*, int, const float* const*, int64_t, float*, void*) { absent("mmq::swiglu_quant (fork)"); }
size_t direct_slot_guard() { return 0; }
void prep_group(const GuardTab&, size_t, void*, void*, size_t, void*) { absent("mmq::prep_group"); }
int fp4g_mode() { return 0; }
int fp4g_group(int, const void* const*, const void* const*, const float* const*, const int32_t*, const int32_t*, int64_t,
               int64_t, int64_t, const void*, const float*, const int32_t*, int64_t, float*, void*, float*, float*, float*,
               void*) { return 0; }
void fp4g_set_dm16(bool) {}

}  // namespace strata::prefill::mmq
