// the candidates and the launch table of hc_v5_proto.cu
template <typename K, typename... Args>
static cudaError_t lx(K k, dim3 grid, size_t sm, cudaStream_t st, bool pdl, Args... args) {
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t c = {};
    c.gridDim = grid; c.blockDim = dim3(THREADS); c.dynamicSmemBytes = sm; c.stream = st;
    c.attrs = pdl ? attr : nullptr; c.numAttrs = pdl ? 1 : 0;
    return cudaLaunchKernelEx(&c, k, args...);
}
// v4 at one T (R2 = 2, the default); part: 0 both, 1 down only, 2 up only
template <int TT, bool EX>
static cudaError_t v4_t(const GrMulti& m, float* scratch, int T, cudaStream_t st, int part) {
    float* pp = scratch;
    float* ssg = scratch + (size_t) T * HC * 2 * PR;
    const bool pdl = part == 0;
    cudaError_t e = cudaSuccess;
    if (part != 2) e = lx(gr_down_v4_kernel<TT, EX, true, 2>, dim3(LR / (WARPS * 2) + 1, HC), (size_t) T * N * 4, st, pdl, m, pp, ssg);
    if (e == cudaSuccess && part != 1) e = lx(gr_up_v4_kernel<TT, EX, true>, dim3(UPM_BLOCKS), 0, st, pdl, m, (const float*) pp, (const float*) ssg);
    return e;
}
template <int TT, bool EX>
static void v4_attr() {
    cudaFuncSetAttribute(gr_down_v4_kernel<TT, EX, true, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, TT * N * 4);
    cudaFuncSetAttribute(gr_down_v4_kernel<TT, EX, true, 2>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}
static cudaError_t v4_any(const GrMulti& m, float* s, int T, cudaStream_t st, int part, bool exact78) {
    switch (T) {
        case 1: return v4_t<1, true>(m, s, T, st, part);
        case 2: return v4_t<2, true>(m, s, T, st, part);
        case 3: return v4_t<3, true>(m, s, T, st, part);
        case 4: return v4_t<4, true>(m, s, T, st, part);
        case 5: return v4_t<5, true>(m, s, T, st, part);
        case 6: return v4_t<6, true>(m, s, T, st, part);
        case 7: if (exact78) return v4_t<7, true>(m, s, T, st, part); break;
        case 8: if (exact78) return v4_t<8, true>(m, s, T, st, part); break;
    }
    return v4_t<8, false>(m, s, T, st, part);
}

#include "hc_diag.cuh"
#include "hc_v5_up.cuh"
// variant 2: v4's down (exact T) + v5's up
template <int TT>
static cudaError_t v52_t(const GrMulti& m, float* scratch, int T, cudaStream_t st, int part) {
    float* pp = scratch;
    float* ssg = scratch + (size_t) T * HC * 2 * PR;
    const bool pdl = part == 0;
    cudaError_t e = cudaSuccess;
    if (part != 2) e = lx(gr_down_v4_kernel<TT, true, true, 2>, dim3(LR / (WARPS * 2) + 1, HC), (size_t) T * N * 4, st, pdl, m, pp, ssg);
    if (e == cudaSuccess && part != 1) e = lx(gr_up_v5_kernel<TT, true, true>, dim3(UPM_BLOCKS), 0, st, pdl, m, (const float*) pp, (const float*) ssg);
    return e;
}
#include "hc_v5_down.cuh"
template <int TT, int P, int DG = 0>
static cudaError_t v5_t(const GrMulti& m, float* scratch, int T, cudaStream_t st, int part) {
    float* pp = scratch;
    float* ssg = scratch + (size_t) T * HC * 2 * PR;
    const bool pdl = part == 0;
    cudaError_t e = cudaSuccess;
    const int gx = P * (d5_nrg<P>() + (m.a[0].w_inject != nullptr ? 1 : 0));
    if (part != 2) {
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t c = {};
        c.gridDim = dim3(gx, HC); c.blockDim = dim3(D5_THREADS); c.dynamicSmemBytes = d5_smem<P>(TT); c.stream = st;
        c.attrs = pdl ? attr : nullptr; c.numAttrs = pdl ? 1 : 0;
        e = cudaLaunchKernelEx(&c, gr_down_v5_kernel<TT, P, true, DG>, m, pp, ssg);
    }
    if (e == cudaSuccess && part != 1) e = lx(gr_up_v5_kernel<TT, true, true>, dim3(UPM_BLOCKS), 0, st, pdl, m, (const float*) pp, (const float*) ssg);
    return e;
}
template <int TT> static cudaError_t v5p2(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v5_t<TT, 2>(m, s, T, st, part); }
template <int TT> static cudaError_t v5p4(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v5_t<TT, 4>(m, s, T, st, part); }
template <int TT> static cudaError_t v5p8(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v5_t<TT, 8>(m, s, T, st, part); }
template <int P, int DG = 0> static void d5_attr() {
    auto f = [](auto k, int tt) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) d5_smem<P>(tt)); cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 100); };
    f(gr_down_v5_kernel<1, P, true, DG>, 1); f(gr_down_v5_kernel<2, P, true, DG>, 2); f(gr_down_v5_kernel<3, P, true, DG>, 3); f(gr_down_v5_kernel<4, P, true, DG>, 4);
    f(gr_down_v5_kernel<5, P, true, DG>, 5); f(gr_down_v5_kernel<6, P, true, DG>, 6); f(gr_down_v5_kernel<7, P, true, DG>, 7); f(gr_down_v5_kernel<8, P, true, DG>, 8);
}
template <int TT> static cudaError_t v5d1(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v5_t<TT, 2, 1>(m, s, T, st, part); }
template <int TT> static cudaError_t v5d2(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v5_t<TT, 2, 2>(m, s, T, st, part); }
template <int TT> static cudaError_t v5d4(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v5_t<TT, 2, 4>(m, s, T, st, part); }
template <int TT> static cudaError_t v5d8(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v5_t<TT, 2, 8>(m, s, T, st, part); }
template <int TT> static cudaError_t v5d16(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v5_t<TT, 2, 16>(m, s, T, st, part); }
template <int TT> static cudaError_t v5d48(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v5_t<TT, 2, 48>(m, s, T, st, part); }
template <int TT> static cudaError_t v5d36(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v5_t<TT, 2, 36>(m, s, T, st, part); }
template <int TT> static cudaError_t v5d64(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v5_t<TT, 2, 64>(m, s, T, st, part); }
template <int TT> static cudaError_t v5d32(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v5_t<TT, 2, 32>(m, s, T, st, part); }
template <int TT> static cudaError_t v5d128(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v5_t<TT, 2, 128>(m, s, T, st, part); }
template <int TT> static cudaError_t v5d288(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v5_t<TT, 2, 288>(m, s, T, st, part); }
template <int TT> static cudaError_t v5d544(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v5_t<TT, 2, 544>(m, s, T, st, part); }
#include "hc_v4b_down.cuh"
template <int TT, int KB, int R2 = 2, int TH = 256, int DG = 0>
static cudaError_t v4b_t(const GrMulti& m, float* scratch, int T, cudaStream_t st, int part) {
    float* pp = scratch;
    float* ssg = scratch + (size_t) T * HC * 2 * PR;
    const bool pdl = part == 0;
    cudaError_t e = cudaSuccess;
    if (part != 2) {
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t c = {};
        c.gridDim = dim3(LR / (TH / 32 * R2) + 1, HC); c.blockDim = dim3(TH); c.dynamicSmemBytes = (size_t) T * N * 4; c.stream = st;
        c.attrs = pdl ? attr : nullptr; c.numAttrs = pdl ? 1 : 0;
        e = cudaLaunchKernelEx(&c, gr_down_v4b_kernel<TT, true, R2, KB, TH, DG>, m, pp, ssg);
    }
    if (e == cudaSuccess && part != 1) e = lx(gr_up_v5_kernel<TT, true, true>, dim3(UPM_BLOCKS), 0, st, pdl, m, (const float*) pp, (const float*) ssg);
    return e;
}
template <int TT> static cudaError_t v4bk1(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v4b_t<TT, 1>(m, s, T, st, part); }
template <int TT> static cudaError_t v4bk2(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v4b_t<TT, 2>(m, s, T, st, part); }
template <int TT> static cudaError_t v4bk4(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v4b_t<TT, 4>(m, s, T, st, part); }
template <int TT> static cudaError_t v4bk2r1(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v4b_t<TT, 2, 1>(m, s, T, st, part); }
template <int KB, int R2, int TH = 256, int DG = 0> static void v4b_attr() {
    auto f = [](auto k, int tt) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, tt * N * 4); cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 100); };
    f(gr_down_v4b_kernel<1, true, R2, KB, TH, DG>, 1); f(gr_down_v4b_kernel<2, true, R2, KB, TH, DG>, 2); f(gr_down_v4b_kernel<3, true, R2, KB, TH, DG>, 3); f(gr_down_v4b_kernel<4, true, R2, KB, TH, DG>, 4);
    f(gr_down_v4b_kernel<5, true, R2, KB, TH, DG>, 5); f(gr_down_v4b_kernel<6, true, R2, KB, TH, DG>, 6); f(gr_down_v4b_kernel<7, true, R2, KB, TH, DG>, 7); f(gr_down_v4b_kernel<8, true, R2, KB, TH, DG>, 8);
}
template <int TT> static cudaError_t v4ck2(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v4b_t<TT, 2, 2, 128>(m, s, T, st, part); }
template <int TT> static cudaError_t v4ck1(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v4b_t<TT, 1, 2, 128>(m, s, T, st, part); }
template <int TT> static cudaError_t v4ck2r1(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v4b_t<TT, 2, 1, 128>(m, s, T, st, part); }
template <int TT> static cudaError_t v4bd1(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v4b_t<TT, 2, 2, 256, 1>(m, s, T, st, part); }
template <int TT> static cudaError_t v4bd2(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v4b_t<TT, 2, 2, 256, 2>(m, s, T, st, part); }
template <int TT> static cudaError_t v4bd32(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v4b_t<TT, 2, 2, 256, 32>(m, s, T, st, part); }
#include "hc_v5b_up.cuh"
template <int TT, int KB = 2>
static cudaError_t v5b_t(const GrMulti& m, float* scratch, int T, cudaStream_t st, int part) {
    float* pp = scratch;
    float* ssg = scratch + (size_t) T * HC * 2 * PR;
    const bool pdl = part == 0;
    cudaError_t e = cudaSuccess;
    if (part != 2) e = lx(gr_down_v4b_kernel<TT, true, 2, KB>, dim3(LR / (WARPS * 2) + 1, HC), (size_t) T * N * 4, st, pdl, m, pp, ssg);
    if (e == cudaSuccess && part != 1) e = lx(gr_up_v5b_kernel<TT, true, true>, dim3(UPM_BLOCKS), 0, st, pdl, m, (const float*) pp, (const float*) ssg);
    return e;
}
template <int TT> static cudaError_t v5bk2(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v5b_t<TT, 2>(m, s, T, st, part); }
template <int TT> static cudaError_t v5ldg(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v5_t<TT, 2, 1024>(m, s, T, st, part); }
template <int TT> static cudaError_t v5ldg4(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v5_t<TT, 4, 1024>(m, s, T, st, part); }
template <int TT> static cudaError_t v5ldgst(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v5_t<TT, 2, 1056>(m, s, T, st, part); }
template <int TT> static cudaError_t v4br4k1(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v4b_t<TT, 1, 4>(m, s, T, st, part); }
template <int TT> static cudaError_t v4br4k2(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v4b_t<TT, 2, 4>(m, s, T, st, part); }
#include "hc_v4d_down.cuh"
template <int TT, int CS>
static cudaError_t v4d_t(const GrMulti& m, float* scratch, int T, cudaStream_t st, int part) {
    float* pp = scratch;
    float* ssw = scratch + (size_t) T * HC * 2 * PR;
    const bool pdl = part == 0;
    cudaError_t e = cudaSuccess;
    if (part != 2) {
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t c = {};
        const int nb = (LR / (WARPS * 2) + 1 + CS - 1) / CS * CS;
        c.gridDim = dim3(nb, HC); c.blockDim = dim3(THREADS); c.dynamicSmemBytes = (size_t) T * N * 4; c.stream = st;
        c.attrs = pdl ? attr : nullptr; c.numAttrs = pdl ? 1 : 0;
        e = cudaLaunchKernelEx(&c, gr_down_v4d_kernel<TT, CS, true>, m, pp, ssw);
    }
    if (e == cudaSuccess && part != 1) e = lx(gr_up_v5b_kernel<TT, true, true, true>, dim3(UPM_BLOCKS), 0, st, pdl, m, (const float*) pp, (const float*) ssw);
    return e;
}
template <int TT> static cudaError_t v4dc2(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v4d_t<TT, 2>(m, s, T, st, part); }
template <int TT> static cudaError_t v4dc4(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v4d_t<TT, 4>(m, s, T, st, part); }
template <int TT> static cudaError_t v4dc8(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v4d_t<TT, 8>(m, s, T, st, part); }
template <int CS> static void v4d_attr() {
    auto f = [](auto k, int tt) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, tt * N * 4); cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 100); };
    f(gr_down_v4d_kernel<1, CS, true>, 1); f(gr_down_v4d_kernel<2, CS, true>, 2); f(gr_down_v4d_kernel<3, CS, true>, 3); f(gr_down_v4d_kernel<4, CS, true>, 4);
    f(gr_down_v4d_kernel<5, CS, true>, 5); f(gr_down_v4d_kernel<6, CS, true>, 6); f(gr_down_v4d_kernel<7, CS, true>, 7); f(gr_down_v4d_kernel<8, CS, true>, 8);
}
#include "hc_v4e_down.cuh"
template <int TT, int KB>
static cudaError_t v4e_t(const GrMulti& m, float* scratch, int T, cudaStream_t st, int part) {
    float* pp = scratch;
    float* ssg = scratch + (size_t) T * HC * 2 * PR;
    const bool pdl = part == 0;
    cudaError_t e = cudaSuccess;
    if (part != 2) e = lx(gr_down_v4e_kernel<TT, true, 2, KB>, dim3(LR / (WARPS * 2) + 1, HC), (size_t) T * N * 4, st, pdl, m, pp, ssg);
    if (e == cudaSuccess && part != 1) e = lx(gr_up_v5b_kernel<TT, true, true>, dim3(UPM_BLOCKS), 0, st, pdl, m, (const float*) pp, (const float*) ssg);
    return e;
}
template <int TT> static cudaError_t v4ek1(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v4e_t<TT, 1>(m, s, T, st, part); }
template <int TT> static cudaError_t v4ek2(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v4e_t<TT, 2>(m, s, T, st, part); }
template <int TT> static cudaError_t v4ek4(const GrMulti& m, float* s, int T, cudaStream_t st, int part) { return v4e_t<TT, 4>(m, s, T, st, part); }
template <int KB> static void v4e_attr() {
    auto f = [](auto k, int tt) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, tt * N * 4); cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 100); };
    f(gr_down_v4e_kernel<1, true, 2, KB>, 1); f(gr_down_v4e_kernel<2, true, 2, KB>, 2); f(gr_down_v4e_kernel<3, true, 2, KB>, 3); f(gr_down_v4e_kernel<4, true, 2, KB>, 4);
    f(gr_down_v4e_kernel<5, true, 2, KB>, 5); f(gr_down_v4e_kernel<6, true, 2, KB>, 6); f(gr_down_v4e_kernel<7, true, 2, KB>, 7); f(gr_down_v4e_kernel<8, true, 2, KB>, 8);
}
#define TSWITCH(F, ...) switch (T) { case 1: return F<1>(__VA_ARGS__); case 2: return F<2>(__VA_ARGS__); case 3: return F<3>(__VA_ARGS__);     case 4: return F<4>(__VA_ARGS__); case 5: return F<5>(__VA_ARGS__); case 6: return F<6>(__VA_ARGS__); case 7: return F<7>(__VA_ARGS__);     default: return F<8>(__VA_ARGS__); }
template <int TT, int DG>
static cudaError_t dg_t(const GrMulti& m, float* scratch, int T, cudaStream_t st, int part) {
    float* pp = scratch;
    float* ssg = scratch + (size_t) T * HC * 2 * PR;
    const bool pdl = part == 0;
    cudaError_t e = cudaSuccess;
    if (part != 2) e = lx(gr_down_dg_kernel<TT, true, true, 2, DG>, dim3(LR / (WARPS * 2) + 1, HC), (size_t) T * N * 4, st, pdl, m, pp, ssg);
    if (e == cudaSuccess && part != 1) e = lx(gr_up_dg_kernel<TT, true, true, DG>, dim3(UPM_BLOCKS), 0, st, pdl, m, (const float*) pp, (const float*) ssg);
    return e;
}
template <int DG>
static cudaError_t dg_any(const GrMulti& m, float* s, int T, cudaStream_t st, int part) {
    switch (T) {
        case 1: return dg_t<1, DG>(m, s, T, st, part); case 2: return dg_t<2, DG>(m, s, T, st, part);
        case 3: return dg_t<3, DG>(m, s, T, st, part); case 4: return dg_t<4, DG>(m, s, T, st, part);
        case 5: return dg_t<5, DG>(m, s, T, st, part); case 6: return dg_t<6, DG>(m, s, T, st, part);
        case 7: return dg_t<7, DG>(m, s, T, st, part); default: return dg_t<8, DG>(m, s, T, st, part);
    }
}
template <int DG> static void dg_attr1() {
    auto f = [](auto k, int tt) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, tt * N * 4); cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 100); };
    f(gr_down_dg_kernel<1, true, true, 2, DG>, 1); f(gr_down_dg_kernel<2, true, true, 2, DG>, 2); f(gr_down_dg_kernel<3, true, true, 2, DG>, 3);
    f(gr_down_dg_kernel<4, true, true, 2, DG>, 4); f(gr_down_dg_kernel<5, true, true, 2, DG>, 5); f(gr_down_dg_kernel<6, true, true, 2, DG>, 6);
    f(gr_down_dg_kernel<7, true, true, 2, DG>, 7); f(gr_down_dg_kernel<8, true, true, 2, DG>, 8);
}
static void set_attrs() {
    v4b_attr<1, 2>(); v4b_attr<2, 2>(); v4b_attr<4, 2>(); v4b_attr<2, 1>(); v4b_attr<2, 2, 128>(); v4b_attr<1, 2, 128>(); v4b_attr<2, 1, 128>(); v4b_attr<2, 2, 256, 1>(); v4b_attr<2, 2, 256, 2>(); v4b_attr<2, 2, 256, 32>(); v4b_attr<1, 4>(); v4b_attr<2, 4>(); v4d_attr<2>(); v4d_attr<4>(); v4d_attr<8>(); v4e_attr<1>(); v4e_attr<2>(); v4e_attr<4>();
    d5_attr<2>(); d5_attr<4>(); d5_attr<8>(); d5_attr<2, 1>(); d5_attr<2, 2>(); d5_attr<2, 4>(); d5_attr<2, 8>(); d5_attr<2, 16>(); d5_attr<2, 48>(); d5_attr<2, 64>(); d5_attr<2, 128>(); d5_attr<2, 288>(); d5_attr<2, 544>(); d5_attr<2, 1024>(); d5_attr<4, 1024>(); d5_attr<2, 1056>(); d5_attr<2, 32>(); d5_attr<2, 36>();
    dg_attr1<1>(); dg_attr1<2>(); dg_attr1<4>(); dg_attr1<8>();
    v4_attr<1, true>(); v4_attr<2, true>(); v4_attr<3, true>(); v4_attr<4, true>(); v4_attr<5, true>(); v4_attr<6, true>();
    v4_attr<7, true>(); v4_attr<8, true>(); v4_attr<8, false>();
}
static const char* variant_name(int v) {
    switch (v) {
        case 0: return "v4";
        case 1: return "v4 exact78";
        case 2: return "v4dn+v5up"; case 11: return "v4b kb1"; case 12: return "v4b kb2"; case 14: return "v4b kb4"; case 15: return "v4b kb2 r1"; case 21: return "v4c kb2"; case 31: return "v4b distinct"; case 33: return "v4b stamps"; case 40: return "v4b+v5b up"; case 41: return "v4b r4 kb1"; case 60: return "v4d cs2"; case 70: return "v4e kb1"; case 71: return "v4e kb2"; case 72: return "v4e kb4"; case 61: return "v4d cs4"; case 62: return "v4d cs8"; case 42: return "v4b r4 kb2"; case 32: return "v4b bo dist"; case 22: return "v4c kb1"; case 23: return "v4c kb2 r1";
        case 3: return "v5 P2"; case 81: return "P2 nocl"; case 82: return "P2 coalW"; case 84: return "P2 nostage"; case 88: return "P2 ldg"; case 96: return "P2 cp.ca"; case 148: return "P2 ca stamps"; case 164: return "P2 seqdots"; case 228: return "P2 rot"; case 388: return "P2 now stamps"; case 644: return "P2 distinct st"; case 50: return "v5 P2 ldg"; case 51: return "v5 P4 ldg"; case 52: return "P2 ldg stamps"; case 132: return "P2 stamps"; case 136: return "P2 nost stamps"; case 4: return "v5 P4"; case 5: return "v5 P8";
        case 91: return "dg nostage"; case 92: return "dg nodots"; case 94: return "dg up nolo"; case 98: return "dg up 1dot";
    }
    return "?";
}
static cudaError_t launch_variant(int v, const GrMulti& m, float* scratch, int T, cudaStream_t st, int part) {
    switch (v) {
        case 0: return v4_any(m, scratch, T, st, part, false);
        case 1: return v4_any(m, scratch, T, st, part, true);
        case 2: TSWITCH(v52_t, m, scratch, T, st, part)
        case 11: TSWITCH(v4bk1, m, scratch, T, st, part)
        case 12: TSWITCH(v4bk2, m, scratch, T, st, part)
        case 14: TSWITCH(v4bk4, m, scratch, T, st, part)
        case 15: TSWITCH(v4bk2r1, m, scratch, T, st, part)
        case 21: TSWITCH(v4ck2, m, scratch, T, st, part)
        case 31: TSWITCH(v4bd1, m, scratch, T, st, part)
        case 33: TSWITCH(v4bd32, m, scratch, T, st, part)
        case 40: TSWITCH(v5bk2, m, scratch, T, st, part)
        case 41: TSWITCH(v4br4k1, m, scratch, T, st, part)
        case 60: TSWITCH(v4dc2, m, scratch, T, st, part)
        case 70: TSWITCH(v4ek1, m, scratch, T, st, part)
        case 71: TSWITCH(v4ek2, m, scratch, T, st, part)
        case 72: TSWITCH(v4ek4, m, scratch, T, st, part)
        case 61: TSWITCH(v4dc4, m, scratch, T, st, part)
        case 62: TSWITCH(v4dc8, m, scratch, T, st, part)
        case 42: TSWITCH(v4br4k2, m, scratch, T, st, part)
        case 32: TSWITCH(v4bd2, m, scratch, T, st, part)
        case 22: TSWITCH(v4ck1, m, scratch, T, st, part)
        case 23: TSWITCH(v4ck2r1, m, scratch, T, st, part)
        case 3: TSWITCH(v5p2, m, scratch, T, st, part)
        case 81: TSWITCH(v5d1, m, scratch, T, st, part)
        case 82: TSWITCH(v5d2, m, scratch, T, st, part)
        case 84: TSWITCH(v5d4, m, scratch, T, st, part)
        case 88: TSWITCH(v5d8, m, scratch, T, st, part)
        case 96: TSWITCH(v5d16, m, scratch, T, st, part)
        case 148: TSWITCH(v5d48, m, scratch, T, st, part)
        case 164: TSWITCH(v5d64, m, scratch, T, st, part)
        case 228: TSWITCH(v5d128, m, scratch, T, st, part)
        case 388: TSWITCH(v5d288, m, scratch, T, st, part)
        case 644: TSWITCH(v5d544, m, scratch, T, st, part)
        case 50: TSWITCH(v5ldg, m, scratch, T, st, part)
        case 51: TSWITCH(v5ldg4, m, scratch, T, st, part)
        case 52: TSWITCH(v5ldgst, m, scratch, T, st, part)
        case 132: TSWITCH(v5d32, m, scratch, T, st, part)
        case 136: TSWITCH(v5d36, m, scratch, T, st, part)
        case 4: TSWITCH(v5p4, m, scratch, T, st, part)
        case 5: TSWITCH(v5p8, m, scratch, T, st, part)
        case 91: return dg_any<1>(m, scratch, T, st, part); case 92: return dg_any<2>(m, scratch, T, st, part);
        case 94: return dg_any<4>(m, scratch, T, st, part); case 98: return dg_any<8>(m, scratch, T, st, part);
    }
    return cudaErrorInvalidValue;
}
