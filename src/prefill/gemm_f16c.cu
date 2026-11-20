// src/prefill/gemm_f16c.cu - see include/strata/prefill/gemm_f16c.hpp.
//
// Y[T, N] (+)= X[T, K] . W[N, K]^T, f16 operands, f32 output.  The tensor cores' f16-accumulate MMA runs at twice the
// f32-accumulate rate on GeForce Blackwell (sm_120: 1024 vs 512 flop/clk/SM), so the K sum runs in f16 chains of 64
// (four m16n8k16 MMAs: each MMA's result is the correctly rounded f16 of its exact sum), and every chain is added to an
// f32 accumulator.  The relative error grows from cuBLAS's ~3e-4 to ~4.4e-4 (gpu-probe/poc/gemm_poc.cu, a Q8_0 weight
// and activations with outlier channels); sustained at the 510 W power limit 235.5 -> 267.3 TFLOPS on 16384 x 10240 x 2560.
// A CTA tile whose result is not finite (a chain overflowed f16's 65504, or a NaN / Inf input) is computed again with
// f32 accumulation, so an overflow costs time, never a wrong value.
//
// CTA 128 x 128 (rows of X x rows of W), 4 warps of 64 x 64, K in 32-element (64-byte) stages, 3 stages through cp.async,
// two CTAs per SM (48 KiB of shared memory each).  Every T and N; K a multiple of 32; X, W 16-byte aligned.
#include "strata/prefill/gemm_f16c.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::prefill {
namespace {

constexpr int kBK = 64;                       // bytes of K per stage (32 halves)
constexpr int kStages = 3;
constexpr int kTile = 128 * kBK;              // bytes per operand and stage
constexpr int kThreads = 128;
constexpr int kSmem = kStages * 2 * kTile;    // 49152

__device__ __forceinline__ unsigned smem_addr(const void* p) { return (unsigned) __cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16(unsigned d, const void* s) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(s));
}
__device__ __forceinline__ void ldm4(unsigned (&r)[4], unsigned a) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}
// 64-byte rows, 16-byte chunks XOR-swizzled by (row >> 1) & 3: ldmatrix's 8 rows hit 8 distinct bank groups
__device__ __forceinline__ unsigned swz(int row, int chunk) { return row * kBK + ((chunk ^ ((row >> 1) & 3)) << 4); }

template <int CH>   // 0: f32 accumulation; 1, 2, 4: f16 chains of 16, 32, 64 k
__device__ __forceinline__ void mainloop(const uint8_t* __restrict__ X, const uint8_t* __restrict__ W, int T, int N, int K,
                                         int m0, int n0, uint8_t* smem, float (&acc)[4][8][4]) {
    uint8_t* As = smem;
    uint8_t* Bs = smem + kStages * kTile;
    const int Kb = K * 2, nkt = Kb / kBK;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 1, wn = warp & 1;
    auto load = [&](int kt, int st) {
#pragma unroll
        for (int i = 0; i < 4; i++) {   // 128 rows x 4 chunks = 512 / 128 threads, A and B
            const int c = tid + i * kThreads, row = c >> 2, ch = c & 3;
            const int ra = min(m0 + row, T - 1), rb = min(n0 + row, N - 1);
            cp16(smem_addr(As + st * kTile) + swz(row, ch), X + (size_t) ra * Kb + kt * kBK + ch * 16);
            cp16(smem_addr(Bs + st * kTile) + swz(row, ch), W + (size_t) rb * Kb + kt * kBK + ch * 16);
        }
    };
#pragma unroll
    for (int a = 0; a < 4; a++)
#pragma unroll
        for (int b = 0; b < 8; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;
    uint32_t hacc[4][8][2];
    if (CH > 0) {
#pragma unroll
        for (int a = 0; a < 4; a++)
#pragma unroll
            for (int b = 0; b < 8; b++) hacc[a][b][0] = hacc[a][b][1] = 0u;
    }
    for (int s = 0; s < kStages - 1; s++) {
        if (s < nkt) load(s, s);
        asm volatile("cp.async.commit_group;");
    }
    for (int kt = 0; kt < nkt; kt++) {
        asm volatile("cp.async.wait_group %0;" ::"n"(kStages - 2));
        __syncthreads();
        {
            const int nk = kt + kStages - 1;
            if (nk < nkt) load(nk, nk % kStages);
            asm volatile("cp.async.commit_group;");
        }
        const uint8_t* A = As + (kt % kStages) * kTile;
        const uint8_t* B = Bs + (kt % kStages) * kTile;
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            unsigned bf[4][4];
#pragma unroll
            for (int np = 0; np < 4; np++) {
                const int m = lane >> 3, row = wn * 64 + np * 16 + (m >> 1) * 8 + (lane & 7), ch = ks * 2 + (m & 1);
                ldm4(bf[np], smem_addr(B) + swz(row, ch));
            }
#pragma unroll
            for (int mt = 0; mt < 4; mt++) {
                unsigned af[4];
                const int m = lane >> 3, row = wm * 64 + mt * 16 + (m & 1) * 8 + (lane & 7), ch = ks * 2 + (m >> 1);
                ldm4(af, smem_addr(A) + swz(row, ch));
#pragma unroll
                for (int nt = 0; nt < 8; nt++) {
                    const unsigned b0 = bf[nt >> 1][(nt & 1) * 2], b1 = bf[nt >> 1][(nt & 1) * 2 + 1];
                    if (CH > 0)
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                                     : "+r"(hacc[mt][nt][0]), "+r"(hacc[mt][nt][1])
                                     : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(b0), "r"(b1));
                    else
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                     : "+f"(acc[mt][nt][0]), "+f"(acc[mt][nt][1]), "+f"(acc[mt][nt][2]), "+f"(acc[mt][nt][3])
                                     : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]), "r"(b0), "r"(b1));
                }
                if (CH == 1) {   // a chain is one MMA (16 k)
#pragma unroll
                for (int nt = 0; nt < 8; nt++) {
                    const float2 lo = __half22float2(*reinterpret_cast<__half2*>(&hacc[mt][nt][0]));
                    const float2 hi = __half22float2(*reinterpret_cast<__half2*>(&hacc[mt][nt][1]));
                    acc[mt][nt][0] += lo.x;
                    acc[mt][nt][1] += lo.y;
                    acc[mt][nt][2] += hi.x;
                    acc[mt][nt][3] += hi.y;
                    hacc[mt][nt][0] = hacc[mt][nt][1] = 0u;
                }
                }
            }
        }
        if ((CH == 2) || (CH == 4 && ((kt & 1) || kt == nkt - 1))) {   // 32 k (a stage) or 64 k (two; the last may be 32)
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int nt = 0; nt < 8; nt++) {
                    const float2 lo = __half22float2(*reinterpret_cast<__half2*>(&hacc[mt][nt][0]));
                    const float2 hi = __half22float2(*reinterpret_cast<__half2*>(&hacc[mt][nt][1]));
                    acc[mt][nt][0] += lo.x;
                    acc[mt][nt][1] += lo.y;
                    acc[mt][nt][2] += hi.x;
                    acc[mt][nt][3] += hi.y;
                    hacc[mt][nt][0] = hacc[mt][nt][1] = 0u;
                }
        }
    }
    asm volatile("cp.async.wait_all;");
    __syncthreads();   // the shared memory may be refilled by a second pass
}

template <int CH>
__global__ void __launch_bounds__(kThreads, 2) gemm_f16c_kernel(const uint8_t* __restrict__ X, const uint8_t* __restrict__ W,
                                                                float* __restrict__ Y, int T, int N, int K, int64_t ldy,
                                                                float beta, unsigned* __restrict__ redo_count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    __trap();                                     // never launched there: gemm_f16c() returns false below cc 8.0
#else
    extern __shared__ __align__(128) uint8_t smem[];
    const int m0 = blockIdx.y * 128, n0 = blockIdx.x * 128;
    float acc[4][8][4];
    mainloop<CH>(X, W, T, N, K, m0, n0, smem, acc);
    bool bad = false;
#pragma unroll
    for (int a = 0; a < 4; a++)
#pragma unroll
        for (int b = 0; b < 8; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) bad |= !(fabsf(acc[a][b][c]) <= 3.0e38f);
    if (__syncthreads_or(bad)) {
        mainloop<0>(X, W, T, N, K, m0, n0, smem, acc);
        if (threadIdx.x == 0 && redo_count) atomicAdd(redo_count, 1u);
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wm = warp >> 1, wn = warp & 1, g = lane >> 2, t4 = lane & 3;
    const bool vec = (ldy & 1) == 0 && (reinterpret_cast<uintptr_t>(Y) & 7) == 0;
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int r = m0 + wm * 64 + mt * 16 + g + h * 8;
            if (r >= T) continue;
            float* yr = Y + (int64_t) r * ldy;
#pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                const int c = n0 + wn * 64 + nt * 8 + 2 * t4;
                float v0 = acc[mt][nt][h * 2], v1 = acc[mt][nt][h * 2 + 1];
                if (vec && c + 1 < N) {
                    float2* p = reinterpret_cast<float2*>(yr + c);
                    if (beta != 0.0f) {
                        const float2 o = *p;
                        v0 = fmaf(beta, o.x, v0);
                        v1 = fmaf(beta, o.y, v1);
                    }
                    *p = make_float2(v0, v1);
                } else {
                    if (c < N) yr[c] = beta != 0.0f ? fmaf(beta, yr[c], v0) : v0;
                    if (c + 1 < N) yr[c + 1] = beta != 0.0f ? fmaf(beta, yr[c + 1], v1) : v1;
                }
            }
        }
#endif
}

__device__ unsigned g_f16c_redo;

}  // namespace

bool gemm_f16c_fits(int64_t T, int64_t N, int64_t K) {
    return T > 0 && N > 0 && K >= 32 && K % 32 == 0 && T <= 65535LL * 128 && N <= 2147483647LL && K <= 1048576;
}

bool gemm_f16c(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy, float beta,
               void* stream) {
    if (!gemm_f16c_fits(T, N, K)) return false;
    if ((reinterpret_cast<uintptr_t>(X) & 15) != 0 || (reinterpret_cast<uintptr_t>(W) & 15) != 0) return false;
    if (ldy <= 0) ldy = N;
    // STRATA_DENSE_F16C_CHAIN=16|32|64 (k per f16 chain, default 64)
    static const int chain = [] { const char* v = std::getenv("STRATA_DENSE_F16C_CHAIN"); const int c = v ? std::atoi(v) : 64; return c == 16 ? 1 : c == 32 ? 2 : 4; }();
    auto kern = chain == 1 ? gemm_f16c_kernel<1> : chain == 2 ? gemm_f16c_kernel<2> : gemm_f16c_kernel<4>;
    static bool attr[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    static int major[64] = {};                    // the kernel needs sm_80 (cp.async, m16n8k16): older cards keep cuBLAS
    if (major[dev] == 0 && cudaDeviceGetAttribute(&major[dev], cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
        cudaGetLastError();
        major[dev] = -1;
    }
    if (major[dev] < 8) return false;
    if (!attr[dev]) {
        if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        attr[dev] = true;
    }
    unsigned* redo = nullptr;
    if (cudaGetSymbolAddress((void**) &redo, g_f16c_redo) != cudaSuccess) { cudaGetLastError(); redo = nullptr; }
    const dim3 grid((unsigned) ((N + 127) / 128), (unsigned) ((T + 127) / 128));
    kern<<<grid, kThreads, kSmem, (cudaStream_t) stream>>>(reinterpret_cast<const uint8_t*>(X),
                                                                       reinterpret_cast<const uint8_t*>(W), Y, (int) T,
                                                                       (int) N, (int) K, ldy, beta, redo);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "gemm_f16c: %s\n", cudaGetErrorString(e)); std::exit(1); }
    return true;
}

unsigned gemm_f16c_redo_count(bool reset) {
    unsigned v = 0;
    if (cudaMemcpyFromSymbol(&v, g_f16c_redo, sizeof v) != cudaSuccess) { cudaGetLastError(); return 0; }
    if (reset) {
        const unsigned z = 0;
        cudaMemcpyToSymbol(g_f16c_redo, &z, sizeof z);
    }
    return v;
}

}  // namespace strata::prefill

namespace strata::prefill {
namespace {
__global__ void f16c_diff_kernel(const float* __restrict__ a, const float* __restrict__ b, int64_t n, double* __restrict__ out) {
    double d2 = 0, r2 = 0, mx = 0;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x * blockDim.x) {
        const double d = (double) a[i] - (double) b[i];
        d2 += d * d;
        r2 += (double) b[i] * (double) b[i];
        mx = fmax(mx, fabs(d));
    }
    for (int o = 16; o > 0; o >>= 1) {
        d2 += __shfl_xor_sync(0xffffffffu, d2, o);
        r2 += __shfl_xor_sync(0xffffffffu, r2, o);
        mx = fmax(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    }
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&out[0], d2);
        atomicAdd(&out[1], r2);
        unsigned long long* m = reinterpret_cast<unsigned long long*>(&out[2]);
        atomicMax(m, (unsigned long long) __double_as_longlong(mx));   // non-negative doubles order as integers
    }
}
}  // namespace

bool gemm_f16c_diff(const float* a, const float* b, int64_t n, double* host_out3, void* stream) {
    static double* d = nullptr;
    if (d == nullptr && cudaMalloc((void**) &d, 3 * sizeof(double)) != cudaSuccess) { cudaGetLastError(); d = nullptr; return false; }
    const cudaStream_t st = (cudaStream_t) stream;
    cudaMemsetAsync(d, 0, 3 * sizeof(double), st);
    f16c_diff_kernel<<<512, 256, 0, st>>>(a, b, n, d);
    if (cudaMemcpyAsync(host_out3, d, 3 * sizeof(double), cudaMemcpyDeviceToHost, st) != cudaSuccess ||
        cudaStreamSynchronize(st) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}

}  // namespace strata::prefill
