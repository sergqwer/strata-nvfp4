// src/prefill/mmq_nvfp4_fp4g.cu - the prompt path's NVFP4 expert groups on a grouped FP4 tensor-core GEMM with the
// fork's two-term activations (opt-in: STRATA_PF_FP4G, see moe_mmq.hpp fp4g_mode).  Built for 12xa only, in the
// w4a4 unit's library (mmq_nvfp4_a44.cu: the same -use_fast_math command line).
//
// The activations are the w4a4x2 rows the MMQ path already reads (quantize_nvfp4_x2: block_fp4_mmq [kb][row], term 2
// y2_off bytes on, one FP32 scale per row): the same codes and scales, both terms through kind::mxf4nvf4 into one FP32
// sum per output, times the row scale - MMQ's sums (mmq.cuh STRATA_MMQ_Y2) in another accumulation order.  The
// weights are read in place (each expert's NVFP4 rows where they lie), their scale bytes as MMQ passes them.
// Epilogues: the gate/up product's SwiGLU with the expert's {s_gate, s_up} and H's q8_1 rows (swiglu_quant_kernel's
// expressions: the same bytes for the same GU values), or H in floats (swiglu_scaled_kernel's), or the plain product.
//
// Tile: 96 activation rows x 256 weight rows (SwiGLU: 128 gate rows and their 128 up rows), eight warps of 48 x 64,
// three cp.async stages of 128 K values (97 KB of shared memory).  The block-scaled MMA wrapper, the A fragments
// through ldmatrix from 64-byte bank-swizzled rows and the scale registers per 64-wide K step follow Basalt's
// src/prefill/moe_fp4.cu (MIT), which adapts Infernix's grouped NVFP4 GEMM (Ian Ranson and contributors,
// Apache-2.0: src/ops/offloaded_sparse_moe/cuda/wide_expert.cuh, src/ops/common/mma.cuh).  Measured on one
// synthetic 16-expert group (fp4g_parity, NOTES-pfx.md): the kernel is bound by its global->shared copies and its
// ldmatrix/lds feeding about equally; the FP4 MMA itself runs at ~1/3 of its peak.
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill::mmq {
// mmq_nvfp4_a44.cu
void quantize_nvfp4_x2(const float* x, const int32_t* ids, void* xq, float* yscale, int64_t cols, int64_t ld,
                       int64_t rows, int64_t padded, cudaStream_t s);
// the same bytes a warp per row, the padding past cols not written (the FP4 kernel's K ends at cols)
void quantize_nvfp4_x2_rows(const float* x, void* xq, float* yscale, int64_t cols, int64_t ld, int64_t rows, int64_t padded,
                            cudaStream_t s);
// the same sub-blocks' bytes in the K-step-major 144-byte layout of the R144 kernel (cols 640)
void quantize_nvfp4_x2_r144(const float* x, void* xq, float* yscale, int64_t ld, int64_t rows, cudaStream_t s);
}

namespace strata::prefill::fp4g {
namespace {

constexpr int kMaxExperts = 16;
constexpr int kRows = 256;                     // weight rows per block (SwiGLU: 128 gate + their 128 up rows)
constexpr int kTok = 96;                       // activation rows per block (two warp tiles of kTok / 2)
constexpr int kWM = kTok / 2, kMT = kWM / 16;   // a warp's rows and 16-row m-tiles
constexpr int kKT = 128;                       // K values per pipeline stage
constexpr int kStages = 3;
constexpr int kACodes = kTok * kKT / 2;        // 6144: a term's codes, 64 bytes per row
constexpr int kAScales = kTok * kKT / 16;      // 768: its UE4M3 scales, 8 per row
constexpr int kATerm = kACodes + kAScales;     // 6912
constexpr int kAStage = 2 * kATerm;            // both terms
constexpr int kBRowBytes = kKT / 64 * 36;      // 72: two NVFP4 blocks of a weight row
constexpr int kBStride = 72;                   // a weight row's bytes in shared memory
constexpr int kBStage = kRows * kBStride;      // 18432
constexpr int kStage = kAStage + kBStage;      // 32256
constexpr int kSmem = kStages * kStage;        // 96768
constexpr int kThreads = 256;
constexpr int kNT = 8;                         // a warp's 8-row n-tiles (64 weight rows)
constexpr int kOutCols = kRows / 2;            // SwiGLU: the block's outputs
constexpr int kBlk = 144;                      // sizeof(block_fp4_mmq) == sizeof(block_q8_1_mmq)
constexpr int kHsStride = kOutCols + 4;        // the SwiGLU tile's floats per row (float4 rows, fewer conflicts)
static_assert(kTok * kHsStride * 4 <= kSmem, "the epilogue tile reuses the pipeline's shared memory");

enum Epi : int { kPlain = 0, kSwigluQ8 = 1, kSwigluF32 = 2 };

struct Args {
    const uint8_t* w[kMaxExperts];      // expert z's weight rows (NVFP4 blocks, K / 64 x 36 bytes a row)
    const float* tail[kMaxExperts];     // its {s_gate, s_up, s_down, 0} (the SwiGLU epilogues)
    int K;                              // weight columns
    int half;                           // SwiGLU: the gate rows (the up rows follow)
    const uint8_t* y;                   // the activation rows: block_fp4_mmq [kb][ne1]
    int64_t ne1;                        // rows of that layout
    int64_t y2_off;                     // term 2, bytes after term 1
    const float* ys;                    // the row scale of each activation row
    const int32_t* bounds;              // n + 1 row bounds
    const int32_t* yrows;               // row r -> activation row (null: r - y_base)
    int64_t y_base;
    int64_t o_base;                     // output row of row r: r - o_base
    float* dst;                         // kPlain: [rows][ld]; kSwigluF32: H [rows][half]
    int64_t ld;
    uint8_t* hq;                        // kSwigluQ8: block_q8_1_mmq [k_block][hq_rows]
    int64_t hq_rows;
    float* sd;                          // SwiGLU: s_down per output row (null: none)
    int bulk;                           // R144: the stage's rows by one bulk copy (else 16-byte cp.async)
    int flat;                           // >0: the group's experts (this many) as one list of row tiles in grid y
};

__device__ __forceinline__ uint32_t smem_addr(const void* p) { return (uint32_t) __cvta_generic_to_shared(p); }

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

// c += A (16 rows x 64 values, E2M1 with a UE4M3 scale per 16) * B (64 x 8 weight rows, same format)
__device__ __forceinline__ void mma_fp4(float* c, const unsigned* a, unsigned b0, unsigned b1, unsigned sfa,
                                        unsigned sfb) {
    constexpr unsigned short kId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "h"(kId), "h"(kId),
                   "r"(sfb), "h"(kId), "h"(kId));
}

__device__ __forceinline__ void cp_async8(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(smem_addr(dst)), "l"(src));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// the byte of a 64-byte tile row holding logical byte lb of row rr: the four 16-byte chunks rotate with the row so
// that eight rows of an ldmatrix read different banks
__device__ __forceinline__ int swz(int rr, int lb) { return (((lb >> 4) ^ ((rr >> 1) & 3)) << 4) + (lb & 15); }

// The weight row behind B tile row sr of block rb: 256 consecutive rows, or (SwiGLU) the block's 128 gate rows and
// the 128 up rows of the same outputs
template <bool SWIGLU> __device__ __forceinline__ int64_t b_row(int rb, int sr, int half) {
    if (!SWIGLU) return (int64_t) rb * kRows + sr;
    return sr < kOutCols ? (int64_t) rb * kOutCols + sr : (int64_t) half + rb * kOutCols + (sr - kOutCols);
}
// The first B tile row of a warp's n-tile r: with SwiGLU tiles 0-3 are gate rows, 4-7 the up rows of the same outputs
template <bool SWIGLU> __device__ __forceinline__ int b_tile_row(int warp_n, int r) {
    if (!SWIGLU) return warp_n * 64 + r * 8;
    return r < 4 ? warp_n * 32 + r * 8 : kOutCols + warp_n * 32 + (r - 4) * 8;
}

// mmq::swiglu_quant_kernel's and swiglu_scaled_kernel's value: g, u the gate/up product as MMQ stores it (row scale
// times the sum), then the expert's tails
__device__ __forceinline__ float swiglu_h(float g_sum, float u_sum, float rs, float sg, float su) {
    const float g0 = rs * g_sum, u0 = rs * u_sum;
    const float g = g0 * sg, u = u0 * su;
    return g / (1.0f + __expf(-g)) * u;
}

__device__ __forceinline__ void mbar_init(uint64_t* b, unsigned count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(smem_addr(b)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" ::"r"(smem_addr(b)) : "memory");
}
// the arrival comes when every cp.async this thread issued before it has landed (counts against the expected count)
__device__ __forceinline__ void mbar_cp_arrive(uint64_t* b) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(smem_addr(b)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, unsigned parity) {
    asm volatile("{\n .reg .pred P;\n WAIT_%=:\n mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n @!P bra WAIT_%=;\n}\n"
                 ::"r"(smem_addr(b)), "r"(parity) : "memory");
}
// one bulk copy (TMA engine, no tensor map) of `bytes` (16-byte multiple, both ends 16-byte aligned), counted in tx bytes
__device__ __forceinline__ void bulk_copy(void* dst, const void* src, unsigned bytes, uint64_t* b) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 ::"r"(smem_addr(dst)), "l"(src), "r"(bytes), "r"(smem_addr(b)) : "memory");
}
__device__ __forceinline__ void mbar_arrive_tx(uint64_t* b, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_addr(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void consumers_sync() { asm volatile("bar.sync 1, %0;\n" ::"n"(kThreads) : "memory"); }

// Warp-specialized: warps 0-7 multiply (the tile of before), warp 8 issues every global->shared copy.  Stage s has a
// "full" barrier (the producer's 32 lanes arrive as their copies land) and an "empty" one (the 8 consumer warps arrive
// when they are done reading it); no block-wide barrier in the loop, so a stage is refilled as soon as it is free.
// R144: the activation rows are H's two FP4 terms in the K-step-major layout of quantize_nvfp4_x2_r144 (row r of K
// step kt at (kt x ne1 + r) x 144: term 1's 64 code bytes, term 2's, term 1's 8 scales, term 2's): a stage's rows are
// one contiguous run, one bulk copy; in shared memory a row's 144 bytes as they are (a 36-word stride: the eight rows
// of an ldmatrix hit different banks without a swizzle).
template <int EPI, bool R144>
__global__ void __launch_bounds__(kThreads + 128, 1) gemm_kernel(const Args a) {
    constexpr bool SWIGLU = EPI != kPlain;
    extern __shared__ __align__(1024) uint8_t smem[];
    __shared__ int32_t yrow_s[kTok];
    __shared__ float ys_s[kTok];
    __shared__ __align__(8) uint64_t full_b[kStages], empty_b[kStages];
    const int rb = blockIdx.x;
    int z = blockIdx.z, first = blockIdx.y * kTok;   // the expert, the block's first row inside it
    if (a.flat > 0) {
        // grid y walks every expert's ceil(rows / kTok) tiles in turn (no blocks for the empty tail of small experts)
        int t = blockIdx.y;
        for (z = 0; z < a.flat; ++z) {
            const int nt = (a.bounds[z + 1] - a.bounds[z] + kTok - 1) / kTok;
            if (t < nt) break;
            t -= nt;
        }
        if (z == a.flat) return;
        first = t * kTok;
    }
    const int64_t b0r = a.bounds[z];
    const int cnt = a.bounds[z + 1] - (int) b0r;
    if (first >= cnt) return;
    const int nkt = a.K / kKT;
    const size_t row_bytes = (size_t) (a.K / 64) * 36;
    const uint8_t* w_src = a.w[z];
    const int tid = threadIdx.x;
    if (tid < kTok) {
        // a padded row reads the expert's last row (valid memory): its outputs are never stored
        const int64_t r = b0r + (first + tid < cnt ? first + tid : cnt - 1);
        const int32_t yr = a.yrows ? a.yrows[r] : (int32_t) (r - a.y_base);
        yrow_s[tid] = yr;
        ys_s[tid] = a.ys[yr];
    }
    if (tid == 0)
        for (int s = 0; s < kStages; ++s) {
            mbar_init(&full_b[s], R144 && a.bulk ? 129 : 128);
            mbar_init(&empty_b[s], kThreads / 32);
        }
    __syncthreads();

    if (tid >= kThreads) {   // the producer warp
        const int lane = tid - kThreads;
        const int64_t kb_stride = a.ne1 * kBlk;
        for (int kt = 0; kt < nkt; ++kt) {
            const int stage = kt % kStages;
            if (kt >= kStages) mbar_wait(&empty_b[stage], (unsigned) ((kt / kStages - 1) & 1));
            uint8_t* as = smem + stage * kStage;
            const int64_t ko = (int64_t) (kt >> 1) * kb_stride;
            const int h = kt & 1;
            if constexpr (R144) {
                const int64_t r0 = b0r + first;
                const int64_t n = a.ne1 - r0 < kTok ? a.ne1 - r0 : kTok;   // rows past the end: not read (not stored)
                const uint8_t* src = a.y + ((int64_t) kt * a.ne1 + r0) * kBlk;
                if (a.bulk) {
                    if (lane == 0) {
                        mbar_arrive_tx(&full_b[stage], (unsigned) (n * kBlk));
                        bulk_copy(as, src, (unsigned) (n * kBlk), &full_b[stage]);
                    }
                } else {
                    for (int c = lane; c < (int) n * (kBlk / 16); c += 128) cp_async16(as + 16 * c, src + 16 * c);
                }
            }
            // activation codes: 2 terms x kTok rows x 4 chunks of 16 bytes, then the 8-byte scale runs
            if (!R144)
            for (int c = lane; c < 2 * kTok * 4; c += 128) {
                const int term = c / (4 * kTok), rr = c % (4 * kTok), r = rr >> 2, q = rr & 3;
                cp_async16(as + term * kATerm + r * 64 + ((q ^ ((r >> 1) & 3)) << 4),
                           a.y + (term ? a.y2_off : 0) + (int64_t) yrow_s[r] * kBlk + 16 + 16 * q + ko + 64 * h);
            }
            if (!R144)
            for (int c = lane; c < 2 * kTok; c += 128) {
                const int term = c / kTok, r = c % kTok;
                cp_async8(as + term * kATerm + kACodes + r * 8,
                          a.y + (term ? a.y2_off : 0) + (int64_t) yrow_s[r] * kBlk + ko + 8 * h);
            }
            // weights: kRows rows x 9 pieces of 8 bytes
            uint8_t* bs = as + kAStage;
            const size_t bo = (size_t) kt * kBRowBytes;
            for (int i = lane; i < kRows * (kBRowBytes / 8); i += 128) {
                const int row = i / (kBRowBytes / 8), c = i % (kBRowBytes / 8);
                cp_async8(bs + row * kBStride + 8 * c, w_src + (size_t) b_row<SWIGLU>(rb, row, a.half) * row_bytes + bo + 8 * c);
            }
            mbar_cp_arrive(&full_b[stage]);
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        return;
    }

    const int lane = tid & 31, warp = tid >> 5;
    const int warp_m = warp >> 2, warp_n = warp & 3;   // kWM-row tile x 64 weight rows
    const bool live = first + warp_m * kWM < cnt;
    const int group = lane >> 2, tig = lane & 3;
    const int a_row = (lane & 7) + (((lane >> 3) & 1) << 3);
    const int a_col = (lane >> 4) * 16;
    const int sfa_row = ((lane & 1) << 3) | (lane >> 2);

    float acc[kMT][kNT][4] = {};
    for (int kt = 0; kt < nkt; ++kt) {
        const int stage = kt % kStages;
        mbar_wait(&full_b[stage], (unsigned) ((kt / kStages) & 1));
        if (live) {
            const uint8_t* st = smem + stage * kStage;
            const uint8_t* bs = st + kAStage;
#pragma unroll
            for (int s = 0; s < 2; ++s) {
                unsigned b0[kNT], b1[kNT], sb[kNT];
#pragma unroll
                for (int r = 0; r < kNT; ++r) {
                    const uint8_t* p = bs + (b_tile_row<SWIGLU>(warp_n, r) + group) * kBStride + s * 36;
                    b0[r] = *(const uint32_t*) (p + 4 + 4 * tig);
                    b1[r] = *(const uint32_t*) (p + 20 + 4 * tig);
                    sb[r] = *(const uint32_t*) p;   // the scale bytes as MMQ passes them
                }
#pragma unroll
                for (int term = 0; term < 2; ++term) {
                    const uint8_t* as = st + term * kATerm;
                    unsigned af[kMT][4], asc[kMT];
#pragma unroll
                    for (int mt = 0; mt < kMT; ++mt) {
                        const int row = warp_m * kWM + mt * 16 + a_row, srow = warp_m * kWM + mt * 16 + sfa_row;
                        if constexpr (R144) {
                            ldmatrix_x4(af[mt][0], af[mt][1], af[mt][2], af[mt][3],
                                        smem_addr(st + row * kBlk + term * 64 + s * 32 + a_col));
                            asc[mt] = *(const uint32_t*) (st + srow * kBlk + 128 + term * 8 + 4 * s);
                        } else {
                            ldmatrix_x4(af[mt][0], af[mt][1], af[mt][2], af[mt][3],
                                        smem_addr(as + row * 64 + swz(row, s * 32 + a_col)));
                            asc[mt] = *(const uint32_t*) (as + kACodes + srow * 8 + 4 * s);
                        }
                    }
#pragma unroll
                    for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
                        for (int r = 0; r < kNT; ++r) mma_fp4(acc[mt][r], af[mt], b0[r], b1[r], asc[mt], sb[r]);
                }
            }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty_b[stage]);
    }

    if constexpr (EPI == kPlain) {
        if (!live) return;
#pragma unroll
        for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int t = warp_m * kWM + mt * 16 + group + 8 * h;
                if (first + t >= cnt) continue;
                const float rs = ys_s[t];
                float* out = a.dst + (b0r + first + t - a.o_base) * a.ld + rb * kRows + warp_n * 64 + 2 * tig;
#pragma unroll
                for (int r = 0; r < kNT; ++r)
                    *(float2*) (out + 8 * r) = make_float2(rs * acc[mt][r][2 * h], rs * acc[mt][r][2 * h + 1]);
            }
        return;
    } else {
        const float4 tl = *(const float4*) a.tail[z];
        if (a.sd != nullptr && rb == 0 && tid < kTok && first + tid < cnt) a.sd[b0r + first + tid - a.o_base] = tl.z;
        if constexpr (EPI == kSwigluF32) {
            if (!live) return;
#pragma unroll
            for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int t = warp_m * kWM + mt * 16 + group + 8 * h;
                    if (first + t >= cnt) continue;
                    const float rs = ys_s[t];
                    float* out = a.dst + (b0r + first + t - a.o_base) * a.ld + rb * kOutCols + warp_n * 32 + 2 * tig;
#pragma unroll
                    for (int r = 0; r < kNT / 2; ++r)
                        *(float2*) (out + 8 * r) =
                            make_float2(swiglu_h(acc[mt][r][2 * h], acc[mt][r + 4][2 * h], rs, tl.x, tl.y),
                                        swiglu_h(acc[mt][r][2 * h + 1], acc[mt][r + 4][2 * h + 1], rs, tl.x, tl.y));
                }
            return;
        } else {
            // the block's kTok x 128 values of H through shared memory, then per row and 32-value block:
            // swiglu_quant_kernel's q8_1 (127 / amax, roundf, d = 1 / d_inv)
            consumers_sync();   // every consumer warp is past its last read of the pipeline's tiles
            float* hs = (float*) smem;
            if (live) {
#pragma unroll
                for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int t = warp_m * kWM + mt * 16 + group + 8 * h;
                        const float rs = ys_s[t];
                        float* row = hs + t * kHsStride + warp_n * 32 + 2 * tig;
#pragma unroll
                        for (int r = 0; r < kNT / 2; ++r)
                            *(float2*) (row + 8 * r) =
                                make_float2(swiglu_h(acc[mt][r][2 * h], acc[mt][r + 4][2 * h], rs, tl.x, tl.y),
                                            swiglu_h(acc[mt][r][2 * h + 1], acc[mt][r + 4][2 * h + 1], rs, tl.x, tl.y));
                    }
            }
            consumers_sync();
#pragma unroll
            for (int it = 0; it < 2; ++it) {
                const int task = tid + it * kThreads, t = task >> 2, blk = task & 3;
                if (t >= kTok || first + t >= cnt) continue;
                const float* v = hs + t * kHsStride + blk * 32;
                float amax = 0.0f;
#pragma unroll
                for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
                const float d_inv = 127.0f / amax;
                const int col0 = rb * kOutCols + blk * 32;
                const int64_t o = b0r + first + t - a.o_base;
                uint8_t* yb = a.hq + ((int64_t) (col0 / 128) * a.hq_rows + o) * kBlk;
                const int iqs = col0 % 128;
                uint32_t q[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    char4 c;
                    c.x = roundf(v[4 * i + 0] * d_inv);
                    c.y = roundf(v[4 * i + 1] * d_inv);
                    c.z = roundf(v[4 * i + 2] * d_inv);
                    c.w = roundf(v[4 * i + 3] * d_inv);
                    q[i] = *(const uint32_t*) &c;
                }
                uint4* qd = (uint4*) (yb + 16 + iqs);   // block_q8_1_mmq: d4[4] first, then qs[128]
                qd[0] = make_uint4(q[0], q[1], q[2], q[3]);
                qd[1] = make_uint4(q[4], q[5], q[6], q[7]);
                ((float*) yb)[iqs / 32] = 1.0f / d_inv;
            }
        }
    }
}

// strata-nvfp4 (pfx round 3): the same tile and MMA sequence as gemm_kernel, persistent: one block per SM walks the
// group's tiles (x the weight-row block, y the flat list of expert row tiles), the producer warps run on into the
// next tile's stages while the MMA warps store the last one's outputs (no prologue / epilogue bubble per tile: K is
// only 5 stages for down).  The plain and the SwiGLU-F32 epilogues (registers -> global); per accumulator the same
// sums in the same order as gemm_kernel.
struct Tile { int z, first, rb; };
// tile i of the walk: rb = i % nx, the expert row tile i / nx of the flat list; false past the list's end
__device__ __forceinline__ bool tile_at(const Args& a, int i, int nx, Tile& t) {
    t.rb = i % nx;
    int y = i / nx;
    for (int z = 0; z < a.flat; ++z) {
        const int nt = (a.bounds[z + 1] - a.bounds[z] + kTok - 1) / kTok;
        if (y < nt) { t.z = z; t.first = y * kTok; return true; }
        y -= nt;
    }
    return false;
}
__device__ __forceinline__ void producers_sync() { asm volatile("bar.sync 2, 128;\n" ::: "memory"); }

template <int EPI>
__global__ void __launch_bounds__(kThreads + 128, 1) gemm_persist(const Args a, const int nx, const int ntiles) {
    constexpr bool SWIGLU = EPI != kPlain;
    extern __shared__ __align__(1024) uint8_t smem[];
    __shared__ int32_t yrow_p[kTok];   // the producers' row table of the tile they load
    __shared__ __align__(8) uint64_t full_b[kStages], empty_b[kStages];
    const int nkt = a.K / kKT;
    const size_t row_bytes = (size_t) (a.K / 64) * 36;
    const int tid = threadIdx.x;
    if (tid == 0)
        for (int s = 0; s < kStages; ++s) {
            mbar_init(&full_b[s], 128);
            mbar_init(&empty_b[s], kThreads / 32);
        }
    __syncthreads();

    if (tid >= kThreads) {   // producer warps
        const int lane = tid - kThreads;
        const int64_t kb_stride = a.ne1 * kBlk;
        int g = 0;   // the stage counter over every tile of this block
        for (int i = blockIdx.x; i < ntiles; i += gridDim.x) {
            Tile t;
            if (!tile_at(a, i, nx, t)) break;
            const int64_t b0r = a.bounds[t.z];
            const int cnt = a.bounds[t.z + 1] - (int) b0r;
            producers_sync();   // every producer is done with the last tile's yrow_p
            if (lane < kTok) {
                const int64_t r = b0r + (t.first + lane < cnt ? t.first + lane : cnt - 1);
                yrow_p[lane] = a.yrows ? a.yrows[r] : (int32_t) (r - a.y_base);
            }
            producers_sync();
            const uint8_t* w_src = a.w[t.z];
            for (int kt = 0; kt < nkt; ++kt, ++g) {
                const int stage = g % kStages;
                if (g >= kStages) mbar_wait(&empty_b[stage], (unsigned) ((g / kStages - 1) & 1));
                uint8_t* as = smem + stage * kStage;
                const int64_t ko = (int64_t) (kt >> 1) * kb_stride;
                const int h = kt & 1;
                for (int c = lane; c < 2 * kTok * 4; c += 128) {
                    const int term = c / (4 * kTok), rr = c % (4 * kTok), r = rr >> 2, q = rr & 3;
                    cp_async16(as + term * kATerm + r * 64 + ((q ^ ((r >> 1) & 3)) << 4),
                               a.y + (term ? a.y2_off : 0) + (int64_t) yrow_p[r] * kBlk + 16 + 16 * q + ko + 64 * h);
                }
                for (int c = lane; c < 2 * kTok; c += 128) {
                    const int term = c / kTok, r = c % kTok;
                    cp_async8(as + term * kATerm + kACodes + r * 8,
                              a.y + (term ? a.y2_off : 0) + (int64_t) yrow_p[r] * kBlk + ko + 8 * h);
                }
                uint8_t* bs = as + kAStage;
                const size_t bo = (size_t) kt * kBRowBytes;
                for (int j = lane; j < kRows * (kBRowBytes / 8); j += 128) {
                    const int row = j / (kBRowBytes / 8), c = j % (kBRowBytes / 8);
                    cp_async8(bs + row * kBStride + 8 * c, w_src + (size_t) b_row<SWIGLU>(t.rb, row, a.half) * row_bytes + bo + 8 * c);
                }
                mbar_cp_arrive(&full_b[stage]);
            }
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        return;
    }

    const int lane = tid & 31, warp = tid >> 5;
    const int warp_m = warp >> 2, warp_n = warp & 3;
    const int group = lane >> 2, tig = lane & 3;
    const int a_row = (lane & 7) + (((lane >> 3) & 1) << 3);
    const int a_col = (lane >> 4) * 16;
    const int sfa_row = ((lane & 1) << 3) | (lane >> 2);
    int g = 0;
    for (int i = blockIdx.x; i < ntiles; i += gridDim.x) {
        Tile t;
        if (!tile_at(a, i, nx, t)) break;
        const int64_t b0r = a.bounds[t.z];
        const int cnt = a.bounds[t.z + 1] - (int) b0r;
        const bool live = t.first + warp_m * kWM < cnt;
        float acc[kMT][kNT][4] = {};
        for (int kt = 0; kt < nkt; ++kt, ++g) {
            const int stage = g % kStages;
            mbar_wait(&full_b[stage], (unsigned) ((g / kStages) & 1));
            if (live) {
                const uint8_t* st = smem + stage * kStage;
                const uint8_t* bs = st + kAStage;
#pragma unroll
                for (int s = 0; s < 2; ++s) {
                    unsigned b0[kNT], b1[kNT], sb[kNT];
#pragma unroll
                    for (int r = 0; r < kNT; ++r) {
                        const uint8_t* p = bs + (b_tile_row<SWIGLU>(warp_n, r) + group) * kBStride + s * 36;
                        b0[r] = *(const uint32_t*) (p + 4 + 4 * tig);
                        b1[r] = *(const uint32_t*) (p + 20 + 4 * tig);
                        sb[r] = *(const uint32_t*) p;
                    }
#pragma unroll
                    for (int term = 0; term < 2; ++term) {
                        const uint8_t* as = st + term * kATerm;
                        unsigned af[kMT][4], asc[kMT];
#pragma unroll
                        for (int mt = 0; mt < kMT; ++mt) {
                            const int row = warp_m * kWM + mt * 16 + a_row;
                            ldmatrix_x4(af[mt][0], af[mt][1], af[mt][2], af[mt][3],
                                        smem_addr(as + row * 64 + swz(row, s * 32 + a_col)));
                            asc[mt] = *(const uint32_t*) (as + kACodes + (warp_m * kWM + mt * 16 + sfa_row) * 8 + 4 * s);
                        }
#pragma unroll
                        for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
                            for (int r = 0; r < kNT; ++r) mma_fp4(acc[mt][r], af[mt], b0[r], b1[r], asc[mt], sb[r]);
                    }
                }
            }
            __syncwarp();
            if (lane == 0) mbar_arrive(&empty_b[stage]);
        }
        // the outputs straight from the registers (the producers already fill the next tile's stages)
        float4 tl = make_float4(0.f, 0.f, 0.f, 0.f);
        if constexpr (SWIGLU) {
            tl = *(const float4*) a.tail[t.z];
            if (a.sd != nullptr && t.rb == 0 && tid < kTok && t.first + tid < cnt) a.sd[b0r + t.first + tid - a.o_base] = tl.z;
        }
        if (!live) continue;
#pragma unroll
        for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int tr = warp_m * kWM + mt * 16 + group + 8 * h;
                if (t.first + tr >= cnt) continue;
                const int64_t r = b0r + t.first + tr;
                const float rs = a.ys[a.yrows ? a.yrows[r] : (int32_t) (r - a.y_base)];
                if constexpr (EPI == kPlain) {
                    float* out = a.dst + (r - a.o_base) * a.ld + t.rb * kRows + warp_n * 64 + 2 * tig;
#pragma unroll
                    for (int rr = 0; rr < kNT; ++rr)
                        *(float2*) (out + 8 * rr) = make_float2(rs * acc[mt][rr][2 * h], rs * acc[mt][rr][2 * h + 1]);
                } else {
                    float* out = a.dst + (r - a.o_base) * a.ld + t.rb * kOutCols + warp_n * 32 + 2 * tig;
#pragma unroll
                    for (int rr = 0; rr < kNT / 2; ++rr)
                        *(float2*) (out + 8 * rr) =
                            make_float2(swiglu_h(acc[mt][rr][2 * h], acc[mt][rr + 4][2 * h], rs, tl.x, tl.y),
                                        swiglu_h(acc[mt][rr][2 * h + 1], acc[mt][rr + 4][2 * h + 1], rs, tl.x, tl.y));
                }
            }
    }
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill fp4g: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <int EPI, bool R144 = false> void launch(const Args& a, dim3 grid, cudaStream_t s) {
    static std::atomic<bool> configured[64];
    int dev = 0;
    ck(cudaGetDevice(&dev), "device");
    if (dev < 0 || dev >= 64) ck(cudaErrorInvalidDevice, "device range");
    if (!configured[dev].load(std::memory_order_acquire)) {
        ck(cudaFuncSetAttribute(gemm_kernel<EPI, R144>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem), "smem");
        configured[dev].store(true, std::memory_order_release);
    }
    gemm_kernel<EPI, R144><<<grid, kThreads + 128, kSmem, s>>>(a);
    ck(cudaGetLastError(), "gemm");
}

// the persistent kernel over the group's tiles: one block per SM (or fewer when there are fewer tiles)
template <int EPI> void launch_persist(const Args& a, int nx, int ntiles, cudaStream_t s) {
    static std::atomic<bool> configured[64];
    static int sms[64] = {};
    int dev = 0;
    ck(cudaGetDevice(&dev), "device");
    if (dev < 0 || dev >= 64) ck(cudaErrorInvalidDevice, "device range");
    if (!configured[dev].load(std::memory_order_acquire)) {
        ck(cudaFuncSetAttribute(gemm_persist<EPI>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem), "smem");
        ck(cudaDeviceGetAttribute(&sms[dev], cudaDevAttrMultiProcessorCount, dev), "SMs");
        configured[dev].store(true, std::memory_order_release);
    }
    const int grid = ntiles < sms[dev] ? ntiles : sms[dev];
    gemm_persist<EPI><<<grid, kThreads + 128, kSmem, s>>>(a, nx, ntiles);
    ck(cudaGetLastError(), "gemm persist");
}

__global__ void probe_kernel(int* out) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
    *out = 1;
#else
    *out = 0;
#endif
}

}  // namespace

bool available() {
    static const bool ok = [] {
        int* d = nullptr;
        int h = 0;
        if (cudaMalloc(&d, sizeof(int)) != cudaSuccess) { (void) cudaGetLastError(); return false; }
        probe_kernel<<<1, 1>>>(d);
        const bool ran = cudaGetLastError() == cudaSuccess && cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess;
        (void) cudaGetLastError();
        cudaFree(d);
        return ran && h == 1;
    }();
    return ok;
}

// see moe_mmq.hpp (mmq::Fp4gGroup); n_embd 2560, n_ff 640
int run_group(int n, const void* const* gu, const void* const* dn, const float* const* tail, const int32_t* bounds_gu,
              const int32_t* bounds_d, int64_t r0, int64_t nr, int64_t max_rows, const void* xq, const float* xscale,
              const int32_t* yrows, int64_t x_rows, float* h, void* hq, float* hscale, float* sd, float* dm, int mode,
              void* stream) {
    constexpr int kN = 2560, kFF = 640;
    if (n <= 0 || n > kMaxExperts || nr <= 0 || max_rows <= 0) return 0;
    uintptr_t aw = 0, at = 0;   // cp.async 8 reads the weight rows in 8-byte pieces; the tail is one float4
    for (int i = 0; i < n; ++i) {
        aw |= (uintptr_t) gu[i] | (mode == 2 ? (uintptr_t) dn[i] : 0);
        at |= (uintptr_t) tail[i];
    }
    if (aw % 8 != 0 || at % 16 != 0) {
        static bool said = false;
        if (!said) { said = true; std::fprintf(stderr, "strata: STRATA_PF_FP4G: an expert blob is not aligned for the FP4 kernel; such groups take MMQ\n"); }
        return 0;
    }
    if (mode == 2 && h == nullptr) return 0;
    const cudaStream_t s = (cudaStream_t) stream;
    // STRATA_PF_FP4G_FLAT=0: grid y = the largest expert's tiles for every expert (round 2); else one list of tiles
    static const bool flat = [] { const char* v = std::getenv("STRATA_PF_FP4G_FLAT"); return v == nullptr || v[0] != '0'; }();
    const unsigned ty = flat ? (unsigned) ((nr + kTok - 1) / kTok + n) : (unsigned) ((max_rows + kTok - 1) / kTok);
    const unsigned tz = flat ? 1u : (unsigned) n;
    // STRATA_PF_FP4G_PERSIST (with the flat list): bit 1 the down product, bit 2 mode 2's gate/up on the persistent
    // kernel; default 1 (64K nsys: down 364 -> 326 ms; gate/up 519 -> 542, so it stays one block per tile)
    static const int persist = [] { const char* v = std::getenv("STRATA_PF_FP4G_PERSIST"); return v != nullptr ? std::atoi(v) : 1; }();
    const int pdown = flat ? persist & 1 : 0, pgu = flat ? persist & 2 : 0;
    Args g{};
    for (int i = 0; i < n; ++i) { g.w[i] = (const uint8_t*) gu[i]; g.tail[i] = tail[i]; }
    g.K = kN; g.half = kFF;
    g.y = (const uint8_t*) xq; g.ne1 = x_rows;
    g.y2_off = (int64_t) (((kN + 511) / 512 * 512) / 256) * x_rows * kBlk;   // quantize_nvfp4_x2: blocks_per_col x ne1
    g.ys = xscale; g.bounds = bounds_gu; g.yrows = yrows; g.y_base = 0; g.o_base = r0;
    g.sd = sd;
    g.flat = flat ? n : 0;
    if (mode == 1) {
        g.hq = (uint8_t*) hq; g.hq_rows = nr;
        launch<kSwigluQ8>(g, dim3(kFF / kOutCols, ty, tz), s);
        return 1;
    }
    g.dst = h; g.ld = kFF;
    if (pgu) launch_persist<kSwigluF32>(g, kFF / kOutCols, (int) (kFF / kOutCols * ty), s);
    else launch<kSwigluF32>(g, dim3(kFF / kOutCols, ty, tz), s);
    // H to two FP4 terms per row (the rows padded to 1024 values: 4 blocks a term), then down
    // the same bytes three ways: quantize_nvfp4_x2_rows (default: a thread per sub-block; STRATA_PF_FP4G_QOLD=2 a warp
    // per row), STRATA_PF_FP4G_QOLD=1 quantize_nvfp4_x2 itself, STRATA_PF_FP4G_R144=1 the 144-byte K-step layout read
    // by the R144 down kernel (faster kernels in nsys but a slower prompt in the A/B: see NOTES-pfx.md)
    static const int old_q = [] { const char* v = std::getenv("STRATA_PF_FP4G_QOLD"); return v != nullptr ? std::atoi(v) : 0; }();
    static const bool r144 = [] { const char* v = std::getenv("STRATA_PF_FP4G_R144"); return v != nullptr && v[0] == '1'; }();
    if (old_q == 1) strata::prefill::mmq::quantize_nvfp4_x2(h, nullptr, hq, hscale, kFF, kFF, nr, (kFF + 511) / 512 * 512, s);
    else if (r144) strata::prefill::mmq::quantize_nvfp4_x2_r144(h, hq, hscale, kFF, nr, s);
    else strata::prefill::mmq::quantize_nvfp4_x2_rows(h, hq, hscale, kFF, kFF, nr, (kFF + 511) / 512 * 512, s);
    ck(cudaGetLastError(), "quantize H");
    Args d{};
    for (int i = 0; i < n; ++i) { d.w[i] = (const uint8_t*) dn[i]; d.tail[i] = tail[i]; }
    d.K = kFF; d.half = 0;
    d.y = (const uint8_t*) hq; d.ne1 = nr;
    d.y2_off = (int64_t) (((kFF + 511) / 512 * 512) / 256) * nr * kBlk;
    d.ys = hscale; d.bounds = bounds_d; d.yrows = nullptr; d.y_base = 0; d.o_base = 0;
    d.dst = dm; d.ld = kN;
    static const int bulk = [] { const char* v = std::getenv("STRATA_PF_FP4G_BULK"); return v != nullptr && v[0] == '1'; }();
    d.bulk = bulk;
    d.flat = flat ? n : 0;
    if (r144 && old_q != 1) launch<kPlain, true>(d, dim3(kN / kRows, ty, tz), s);
    else if (pdown) launch_persist<kPlain>(d, kN / kRows, (int) (kN / kRows * ty), s);
    else launch<kPlain>(d, dim3(kN / kRows, ty, tz), s);
    return 2;
}

}  // namespace strata::prefill::fp4g
