// v5 `down`: v4's sums, split along K by v4's lane bits 1..log2 P.  v4: a warp per row, lane l accumulates chunks
// l + 32 q (q = 0..9) in order, then warp_sum (xor 16, 8, 4, 2, 1).  Part p of P (a cluster of P blocks) owns the
// lanes with (l >> 1) mod P = p: chunk pairs (32 bytes: whole sectors) of 1/P of the stream.  It runs v4's per-lane
// sums and the xor steps over the lane bits above log2 P + 1 (v4's order); the cluster finishes over distributed
// shared memory: the part bits (distance P/2 first, as warp_sum's xor P .. 2) for each lane bit 0, then bit 0 (xor 1).
// Every sum is v4's own + partner, so `part` and `ssg` are v4's bits; the sums of squares likewise (v4 thread tt
// owns float4s tt + 256 u, all in part (tt >> 2) mod P: lane bits 2..log2 P + 1, then 1 and 0 after the parts).
// 128 threads: 4 warps; a warp = 32/P v4 lanes x P row slots, 2 rows a slot (8P rows a block); 40/P row groups + 1
// inject group (rows LR..LR+3) per part and stream.  Activations: cp.async of R and bo for all T tokens at once into
// v4's two-plane layout (of this part's chunks), transformed in place by v4's norm threads.
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__device__ __forceinline__ void cp16(void* sdst, const void* gsrc) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(sdst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gsrc) : "memory");   // .cg measured 1.5x slower here
}
__device__ __forceinline__ void cp_wait_all() {
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}

constexpr int D5_THREADS = 128, D5_WARPS = 4;
template <int P> __host__ __device__ constexpr int d5_lg() { return P == 2 ? 1 : P == 4 ? 2 : 3; }
template <int P> __host__ __device__ constexpr int d5_rows() { return 8 * P; }              // rows per block
template <int P> __host__ __device__ constexpr int d5_nrg() { return LR / d5_rows<P>(); }   // row groups per stream
template <int P> __host__ __device__ constexpr size_t d5_smem(int T) { return (size_t) T * 2 * (N / P) * 4; }

// the cross-part butterfly of P values (distance P/2 first)
template <int P> __device__ __forceinline__ float d5_tree(float (&v)[P]) {
#pragma unroll
    for (int h = P / 2; h > 0; h >>= 1)
#pragma unroll
        for (int q = 0; q < h; ++q) v[q] = v[q] + v[q + h];
    return v[0];
}

__device__ long long g_stamp[256 * 10];
template <int MAX_T, int P, bool PDL, int DIAG = 0>
__global__ void __cluster_dims__(P, 1, 1) __launch_bounds__(D5_THREADS)
gr_down_v5_kernel(GrMulti m, float* __restrict__ part, float* __restrict__ ssg) {
    constexpr int LG = d5_lg<P>(), VS = 32 / P;          // v4 lanes per part
    constexpr int NJ = N / 8 / P;                         // chunks of this part (320 / P)
    constexpr int SLP = 2 * NJ;                           // float4s of this part per token (two planes of NJ)
    constexpr int RPB = d5_rows<P>(), NRG = d5_nrg<P>();
    constexpr int TQS = N / 8 / 32;                       // 10 chunks per lane
    constexpr int NTH = 256 / P;                          // v4's norm threads in this part
    constexpr int NU = 3;                                 // float4s per v4 norm thread (tt + 256 u < 640)
    constexpr int T = MAX_T;
    static_assert(LR % RPB == 0 && NTH <= D5_THREADS && VS >= 2, "geometry");
    extern __shared__ __align__(16) float4 sm5[];        // [T][SLP] activations, then [T][SLP] bo
    __shared__ float red[RPB][MAX_T][2];                  // per-part sums of this block's rows, per v4 lane bit 0
    __shared__ float ssx[MAX_T][8][4];                    // per-part sums of squares: [token][v4 warp][lane bits 1, 0]
    cg::cluster_group cl = cg::this_cluster();
    long long st0 = 0;
    if constexpr ((DIAG & 32) != 0) st0 = clock64();
#define D5STAMP(ph) do { if constexpr ((DIAG & 32) != 0) { if (threadIdx.x == 0) g_stamp[(blockIdx.y * gridDim.x + blockIdx.x) * 10 + (ph)] = clock64() - st0; } } while (0)
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int p = (int) cl.block_rank();
    const int rg = blockIdx.x / P, c = blockIdx.y;
    const bool inject = rg == NRG;
    const int i = lane % VS, sub = lane / VS;             // lane slot and row slot
    const int vl = ((i >> 1) << (LG + 1)) | (p << 1) | (i & 1);   // the v4 lane this thread plays
    const int slot = warp * P + sub;                      // 0 .. 4P - 1
    const int nrows = inject ? (slot < HC ? 1 : 0) : 2;
    const uint16_t* wbase = inject ? m.a[0].w_inject : m.a[0].w_down;
    const int row0 = inject ? slot : rg * RPB + slot * 2;
    // every token's parameters up front (constant-bank loads, issued together)
    const float* Rp[MAX_T];
    const float* Bp[MAX_T];
    const float* Ip[MAX_T];
    bool ap[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        ap[k] = m.a[k].apply;
        Rp[k] = m.a[k].R + (size_t) c * N;
        Bp[k] = ap[k] ? m.a[k].bo_prev : Rp[k];
        Ip[k] = ap[k] ? m.a[k].inj_prev + c : Rp[k];
    }
    uint4 wq[2][TQS];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        if (r >= nrows) break;
        const uint4* w4 = reinterpret_cast<const uint4*>(wbase + (size_t) (row0 + r) * D + (size_t) c * N);
#pragma unroll
        for (int q = 0; q < TQS; ++q) wq[r][q] = (DIAG & 256) ? make_uint4(q, r, 0, 0) : __ldg(w4 + vl + 32 * q);
    }
    // v4's norm thread tt of this part: tt bits 0-1 = t bits 0-1, bits 2..LG+1 = p, bits LG+2.. = t bits 2..
    const int tt = (t & 3) | (p << 2) | ((t >> 2) << (LG + 2));
    const bool nth = t < NTH;
    float4 gv[NU];
    {
        const float4* G4 = reinterpret_cast<const float4*>(m.a[0].w_norm + (size_t) c * N);
#pragma unroll
        for (int u = 0; u < NU; ++u)
            if (nth && tt + u * 256 < N / 4) gv[u] = __ldg(G4 + tt + u * 256);
    }
    // float4 f of the stream (chunk j = f >> 1, plane f & 1) sits at plane (f & 1), index 2 ((j >> 1) >> LG) + (j & 1)
    auto sidx = [](int f) { const int j = f >> 1; return (f & 1) * NJ + 2 * ((j >> 1) >> LG) + (j & 1); };
    if constexpr (PDL) {
        gr_pdl_launch_dependents();
        gr_pdl_wait();
        D5STAMP(1);
    }
    float4* xs = sm5;
    float4* bs = sm5 + (size_t) T * SLP;
    D5STAMP(2);
    float gwk[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) gwk[k] = __ldg(Ip[k]);
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        const float g = 2.0f * sigmoidf_(gwk[k] / (float) HC);
        gwk[k] = ap[k] ? g : 0.0f;
    }
    float ssp[MAX_T];
    if constexpr ((DIAG & 1024) != 0) {
        // ld.global.nc in batches of KB tokens (all of a batch's loads before its stores), transformed on the way in;
        // no branch on a token's `apply` (bo is R again without a pending write; the write selected)
        constexpr int KB = 2;
#pragma unroll
        for (int k0 = 0; k0 < MAX_T; k0 += KB) {
            float4 rr[KB][NU], bb[KB][NU];
#pragma unroll
            for (int kk = 0; kk < KB; ++kk) {
                const int k = k0 + kk;
                if (k >= MAX_T) break;
                const float4* R4 = reinterpret_cast<const float4*>(Rp[k]);
                const float4* B4 = reinterpret_cast<const float4*>(Bp[k]);
#pragma unroll
                for (int u = 0; u < NU; ++u) {
                    const int f = tt + u * 256;
                    if (nth && f < N / 4) { rr[kk][u] = __ldg(R4 + f); bb[kk][u] = __ldg(B4 + f); }
                }
            }
#pragma unroll
            for (int kk = 0; kk < KB; ++kk) {
                const int k = k0 + kk;
                if (k >= MAX_T) break;
                ssp[k] = 0.0f;
#pragma unroll
                for (int u = 0; u < NU; ++u) {
                    const int f = tt + u * 256;
                    if (!nth || f >= N / 4) break;
                    float4 r = rr[kk][u];
                    const float4 b = bb[kk][u];
                    const float gw = gwk[k];
                    const float4 ra = make_float4(fmaf(b.x, gw, r.x), fmaf(b.y, gw, r.y), fmaf(b.z, gw, r.z), fmaf(b.w, gw, r.w));
                    if (ap[k]) r = ra;
                    const float4 g = gv[u];
                    ssp[k] += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
                    xs[k * SLP + sidx(f)] = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
                }
            }
        }
        D5STAMP(3);
    } else {
    // cp.async of R and bo for all tokens (bo is R again without a pending write), then transformed in place
    if (nth) {
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            const float4* R4 = reinterpret_cast<const float4*>(Rp[k]);
            const float4* B4 = reinterpret_cast<const float4*>(Bp[k]);
#pragma unroll
            for (int u = 0; u < NU; ++u) {
                const int f = tt + u * 256;
                if (f >= N / 4) break;
                if (DIAG & 4) continue;
                cp16(xs + k * SLP + sidx(f), R4 + f);
                cp16(bs + k * SLP + sidx(f), B4 + f);
            }
        }
    }
    cp_wait_all();
    D5STAMP(3);
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        ssp[k] = 0.0f;
#pragma unroll
        for (int u = 0; u < NU; ++u) {
            const int f = tt + u * 256;
            if (!nth || f >= N / 4) break;
            const int e = k * SLP + sidx(f);
            float4 r = xs[e];
            const float4 b = bs[e];
            const float gw = gwk[k];
            const float4 ra = make_float4(fmaf(b.x, gw, r.x), fmaf(b.y, gw, r.y), fmaf(b.z, gw, r.z), fmaf(b.w, gw, r.w));
            if (ap[k]) r = ra;
            const float4 g = gv[u];
            ssp[k] += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
            xs[e] = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
        }
    }
    }
    const bool ss_block = rg == 0;
    if (ss_block) {
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            float v = ssp[k];
            // v4 lane bits 4 .. LG+2 = t bits 4-LG .. 2
#pragma unroll
            for (int b = 4; b >= LG + 2; --b) v += __shfl_xor_sync(0xffffffffu, v, 1 << (b - LG));
            if (nth && ((t >> 2) & ((1 << (3 - LG)) - 1)) == 0) ssx[k][tt >> 5][t & 3] = v;
        }
    }
    __syncthreads();
    D5STAMP(4);
    float acc[2][MAX_T];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) acc[r][k] = 0.0f;
    if (nrows > 0) {
#pragma unroll
        for (int q = 0; q < TQS; ++q) {
            const int jl = i + VS * q;
            Bf16x8 w[2];
            w[0] = unpack8(wq[0][q]);
            w[1] = inject ? w[0] : unpack8(wq[1][q]);
            float4 x0[MAX_T], x1[MAX_T];
#pragma unroll
            for (int k = 0; k < MAX_T; ++k) { x0[k] = xs[k * SLP + jl]; x1[k] = xs[k * SLP + NJ + jl]; }
            if constexpr ((DIAG & 64) != 0) {
#pragma unroll
                for (int k = 0; k < MAX_T; ++k) {
                    acc[0][k] += dot8u(w[0], x0[k], x1[k]);
                    if (!inject) acc[1][k] += dot8u(w[1], x0[k], x1[k]);
                }
            } else {
                // dot8u of every (row, token) at once: each chain's FMAs in dot8u's order, the chains interleaved
                float d[2][MAX_T];
#pragma unroll
                for (int r = 0; r < 2; ++r)
#pragma unroll
                    for (int k = 0; k < MAX_T; ++k) d[r][k] = 0.0f;
#define D5STEP(WF, XF) _Pragma("unroll") for (int r = 0; r < 2; ++r) _Pragma("unroll") for (int k = 0; k < MAX_T; ++k) d[r][k] = fmaf(w[r].WF, XF[k], d[r][k]);
                {
                    float xv[MAX_T];
#define D5X(V, C) _Pragma("unroll") for (int k = 0; k < MAX_T; ++k) xv[k] = V[k].C;
                    D5X(x0, x) D5STEP(a.x, xv) D5X(x0, y) D5STEP(a.y, xv) D5X(x0, z) D5STEP(a.z, xv) D5X(x0, w) D5STEP(a.w, xv)
                    D5X(x1, x) D5STEP(b.x, xv) D5X(x1, y) D5STEP(b.y, xv) D5X(x1, z) D5STEP(b.z, xv) D5X(x1, w) D5STEP(b.w, xv)
#undef D5X
                }
#undef D5STEP
#pragma unroll
                for (int k = 0; k < MAX_T; ++k) {
                    acc[0][k] += d[0][k];
                    if (!inject) acc[1][k] += d[1][k];
                }
            }
        }
    }
    D5STAMP(5);
    // xor over v4 lane bits 4 .. LG+1 = slot bits 4-LG .. 1 (highest first); lanes i = 0, 1 keep lane bit 0 = 0, 1
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            float v = acc[r][k];
#pragma unroll
            for (int o = VS / 2; o > 1; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (i < 2 && r < nrows) red[inject ? slot : slot * 2 + r][k][i] = v;
        }
    if (DIAG & 1) {
        if (i == 0 && nrows > 0) part[c * PR + row0] = acc[0][0];
        return;
    }
    cl.sync();
    D5STAMP(6);
    // block p finishes rows p, p + P, ..: the part bits for lane bit 0 = 0 and 1, then bit 0
    const int nfin = inject ? HC : RPB;
    for (int e = t; e < nfin * T; e += D5_THREADS) {
        const int rr = e / T, k = e - rr * T;
        if (rr % P != p) continue;
        float y[2];
#pragma unroll
        for (int b0 = 0; b0 < 2; ++b0) {
            float v[P];
#pragma unroll
            for (int q = 0; q < P; ++q) v[q] = cl.map_shared_rank(&red[0][0][0], q)[(rr * MAX_T + k) * 2 + b0];
            y[b0] = d5_tree<P>(v);
        }
        const int prow = inject ? LR + rr : rg * RPB + rr;
        part[((size_t) k * HC + c) * PR + prow] = y[0] + y[1];
    }
    if (ss_block && p == 0 && t < T) {
        const int k = t;
        float sum = 0.0f;
        for (int w = 0; w < 8; ++w) {
            float y[4];
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                float v[P];
#pragma unroll
                for (int q = 0; q < P; ++q) v[q] = cl.map_shared_rank(&ssx[0][0][0], q)[(k * 8 + w) * 4 + b];
                y[b] = d5_tree<P>(v);
            }
            // lane bit 1 (xor 2), then bit 0 (xor 1): y index = b1 * 2 + b0
            const float z0 = y[0] + y[2], z1 = y[1] + y[3];
            sum += z0 + z1;
        }
        ssg[k * HC + c] = sum;
    }
    D5STAMP(7);
    cl.sync();   // the peers' shared memory stays until every block has read it
    D5STAMP(8);
}
