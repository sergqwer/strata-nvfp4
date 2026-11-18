// v4e `down`: v4b software-pipelined - batch b+1's activation loads in flight while batch b's dots run: v4's kernel, branch-free per token, loads batched, the dot chains interleaved (same arithmetic)
template <int MAX_T, bool PDL, int R2, int KB = 2, int TH = 256, int DIAG = 0>  // TH must be 256
__global__ void __launch_bounds__(TH) gr_down_v4e_kernel(GrMulti m, float* __restrict__ part, float* __restrict__ ssg) {
    extern __shared__ __align__(16) float xs[];    // [T][N]
    constexpr int W_ = TH / 32, HV = 256 / TH;    // warps; v4 threads per thread (their sums of squares apart)
    __shared__ float red[8][MAX_T];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    long long st0 = 0;
    if constexpr ((DIAG & 32) != 0) st0 = clock64();
#define V4BSTAMP(ph) do { if constexpr ((DIAG & 32) != 0) { if (threadIdx.x == 0) g_stamp[(blockIdx.y * gridDim.x + blockIdx.x) * 10 + (ph)] = clock64() - st0; } } while (0)
    constexpr bool EXACT_T = true;
    constexpr int T = MAX_T;
    constexpr int SL = N, TQS = SL / 8 / 32;      // 10 uint4 chunks per lane
    const int rg = blockIdx.x, c = blockIdx.y;
    // R2 rows per warp (rows row0 .. row0 + R2 - 1): fewer blocks stage the same activations, and every staged float4
    // read serves R2 rows; the inject block keeps one row per warp (warps 0-3)
    constexpr int NDB = LR / (W_ * R2);        // down row blocks per stream; block NDB = the inject rows
    const bool inject_block = rg == NDB;
    const int row0 = inject_block ? warp : (rg * W_ + warp) * R2;
    const bool active = inject_block ? (m.a[0].w_inject != nullptr && warp < HC) : true;
    const int nrows = inject_block ? (active ? 1 : 0) : R2;
    const uint16_t* wbase = inject_block ? m.a[0].w_inject : m.a[0].w_down;
    uint4 wq[R2][TQS];
#pragma unroll
    for (int r = 0; r < R2; ++r) {
        if (r >= nrows) break;
        const uint4* w4 = reinterpret_cast<const uint4*>(wbase + (size_t) (row0 + r) * D + (size_t) c * N);
#pragma unroll
        for (int q = 0; q < TQS; ++q) wq[r][q] = __ldg(w4 + lane + 32 * q);
    }
    // w_norm is the same for every token and constant: read once, before the wait.  The activations go through the
    // read-only path (ld.global.nc): unlike generic loads they cannot alias the shared-memory stores, so the next
    // token's loads are not held behind this token's stores.
    constexpr int NI = (SL / 4 + TH - 1) / TH;   // float4s per thread and token (2.5 -> 3)
    float4 gv[NI];
    {
        const float4* G4 = reinterpret_cast<const float4*>(m.a[0].w_norm + (size_t) c * N);
#pragma unroll
        for (int u = 0; u < NI; ++u)
            if (t + u * TH < SL / 4) gv[u] = __ldg(G4 + t + u * TH);
    }
    if constexpr (PDL) {
        gr_pdl_launch_dependents();               // `up` may start loading its weights now
        gr_pdl_wait();                            // R, bo_prev, inj_prev and the scratch are the kernels' before
    }
    V4BSTAMP(1);
    // v4b: no branch on a token's `apply` (R again in place of bo without a pending write; the write selected), and
    // every token's loads issued before any token's stores (DIAG 1: v4's per-token order)
    float ssp[MAX_T][HV];
    float gwk[MAX_T], ipv[MAX_T];
    bool ap[MAX_T];
    const float4* R4k[MAX_T];
    const float4* B4k[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        const FusedGrArgs& a = m.a[k];
        ap[k] = a.apply;
        const float* Rk = a.R + (size_t) c * N;
        R4k[k] = reinterpret_cast<const float4*>(Rk);
        B4k[k] = reinterpret_cast<const float4*>(ap[k] ? a.bo_prev : Rk);
        if constexpr ((DIAG & 1) != 0) {   // timing only: distinct data per block
            const unsigned blk = blockIdx.y * gridDim.x + blockIdx.x;
            R4k[k] = reinterpret_cast<const float4*>(m.a[0].w_up) + ((blk * MAX_T + k) * 1280u) % 400000u;
            B4k[k] = R4k[k] + 640;
        }
        if constexpr ((DIAG & 2) != 0) {   // timing only: bo distinct per block, R shared
            const unsigned blk = blockIdx.y * gridDim.x + blockIdx.x;
            B4k[k] = reinterpret_cast<const float4*>(m.a[0].w_up) + ((blk * MAX_T + k) * 640u) % 400000u;
        }
        ipv[k] = __ldg(ap[k] ? a.inj_prev + c : Rk);
    }
    // the gate of every token computed unconditionally, then selected: a branch per token around its load (what
    // `apply ? f(load) : 0` compiles to) serialises one L2 round trip per token
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        const float g = 2.0f * sigmoidf_(ipv[k] / (float) HC);
        gwk[k] = ap[k] ? g : 0.0f;
    }
    V4BSTAMP(2);
    float acc[R2][MAX_T];
#pragma unroll
    for (int r = 0; r < R2; ++r)
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) acc[r][k] = 0.0f;
    constexpr int NB = (MAX_T + KB - 1) / KB;
    float4 rr[KB][NI], bb[KB][NI];
    auto load = [&](int k0) {
#pragma unroll
        for (int kk = 0; kk < KB; ++kk) {
            const int k = k0 + kk;
            if (k >= MAX_T) break;
#pragma unroll
            for (int u = 0; u < NI; ++u) {
                const int i = t + u * TH;
                if (i < SL / 4) { rr[kk][u] = __ldg(R4k[k] + i); bb[kk][u] = __ldg(B4k[k] + i); }
            }
        }
    };
    auto store = [&](int k0) {
#pragma unroll
        for (int kk = 0; kk < KB; ++kk) {
            const int k = k0 + kk;
            if (k >= MAX_T) break;
            ssp[k][0] = 0.0f;
            float4* X4 = reinterpret_cast<float4*>(xs + (size_t) k * SL);
#pragma unroll
            for (int u = 0; u < NI; ++u) {
                const int i = t + u * TH;
                if (i >= SL / 4) break;
                float4 r = rr[kk][u];
                const float4 b = bb[kk][u];
                const float gw = gwk[k];
                const float4 ra = make_float4(fmaf(b.x, gw, r.x), fmaf(b.y, gw, r.y), fmaf(b.z, gw, r.z), fmaf(b.w, gw, r.w));
                if (ap[k]) r = ra;
                const float4 g = gv[u];
                ssp[k][0] += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
                X4[(i & 1) * (SL / 8) + (i >> 1)] = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
            }
        }
    };
    auto dots = [&](int k0) {
#pragma unroll
        for (int q = 0; q < TQS; ++q) {
            const int j = lane + 32 * q;
            Bf16x8 w[R2];
#pragma unroll
            for (int r = 0; r < R2; ++r) w[r] = unpack8(wq[r][q]);
            float4 x0[KB], x1[KB];
#pragma unroll
            for (int kk = 0; kk < KB; ++kk) {
                if (k0 + kk >= MAX_T) break;
                const float4* xp = reinterpret_cast<const float4*>(xs + (size_t) (k0 + kk) * SL);
                x0[kk] = xp[j]; x1[kk] = xp[SL / 8 + j];
            }
            float d[R2][KB];
#pragma unroll
            for (int r = 0; r < R2; ++r)
#pragma unroll
                for (int kk = 0; kk < KB; ++kk) d[r][kk] = 0.0f;
#define V4ESTEP(WF, XV, C) _Pragma("unroll") for (int r = 0; r < R2; ++r) _Pragma("unroll") for (int kk = 0; kk < KB; ++kk) if (k0 + kk < MAX_T) d[r][kk] = fmaf(w[r].WF, XV[kk].C, d[r][kk]);
            V4ESTEP(a.x, x0, x) V4ESTEP(a.y, x0, y) V4ESTEP(a.z, x0, z) V4ESTEP(a.w, x0, w)
            V4ESTEP(b.x, x1, x) V4ESTEP(b.y, x1, y) V4ESTEP(b.z, x1, z) V4ESTEP(b.w, x1, w)
#undef V4ESTEP
#pragma unroll
            for (int r = 0; r < R2; ++r)
                if (r < nrows)
#pragma unroll
                    for (int kk = 0; kk < KB; ++kk)
                        if (k0 + kk < MAX_T) acc[r][k0 + kk] += d[r][kk];
        }
    };
    load(0);
    store(0);
    __syncthreads();
#pragma unroll
    for (int b = 0; b < NB; ++b) {
        if (b + 1 < NB) load((b + 1) * KB);
        if (active) dots(b * KB);
        if (b + 1 < NB) { store((b + 1) * KB); __syncthreads(); }
    }
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        const float v = warp_sum(ssp[k][0]);
        if (lane == 0) red[warp][k] = v;
    }
    __syncthreads();
    if (rg == 0 && t < T) {
        float sum = 0.0f;
        for (int w = 0; w < 8; ++w) sum += red[w][t];
        ssg[t * HC + c] = sum;
    }
    if (!active) return;
#pragma unroll
    for (int r = 0; r < R2; ++r) {
        if (r >= nrows) break;
        const int prow = inject_block ? LR + warp : row0 + r;
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            if (!EXACT_T && k >= T) break;
            const float v = warp_sum(acc[r][k]);
            if (lane == 0) part[((size_t) k * HC + c) * PR + prow] = v;
        }
    }
    V4BSTAMP(6);
}

