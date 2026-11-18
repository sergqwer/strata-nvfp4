// v4d `down`: v4b's rows and dots, the activations staged once per cluster of CS row groups of a stream.  Every block
// of a cluster loads 1/CS of the stream (v4's norm threads tt = rank * 256/CS .. : each one's float4s tt + 256 u and
// its sum of squares in v4's order), transforms it and stores it into the shared memory of all CS blocks (DSMEM);
// a cluster barrier, then v4b's dots on the whole stream.  The sums of squares leave as v4's 8 per-warp partials
// (ssw[token][stream][warp]); `up` (SSW) adds them in v4's order (0 + w0 + .. + w7).  Rows: v4's (R2 = 2, 21 row
// blocks a stream: 20 + the inject rows), padded to a multiple of CS with blocks that only stage.
template <int MAX_T, int CS, bool PDL, int KB = 2>
__global__ void __cluster_dims__(CS, 1, 1) __launch_bounds__(THREADS)
gr_down_v4d_kernel(GrMulti m, float* __restrict__ part, float* __restrict__ ssw) {
    extern __shared__ __align__(16) float xs[];    // [T][N]
    constexpr int R2 = 2, T = MAX_T, SL = N, TQS = SL / 8 / 32, NI = 3;
    constexpr int NDB = LR / (WARPS * R2);        // 20 row blocks; block NDB = the inject rows; beyond: staging only
    constexpr int NST = 256 / CS;                 // v4 norm threads per block
    cg::cluster_group cl = cg::this_cluster();
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int rg = blockIdx.x, c = blockIdx.y;
    const int rank = (int) cl.block_rank();
    // DSMEM may be written only once every block of the cluster runs: arrive now, wait before the first remote store
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    const bool inject_block = rg == NDB;
    const bool active = rg < NDB || (inject_block && m.a[0].w_inject != nullptr && warp < HC);
    const int row0 = inject_block ? warp : (rg * WARPS + warp) * R2;
    const int nrows = !active ? 0 : inject_block ? 1 : R2;
    const uint16_t* wbase = inject_block ? m.a[0].w_inject : m.a[0].w_down;
    uint4 wq[R2][TQS];
#pragma unroll
    for (int r = 0; r < R2; ++r) {
        if (r >= nrows) break;
        const uint4* w4 = reinterpret_cast<const uint4*>(wbase + (size_t) (row0 + r) * D + (size_t) c * N);
#pragma unroll
        for (int q = 0; q < TQS; ++q) wq[r][q] = __ldg(w4 + lane + 32 * q);
    }
    const int tt = rank * NST + t;                // this thread's v4 norm thread (t < NST)
    const bool st_ = t < NST;
    float4 gv[NI];
    {
        const float4* G4 = reinterpret_cast<const float4*>(m.a[0].w_norm + (size_t) c * N);
#pragma unroll
        for (int u = 0; u < NI; ++u)
            if (st_ && tt + u * 256 < SL / 4) gv[u] = __ldg(G4 + tt + u * 256);
    }
    if constexpr (PDL) {
        gr_pdl_launch_dependents();
        gr_pdl_wait();
    }
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
        ipv[k] = __ldg(ap[k] ? a.inj_prev + c : Rk);
    }
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        const float g = 2.0f * sigmoidf_(ipv[k] / (float) HC);
        gwk[k] = ap[k] ? g : 0.0f;
    }
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    float* xr[CS];
#pragma unroll
    for (int r = 0; r < CS; ++r) xr[r] = cl.map_shared_rank(xs, r);
    float ssp[MAX_T];
#pragma unroll
    for (int k0 = 0; k0 < MAX_T; k0 += KB) {
        float4 rr[KB][NI], bb[KB][NI];
#pragma unroll
        for (int kk = 0; kk < KB; ++kk) {
            const int k = k0 + kk;
            if (k >= MAX_T) break;
#pragma unroll
            for (int u = 0; u < NI; ++u) {
                const int i = tt + u * 256;
                if (st_ && i < SL / 4) { rr[kk][u] = __ldg(R4k[k] + i); bb[kk][u] = __ldg(B4k[k] + i); }
            }
        }
#pragma unroll
        for (int kk = 0; kk < KB; ++kk) {
            const int k = k0 + kk;
            if (k >= MAX_T) break;
            ssp[k] = 0.0f;
#pragma unroll
            for (int u = 0; u < NI; ++u) {
                const int i = tt + u * 256;
                if (!st_ || i >= SL / 4) break;
                float4 r = rr[kk][u];
                const float4 b = bb[kk][u];
                const float gw = gwk[k];
                const float4 ra = make_float4(fmaf(b.x, gw, r.x), fmaf(b.y, gw, r.y), fmaf(b.z, gw, r.z), fmaf(b.w, gw, r.w));
                if (ap[k]) r = ra;
                const float4 g = gv[u];
                ssp[k] += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
                const float4 x = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
                const size_t e = (size_t) k * SL + (size_t) ((i & 1) * (SL / 8) + (i >> 1)) * 4;
#pragma unroll
                for (int q = 0; q < CS; ++q) *reinterpret_cast<float4*>(xr[q] + e) = x;
            }
        }
    }
    if (blockIdx.x < CS && st_) {                 // the cluster of row block 0: v4's per-warp sums of squares
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            const float v = warp_sum(ssp[k]);
            if (lane == 0) ssw[((size_t) k * HC + c) * 8 + (tt >> 5)] = v;
        }
    }
    cl.sync();                                    // every block has the whole stream
    if (nrows == 0) return;
    float acc[R2][MAX_T];
#pragma unroll
    for (int r = 0; r < R2; ++r)
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) acc[r][k] = 0.0f;
#pragma unroll
    for (int q = 0; q < TQS; ++q) {
        const int j = lane + 32 * q;
        Bf16x8 w[R2];
#pragma unroll
        for (int r = 0; r < R2; ++r) w[r] = unpack8(wq[r][q]);
        float4 x0[MAX_T], x1[MAX_T];
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            const float4* xp = reinterpret_cast<const float4*>(xs + (size_t) k * SL);
            x0[k] = xp[j]; x1[k] = xp[SL / 8 + j];
        }
        float d[R2][MAX_T];
#pragma unroll
        for (int r = 0; r < R2; ++r)
#pragma unroll
            for (int k = 0; k < MAX_T; ++k) d[r][k] = 0.0f;
#define V4DSTEP(WF, XV, C) _Pragma("unroll") for (int r = 0; r < R2; ++r) _Pragma("unroll") for (int k = 0; k < MAX_T; ++k) d[r][k] = fmaf(w[r].WF, XV[k].C, d[r][k]);
        V4DSTEP(a.x, x0, x) V4DSTEP(a.y, x0, y) V4DSTEP(a.z, x0, z) V4DSTEP(a.w, x0, w)
        V4DSTEP(b.x, x1, x) V4DSTEP(b.y, x1, y) V4DSTEP(b.z, x1, z) V4DSTEP(b.w, x1, w)
#undef V4DSTEP
#pragma unroll
        for (int r = 0; r < R2; ++r)
            if (r < nrows)
#pragma unroll
                for (int k = 0; k < MAX_T; ++k) acc[r][k] += d[r][k];
    }
#pragma unroll
    for (int r = 0; r < R2; ++r) {
        if (r >= nrows) break;
        const int prow = inject_block ? LR + warp : row0 + r;
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            const float v = warp_sum(acc[r][k]);
            if (lane == 0) part[((size_t) k * HC + c) * PR + prow] = v;
        }
    }
}
