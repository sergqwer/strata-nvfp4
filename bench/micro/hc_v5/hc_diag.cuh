template <int MAX_T, bool EXACT_T, bool PDL, int R2, int DIAG>
__global__ void __launch_bounds__(THREADS) gr_down_dg_kernel(GrMulti m, float* __restrict__ part, float* __restrict__ ssg) {
    extern __shared__ __align__(16) float xs[];    // [T][N]
    __shared__ float red[WARPS][MAX_T];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    constexpr int SL = N, TQS = SL / 8 / 32;      // 10 uint4 chunks per lane
    const int rg = blockIdx.x, c = blockIdx.y;
    // R2 rows per warp (rows row0 .. row0 + R2 - 1): fewer blocks stage the same activations, and every staged float4
    // read serves R2 rows; the inject block keeps one row per warp (warps 0-3)
    constexpr int NDB = LR / (WARPS * R2);        // down row blocks per stream; block NDB = the inject rows
    const bool inject_block = rg == NDB;
    const int row0 = inject_block ? warp : (rg * WARPS + warp) * R2;
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
    constexpr int NI = (SL / 4 + THREADS - 1) / THREADS;   // float4s per thread and token (2.5 -> 3)
    float4 gv[NI];
    {
        const float4* G4 = reinterpret_cast<const float4*>(m.a[0].w_norm + (size_t) c * N);
#pragma unroll
        for (int u = 0; u < NI; ++u)
            if (t + u * THREADS < SL / 4) gv[u] = __ldg(G4 + t + u * THREADS);
    }
    if constexpr (PDL) {
        gr_pdl_launch_dependents();               // `up` may start loading its weights now
        gr_pdl_wait();                            // R, bo_prev, inj_prev and the scratch are the kernels' before
    }
    float ssp[MAX_T];
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        ssp[k] = 0.0f;
        if (!EXACT_T && k >= T) continue;
        const FusedGrArgs& a = m.a[k];
        const bool apply = a.apply;
        const float gw = apply ? 2.0f * sigmoidf_(__ldg(a.inj_prev + c) / (float) HC) : 0.0f;
        const float4* R4 = reinterpret_cast<const float4*>(a.R + (size_t) c * N);
        const float4* B4 = reinterpret_cast<const float4*>(a.bo_prev);
        float4* X4 = reinterpret_cast<float4*>(xs + (size_t) k * SL);
        float4 rr[NI], bb[NI];
#pragma unroll
        for (int u = 0; u < NI; ++u) {
            const int i = t + u * THREADS;
            if (i < SL / 4) {
                if constexpr ((DIAG & 1) != 0) { rr[u] = gv[u]; bb[u] = gv[u]; }
                else {
                rr[u] = __ldg(R4 + i);
                if (apply) bb[u] = __ldg(B4 + i);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < NI; ++u) {
            const int i = t + u * THREADS;
            if (i >= SL / 4) break;
            float4 r = rr[u];
            if (apply) {
                const float4 b = bb[u];
                r.x = fmaf(b.x, gw, r.x); r.y = fmaf(b.y, gw, r.y);
                r.z = fmaf(b.z, gw, r.z); r.w = fmaf(b.w, gw, r.w);
            }
            const float4 g = gv[u];
            ssp[k] += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
            // two planes: halves 0 and 1 of every chunk of 8 apart, so the dots' float4 reads are consecutive
            X4[(i & 1) * (SL / 8) + (i >> 1)] = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
        }
    }
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if (!EXACT_T && k >= T) break;
        const float v = warp_sum(ssp[k]);
        if (lane == 0) red[warp][k] = v;
    }
    __syncthreads();
    if (rg == 0 && t < T) {
        float sum = 0.0f;
        for (int w = 0; w < WARPS; ++w) sum += red[w][t];
        ssg[t * HC + c] = sum;
    }
    if (!active) return;
    float acc[R2][MAX_T];
#pragma unroll
    for (int r = 0; r < R2; ++r)
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) acc[r][k] = 0.0f;
    if constexpr ((DIAG & 2) != 0) {
        uint32_t h = 0;
#pragma unroll
        for (int r = 0; r < R2; ++r)
#pragma unroll
            for (int q = 0; q < TQS; ++q) h ^= wq[r][q].x ^ wq[r][q].y ^ wq[r][q].z ^ wq[r][q].w;
        acc[0][0] = __uint_as_float(h & 0x3fffffffu) + xs[lane];
    } else {
#pragma unroll
    for (int q = 0; q < TQS; ++q) {
        const int j = lane + 32 * q;
#pragma unroll
        for (int k = 0; k < MAX_T; ++k)
            if (EXACT_T || k < T) {
                const float4* xp = reinterpret_cast<const float4*>(xs + (size_t) k * SL);
                const float4 x0 = xp[j], x1 = xp[SL / 8 + j];
#pragma unroll
                for (int r = 0; r < R2; ++r)
                    if (r < nrows) acc[r][k] += dot8u(unpack8(wq[r][q]), x0, x1);
            }
    }
    }
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
}

template <int MAX_T, bool EXACT_T, bool PDL, int DIAG>
__global__ void __launch_bounds__(THREADS) gr_up_dg_kernel(GrMulti m, const float* __restrict__ part,
                                                           const float* __restrict__ ssg) {
    __shared__ __align__(16) float lo[MAX_T][LR];  // two planes: floats 0-3 and 4-7 of every chunk of 8 apart
    __shared__ float rsS[MAX_T][HC];
    __shared__ float g[MAX_T][HC][UPM_COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const int d0 = blockIdx.x * UPM_COLS;
    constexpr int RPW = HC * UPM_COLS / WARPS;     // 8 rows per warp: row warp + q * WARPS
    static_assert(RPW == 8, "the butterfly below reduces 8 rows");
    uint4 wa4[RPW], wb4[RPW];
#pragma unroll
    for (int q = 0; q < RPW; ++q) {
        const int r = warp + q * WARPS;
        const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
        const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
        wa4[q] = __ldg(w4 + lane);
        wb4[q] = lane < LR / 8 - 32 ? __ldg(w4 + 32 + lane) : make_uint4(0, 0, 0, 0);
    }
    // the epilogue's share of this lane: row qo = (lane >> 2) & 7 of the warp, tokens ko = lane & 3 and ko + 4
    const int qo = (lane >> 2) & 7, ko = lane & 3;
    const int ro = warp + qo * WARPS, co = ro / UPM_COLS, ddo = ro - co * UPM_COLS, io = co * N + d0 + ddo;
    const float wn = __ldg(m.a[0].w_norm + io);   // the same w_norm for every token; constant: before the wait
    if constexpr (PDL) gr_pdl_wait();             // part / ssg are `down`'s, R and the rest the kernels' before
    float rvq[2] = {0.0f, 0.0f}, boq[2] = {0.0f, 0.0f}, ipq[2] = {0.0f, 0.0f};
    bool apq[2] = {false, false};
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int k = ko + 4 * j;
        if (k < T) {
            const FusedGrArgs& a = m.a[k];
            apq[j] = a.apply;
            rvq[j] = a.R[io];
            if (apq[j]) { boq[j] = __ldg(a.bo_prev + d0 + ddo); ipq[j] = __ldg(a.inj_prev + co); }
        }
    }
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        float ss = 0.0f;
        ss += __ldg(ssg + t);
        const float r = rsqrtf(ss / (float) N + m.a[k].eps);
        rsS[k][c] = r;
        if (blockIdx.x == 0) m.a[k].rs[c] = r;
    }
    __syncthreads();
    for (int i = t; i < T * LR; i += THREADS) {
        const int k = i / LR, r = i - k * LR;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
            if constexpr ((DIAG & 4) != 0) p = (float) (r + c); else
            p += __ldg(part + ((size_t) k * HC + c) * PR + r);
            sum = fmaf(rsS[k][c], p, sum);
        }
        const float x = sum / (float) HC;
        lo[k][((r & 7) >> 2) * (LR / 2) + (r >> 3) * 4 + (r & 3)] = x / (1.0f + __expf(-x));   // two planes
    }
    if (blockIdx.x == 0 && t < T * HC) {
        const int k = t / HC, cc = t - k * HC;
        if (m.a[k].w_inject != nullptr) {
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                float p = 0.0f;
                p += __ldg(part + ((size_t) k * HC + c) * PR + LR + cc);
                sum = fmaf(rsS[k][c], p, sum);
            }
            m.a[k].inject_out[cc] = sum;
        }
    }
    __syncthreads();
    float mine[2] = {0.0f, 0.0f};
    const bool b4 = (lane & 16) != 0, b3 = (lane & 8) != 0, b2 = (lane & 4) != 0;
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if (!EXACT_T && k >= T) break;
        if constexpr ((DIAG & 8) != 0) { if (k > 0) break; }
        // lo for token k in registers once (the rows below all use it)
        const float4* lp = reinterpret_cast<const float4*>(lo[k]);
        const float4 la0 = lp[lane], la1 = lp[LR / 8 + lane];
        float4 lb0 = make_float4(0.f, 0.f, 0.f, 0.f), lb1 = lb0;
        if (lane < LR / 8 - 32) { lb0 = lp[32 + lane]; lb1 = lp[LR / 8 + 32 + lane]; }
        float sv[RPW];
#pragma unroll
        for (int q = 0; q < RPW; ++q) {
            float acc = dot8u(unpack8(wa4[q]), la0, la1);
            if (lane < LR / 8 - 32) acc += dot8u(unpack8(wb4[q]), lb0, lb1);
            sv[q] = acc;
        }
        // warp_sum of all 8 rows at once: at each butterfly level a lane keeps half of its rows and adds the
        // partner's partial of the same row - every sum is warp_sum's own + partner (addition commutes bit for bit)
        float h4[4], h2[2];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const float send = b4 ? sv[r] : sv[4 + r];
            const float got = __shfl_xor_sync(0xffffffffu, send, 16);
            h4[r] = (b4 ? sv[4 + r] : sv[r]) + got;
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const float send = b3 ? h4[r] : h4[2 + r];
            const float got = __shfl_xor_sync(0xffffffffu, send, 8);
            h2[r] = (b3 ? h4[2 + r] : h4[r]) + got;
        }
        float fin;
        {
            const float send = b2 ? h2[0] : h2[1];
            const float got = __shfl_xor_sync(0xffffffffu, send, 4);
            fin = (b2 ? h2[1] : h2[0]) + got;
        }
        fin += __shfl_xor_sync(0xffffffffu, fin, 2);
        fin += __shfl_xor_sync(0xffffffffu, fin, 1);   // row qo's sum for token k, in lanes 4 qo .. 4 qo + 3
        if ((k & 3) == ko) mine[k >> 2] = fin;
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int k = ko + 4 * j;
        if (k < T) {
            float rv = rvq[j];
            if (apq[j]) {
                rv = fmaf(boq[j], 2.0f * sigmoidf_(ipq[j] / (float) HC), rv);
                m.a[k].R_out[io] = rv;
            }
            const float x = rv * wn * rsS[k][co];
            g[k][co][ddo] = x * sigmoidf_(mine[j]);
        }
    }
    __syncthreads();
    for (int i = t; i < T * UPM_COLS; i += THREADS) {
        const int k = i / UPM_COLS, col = i - k * UPM_COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        m.a[k].mixed[d0 + col] = sum / (float) HC;
    }
}
