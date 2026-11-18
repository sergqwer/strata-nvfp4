// v5 `up`: v4's with the prologue's loads issued together
template <int MAX_T, bool EXACT_T, bool PDL>
__global__ void __launch_bounds__(THREADS) gr_up_v5_kernel(GrMulti m, const float* __restrict__ part,
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
    // v5: every load the prologue needs issued at once (one L2 round trip instead of one per loop trip): this
    // thread's sum of squares, its lo entries' 4 partials, and block 0's inject partials; then the same arithmetic
    constexpr int NLO = (MAX_T * LR + THREADS - 1) / THREADS;
    float pv[NLO][HC];
    float ssv = 0.0f, pj[HC];
    if (t < T * HC) ssv = __ldg(ssg + t);
#pragma unroll
    for (int u = 0; u < NLO; ++u) {
        const int i = t + u * THREADS;
        if (i < T * LR) {
            const int k = i / LR, r = i - k * LR;
#pragma unroll
            for (int c = 0; c < HC; ++c) pv[u][c] = __ldg(part + ((size_t) k * HC + c) * PR + r);
        }
    }
    const bool inj = blockIdx.x == 0 && t < T * HC && m.a[t / HC].w_inject != nullptr;
    if (inj) {
        const int k = t / HC, cc = t - k * HC;
#pragma unroll
        for (int c = 0; c < HC; ++c) pj[c] = __ldg(part + ((size_t) k * HC + c) * PR + LR + cc);
    }
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        float ss = 0.0f;
        ss += ssv;
        const float r = rsqrtf(ss / (float) N + m.a[k].eps);
        rsS[k][c] = r;
        if (blockIdx.x == 0) m.a[k].rs[c] = r;
    }
    __syncthreads();
#pragma unroll
    for (int u = 0; u < NLO; ++u) {
        const int i = t + u * THREADS;
        if (i >= T * LR) break;
        const int k = i / LR, r = i - k * LR;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
            p += pv[u][c];
            sum = fmaf(rsS[k][c], p, sum);
        }
        const float x = sum / (float) HC;
        lo[k][((r & 7) >> 2) * (LR / 2) + (r >> 3) * 4 + (r & 3)] = x / (1.0f + __expf(-x));   // two planes
    }
    if (inj) {
        const int k = t / HC, cc = t - k * HC;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
            p += pj[c];
            sum = fmaf(rsS[k][c], p, sum);
        }
        m.a[k].inject_out[cc] = sum;
    }
    __syncthreads();
    float mine[2] = {0.0f, 0.0f};
    const bool b4 = (lane & 16) != 0, b3 = (lane & 8) != 0, b2 = (lane & 4) != 0;
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if (!EXACT_T && k >= T) break;
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
