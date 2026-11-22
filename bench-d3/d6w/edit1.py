import re
R = r"D:\Projects\strata-dec3"
def sub(path, old, new, n=1):
    p = R + "\\" + path
    b = open(p, "rb").read().decode("utf-8")
    old = old.replace("\n", "\r\n"); new = new.replace("\n", "\r\n")
    c = b.count(old)
    assert c == n, (path, c, old[:80])
    b = b.replace(old, new)
    open(p, "wb").write(b.encode("utf-8"))

sub(r"include\strata\kernels\draft_proxy.hpp",
"""void draft_proxy_final2(bool on);
""",
"""void draft_proxy_final2(bool on);
/// STRATA_MTP_HEAD_L2: prefetch the 2-bit copy's rows [0, *n_rows_dev) into L2 (`blocks` blocks of 256 threads); reads no value.
void draft_proxy_l2(const void* proxy, const int32_t* n_rows_dev, int64_t n_cap, int blocks, cudaStream_t stream);
""")

sub(r"include\strata\core\mtp.hpp",
"""    cudaEvent_t pre_join_ = nullptr;
""",
"""    cudaEvent_t pre_join_ = nullptr;
    // STRATA_MTP_HEAD_L2: the 2-bit head copy prefetched into L2 on pf_ beside the step's middle, joined after the head
    cudaStream_t pf_ = nullptr;
    cudaEvent_t pf_fork_ = nullptr, pf_join_ = nullptr;
""")

# streams / events
sub(r"src\core\mtp.cpp",
"""        cudaEventCreateWithFlags(&pre_join_, cudaEventDisableTiming) != cudaSuccess) {
        err = "mtp: streams";
        return false;
    }
""",
"""        cudaEventCreateWithFlags(&pre_join_, cudaEventDisableTiming) != cudaSuccess ||
        cudaStreamCreateWithFlags(&pf_, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&pf_fork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&pf_join_, cudaEventDisableTiming) != cudaSuccess) {
        err = "mtp: streams";
        return false;
    }
""")
sub(r"src\core\mtp.cpp",
"""    if (pre_join_) cudaEventDestroy(pre_join_);
""",
"""    if (pre_join_) cudaEventDestroy(pre_join_);
    if (pf_) cudaStreamDestroy(pf_);
    if (pf_fork_) cudaEventDestroy(pf_fork_);
    if (pf_join_) cudaEventDestroy(pf_join_);
""")

# env switch
sub(r"src\core\mtp.cpp",
"""bool mtp_qfuse_hc_on() {""",
"""// STRATA_MTP_HEAD_L2 (D6, decode round 3): the proxy head's 2-bit rows prefetched into L2 on a branch that forks at point P of the
// one-row rest (1 after the query projection, 2 after the attention, 3 after o_proj, 4 after the MLP read) and joins after the
// head, so the scan reads L2 instead of DRAM; B blocks.  STRATA_MTP_HEAD_L2=P[:B]; =0 off.  No value changes: same drafts.
struct MtpHeadL2 { int at = 0, blocks = 64; };
const MtpHeadL2& mtp_head_l2() {
    static const MtpHeadL2 c = [] {
        MtpHeadL2 r;
#if !defined(STRATA_USE_HIP)
        r.at = 1;
        if (const char* v = std::getenv("STRATA_MTP_HEAD_L2")) {
            int a = 0, b = 0;
            const int k = std::sscanf(v, "%d:%d", &a, &b);
            if (k >= 1) r.at = a;
            if (k >= 2 && b > 0) r.blocks = b;
        }
#endif
        return r;
    }();
    return c;
}

bool mtp_qfuse_hc_on() {""")

# fork helper usage in record_rest
sub(r"src\core\mtp.cpp",
"""    const bool fuse_nr = native_rope_enabled() && native_norm_rope_usable((int) HD, (int) s.n_rot);   // #783 PR-f
    try {
        // ---- dense attention over every cell
""",
"""    const bool fuse_nr = native_rope_enabled() && native_norm_rope_usable((int) HD, (int) s.n_rot);   // #783 PR-f
    // STRATA_MTP_HEAD_L2: fork the head copy's prefetch at point `at` (joined after the head below)
    const MtpHeadL2& hl2 = mtp_head_l2();
    const bool hl2_on = hl2.at > 0 && proxy_on_ && !coupled_rec_ && pf_ != nullptr;
    bool hl2_open = false;
    auto hl2_fork = [&](int at) -> bool {
        if (!hl2_on || hl2.at != at) return true;
        if (cudaEventRecord(pf_fork_, cs) != cudaSuccess || cudaStreamWaitEvent(pf_, pf_fork_, 0) != cudaSuccess) return false;
        strata::kernels::draft_proxy_l2(proxy_, row_ + 4, n_dvocab_, hl2.blocks, pf_);
        if (cudaEventRecord(pf_join_, pf_) != cudaSuccess) return false;
        hl2_open = true;
        return true;
    };
    try {
        // ---- dense attention over every cell
""")
sub(r"src\core\mtp.cpp",
"""            native_mmvq(wt_q_proj, wp_q_proj, xq_, qfull_, (int) N, (int) (NH * 2 * HD), T, cs);
        // all T tokens' q rows at once""",
"""            native_mmvq(wt_q_proj, wp_q_proj, xq_, qfull_, (int) N, (int) (NH * 2 * HD), T, cs);
        if (!hl2_fork(1)) { err = "mtp: the head's L2 branch"; return false; }
        // all T tokens' q rows at once""")
sub(r"src\core\mtp.cpp",
"""        if (st_.kv_rot && !gq) fwht256_inplace_cuda(attn_, (int64_t) T * NH, cs);
        g_mp.stamp(6, cs);
""",
"""        if (st_.kv_rot && !gq) fwht256_inplace_cuda(attn_, (int64_t) T * NH, cs);
        g_mp.stamp(6, cs);
        if (!hl2_fork(2)) { err = "mtp: the head's L2 branch"; return false; }
""")
sub(r"src\core\mtp.cpp",
"""        mtp_gemv(wt_o_proj, wp_o_proj, xq_, bo_, (int) (NH * HD), (int) N, T, cs);
        g_mp.stamp(7, cs);
""",
"""        mtp_gemv(wt_o_proj, wp_o_proj, xq_, bo_, (int) (NH * HD), (int) N, T, cs);
        g_mp.stamp(7, cs);
        if (!hl2_fork(3)) { err = "mtp: the head's L2 branch"; return false; }
""")
sub(r"src\core\mtp.cpp",
"""            q8_mlp = fused_gr_read_multi(fa, T, xn_, cs);
        }
        g_mp.stamp(8, cs);
""",
"""            q8_mlp = fused_gr_read_multi(fa, T, xn_, cs);
        }
        g_mp.stamp(8, cs);
        if (!hl2_fork(4)) { err = "mtp: the head's L2 branch"; return false; }
""")
sub(r"src\core\mtp.cpp",
"""            strata::kernels::draft_proxy_top1(proxy_, head_->weights(), head_->planes(), xq_, row_ + 4, n_dvocab_, dvocab_, out_ids_, probs_,
                                              proxy_scratch_, cs);
""",
"""            strata::kernels::draft_proxy_top1(proxy_, head_->weights(), head_->planes(), xq_, row_ + 4, n_dvocab_, dvocab_, out_ids_, probs_,
                                              proxy_scratch_, cs);
            if (hl2_open && cudaStreamWaitEvent(cs, pf_join_, 0) != cudaSuccess) { err = "mtp: the head's L2 join"; return false; }
""")
print("ok")
