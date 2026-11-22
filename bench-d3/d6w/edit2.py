R = r"D:\Projects\strata-dec3"
def sub(path, old, new, n=1):
    p = R + "\\" + path
    b = open(p, "rb").read().decode("utf-8")
    old = old.replace("\n", "\r\n"); new = new.replace("\n", "\r\n")
    c = b.count(old)
    assert c == n, (path, c, old[:80])
    b = b.replace(old, new)
    open(p, "wb").write(b.encode("utf-8"))

sub(r"include\strata\core\mtp.hpp",
"""    cudaEvent_t pre_join_ = nullptr;
""",
"""    cudaEvent_t pre_join_ = nullptr;
    // STRATA_MTP_WIN_FORK: the attention's cell list (window_ids, a function of the staged step row only) on win_ beside the
    // query projection, joined before the attention
    cudaStream_t win_ = nullptr;
    cudaEvent_t win_fork_ = nullptr, win_join_ = nullptr;
""")
sub(r"src\core\mtp.cpp",
"""        cudaEventCreateWithFlags(&pre_join_, cudaEventDisableTiming) != cudaSuccess) {
        err = "mtp: streams";
        return false;
    }
""",
"""        cudaEventCreateWithFlags(&pre_join_, cudaEventDisableTiming) != cudaSuccess ||
        (side_prio != 0 ? cudaStreamCreateWithPriority(&win_, cudaStreamNonBlocking, side_prio)
                        : cudaStreamCreateWithFlags(&win_, cudaStreamNonBlocking)) != cudaSuccess ||
        cudaEventCreateWithFlags(&win_fork_, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&win_join_, cudaEventDisableTiming) != cudaSuccess) {
        err = "mtp: streams";
        return false;
    }
""")
sub(r"src\core\mtp.cpp",
"""    if (pre_join_) cudaEventDestroy(pre_join_);
""",
"""    if (pre_join_) cudaEventDestroy(pre_join_);
    if (win_) cudaStreamDestroy(win_);
    if (win_fork_) cudaEventDestroy(win_fork_);
    if (win_join_) cudaEventDestroy(win_join_);
""")
sub(r"src\core\mtp.cpp",
"""bool mtp_qfuse_hc_on() {""",
"""// STRATA_MTP_WIN_FORK (D6, decode round 3; =0 off): a one-row rest builds the attention's cell list on a branch beside the
// query projection instead of between the q rotation and the attention.  The same kernel on the same step row: same drafts.
bool mtp_win_fork_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_MTP_WIN_FORK"); return v == nullptr || std::atoi(v) != 0; }();
    return on;
}

bool mtp_qfuse_hc_on() {""")
sub(r"src\core\mtp.cpp",
"""    try {
        // ---- dense attention over every cell
""",
"""    try {
        // STRATA_MTP_WIN_FORK: the cell list only reads the staged step row, so it runs beside the query side
        const bool win_fork = mtp_win_fork_on() && win_ != nullptr && (floor_on_ && m_floor_ != nullptr || window_ > 0);
        if (win_fork) {
            if (cudaEventRecord(win_fork_, cs) != cudaSuccess || cudaStreamWaitEvent(win_, win_fork_, 0) != cudaSuccess) {
                err = "mtp: the cell list's branch";
                return false;
            }
            if (floor_on_ && m_floor_ != nullptr)
                window_ids_floor(const_cast<int32_t*>(step), T, window_ > 0 ? (int) window_ : 0x7fffffff, ident_, cap_, m_floor_, win_);
            else
                window_ids(const_cast<int32_t*>(step), T, (int) window_, ident_, cap_, win_);
            if (cudaEventRecord(win_join_, win_) != cudaSuccess) { err = "mtp: the cell list's branch"; return false; }
        }
        // ---- dense attention over every cell
""")
sub(r"src\core\mtp.cpp",
"""        const QsaAttnPools pools = qsa_attn_pools(st_);
        if (floor_on_ && m_floor_ != nullptr)   // KVA""",
"""        const QsaAttnPools pools = qsa_attn_pools(st_);
        if (win_fork) {
            if (cudaStreamWaitEvent(cs, win_join_, 0) != cudaSuccess) { err = "mtp: the cell list's join"; return false; }
        } else if (floor_on_ && m_floor_ != nullptr)   // KVA""")
print("ok")
