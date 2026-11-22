"""E1: the MTP chain decoupled from the round's rows (C_ drafts, max_t_ round rows). Binary edit, CRLF kept."""
p = r"D:\Projects\strata-dec3\src\core\mtp.cpp"
b = open(p, "rb").read()
n_crlf = b.count(b"\r\n")
R = []
def rep(old, new, count=1):
    global b
    old = old.replace("\n", "\r\n").encode()
    new = new.replace("\n", "\r\n").encode()
    c = b.count(old)
    assert c == count, (c, old[:120])
    b = b.replace(old, new)

# load: the chain capacity
rep("""    max_t_ = max_t;
    rt_dir_ = rt_dir;
    // (the draft layer's kernels - the resident grouping, the grouped experts, the fused reads - are built for the narrow window)
    if (max_t < 1 || max_t > strata::kernels::kVerifyNarrowT) { err = "mtp: max_t out of range"; return false; }
""", """    max_t_ = max_t;
    rt_dir_ = rt_dir;
    // (the draft layer's kernels - the resident grouping, the grouped experts, the fused reads - are built for the narrow window)
    if (max_t < 1 || max_t > strata::kernels::kVerifyNarrowT) { err = "mtp: max_t out of range"; return false; }
    // E1: the chain's drafts (one-row steps) may outrun the round's rows on a wide build (set_chain_cap)
    C_ = std::max(max_t - 1, std::min(chain_req_, (int) strata::kernels::kWindowMaxT - 1));
""")
# staging rows / outputs
rep("""    const uint64_t R2 = 2 * T;   // step/pos rows: T catch-up rows + up to T-2 chain steps
    bool ok = mapped(T * 4 + 64, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(R2 * 4 * 4 + 64, (void**) &h_step_, (void**) &m_step_) &&
              mapped(R2 * NH * 4 + 64, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped(64, (void**) &h_row_, (void**) &m_row_) &&
              mapped(T * 4 + 64, (void**) &h_out_, (void**) &m_out_) &&
              mapped(T * 4 + 64, (void**) &h_prob_, (void**) &m_prob_) &&""",
"""    // step/pos rows: T catch-up rows + up to C_-1 chain steps + draft 0's cell (2 * T when C_ = T - 1)
    const uint64_t R2 = T + (uint64_t) C_ + 1;
    const uint64_t TO = std::max<uint64_t>(T, (uint64_t) C_ + 1);   // the chain's outputs (T when C_ = T - 1)
    bool ok = mapped(T * 4 + 64, (void**) &h_tok_, (void**) &m_tok_) &&
              mapped(R2 * 4 * 4 + 64, (void**) &h_step_, (void**) &m_step_) &&
              mapped(R2 * NH * 4 + 64, (void**) &h_pos_, (void**) &m_pos_) &&
              mapped(64, (void**) &h_row_, (void**) &m_row_) &&
              mapped(TO * 4 + 64, (void**) &h_out_, (void**) &m_out_) &&
              mapped(TO * 4 + 64, (void**) &h_prob_, (void**) &m_prob_) &&""")
rep("""        out_ids_ = b.take<int32_t>(T + 4);
        probs_ = b.take<float>(T + 4);""", """        out_ids_ = b.take<int32_t>(TO + 4);
        probs_ = b.take<float>(TO + 4);""")
# top2 diagnostic
rep("""    if ((int) top2_.size() < max_t_) top2_.assign((size_t) max_t_, -1);
    if (j < 0 || j >= max_t_) return;""", """    if ((int) top2_.size() < C_ + 1) top2_.assign((size_t) C_ + 1, -1);
    if (j < 0 || j > C_) return;""")
# coupled ring / q lists
rep("""(uint64_t) (kCoupledHistCap + max_t_) * 4 + 64ull""", """(uint64_t) (kCoupledHistCap + std::max(max_t_, C_ + 1)) * 4 + 64ull""")
rep("""    const size_t ring = (size_t) (kCoupledHistCap + max_t_) * sizeof(int32_t);""",
    """    const size_t ring = (size_t) (kCoupledHistCap + std::max(max_t_, C_ + 1)) * sizeof(int32_t);""")
rep("""        !mapped((size_t) max_t_ * (size_t) kSpecQStride * sizeof(int32_t), (void**) &h_q_, (void**) &m_q_)) {""",
    """        !mapped((size_t) std::max(max_t_, C_ + 1) * (size_t) kSpecQStride * sizeof(int32_t), (void**) &h_q_, (void**) &m_q_)) {""")
# capture_round: draft 0's row and the all-rows copy
rep("""    const int ra = 2 * max_t_ - 1;""", """    const int ra = draft0_row();""")
rep("""        const MappedCopy all[5] = {{tok_, m_tok_, T}, {step_, m_step_, (int64_t) 2 * max_t_ * 4}, {pos_, m_pos_, 2 * max_t_ * NH},""",
    """        const MappedCopy all[5] = {{tok_, m_tok_, T}, {step_, m_step_, (int64_t) (draft0_row() + 1) * 4}, {pos_, m_pos_, (int64_t) (draft0_row() + 1) * NH},""")
# draft(): the chain length and draft 0's row
rep("""    const int max_steps = (int) std::min<int64_t>(std::min(max_t_ - 1, max_drafts_), st_.max_cells - (p + a));""",
    """    const int max_steps = (int) std::min<int64_t>(std::min(C_, max_drafts_), st_.max_cells - (p + a));""")
rep("""    put(2 * max_t_ - 1, coupled_draft_cell(p, a, 0));   // p + a: draft 0's cell""",
    """    put(draft0_row(), coupled_draft_cell(p, a, 0));   // p + a: draft 0's cell""")
rep("""    if (probs != nullptr) mtp_dump_round(p, a, drafts, probs, n);
    for (int j = n; j < max_t_ - 1; ++j) { drafts[j] = 0; if (probs) probs[j] = 0.0f; }""",
    """    if (probs != nullptr) mtp_dump_round(p, a, drafts, probs, n);
    for (int j = n; j < C_; ++j) { drafts[j] = 0; if (probs) probs[j] = 0.0f; }""")
# chain_launch (pipeline): draft 0's row
rep("""    put(2 * max_t_ - 1, p + a);                                      // output 0's cell""",
    """    put(draft0_row(), p + a);                                        // output 0's cell""")
# draft_first: the round keeps max_t_ rows
rep("""    const int64_t HCN = g_->hc * g_->n_embd;
    for (int t = 0; t < T; ++t)
        if (cudaMemcpy((void*) (window_R_ + (size_t) t * HCN), R_row,""", """    const int64_t HCN = g_->hc * g_->n_embd;
    T = std::min(T, max_t_);   // E1: a wider window's chain comes from C_, not from the round's rows
    for (int t = 0; t < T; ++t)
        if (cudaMemcpy((void*) (window_R_ + (size_t) t * HCN), R_row,""")
open(p, "wb").write(b)
print("crlf", n_crlf, "->", b.count(b"\r\n"), "lf", b.count(b"\n"))
