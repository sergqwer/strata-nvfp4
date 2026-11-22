"""E1: MTP windows past 8 rows on a wide build (STRATA_WIDE_MTP, =0 the old cap). Binary edit, CRLF kept."""
p = r"D:\Projects\strata-dec3\src\program\generate.cpp"
b = open(p, "rb").read()
n_crlf = b.count(b"\r\n")
def rep(old, new, count=1):
    global b
    old = old.replace("\n", "\r\n").encode()
    new = new.replace("\n", "\r\n").encode()
    c = b.count(old)
    assert c == count, (c, old[:120])
    b = b.replace(old, new)

# the switch, beside gated_proposal_usable
rep("""static bool gated_proposal_usable(bool serial_single_device,""",
"""// decode round 4 E1: on a build whose windows hold more than 8 rows (-DSTRATA_WINDOW_MAX_T=12 / 16) an MTP window may be as wide
// as the build: --spec N > 8 makes the drafter's chain N - 1 drafts long (MtpDrafter::set_chain_cap; its round keeps 8 rows).
// STRATA_WIDE_MTP=0: the old cap (a speculative window of at most 8 rows).  Always off on the 8-row build.
static bool wide_mtp_on() {
    if (strata::kernels::kWindowMaxT <= strata::kernels::kVerifyNarrowT) return false;
    static const bool on = [] { const char* v = std::getenv("STRATA_WIDE_MTP"); return v == nullptr || v[0] != '0'; }();
    return on;
}

static bool gated_proposal_usable(bool serial_single_device,""")
# suffix: lookup windows up to --spec + 2 (the build's width for a wide MTP window)
rep("""        o.spec = std::min(o.spec + 2, 8);   // kVerifyMaxT""",
    """        o.spec = std::min(o.spec + 2, wide_mtp_on() && o.spec > 8 ? (int) strata::kernels::kWindowMaxT : 8);   // kVerifyMaxT (E1: wide)""")
rep("""    const int spec_cap = o.lookup_max_t > 0 && o.suffix_draft > 0 ? (int) strata::kernels::kVerifyLongT
                                                                  : (int) strata::kernels::kVerifyNarrowT;""",
    """    const int spec_cap = o.lookup_max_t > 0 && o.suffix_draft > 0 ? (int) strata::kernels::kVerifyLongT
                         : wide_mtp_on() ? (int) strata::kernels::kWindowMaxT   // E1
                                                                  : (int) strata::kernels::kVerifyNarrowT;""")
# the drafter: a chain as long as the MTP's widest window
rep("""        if (o.pipeline_windows >= 2) mtp.set_force_capture(true);
        if (!o.mtp.empty() && !mtp.load(""", """        if (o.pipeline_windows >= 2) mtp.set_force_capture(true);
        if (wide_mtp_on() && o.pipeline_windows < 2) {   // E1: the chain may outrun the round's rows
            const int s_mtp = o.mtp_max_t > 0 ? std::min(o.mtp_max_t, o.spec) : o.spec;
            if (s_mtp > mtp_t) mtp.set_chain_cap(s_mtp - 1);
        }
        if (!o.mtp.empty() && !mtp.load(""")
# serve: the drafts buffer holds the whole chain; the window's probabilities up to the build's width
rep("""            const size_t DS = (size_t) std::max(S, use_mtp ? mtp.max_t() : 0);""",
    """            const size_t DS = (size_t) std::max({S, use_mtp ? mtp.max_t() : 0, use_mtp ? mtp.chain_cap() + 1 : 0});""")
rep("""            std::array<float, 8> dwin_s{};   // the probabilities the current window was chosen from (STRATA_SPEC_COST)""",
    """            std::array<float, strata::kernels::kWindowMaxT> dwin_s{};   // the probabilities the current window was chosen from (STRATA_SPEC_COST)""")
rep("""            std::array<int32_t, 8> tr_d{};""", """            std::array<int32_t, strata::kernels::kWindowMaxT> tr_d{};""")
rep("""                std::copy(dprob.begin(), dprob.begin() + (std::ptrdiff_t) std::min<size_t>(dprob.size(), 8), dwin_s.begin());
                if (df2_trace != nullptr) std::copy(drafts.begin(), drafts.begin() + (std::ptrdiff_t) std::min<size_t>(drafts.size(), 8), tr_d.begin());""",
    """                std::copy(dprob.begin(), dprob.begin() + (std::ptrdiff_t) std::min<size_t>(dprob.size(), dwin_s.size()), dwin_s.begin());
                if (df2_trace != nullptr) std::copy(drafts.begin(), drafts.begin() + (std::ptrdiff_t) std::min<size_t>(drafts.size(), tr_d.size()), tr_d.begin());""")
open(p, "wb").write(b)
print("crlf", n_crlf, "->", b.count(b"\r\n"), "lf", b.count(b"\n"))
