// src/core/nv_defaults.cpp - see include/strata/core/nv_defaults.hpp.
#include "strata/core/nv_defaults.hpp"

#include "strata/core/defaults_026.hpp"
#include "strata/core/emulate.hpp"
#include "strata/core/sm120_series.hpp"
#include "strata/prefill/fused_tile_rule.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::core {

namespace {
bool off(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '0';
}
}  // namespace

std::vector<std::pair<std::string, std::string>> nv_default_env(int major, int minor) {
    return nv_default_env(major, minor, strata::qwen_defaults_026());
}

std::vector<std::pair<std::string, std::string>> nv_default_env(int major, int minor, bool defaults026) {
    std::vector<std::pair<std::string, std::string>> t;
    const int mj = strata::cc_major_of(major), mn = strata::cc_minor_of(minor);
    if (mj == 8 && mn == 6) {
        if (off("STRATA_SM86_PERF")) return t;
        t = {
            // 148-nv, RTX 3060 + RTX A4000 (IQ3_XXS, default flags, 8 pairs): the stack +9.6% / +14.6% (8/8 each), identity gate IDENTICAL.
            // Each is in because removing it from the full stack cost speed on a card (drop-one, 4 rounds) and saved none on the other.
            // Repeatable output (0.2.15, option 2 of phaseA-tests/par/RESULTS-REPEAT.md (d)): STRATA_ADAPT_NOWAIT=1 is no longer a default here.
            // With it, a verify window ran a swapped-in expert on the GPU or the CPU by its copy's timing, and two starts of the same request
            // differed (RTX A6000, fixed draft table: 7 of 24 requests identical; without it 24 of 24 in 3 of 3 starts).  Its cost there:
            // decode 0.988 [0.969, 1.002] over 3 pairs.  It was in the stack for "A4000 -6.3 points when removed (with ADAPT_LAG), 3060 -0.8";
            // ADAPT_LAG=2 stays, so the copies still get two windows to land.  STRATA_ADAPT_NOWAIT=1 brings it back.
            {"STRATA_ADAPT_LAG", "2"},
            {"STRATA_CPU_IQ_ALL_EXACT1", "1"},   // A4000 -3.3, 3060 -0.6
            {"STRATA_SIDE_MIXER", "2"},          // A4000 -1.2, 3060 -2.3
            {"STRATA_ATTN_DAILY", "1"},          // -1.2, -1.3
            {"STRATA_GR_UP75", "1"},             // -0.9, -1.8
            {"STRATA_GU_SWIGLU", "1"},           // -0.8, -1.4
            {"STRATA_QSA_ROWS", "1"},            // -0.5, -1.2
            {"STRATA_B6_MMVQ_ROWS", "1"},        // A4000 0, 3060 -1.7
        };
    } else if (mj == 6 && mn == 0) {
        if (off("STRATA_SM60_PERF")) return t;
        t = {
            // 148-nv, Tesla P100 (IQ3_XXS, default flags, 8 pairs): the stack +17.8% (8/8), identity gate IDENTICAL.
            {"STRATA_ADAPT_NOWAIT", "1"},        // -10 points when removed (with ADAPT_LAG)
            {"STRATA_ADAPT_LAG", "2"},
            {"STRATA_SIDE_MIXER", "2"},          // -1.8
            {"STRATA_FUSE_WAIT", "2"},           // -1.6
            {"STRATA_B6_MMVQ_ROWS", "1"},        // -1.3
            {"STRATA_FUSE_DECODE", "1"},         // -1.3
            {"STRATA_GR_UP75", "1"},             // -1.2
        };
    } else if (mj == 12 && mn == 1) {
        // GB10 (NVIDIA DGX Spark, 48 SMs, unified memory): only what was measured on that part goes in.  New in 0.2.7.1, so
        // STRATA_DEFAULTS=026 (the master switch of everything since 0.2.6) leaves the table out, like STRATA_SM121_PERF=0.
        if (off("STRATA_SM121_PERF") || defaults026) return t;
        t = {
            // the fused IQ prompt kernels.  IQ3_S, 64K context, fresh 4K/16K/32K prompts at T=0, 3 paired rounds against
            // the same tree without it: prompt 1.139 / 1.149 / 1.121, decode unchanged; 5 of 5 greedy answers byte-
            // identical with it on and off (docs/DGX_SPARK_LINUX.md; HANDOVER §6.2).  It rounds differently from MMQ,
            // so it is a default with an off switch, like the sm_86 table.
            {"STRATA_PF_FUSED", "1"},
            // the same fused prompt kernels for the K-quant experts of Unsloth's UD-Q4_K_XL files (Q4_K / Q5_K gate/up, Q5_1 / Q8_0 down) instead of llama.cpp's MMQ.
            // UD-Q4_K_XL (Flash-Next), expert cache budget 12000 (15,283 of 24,576 experts resident, so the box keeps over 16 GiB of its 121 GiB free), two starts a pair,
            // against the FP16 prompt path's text, 48 real-text prompts (24 at 2,048 and 24 at 4,096 tokens, 256 forced tokens each, 12,288 positions): forced-token
            // KL kq / MMQ 0.895 [0.808, 0.989] (0.864 [0.756, 0.976] at 2K, 0.928 [0.791, 1.088] at 4K), top-1 against the FP16 path's 96.31% / 95.74%, the same
            // text twice byte for byte on both sides (repeat floor 0); prompt speed kq / MMQ, 10 interleaved pairs, median of the per-round ratios 1.10 / 1.26 / 1.18 / 1.53 at 2K / 4K / 8K / 20K (faster in 7 / 8 / 8 / 9 of 10; the rounds scatter 0.5 to 2.0)
            // (phaseA-tests/par/RESULTS-SPARK-KQ.md).  STRATA_PF_FUSED_KQ=0 is the way back (a switch you set is kept); it is off with the rest of the table by STRATA_SM121_PERF=0.
            {"STRATA_PF_FUSED_KQ", "1"},
            // NVFP4 prompt MMQ with FP4 activations (W4A4), the Blackwell path the engine keeps opt-in.  On this part, the
            // NVFP4 pack measured prompt 1.106x at 4K and 1.229x at 16K against the default W4A8 MMQ, decode unchanged
            // (1.015x), and 24 of 24 greedy answers byte-identical with it on and off.  Opt-in upstream because it is only
            // checked on Blackwell; switched on here for the same reason as the row above, with the same off switch.
            {"STRATA_PREFILL_NVFP4", "w4a4"},
            // the hyper-connection read's fused up projection + mix (gr_upmix).  Measured on IQ3_S against the unfused read:
            // decode 1.054x [1.007, 1.089], prompt unchanged, and 24 of 24 greedy answers byte-identical (the parity test
            // src/prefill/gr_upmix_parity.cu covers it upstream).  The HC read is ~13% of a decode round here, which is
            // why a fusion on it moves the number at all.
            {"STRATA_HC_UPMIX", "1"},
            // Run the mixer's input-only QSA/GDN work on graph branches.  With the grouped experts' mode 8 selected,
            // fixed-work IQ3_S decode measured +6.68% paired median and +6.67% geometric mean over 8 balanced pairs
            // (AB +11.60%, BA +6.68%, identical 512-token output and verifier work).  The branch changes no arithmetic.
            {"STRATA_SIDE_MIXER", "2"},
            // the cost-aware MTP window: pick the verify window from what a draft costs against what it has been paying.
            // Measured on IQ3_S: decode 1.047x [1.019, 1.079], prompt unchanged, 24 of 24 greedy answers byte-identical.
            // It only changes which drafts are attempted, and every draft is verified, so the answers cannot drift.
            {"STRATA_SPEC_COST", "1"},
        };
    } else if (mj == 12 && mn == 0) {
        // strata-nvfp4 (RTX 5090 class, CUDA 12.0): the speed switches the owner accepted on 2026-11-19 ("on by default when they
        // cost quality not much more than the run-to-run noise"); STRATA_NVFP4_SM120=0 turns the whole table off, each switch's
        // own =0 turns it off alone.  Measured on an RTX 5090 with the GPTQ pack (dec/int NOTES-int.md, NOTES-hc8.md, NOTES-gdn.md):
        if (off("STRATA_NVFP4_SM120") || defaults026) return t;
        t = {
            // hc8: the 8-bit hyper-connection read (fused_gr.cu hc8_read.cuh).  Decode struct +3.6% (DFlash2) / +5.8% (MTP spec 8);
            // teacher-forced argmax 96.10% / KL 0.0152 against the noise floor 96.52% / 0.0118, NLL not worse.
            {"STRATA_HC_Q8", "1"},
            // equal prompt chunks, a sub-floor tail included: 66,617 tokens 7,825 -> ~7,080 ms; inside the chunk-order noise.
            {"STRATA_PREFILL_EQUAL", "2"},
            // the prompt path's GDN scan on tensor cores, conv + l2 fused (the Linux CUDA default since 2026-11-07), and the
            // one-pass f16 products of the 2-tile scan: 64K read -2.5% / -0.6% / -2%; first-token KL ~0.001.
            {"STRATA_GDN_TC", "1"},
            {"STRATA_GDN_CONVL2", "2"},
            {"STRATA_GDN_TC_NP", "1"},
            // the prompt path's smaller buffers (NOTES-gdn.md, the end-of-prompt refill): one expert group's gate/up rows and the
            // HC scratch in the attention/MoE region with xn16 outside it - fewer borrowed slots (64K 3,663 -> 2,744), refill
            // 205 -> 154 ms at 64K, 16K -2.1%; same bits by construction (NOTES-int.md: checked)
            {"STRATA_PF_GROUP_SCRATCH", "1"},
            {"STRATA_PF_HC_REGION", "2"},
            // the serve tier as dec/host round 2 measured it (NOTES-host.md R2.3-R2.5, "x3p"): the PCIe share admits AND the tier
            // ranks every layer's misses against every layer's coldest slots, a lower candidate floor, a slower decay, and the PCIe
            // share fixed at 0.45 (PcieAuto settles at 0.35).  Struct +10.4%, prose +3.9%, conv +2.7%; misses 4.80 -> 3.91 a layer-window.
            // It moves experts between GPU and CPU arithmetic, not the numerics (NOTES-int.md: KL against the noise floor).
            {"STRATA_ADAPT_FETCH", "1"},
            {"STRATA_ADAPT_ACROSS", "1"},
            {"STRATA_ADAPT_CAND_MIN", "1.5"},
            {"STRATA_ADAPT_DECAY", "0.96"},
            {"STRATA_PCIE_FIXED", "0.45"},   // fixed beat PcieAuto + 0.10 (STRATA_PCIE_AUTO_ADD) on struct and conv
            // dense_tc (Basalt's int8 tensor-core Q8_0 GEMV, MIT): every dense Q8_0 matrix of a 1..8-row window, exact int32 per
            // 32-value block, so a column's value does not depend on the window width.  Struct T=8 17.66 -> 17.29 ms/window
            // (+2.0%), prose unchanged; teacher-forced KL 0.0117-0.0130 against the noise floor 0.0120, NLL not worse.
            {"STRATA_DENSE_TC", "1"},
            // staged INT8 attention chunks (Basalt's attn_chunk_v2, MIT): bitwise the Daily kernel's partials, K/V staged by cp.async,
            // two blocks per SM.  Chunk pass 14.6 -> 7.5 us an attention layer (T~3), conv/prose/uk -0.6..-1.3% ms/window, struct flat.
            {"STRATA_ATTN_STAGED", "1"},
            // gdn_mid (host rounds 6-7, Basalt's k_mid structure on our arithmetic): the GDN middle in one persistent launch,
            // bitwise the chain (gdn_parity section 8; needs DENSE_TC=1); GDN front 81.8 -> 78.4 us a layer (prose).
            {"STRATA_GDN_MID", "1"},
            // attention-layer layout (round 8, Basalt's qhalf placement): the q rows of q|gate first, the gate rows on the
            // queries' branch after the q post on 48 persistent blocks (GATE_GRID; the full grid starved the indexer and K/V
            // chains), the window graph's nodes at their streams' priorities (indexer branch highest).  Bitwise (dense_tc_rows);
            // attention front 89 -> 84 us, ms/window struct -1.4%, prose -1.6%, uk -0.5%, conv 0.
            {"STRATA_QPROJ_HALF", "1"},
            {"STRATA_QPROJ_GATE_AT", "3"},
            {"STRATA_QPROJ_GATE_GRID", "48"},
            {"STRATA_GRAPH_PRIO", "1"},
            // PDL edges in the doorbell graphs too (until now the all-resident graph only; E21's -1.6..-4.5% there is gone):
            // bitwise; prose 8.99 -> 8.70 ms/window, struct 15.96 -> 16.03, 16 GB budget struct 48.98 -> 48.42.
            {"STRATA_DF_PDL", "1"},
            // dec/front (the layer front at T~3): the next layer's first HC read launched before the CPU-row wait and gated on the
            // combine's count (with the next attention layer's indexer-tail snapshot on that branch); gdn_mid's tile copies after
            // its grid wait (their flood held the read up); route + doorbell in one launch, the rows first.  All bitwise;
            // nsys prose front GDN 77.6 -> ~72, attention 79.1 -> ~75 us.
            // STRATA_EARLY_READ (the early read above) left the table in decode round 4 E4: serve e2 vs e2 + =0, 5 ABAB pairs, 600 W:
            // struct 16.98 -> 16.73 ms/window (5 of 5), prose 8.83 -> 8.90 (noise); bitwise; =1 still turns it on (bench-d4 NOTES-d4.md E4).
            // decode round 3 D6: PDL edges in the MTP draft graphs (their kernels read only weights before the wait): the same drafts
            // (dumps identical), CLI fixed struct: chain steps 0.791 -> 0.774 ms a round (6 runs each), round graph 0.309 -> 0.305 (bench-d3 NOTES-d3.md D6).
            {"STRATA_MTP_PDL", "1"},
            {"STRATA_GDN_MID_HEAD", "2"},
            {"STRATA_ROUTE_DB", "2"},
            // dec/pf-num (NOTES-pfnum.md): the prompt path's QSA block scores on int8 tensor cores (i8x6, ~2x the 3xTF32 kernel,
            // 9x closer to FP64) and the dense FP16 GEMMs with f16-accumulate chains of 64 promoted to f32 (1.2x cuBLAS at chunk
            // size, rel. error ~3e-4 more); first-token / all-rows KL on the long-prompt sets at the run-to-run floor.
            {"STRATA_QSA_I8", "1"},
            {"STRATA_DENSE_F16C", "1"},
            // dec/pf-host: the next streamed attention layer's K/V prefix staged on its own stream, and (STRATA_PF_KV_UNSTAGE_SIDE)
            // the chunk's K/V download there too, so the compute stream never waits behind the expert copies (one copy engine
            // under WDDM).  Same bytes; 64K read 6098 -> 6007 ms at 600 W with the pf-host scheduling changes (NOTES-pfhost.md).
            {"STRATA_KV_PREFETCH", "1"},
            // dec/pf-hc (NOTES-hc.md): the next half's HC inject projection folded into the residual writes (FP32 sums of the
            // stored xn16 bits; R / rs / xn16 bitwise) and the FP4G down rows stored FP16 with s_down applied (real |Dm| <= 23,
            // 2800x under FP16's range).  64K read -3.3% at 600 W; long-set / 131K / TF-corpus KL at the run-to-run floor.
            {"STRATA_PF_HC_INJ", "1"},
            {"STRATA_PF_DM16H", "1"},
            // dec/pf-graph (NOTES-graph.md): the prompt path's QSA indexer chain (a layer's scores + top-k per 256 queries) and each
            // expert group's chain (gate/up, H quantize, down) replayed as CUDA graphs: under WDDM an eager launch costs ~10 us of
            // host and ~3 us of GPU idle.  The same kernels and arguments (bitwise); 64K read -2.7% at 600 W.
            {"STRATA_PF_GRAPH", "7"},
            // decode round 5 F2 (= R5-1): past 32768 cells the decode windows take the long-context select graphs with the
            // shuffle-free DX block scorer (block_scores_dx_kernel: each pooled key read once for all of the window's queries,
            // bitwise the pairs kernel's scores, so the same ids); the cluster top-k is kept (STRATA_TOPK_MULTI stays off).
            // Indexer scores 286 us/layer at T 8 at 370K were the long-context loss (gpu-probe DECODE-PLAN-R5.md); serve
            // A/B in strata-r5 bench-r5 NOTES-r5.md F2.  STRATA_SELECT_LONG=0 (or STRATA_SCORES_DX=0) gives the old path.
            // The threshold from qsa_front_bench (us/layer, pairs vs DX): DX is faster for T >= 2 from 24K cells (T 3: 8.43 vs
            // 7.07, T 8: 19.7 vs 10.3) and slower below 16K (8K T 3: 4.17 vs 5.70).
            {"STRATA_SELECT_LONG", "1"},
            {"STRATA_SCORES_DX", "1"},
            {"STRATA_SELECT_LONG_AT", "24576"},
            // decode round 5 F6 (= R5-2): the MTP drafter's attention over its 32768-cell window on the pipelined chunk kernel
            // (attn_chunk_draft_pipe_kernel: one wave of blocks looping over chunks, the next chunk's K/V copied during this
            // one's softmax and fold; the same partials bit for bit).  Its fast merge (STRATA_MTP_MERGE_FAST) is on for every
            // CUDA card.  Bench and serve A/B in strata-r5 bench-r5 NOTES-r5.md F6; STRATA_MTP_ATTN_PIPE=0 gives the old kernel.
            {"STRATA_MTP_ATTN_PIPE", "1"},
        };
    }
    return t;
}

std::vector<std::string> apply_nv_defaults(int major, int minor) {
    const int mj = strata::cc_major_of(major), mn = strata::cc_minor_of(minor);
    std::vector<std::string> set;
    for (const auto& kv : nv_default_env(major, minor)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;   // the user's setting wins, whatever it is
#if defined(_WIN32)
        _putenv_s(kv.first.c_str(), kv.second.c_str());
#else
        setenv(kv.first.c_str(), kv.second.c_str(), 0);
#endif
        set.push_back(kv.first);
    }
    if (!set.empty()) {
        const bool sm86 = mj == 8;
        const bool sm121 = mj == 12 && mn == 1;
        const bool sm120 = mj == 12 && mn == 0;
        // sm_86 / sm_60: bit-exact switches.  sm_121: the fused prompt path and the FP4-activation NVFP4 prompt round differently from the
        // paths they replace (24 of 24 greedy answers were identical on the Spark; no KL run), so the line does not say "exact".
        std::fprintf(stderr, "strata: %s: %zu %sspeed switches on by default (%s=0 turns them off%s; a switch you set is kept): ",
                     sm86 ? "sm_86 (Ampere)" : sm121 ? "sm_121 (GB10)" : sm120 ? "sm_120 (strata-nvfp4)" : "sm_60 (Pascal)", set.size(),
                     (sm121 || sm120) ? "" : "exact ",
                     sm86 ? "STRATA_SM86_PERF" : sm121 ? "STRATA_SM121_PERF" : sm120 ? "STRATA_NVFP4_SM120" : "STRATA_SM60_PERF",
                     (sm121 || sm120) ? ", as does STRATA_DEFAULTS=026" : "");
        for (size_t i = 0; i < set.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", set[i].c_str() + 7);
        std::fprintf(stderr, "\n");
    }
    return set;
}

std::vector<std::pair<std::string, std::string>> nv_kq_default_env(int min_major, int min_minor) {
    return nv_kq_default_env(min_major, min_minor, strata::qwen_defaults_026());
}

std::vector<std::pair<std::string, std::string>> nv_kq_default_env(int min_major, int min_minor, bool defaults026) {
    std::vector<std::pair<std::string, std::string>> t;
    if (defaults026) return t;
    if (strata::cc_major_of(min_major) < 8) return t;   // Volta / Turing / Pascal: not measured, keep the FP16 prompt path
    const char* sw = std::getenv("STRATA_NV_KQ_DEFAULT");
    if (sw != nullptr && sw[0] == '0') return t;
    if (!kNvKqDefaultOn && !(sw != nullptr && sw[0] == '1')) return t;
    t.push_back({"STRATA_MMQ_KQ", "1"});
    (void) min_minor;
    return t;
}

bool nv_kq_fused_here(int major, int minor, int sms) {
    namespace ft = strata::prefill::fused_tile;
    if (ft::pf_env(std::getenv("STRATA_PF_FUSED")) == 0) return false;
    const ft::Card c{strata::sm120_default(major, minor), major == 12 && minor == 1, sms, ft::s86_class(major, minor, off("STRATA_PF_FUSED_SM86"))};
    return ft::kq_takes(ft::pf_env(std::getenv("STRATA_PF_FUSED_KQ")), c, ft::sm40_env_of(std::getenv("STRATA_PF_FUSED_SM40")));
}

std::string nv_kq_default_line(bool fused_here) {
    return nv_kq_default_line(fused_here, true);
}

std::string nv_kq_default_line(bool fused_here, bool mmq_built) {
    if (!mmq_built) {   // a build without MMQ: Strata's own kernels read every K-quant layer
        if (fused_here)
            return "strata: a model with K-quant experts (Unsloth UD-Q4/Q5/Q6_K_XL) reads its prompt on Strata's own kernels "
                   "(Q4_K and Q5_K gate/up with a Q5_1 or Q8_0 down); the 'prompt kernels' line below gives the counts";
        return "strata: a model with K-quant experts (Unsloth UD-Q4/Q5/Q6_K_XL) reads its prompt on Strata's own int8 kernels (this engine has no MMQ kernels)";
    }
    if (fused_here)
        return "strata: a model with K-quant experts (Unsloth UD-Q4/Q5/Q6_K_XL) reads its prompt on Strata's own kernels where they cover the layer "
               "(Q4_K and Q5_K gate/up with a Q5_1 or Q8_0 down) and on llama.cpp's MMQ kernels for the rest (STRATA_PF_FUSED_KQ=0 gives all of them back to MMQ)";
    return "strata: a model with K-quant experts (Unsloth UD-Q4/Q5/Q6_K_XL) reads its prompt on llama.cpp's MMQ kernels "
           "(STRATA_MMQ_KQ=0 or STRATA_NV_KQ_DEFAULT=0 turns it off)";
}

bool nv_kq_expert_type(int ggml_type) {
    // ggml type ids: Q5_1 = 7, Q4_K = 12, Q5_K = 13, Q6_K = 14 (the cases of prefill::mmq::supported() behind kq_ok())
    return ggml_type == 7 || ggml_type == 12 || ggml_type == 13 || ggml_type == 14;
}

bool nv_kq_line_wanted(const std::vector<std::pair<int, int>>& layer_gate_up_down_types) {
    for (const auto& l : layer_gate_up_down_types)
        if (nv_kq_expert_type(l.first) || nv_kq_expert_type(l.second)) return true;
    return false;
}

std::vector<std::string> apply_nv_kq_default(int min_major, int min_minor, bool fused_here, bool say) {
    std::vector<std::string> set;
    for (const auto& kv : nv_kq_default_env(min_major, min_minor)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;   // the user's setting wins, whatever it is
#if defined(_WIN32)
        _putenv_s(kv.first.c_str(), kv.second.c_str());
#else
        setenv(kv.first.c_str(), kv.second.c_str(), 0);
#endif
        set.push_back(kv.first);
    }
    if (say && !set.empty())   // the engine passes say = false and prints the line after the expert layout is loaded, only for a set with K-quant experts (nv_kq_line_wanted)
        std::fprintf(stderr, "%s\n", nv_kq_default_line(fused_here).c_str());
    return set;
}

void apply_spec_adapt_default(int major, int minor) {
    const int mj = strata::cc_major_of(major), mn = strata::cc_minor_of(minor);
    // sm_86 only: the P100 (sm_60) gained 4-16% on JSON/code/chat/tool calls but lost 3% on the 4K agent transcript (5/5 pairs), so there it stays opt-in
    // Owner round 9 (2026-10-22): on the z440 A4000s the 5-pair A/B against 0.2.2 was 0.995 (code 0.996, json 1.019, tool 0.973,
    // agent 0.991) and the 3060 was length-confounded, so the earlier +9..17% did not reproduce: opt-in (STRATA_SPEC_ADAPT_SM86=1)
    // until a clean A/B shows it.
    const bool sm86 = mj == 8 && mn == 6;
    if (!sm86) return;
    if (const char* v = std::getenv("STRATA_SPEC_ADAPT_SM86"); v == nullptr || v[0] != '1') return;
    if (off("STRATA_SM86_PERF")) return;
    const std::pair<const char*, const char*> kv[2] = {{"STRATA_SPEC_ADAPT_EVERY", "2"}, {"STRATA_SPEC_ADAPT_WARM", "200"}};
    for (const auto& p : kv) {
        if (std::getenv(p.first) != nullptr) continue;
#if defined(_WIN32)
        _putenv_s(p.first, p.second);
#else
        setenv(p.first, p.second, 0);
#endif
    }
}

}  // namespace strata::core
