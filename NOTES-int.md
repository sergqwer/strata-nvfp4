# dec/int - the decode ports integrated (two-stage head, Basalt MTP, hc8) + the remaining gap to Basalt

Worktree D:\Projects\strata-dec-int, branch dec/int from 0.2.15-nvfp4.3 (2572b0e0f). **Nothing committed.**
Base reference exe: D:\Projects\strata-dec\build-dev\strata.exe (same tree, unchanged code).
Scripts and logs of this work: D:\Projects\strata-dec-int\bench-int\ (exact*.sh/txt, serve_all.sh, ab.sh, sum.py,
nsys_dec.sh, nsys_win.py, serve/*.log).

## Defaults on sm_120 (state 2026-11-20 10:40) - read this first

The cc 12.0 table is in src/core/nv_defaults.cpp (20 entries). `STRATA_NVFP4_SM120=0` (or STRATA_DEFAULTS=026)
turns the whole table off. Each switch's own `=0` turns that switch off alone, and a switch you set yourself is kept.
Start-up prints: `strata: sm_120 (strata-nvfp4): 20 speed switches on by default ...`.
"exact" = bit-identical output, proved by parity tests and the engine md5 matrix (exact*.txt).

### The switches

| switch(es) | what it does | from | exact / rounding | measured gain | off |
|---|---|---|---|---|---|
| STRATA_HC_Q8=1 | 8-bit hyper-connection read (hc8; int8 rows + fp16 scales) | Basalt port (NOTES-hc8.md) | rounding: TF argmax 96.10% / KL 0.0152 vs noise 96.52% / 0.0118, NLL not worse | decode struct +3.6% (DFlash2) / +5.8% (MTP 8) | =0 |
| STRATA_PREFILL_EQUAL=2 | equal prompt chunks, tail included | own | chunk-order rounding | 64K read 7825 -> ~7080 ms | =0 |
| STRATA_GDN_TC=1, STRATA_GDN_CONVL2=2, STRATA_GDN_TC_NP=1 | prompt GDN scan on tensor cores, conv + l2 fused, one-pass f16 products | own (GDN agent) | rounding: first-token KL ~0.001; long TF in Milestone B | 64K read -2.5% / -0.6% / -2% | =0 (NP: =3) |
| STRATA_PF_GROUP_SCRATCH=1, STRATA_PF_HC_REGION=2 | smaller prompt-path buffers (fewer borrowed slots) | own (GDN agent) | exact | 64K refill 205 -> 154 ms, 16K -2.1% | =0 |
| STRATA_ADAPT_FETCH=1, ADAPT_ACROSS=1, ADAPT_CAND_MIN=1.5, ADAPT_DECAY=0.96, PCIE_FIXED=0.45 | x3p serve tier: PCIe share admits, every layer's misses ranked against every layer's coldest slots, PCIe share fixed at 0.45 | own (host agent R2) | expert placement only (GPU vs CPU arithmetic per expert): TF 96.03% / 0.0156 (x3p), 96.11% / 0.0150 (all defaults) vs noise 96.52% / 0.0118 | struct +10.4%, prose +3.9%, conv +2.7% | each =0 (PCIE_FIXED unset -> PcieAuto) |
| STRATA_DENSE_TC=1 | int8 tensor-core GEMV for every dense Q8_0 matrix of a 1..8-row window | Basalt port (dense_tc) | rounding vs mmvq, but width-invariant: TF KL 0.0117-0.0130 vs noise 0.0120, NLL not worse | struct T=8 -2.0% ms/window | =0 |
| STRATA_ATTN_STAGED=1 | INT8 attention chunks staged by cp.async, 2 blocks per SM | Basalt port (attn_chunk_v2) | exact | chunk 14.6 -> 7.5 us per attention layer; -0.6..-1.3% ms/window at T 2-3.5 | =0 |
| STRATA_GDN_MID=1 | GDN layer middle (qkv, z, gates, recurrence, norm, out-proj) in one persistent launch; the first HC read writes its q8_1 | Basalt structure (k_mid) on our arithmetic, host agent R6/R7 | exact (gdn_parity section 8) | GDN front 82.4 -> 79.2 us per layer | =0 |
| STRATA_QPROJ_HALF=1, QPROJ_GATE_AT=3, QPROJ_GATE_GRID=48, GRAPH_PRIO=1 | attention layer: q rows of q/gate first, gate rows after q post on 48 blocks, graph nodes keep their stream priorities | Basalt placement (qhalf) + own grid cap | exact | attention front 89 -> 84 us; struct -1.4%, prose -1.6%, uk -0.5%, conv 0 | HALF=0 (PRIO=0 separately) |
| STRATA_DF_PDL=1 | PDL edges in the doorbell graphs too (before: all-resident graph only) | own (existing switch) | exact | prose -3.2% ms/window, 16 GB budget struct -1.1%, struct flat | =0 |

Also on by default but not in the table (code defaults):
- two-stage bit-plane head (q80p_vhead): STRATA_HEAD_PLANES=0 turns it off; exact.
- the fused attention post (kv_rot_append_q8, merge with the inverse rotation, gate + q8_1): STRATA_KV_POST=0; exact.
- FP4G grouped prompt expert GEMM: STRATA_PF_FP4G=0; first-token KL not worse than MMQ.
- the dense slab: STRATA_DENSE_SLAB=0; exact.
- the drafter's round-2 kernels (MTP_ATTN_STAGED / ATTN_GQ / ROUTE_GROUP / OVERLAP / QSTREAM / SELECT_WIDE / GEMV_PF /
  PROXY_FINAL2): each =0; a drafter never changes output.

Auto drafter rule (strata_control/drafting.py; STRATA_AUTO_NEW_MTP=0 = the old Auto): for Qwen3.8-Flash-Next sets that
draft today, on CUDA, with one request at a time and no layer split, Auto = MTP with `--mtp-head proxy --mtp-cum auto
--spec 8` and the rt-tuned draft layer where present (licence check needed before shipping it). `--mtp-cum auto` sets
Basalt's draft_cum_scale from the CPU-served MB per expert (0.399:0.285 on the 5090). Details in Milestone B.

### Accuracy

Teacher-forced, corpus/set.json vs tf/strata_A.tsv (round 5, every default of that time: hc8 + x3p + dense_tc):

| arm | argmax | KL | NLL vs ref |
|---|---|---|---|
| this stack | 96.11% | 0.0150 | not worse (-0.0025) |
| noise floor | 96.52% | 0.0118 | -0.0024 |

Everything added after that measurement is bit-exact, so the TF result still holds:
- ATTN_STAGED;
- the QPROJ layout and GRAPH_PRIO;
- GDN_MID;
- DF_PDL;
- KV_POST;
- the slab.

The engine md5 matrix is identical through exact11 -> exact17 (conv 26c5e96e8e, struct 30b9f645ca, prose 10df564211,
table off 293b92d3b9). Long prompts (Milestone B): first-token KL against fp16 is the same as the base exe's at 8-32K
and at 131K.

### Same hour, 2026-11-20 10:00-10:40

Decode, serve, 1 round, tok/s (serve_r10.txt):
- base = base exe with DFlash2 as the tray runs it;
- int = this build with serve Auto;
- Basalt = Basalt default.

| fixture | base | int | Basalt | int vs base | int vs Basalt |
|---|---|---|---|---|---|
| struct | 329.7 | 486.7 | 474.5 | +47.6% | +2.6% |
| conv | 218.5 | 269.2 | 248.3 | +23.2% | +8.4% |
| prose | 205.1 | 261.4 | 276.2 | +27.4% | -5.4% |
| uk | 154.3 | 241.3 | 234.9 | +56.4% | +2.7% |

The 2-round table (r6 / int / Basalt) is in Round 9. With 16 GB reserved, struct 155.3 (r6) -> 155.9 / 157.5.

Prompt reads, serve, 1 request each, ctx 140000, ms (prefill_r9.txt), against round 5's int numbers:

| prompt tokens | base | int now | int round 5 |
|---|---|---|---|
| 3,072 | 924 | 853 | 872 / 857-886 |
| 16,676 | 1938 | 1657 | 1659 |
| 66,617 | 7845 | 6163 | 6178 |
| 131,000 | 14460 | 12471 | 12495 |

No decode-side default since round 5 moved the prompt path (all within 0.3%). Basalt's prompt reads were not measured
this hour: its context is 66K.

### Opt-in switches left in the tree, off by default (keep or remove)

Measured and rejected or neutral:
- STRATA_QPROJ_GATE_AT=0/1/2/4: other gate placements.
- STRATA_MMVQ_CSPLIT=1.
- STRATA_QFUSE=1 (exact, no gain).
- STRATA_QIL.
- STRATA_PCIE_AUTO_ADD.
- STRATA_HC8_DOWN / HC8_UP variants.
- STRATA_PF_FP4G_BULK / _R144 / _QOLD.
- STRATA_PROXY_L2.
- STRATA_MTP_HIGH_PRIO, STRATA_MTP_PDL, STRATA_MTP_COND (conditional graph), STRATA_MTP_AHEAD.
- STRATA_MTP_STEP_COPY.
- STRATA_L2PF (MB of read-ahead).

Diagnostics:
- STRATA_ADAPT_LOG, STRATA_WIN_LOG, STRATA_HOST_TL;
- STRATA_HEAD_IN_DUMP, STRATA_MTP_DUMP, STRATA_HEAD_SCREEN_CHECK;
- STRATA_BENCH_STAGED (parity harness only).

On by default with an =0 way back (not opt-in):
- STRATA_ADAPT_EARLY;
- STRATA_PCIE_BRANCH (=1);
- STRATA_FLAGB_DIRECT (=2), STRATA_DMA_FLUSH;
- STRATA_HC8_NV, STRATA_HC8_PDL;
- STRATA_GDN_MID_SPLIT / _EF (1).

Not merged from the host agent: STRATA_HC_RT, round 5's GDN_REC / ROUTE_DB / GDN_LAZY, STRATA_PCIE_CE.

### git diff --stat 2572b0e0f (tracked files; nothing committed)

41 files changed, 3909 insertions(+), 190 deletions(-). The largest:

| file | lines changed |
|---|---|
| src/core/mtp.cpp | 739 |
| src/core/verify.cpp | 629 |
| src/kernels/cuda/qsa_decode_attn.cu | 580 |
| src/program/generate.cpp | 272 |
| src/kernels/cuda/native_mmvq.cu | 262 |
| src/kernels/gdn_parity.cpp | 205 |
| src/prefill/mmq_nvfp4_a44.cu | 203 |
| src/prefill/prefill.cpp | 160 |
| src/kernels/cuda/fused_gr.cu | 139 |
| src/core/native_dense.cpp | 136 |

The full list is `git diff --stat 2572b0e0f`.

New untracked source files (22):

| kind | files |
|---|---|
| headers | include/strata/kernels/{dense_tc,draft_proxy,gdn_mid,mtp_cond,mtp_route,q80_vhead}.hpp, include/strata/spec/draft_cum.hpp |
| kernels | src/kernels/cuda/{dense_tc,draft_proxy,gdn_mid,mtp_cond,mtp_route,q80_vhead}.cu, src/kernels/cuda/hc8_read.cuh |
| prefill | src/prefill/mmq_nvfp4_fp4g.cu |
| tests | src/kernels/{draft_proxy_parity,gr_hc8_check,q80p_parity,qsa_post_parity}.cpp, src/kernels/q81_div_parity.cu, src/prefill/fp4g_parity.cpp |

The licence/attribution of the Basalt-derived kernels (MIT) needs a line in the release notes.

## Milestone A (integration) - 2026-11-19

### What was merged
| source | change | default |
|---|---|---|
| dec-head (NOTES-head.md) | Q8_0 head as bit planes + two-stage exact verify head (q80_vhead.cu, native_head, verify.cpp); readers through NativeHead::gather / mmvq (mtp.cpp, dflash2.cpp) | ON (exact) |
| dec-head | STRATA_MMVQ_V2 k+v grouping beside the query branch; STRATA_FUSE_HITS on CUDA | off (neutral / slower) |
| dec-mtp (NOTES-mtp.md) | `--mtp-head proxy`, `--mtp-cum B[:L]`, draft-graph overlap (STRATA_MTP_OVERLAP, default on), rt-tuned draft layer as an rt dir | flags; serve Auto uses them since milestone B; overlap on |
| dec-hc (NOTES-hc8.md) | hc8: 8-bit HC read for sm_120 under STRATA_HC_Q8=1 | opt-in at milestone A; ON by default since milestone B |
| dec-pool | not merged (all neutral) | - |

Mechanics: files only one worktree touched were copied; the two shared files (cmake/strata_gpu_engine.cmake,
src/core/mtp.cpp) were three-way merged (git merge-file, no textual conflicts). All files CRLF.

### The conflict (head planes vs the MTP proxy head) and its resolution
dec-head rewrites the main Q8_0 head **in place** as bit planes (80 fp16 scales | bits 2..5 | bits 6..7 | bits 0..1 per
row, same 2720 bytes). dec-mtp's proxy draft head read that buffer as plain ggml Q8_0 rows in two places: the 2-bit
copy's pack (`draft_proxy_pack`) and the exact rescoring of the <= 64 candidates (`final_kernel` -> `q80_row_dot`).
On the merged tree both would have read garbage. Fix (src/kernels/cuda/draft_proxy.cu, include/strata/kernels/draft_proxy.hpp,
3 call sites in src/core/mtp.cpp): both functions take `bool planes` (= `NativeHead::planes()`); with planes,
`pl_words()` rebuilds the two Q8_0 words of each quarter block from the planes bit for bit (q80_vhead.cu's `codes()`),
`pl_scale()` reads the fp16 scale; the dot product then runs the identical dp4a / fma order (template `q80_row_dot<PLANES>`).
So the proxy's draft ids and probabilities are the same bits whatever layout the head holds.
New test `draft_proxy_parity --selftest` (ctest): synthetic 65,536-row head with 1,000 near-copy rows (forces many
candidates into the exact pass), a 45,000-row map with empty context rows, the 2-bit copy packed from both layouts
(byte-identical) and 96 columns x both tiers: **PASS, 193 checks, 0 failures**.

### Exactness evidence
Parity tests on the merged build: `q80p_parity --selftest` PASS (121 checks, 0 failures); `gr_hc8_check --selftest`
0 failures; `draft_proxy_parity --selftest` PASS (193/0).

Fixed-placement CLI runs (bench-int/exact.sh: STRATA_PREFILL_CPU_SHARE=0 STRATA_ADAPT_WAIT=1 STRATA_PCIE_BALANCE=0
--expert-cache 8000 --pcie-frac 0 --adapt-every 100000 --suffix-draft 0; the slot count fixed, so the drafter's VRAM
does not move placement), hc8 OFF. md5 of the output ids (bench-int/exact_all.txt):

| fixture | base exe DFlash2 | int DFlash2 | int no drafter | int new MTP (proxy, cum 0.43:0.31, spec 8) | same, STRATA_HEAD_PLANES=0 |
|---|---|---|---|---|---|
| conv greedy 400 | 293b92d3b9 | 293b92d3b9 | 293b92d3b9 | 293b92d3b9 | 293b92d3b9 |
| struct greedy 2000 | 30b9f645ca | 30b9f645ca | 30b9f645ca | 30b9f645ca | 30b9f645ca |
| prose sampled (T 1.0, top_p .95, top_k 20, seed 42) 600, DFlash2 | ac150e0893 | ac150e0893 | | | |
| prose sampled, the profile's MTP | ac150e0893 | ac150e0893 | | | |

- Window statistics match too: base vs int DFlash2 conv 143 rounds / 257 of 379 accepted, struct 350 / 1655 of 1745;
  int MTP planes vs STRATA_HEAD_PLANES=0 conv 143 rounds 257 of 313 and the same window-size histogram, struct
  261 rounds 1742 of 1786 - i.e. the proxy on the bit planes proposes exactly what it proposes on plain Q8_0 rows.
- The screen ran: int windows log `N windows screened ... 0 windows every row exact`; greedy 6-41 exact rows a screen,
  sampled (K = 20) 862-975.

### Combined build, serve (greedy, fresh engine per fixture, ctx 66661, tray profile args), 2026-11-19 12:55-13:45, interleaved per fixture
bench-int/serve_all.sh 1 -> bench-int/serve/*_1_*.log, summarized by bench-int/sum.py. One sample per arm and fixture
(conv/prose/uk change text between runs; struct is the stable one). STRATA_DECODE_TIMING=1 (+ STRATA_DRAFT_TIMING=1) on
every Strata arm. bDF2 = base exe + DFlash2 as the tray runs it; iDF2 = int exe, same flags; iMTP8 = int exe, the
profile's MTP flags + `--mtp Strata-data\mtp\rt-tuned --mtp-head proxy --mtp-cum 0.43:0.31 --spec 8`; iMTP8hc = the same
+ STRATA_HC_Q8=1 (opt-in, rounding differs); B8 = Basalt default (spec 8), our GPTQ DQ8 pack.
Columns: tok/s | tokens/window | ms/window (wall) | avg T.

| fixture | bDF2 | iDF2 | iMTP8 | iMTP8 + hc8 | Basalt |
|---|---|---|---|---|---|
| conv | 214.5 / 2.54 / 11.82 / 3.14 | 223.5 / 2.81 / 12.57 / 3.40 | 221.7 / 2.77 / 12.51 / 3.40 | 239.2 / 2.67 / 11.15 / 3.16 | 249.1 / 2.96 / 11.88 / 3.63 |
| struct | 339.4 / 5.58 / 16.44 / 5.98 | 339.2 / 5.58 / 16.45 / 5.98 | 379.4 / 7.77 / 20.49 / 7.90 | 401.2 / 7.75 / 19.31 / 7.90 | 472.0 / 7.74 / 16.40 / 7.90 |
| prose | 210.2 / 2.29 / 10.90 / 3.19 | 207.4 / 2.19 / 10.54 / 3.08 | 222.2 / 2.33 / 10.50 / 2.96 | 238.9 / 2.36 / 9.87 / 2.93 | 276.2 / 2.26 / 8.19 / 2.90 |
| uk | 154.5 / 1.32 / 8.54 / 1.72 | 156.2 / 1.33 / 8.55 / 1.73 | 206.3 / 1.80 / 8.74 / 2.11 | 214.2 / 1.79 / 8.36 / 2.13 | 239.9 / 1.55 / 6.47 / 1.96 |

Window cost split (ms/window, decode timing line): verify = GPU-reach wait (host waits for the GPU to reach the next
layer's doorbell) + per-layer host (the CPU pool); plus draft.

| fixture | arm | verify | GPU-reach | CPU pool | draft | CPU experts / layer (entries) | PCIe / layer |
|---|---|---|---|---|---|---|---|
| conv | iMTP8 | 11.58 | 5.73 | 5.27 | 0.79 | 2.15 (2.35) | 0.61 |
| conv | Basalt | 11.30 | 4.00 | 6.24 | 0.53 | 2.78 (3.46) | 0 |
| struct | iMTP8 | 18.86 | 9.10 | 8.87 | 1.52 | 3.59 (3.68) | 1.26 |
| struct | iMTP8 + hc8 | 17.69 | 8.12 | 8.69 | 1.51 | 3.43 (3.51) | 1.32 |
| struct | Basalt | 15.40 | 5.40 | 9.10 | 0.99 | 4.09 (4.52) | 0 |
| prose | iMTP8 | 9.67 | 5.69 | 3.45 | 0.71 | 1.46 (1.61) | 0.33 |
| prose | Basalt | 7.70 | 3.88 | 2.94 | 0.45 | 1.32 (1.56) | 0 |
| uk | iMTP8 | 8.05 | 5.40 | 2.17 | 0.55 | 0.96 (1.07) | 0.15 |
| uk | Basalt | 6.08 | 3.73 | 1.63 | 0.35 | 0.76 (0.84) | 0 |

Reading (measured): int vs base with DFlash2 is the same speed (head -0.1 ms/window is inside the noise; conv/prose
texts differ). The new MTP at spec 8: struct +11.8% over base DFlash2, prose +5.7%, uk +33.5%, conv +3.4% (one sample
each). hc8 adds +4..8% on top (opt-in). Basalt stays ahead: struct +24% over iMTP8 (+18% over iMTP8+hc8), prose +24%,
conv +12%, uk +16%. **The gap is almost entirely the GPU-reach wait** (struct 9.10 vs 5.40 ms, prose 5.69 vs 3.88,
uk 5.40 vs 3.73) plus the drafter (struct 1.52 vs 0.99); the CPU pool's time is equal or lower here even though Basalt's
pool computes more experts (no PCIe share in Basalt; its adaptive tier swaps ~32 experts/window, ours ~2).

## Milestone B (the owner's new defaults) - 2026-11-19

Owner's rule (via the coordinator): on by default whatever speeds things up and costs quality not much more than the
run-to-run noise; each piece with its own `=0` way back, only on the card class it was measured on.

### What is on by default now
| piece | where it is decided | way back | rounding |
|---|---|---|---|
| two-stage bit-plane head (q80p_vhead) | always (Q8_0 head) | STRATA_HEAD_PLANES=0 | exact |
| hc8 (8-bit HC read) | src/core/nv_defaults.cpp, cc 12.0 table | STRATA_HC_Q8=0 | changes (KL 0.0152 vs noise 0.0118, accepted) |
| equal prompt chunks | same table, STRATA_PREFILL_EQUAL=2 | =0 | chunk-order rounding |
| GDN prompt scan on tensor cores + fused conv/l2 | same table, STRATA_GDN_TC=1 STRATA_GDN_CONVL2=2 | =0 | ~1e-3 first-token KL |
| one-pass f16 products of the 2-tile scan | same table, STRATA_GDN_TC_NP=1 | STRATA_GDN_TC_NP=3 | f16 (see gdn_rec_parity below) |
| prompt-path buffers: one group's GU/H/Hq + HC scratch in the region, xn16 outside (GDN agent's refill item) | same table, STRATA_PF_GROUP_SCRATCH=1 STRATA_PF_HC_REGION=2 | =0 | exact (checked below) |
| grouped FP4 expert GEMM, mode 2 (pfx agent, round 3 kernels) | src/prefill/moe_mmq.cu fp4g_mode(): 2 where the 12x unit runs | STRATA_PF_FP4G=0 | changes (first-token KL vs fp16 not worse than MMQ) |
| new MTP drafting as serve's Auto | strata_control/drafting.py apply() | STRATA_AUTO_NEW_MTP=0 (old Auto), or drafting chosen by name | exact (drafting never changes output) |

`STRATA_NVFP4_SM120=0` (or STRATA_DEFAULTS=026) turns the whole cc 12.0 table off; FP4G and the Auto drafter have their
own switches. Start-up prints `strata: sm_120 (strata-nvfp4): 7 speed switches on by default ...`.

#### Drafting Auto (strata_control/drafting.py)
- Every Qwen3.8-Flash-Next set where a drafter runs today (the DFlash2 sets, the GPTQ set, its uncensored profile, the
  NVIDIA NVFP4 sets that draft with their own MTP; not Coder / Swift / other sizes), regardless of quantization, on
  CUDA, one request at a time, not a layer split, with an engine that knows the flags (`--mtp-cum auto` in the exe):
  Auto = MTP with `--mtp-head proxy --mtp-cum auto --spec 8` (person's own --spec kept; --spec-min-p dropped).
- Draft layer: the self-distilled `rt-tuned` where one is on this computer (`<mtp>/rt-tuned`, `<data>/mtp/rt-tuned`,
  STRATA_MTP_TUNED_DIR, the dev folder D:\Projects\Strata-data\mtp\rt-tuned), else the set's own layer (clean
  fallback). **Shipping the tuned layer needs a licence check** (it is Basalt's published retraining).
- Proxy head where the head is the Q8_0 2560-column one on the drafter's card; elsewhere the engine prints one line and
  runs the plain draft head with the same rule (MtpDrafter::bind_proxy, unchanged).
- Window threshold per card, not per quant: new engine flag `--mtp-cum auto` (generate.cpp) = Basalt's
  draft_cum_scale: 0.35:0.25 x (1 + 0.5 t), t = (CPU-served MB per expert - 1.1) / 2.4 clamped, CPU-served MB = experts
  outside the VRAM cache x mean slot bytes / (layers x experts). On the 5090 serve start: 1.77 MB -> **0.399:0.285**
  (Basalt's own serve start on the same pack: 39.8 GiB in RAM / 24576 = 1.74 MB -> 0.397). Window: Basalt's fixed 8.
- Tests: tests/terminal/test_drafting_fork_gptq.py (8/8 OK, incl. the new rule, tuned dir, engine without the flags,
  non-CUDA, named DFlash2). The older drafting tests describe the old Auto; tests/terminal/conftest.py runs them with
  STRATA_AUTO_NEW_MTP=0 (their pre-existing failures on this Windows box are the same as on the base tree: test_drafting
  5, hardware 2, switch 1, feed_aarch64 3, feed_models_screens 1).
- Other quantizations: **verified in Basalt only** (tuned layer acceptance on NVIDIA NVFP4 + Q6_K / + Q8_0 / our GPTQ:
  conv 71.4 / 72.1 / 73.8 %, JSON 98.6 / 98.2 / 97.9 %, prose 68.0 / 68.7 / 66.5 %, basalt-work/comm_*.txt); only the
  qwen-nvfp4-gptq pack exists here in Strata format.

### Merged in this milestone
- GDN agent (strata-dec-gdn): src/prefill/kernels.cu (2-tile NP 1/2), the equal_chunk block of generate.cpp,
  docs/DETAILS.md; then its refill item (prefill.cpp, three-way merged with git merge-file onto the pfx changes, no
  conflicts; DETAILS row).
- pfx agent (strata-dec-pfx), rounds 1 + 3: mmq_nvfp4_fp4g.cu (round 3 whole: flat tile list, persistent down),
  mmq_nvfp4_a44.cu, fp4g_parity.cpp, moe_mmq.cu/.hpp, mmq_absent.cpp, prefill.cpp hook, cmake/backends/cuda.cmake.
  STRATA_PF_FP4G_R144 / _BULK stay off.
- **FP4G x GROUP_SCRATCH**: with GROUP_SCRATCH on, the streamed walk's per-range path returned before the FP4G hook, so
  FP4G never ran. Fixed: the range path tries `fp4g_group` per range (range bounds, the range's scratch rows from 0)
  before its MMQ products (prefill.cpp, `if (m.group_cap > 0)` block). Evidence: 16K output with FP4G differs from
  STRATA_PF_FP4G=0 (a13d259b7f vs 0291291206), 64K read 5934 vs 6520 ms.
- **FP4G small-group cutover** (pfx open point): fp4g_parity at 16 experts x R rows, MMQ / FP4G mode 2 ms: 16 0.037 /
  0.064, 32 0.045 / 0.063, 48 0.048 / 0.059, 64 0.056 / 0.071, 96 0.074 / 0.075, 160 0.110 / 0.097 (round 1 kernels;
  round 3: 160 0.110 / 0.085). Chosen: MMQ below 96 rows per expert on average in a group / range
  (STRATA_PF_FP4G_MIN_ROWS, default 96).
- Tried and left OFF: `STRATA_MMVQ_CSPLIT=1` (Q8_0 windows of 5..8 columns as two column groups of neighbouring blocks,
  bitwise the same) - b6_mmvq_bench: slower (gdn qkv 8 columns 0.0264 -> 0.0343 ms, 5 columns 0.0196 -> 0.0258).
  `STRATA_L2PF` (opt-in L2 prefetch of the next layer's weights): its first version allocated inside graph capture
  (fixed: a __device__ sink), not measured since.

### Tests on the final tree
- nv_defaults_test ok (7 switches for cc 12.0; STRATA_DEFAULTS=026 empty).
- q80p_parity PASS 121/0; draft_proxy_parity PASS 193/0; gr_hc8_check 0 failures.
- fp4g_parity (round 3): H quantizer same bytes; m1 relRMS vs MMQ 7.8e-8..1.4e-5, m2 8.2e-3; down vs FP64 w4a8 0.61%,
  two FP4 terms 0.55% - the same numbers as round 1.
- gdn_rec_parity: 0 failures. New: the split plan's variants (kernels.cu `gdn_rec_tc_split_for_test`, linked from
  strata_prefill). NP 3 (1 and 2 tiles): output 1.2-1.5e-6 relative to the FP32 reference (FP64: 1.3e-6). **NP 1 / 2:
  output 5.8e-4 / 7.1e-4, state up to 6.9e-4** (FP64 the same) - the f16 operands, Basalt's BF16 scan is ~1e-3. They
  would fail the 1e-4 bound of the exact variants; they now have a 2e-3 bound and are labelled f16. Whether that shows
  at 128K: the long check below.

### Exactness on the new defaults (fixed placement, bench-int/exact2.sh -> exact2.txt)
| fixture | int D | int none | int new MTP (proxy, cum auto, 8) | same, HEAD_PLANES=0 | same, MTP_OVERLAP=0 | same, CSPLIT=1 | int profile MTP | int, STRATA_NVFP4_SM120=0, DFlash2 | base exe |
|---|---|---|---|---|---|---|---|---|---|
| conv greedy 400 | 5d1c43d97f | 5d1c43d97f | 5d1c43d97f | 5d1c43d97f | 5d1c43d97f | 5d1c43d97f | 5d1c43d97f | 293b92d3b9 | 293b92d3b9 |
| struct greedy 2000 | 30b9f645ca | 30b9f645ca | 30b9f645ca | 30b9f645ca | 30b9f645ca | 30b9f645ca | 30b9f645ca | 30b9f645ca | 30b9f645ca |
| prose sampled 600 | 001bb1c22e (D) | | 001bb1c22e | | | | 001bb1c22e | | |

Drafters, head layout, the draft overlap and CSPLIT do not change the output on the new defaults; the defaults
themselves change conv (rounding, as expected), and with the table off the int build is the base exe bit for bit.
Prompt-path switches (bench-int/pfcheck.sh, greedy 48 after the prompt, no drafter):

| prompt | defaults | GROUP_SCRATCH=0 HC_REGION=0 | FP4G=0 | FP4G=0, both off |
|---|---|---|---|---|
| 16,675 | a13d259b7f, 1502 ms, 2190 slots lent | a13d259b7f, 1507 ms, 2887 | 0291291206, 1639 ms | 0291291206, 1662 ms |
| 66,616 | 400c330c19, 5934 ms, 2744 slots lent | 400c330c19, 5951 ms, 3663 | 400c330c19, 6520 ms | 400c330c19, 6472 ms |

GROUP_SCRATCH / HC_REGION: identical ids at 16K and 64K, with and without FP4G (exact). FP4G changes the 16K
continuation's rounding (as measured by pfx), not the 64K one here.

### Serve A/B (decode), 2026-11-19 15:05-16:00, 2 rounds interleaved, greedy, fresh engine per fixture
bD = base exe + DFlash2 as the tray runs it; iA = int exe with serve's Auto through the worktree's drafting.apply()
(bench-int/pf_client_int.py PF_DRAFT=auto: rt-tuned, proxy, cum auto -> 0.399:0.285, spec 8) and the cc 12.0 defaults;
B = Basalt default. (The decode exe of this A/B predates the prompt-path merges, which do not touch decode.) tok/s, mean
of 2:

| fixture | bD | iA | Basalt | iA vs bD | Basalt vs iA |
|---|---|---|---|---|---|
| conv | 210.9 | 250.0 | 249.7 | +18.5% | -0.1% |
| struct | 340.7 | 403.5 | 476.3 | +18.4% | +18.0% |
| prose | 208.3 | 227.4 | 278.1 | +9.2% | +22.3% |
| uk | 152.5 | 210.8 | 240.3 | +38.2% | +14.0% |

Uncensored tray profile (cvec projection layers 8-33 kept; pf_client_int.py PF_CVEC=1), 1 round: current drafter
(DFlash2, base exe) vs new Auto: struct 338.8 -> 405.6 (+19.7%), conv 221.9 -> 245.1 (+10.5%), prose 211.2 -> 232.6
(+10.1%), uk 150.4 -> 214.6 (+42.7%).

Smaller cards (VRAM emulated with --vram-reserve-mib; the drafter's VRAM is inside the budget), 1 round. bM = base exe
with today's Auto there (the plain MTP: --spec 6 --spec-min-p 0.7, own layer); iM = int, same flags; iA = int Auto:

| budget | fixture | bM | iM | iA | iA vs bM |
|---|---|---|---|---|---|
| ~24 GB (reserve 8144) | struct | 237.8 | 249.3 | 270.4 | +13.7% |
| ~24 GB | conv | 159.0 | 171.3 | 174.7 | +9.9% |
| ~16 GB (reserve 16304) | struct | 130.0 | 136.7 | 144.6 | +11.2% |
| ~16 GB | conv | 108.9 | 114.1 | 120.7 | +10.8% |

Not tested: real 12/16/24 GB cards, other GPU generations (STRATA_EMULATE_CC), several cards, HIP / Metal (Auto keeps
the old rule off CUDA).

### Prompt read A/B (serve, one request, the whole new prompt stack: EQUAL=2 + GDN TC/CONVL2/NP1 + GROUP_SCRATCH/HC_REGION=2 + FP4G=2 round 3), 2026-11-19 ~16:50, 2 runs each, interleaved, ctx 140000
b = base exe, i = int exe (final build). ms (both runs):

| prompt tokens | base | int | change |
|---|---|---|---|
| 3,072 | 889 / 900 | 857 / 886 | -2.4% |
| 16,676 | 1939 / 1941 | 1658 / 1659 | -14.5% |
| 66,617 | 7879 / 7852 | 6175 / 6180 | -21.5% |
| 131,000 | 14483 / 14499 | 12493 / 12497 | -13.8% |

(131K = 4 chunks of 32,768 minus 72, so EQUAL=2 has no tail to even out there; most of its gain is FP4G + GDN.)

### Precision of the new stack (teacher-forced, score_strata.py: the tray args, DFlash2 spec 6, the forced 256 tokens
decode after each prompt; compare_long.py; bench-int/long/tf/)
Reference = base exe with `STRATA_PREFILL_NVFP4=fp16` (as NOTES-pfx.md). base / base2 = base exe twice; int / int2 =
int exe twice (all new defaults, **hc8 included**, so the all-rows KL carries hc8's decode rounding too); intnp3 = int with
STRATA_GDN_TC_NP=3.

corpus/long/set.json (12 items, 8K / 16K / 32K prompts, 3072 rows), against fp16:

| arm | KL all rows [95%] | argmax = ref | NLL - ref [95%] | first token KL mean / max | top-1 |
|---|---|---|---|---|---|
| base | 0.00792 [0.0056, 0.0106] | 96.84% | +0.0001 [-0.0061, +0.0079] | 0.00089 / 0.0064 | 12/12 |
| base2 | 0.00796 [0.0054, 0.0108] | 96.97% | +0.0006 | 0.00095 / 0.0080 | 12/12 |
| int | 0.01018 [0.0070, 0.0135] | 97.07% | -0.0032 [-0.0069, +0.0004] | 0.00135 / 0.0115 | 11/12 |
| int2 | 0.01140 [0.0071, 0.0159] | 96.39% | -0.0005 | 0.00126 / 0.0096 | 12/12 |
| intnp3 | 0.01032 [0.0074, 0.0136] | 96.88% | -0.0000 | 0.00170 / 0.0119 | 12/12 |

Against the base exe (noise floor = base2): base2 0.00566, int 0.01007 / int2 0.01076, intnp3 0.00982.

128K (bench-int/long/set_128k.json: 2 items of 131,000 tokens, en+uk Wikipedia and generate.cpp+en, 512 rows), against fp16:
base 0.01558 / base2 0.01712, int 0.01775 / int2 0.01913, intnp3 0.02256 (CIs all overlap, 2 items); NLL - ref: base
-0.016, int -0.019 / -0.006, intnp3 -0.020; first token mean base 0.00078 / 0.00053, int 0.00057 / 0.00085, intnp3
0.00067; top-1 2/2 everywhere. Against base: base2 0.00496, int 0.00902 / 0.01190, intnp3 0.01098.

Reading: the new defaults sit about 0.004 KL above the base's own run-to-run floor on all rows, the size hc8 alone
measured (0.0152 vs 0.0118, NOTES-hc8.md), with NLL not worse anywhere; the prompt stack's first-token KL against fp16
is the same as the base exe's at 8-32K and at 131K. NP 1 (f16 products, 6e-4 relative in gdn_rec_parity) is not worse
than NP 3 at 131K (0.0178-0.0191 vs 0.0226 against fp16; 2 items, overlapping CIs). Under the owner's rule this stays on.
Not isolated: the prompt stack without hc8 (no HC_Q8=0 arm).

### git diff --stat (2026-11-19, nothing committed)
     cmake/backends/cuda.cmake                 |   6 +-
     cmake/strata_gpu_engine.cmake             |  22 +-
     docs/DETAILS.md                           |   4 +-
     include/strata/core/dflash2.hpp           |   1 +
     include/strata/core/mtp.hpp               |  50 +++++
     include/strata/core/native_head.hpp       |   9 +
     include/strata/core/verify.hpp            |  19 ++
     include/strata/kernels/fused_gr.hpp       |   6 +
     include/strata/kernels/verify_kernels.hpp |  13 ++
     include/strata/prefill/moe_mmq.hpp        |  18 ++
     include/strata/spec/window_rule.hpp       |   7 +
     src/core/dflash2.cpp                      |  12 +-
     src/core/mtp.cpp                          | 350 +++++++++++++++++++++++++++---
     src/core/native_dense.cpp                 |  19 ++
     src/core/native_head.cpp                  |  36 ++-
     src/core/nv_defaults.cpp                  |  30 ++-
     src/core/nv_defaults_test.cpp             |   7 +-
     src/core/verify.cpp                       | 287 +++++++++++++++++++++++-
     src/kernels/b6_mmvq_bench.cpp             |   7 +
     src/kernels/cuda/fused_gr.cu              | 139 ++++++++++++
     src/kernels/cuda/native_mmvq.cu           |  60 ++++-
     src/kernels/cuda/verify_kernels.cu        |  47 ++++
     src/prefill/gdn_rec_parity.cu             |  31 ++-
     src/prefill/kernels.cu                    |  15 +-
     src/prefill/mmq_absent.cpp                |   4 +
     src/prefill/mmq_nvfp4_a44.cu              | 203 +++++++++++++++++
     src/prefill/moe_mmq.cu                    |  49 +++++
     src/prefill/prefill.cpp                   | 160 +++++++++++---
     src/program/generate.cpp                  |  69 +++++-
     strata_control/drafting.py                |  94 ++++++++
     tests/terminal/conftest.py                |   5 +
     tests/terminal/test_drafting_fork_gptq.py |  34 +++
     32 files changed, 1697 insertions(+), 116 deletions(-)
    new: ?? NOTES-int.md
    new: ?? include/strata/kernels/draft_proxy.hpp
    new: ?? include/strata/kernels/q80_vhead.hpp
    new: ?? include/strata/spec/draft_cum.hpp
    new: ?? src/kernels/cuda/draft_proxy.cu
    new: ?? src/kernels/cuda/hc8_read.cuh
    new: ?? src/kernels/cuda/q80_vhead.cu
    new: ?? src/kernels/draft_proxy_parity.cpp
    new: ?? src/kernels/gr_hc8_check.cpp
    new: ?? src/kernels/q80p_parity.cpp
    new: ?? src/prefill/fp4g_parity.cpp
    new: ?? src/prefill/mmq_nvfp4_fp4g.cu

## Part B on the final defaults (verify window) - 2026-11-19 evening

Scope since 14:50 (coordinator): the verify window only; the MTP draft step (mtp.cpp, draft graphs, draft_proxy) is
another agent's (strata-dec-draft).

### PCIe share at wide and narrow windows (serve Auto, 2 rounds interleaved; bench-int/q6_pcie.txt)
| fixture (T) | default share (auto 0.35) | --pcie-frac 0 | change |
|---|---|---|---|
| struct (7.9) | 399.8 / 391.4 | 370.2 / 375.8 | -5.7% |
| prose (2.9) | 234.9 / 229.0 | 221.9 / 223.7 | -4.0% |
| conv (3.5) | 234.8 / 242.0 | 207.4 / 207.0 | -13.1% |

Without the share the GPU-reach wait falls a little (struct 8.07 -> 7.56 ms) but the CPU pool rises more (8.75 ->
10.35 ms; 3.5 -> 5.0 experts a layer on the CPU). The share helps at every window width here: no T-dependent rule.
(Basalt needs none because its adaptive tier swaps ~32 experts a window against our ~2 and its pool is faster per
expert; the min-gain-0 arm of the adaptive tier was neutral: struct 379.1 vs 376.0/373.4, conv 222.5/221.2 vs
221.3/221.3. The opt-in MTP_PDL was neutral too: struct 370.4, conv 216.8 against 379.1 / 222.5.)

### Where the remaining time goes (STRATA_VERIFY_PROFILE=1 vs Basalt --profile, same fixtures, serve, final defaults)
ms per window; S = int at T 7.91 (struct), B = Basalt at 7.90; Sp / Bp prose at 2.90 / 2.90 (bench-int/serve/*b6_prof*).

| | S | B | Sp | Bp |
|---|---|---|---|---|
| wall | 20.08 | 18.28 | 10.92 | 10.06 |
| GPU total | 18.20 | 16.52 | 10.02 | 9.35 |
| GPU waits (waitA + waitB + waitCPU, both mixers) | 5.59 | 5.03 | 2.40 | 1.84 |
| GPU kernels | 12.61 | 11.49 | 7.62 | 7.51 |

At T=8 the kernel gap is 1.12 ms, made of (S - B): GDN qkv/q8 gemv +0.38, GDN conv +0.08, GDN rec -0.14, GDN
out-proj +0.06, GDN shared +0.08, VRAM hits -0.22 (fewer: the PCIe share took some), PCIe group +0.18 (GDN) +0.08
(QSA), copy+combine +0.13, **HC + router now equal** (GDN 1.99 vs 1.93, QSA 0.61 vs 0.61 - hc8 closed the old 1.2 ms
gap), QSA q8+kv-idx +0.10, QSA k/v + norm-rope +0.21, QSA scores/topk +0.08, attention +0.07, small rest +0.1. The
other 0.56 ms is the GPU waiting on the host (waitB: the CPU pool's results) - the CPU side, not a kernel.
At T~3 the kernels are within 0.11 ms of Basalt; the gap there (0.86 ms wall here, 1.4 ms in the decode-timing split)
is the GPU waiting on the host: per-layer host + CPU pool order, not GPU work.

### Exact candidates left, by size (not done)
1. **Q8_0 multi-column mmvq on int8 tensor cores, bit-exact**: each thread's partial in the exact layout is d0 * d1 *
   float(sumi) over an 8-value chunk (SmallTraits<Q80Block, 8>: 2 dp4a); an int8 MMA (m16n8k32, A rows = output row x
   chunk with the other chunks' k zeroed) gives the same int32 chunk sums, which can go through shared memory to the
   thread that owns them in the exact layout, so the float order is unchanged. b6_mmvq_bench (median ms, 1 -> 8 columns):
   gdn qkv 0.0174 -> 0.0264, gate 0.0136 -> 0.0196, out 0.0136 -> 0.0174, qsa q 0.0236 -> 0.0319; if 8 columns cost what
   1 does, ~0.019 ms a GDN layer x 36 + ~0.008 x 12 = **up to ~0.75 ms a window at T=8** (~3.5% struct), ~nothing at T~3.
   Several hours of kernel work + a bitwise test against native_mmvq_multi_kernel; the biggest exact item left.
2. QSA k/v + norm-rope (0.21) and q8+kv-idx (+0.10): already tried by dec-head as STRATA_MMVQ_V2 (k+v grouped, same
   bits): 0.18 -> 0.15 ms, neutral end to end; a norm_rope fused into the k gemv's epilogue needs a block per head (a new
   reduction order for the norm unless copied exactly). <= 0.2 ms.
3. GDN conv (+0.08), QSA scores/topk (+0.08), attention (+0.07): each under 0.1 ms.
Rounding-changing (owner's decision, not done): Basalt's dense_tc (int8 tensor-core GEMMs with its own activation
quantization) for the GDN qkv/z pair is where its 0.38 ms GDN gemv lead at T=8 comes from; other activation scales mean
other rounding, so it would need an accuracy run.

## Round 3 (2026-11-19 night): the drafter's patch, exact TCX mmvq, Basalt's dense_tc

### The drafter agent's draft-step patch (strata-dec-draft, applied by the coordinator)
Rebuilt; tests: nv_defaults_test ok, q80p_parity PASS 121/0, draft_proxy_parity PASS 769/0, gr_hc8_check 0, gdn_rec_parity
0, fp4g_parity same bytes. Identity matrix re-run (bench-int/exact3.sh -> exact3.txt): every md5 and window statistic the
same as exact2 (conv 5d1c43d97f, struct 30b9f645ca, sampled 001bb1c22e; table off = base 293b92d3b9). STRATA_MTP_HC8 /
STRATA_MTP_PDL stay opt-in. Serve, serve Auto, previous int exe (bench-int/exe_prev) vs the new one, 2 rounds:

| fixture | prev tok/s | new tok/s | draft ms/window prev -> new | step ms |
|---|---|---|---|---|
| struct | 392.2 / 401.7 | 402.1 / 397.0 | 1.53 -> 1.30 | 0.190 -> 0.154 |
| conv | 229.4 / 227.7 | 233.2 / 243.0 | 0.79 -> 0.70 | 0.191 -> 0.154 |
| prose | 232.6 / 231.7 | 233.1 / 228.0 | 0.70 -> 0.61 | 0.193 -> 0.155 |

The draft step is 19% shorter; tok/s +1.3% struct, +4% conv, prose within noise (the window's other parts vary more).

### Exact candidate #1: Q8_0 multi-column mmvq on int8 tensor cores (TCX) - built, bit-exact, SLOWER, removed
native_mmvq_q80_tcx_kernel (the exact layout's float order, int32 chunk dots from mma.m16n8k32 with A row 4 i + c = row
i's chunk c, routed through shared memory to the owning lane): b6_mmvq_bench `A=1col ok` for every case (the multi-column
outputs bitwise the single-column kernel's) - but 8 columns gdn qkv 0.0790 ms against 0.0276 (dp4a), gate 0.0524 vs
0.0193, qsa q 0.0995 vs 0.0318; 2 columns already 0.044 vs 0.0175. Keeping the exact float order means one float term per
8-value chunk per (row, column) on the owning lane, plus a shared-memory hop and scattered 2-byte weight loads: the
tensor cores save the dp4a but not the work around it. Removed from the tree (no switch).

### Basalt's dense_tc (int8 tensor cores, one exact int32 dot per 32-value block): opt-in STRATA_DENSE_TC=1
Ported whole from Basalt (MIT, src/kernels/cuda/dense_tc.cu, include/strata/kernels/dense_tc.hpp; namespace, includes,
logging, pdl names changed). Dispatch: native_q8_0_mmvq (native_mmvq.cu), every Q8_0 projection of 1..8 columns with
n_in % 512 == 0, cc >= 9 - all window widths, so a column's value never depends on the window (drafting still cannot
change the output). It uses the same q8_1 activations; it differs from dp4a only in rounding (one term per block, not
per 8-value chunk, and another summation order).

b6_mmvq_bench (ms, mmvq -> dense_tc): gdn qkv 1 col 0.0174 -> 0.0192, 4 0.0181 -> 0.0194, 8 0.0276 -> 0.0196; gate 8
0.0193 -> 0.0155; out 8 0.0172 -> 0.0156; qsa q 1 0.0234 -> 0.0256, 8 0.0318 -> 0.0257. Faster from 5 columns, slower
below.

Serve, serve Auto, build-b, 2 rounds (bench-int/serve_t8.txt):

| fixture | off tok/s | on tok/s | GPU-reach ms off -> on | CPU pool ms off -> on |
|---|---|---|---|---|
| struct (T 7.9) | 402.7 / 405.3 | 406.8 / 401.3 | 8.07 -> 7.62 | 8.47 -> 8.82 |
| prose (T 2.9) | 232.4 / 225.9 | 227.0 / 228.8 | 5.29 -> 5.35 | 3.21 -> 3.19 |
| conv (T 3.4) | 226.0 / 236.9 | 236.3 / 233.7 | 5.29 -> 5.31 | 5.36 -> 5.13 |

At T=8 the GPU side gains 0.45 ms a window, but the window does not get shorter: the CPU pool is then the critical path
(the host agent's area). At T~3 it is neutral.

Accuracy (teacher-forced, score_strata.py, build-b; d0 = off, d1 = on, each twice):

| | KL vs d0 | argmax = d0 | NLL - d0 |
|---|---|---|---|
| d0b (noise) | 0.0120 [0.0082, 0.0174] | 96.29% | -0.0014 |
| d1 | 0.0130 [0.0089, 0.0192] | 96.37% | +0.0004 |
| d1b | 0.0117 [0.0081, 0.0168] | 96.48% | -0.0031 |

Against the tray's strata_A reference: d0 0.0155 / 96.08%, d0b 0.0154 / 96.14%, d1 0.0158 / 96.00%, d1b 0.0150 / 96.12%.
Long set (corpus/long, against d0): d0b 0.00555, d1 0.00616, d1b 0.00616; first token 0.00038 / 0.00033 / 0.00021;
NLL +0.002 / +0.003 / +0.001.

Reading: dense_tc's cost is at the noise floor. It stays **opt-in** for now because it does not speed up serve yet: its
0.45 ms of GPU time at T=8 only turns into tok/s once the host side stops being the critical path. Turn it on (and try
Basalt's dense_tc_pair for GDN qkv + z in one launch, which the port includes) after the host agent's changes land.

## Round 4 (2026-11-19 late): the host agent's round 1 merged (strata-dec-host, NOTES-host.md)
Taken as the host worktree has them (its base was this tree at 17:15; none of these files changed here since):
src/program/generate.cpp (AdaptWorker, adapt_win_ev, adapt_due + set_served_hook around ver.run, joins on error/STOP,
the "adapt rounds" line), include/strata/core/verify.hpp + src/core/verify.cpp (set_served_hook, the PCIe branch pcb_*,
b_direct_ / raise_flag_b_ordered / fetch_dma / publish_plan / flagb_direct_mode / dma_flush_on / write_value32, tl_*
timeline behind STRATA_HOST_TL), fetch_blobs(..., blocks) in verify_kernels.cu/.hpp, metal/src/engine/stubs.cpp. All CRLF.
Defaults as the host agent set them: STRATA_ADAPT_EARLY=1, STRATA_PCIE_BRANCH=1 (16 copy blocks); STRATA_FLAGB_DIRECT=2
and STRATA_DMA_FLUSH=1 (they act in --pcie-mode dma only). Added here (generate.cpp, one env-gated block before
resident_stage_swaps): STRATA_ADAPT_LOG=FILE writes every round's chosen swaps (layer:in:out, in order), for the check below.

Tests: nv_defaults_test, q80p_parity, draft_proxy_parity (769/0), gr_hc8_check, gdn_rec_parity, fp4g_parity - pass.
Identity matrix (bench-int/exact4.sh -> exact4.txt): every md5 as in exact2/exact3.

### Adaptive tier ON: the swaps chosen per window (STRATA_ADAPT_EARLY 1 vs 0)
Serve, struct 2000 greedy, fixed placement start (STRATA_PREFILL_CPU_SHARE=0 STRATA_ADAPT_WAIT=1 STRATA_PCIE_BALANCE=0
--expert-cache 8000 --pcie-frac 0), serve Auto, the tier at its defaults; bench-int/swaps/*.txt (one line a round):

| arm | runs | swap-list md5 (131 rounds) |
|---|---|---|
| EARLY=0 (as before) | 5 | 0134cd10d6 in 5 of 5 |
| EARLY=1 | 5 | 0134cd10d6 in 4 of 5; **one run (e1b) differs from round 7 on** (130 rounds, 1739 of 1795 drafts accepted against 1738 of 1798) |

Rounds 1-6 of e1b equal the others; round 7 picks other swaps (another gain order), so the routing counts (drive.d.usage)
the round read were different - i.e. something before round 7 differed in that run. The logs show nothing else unusual.
**Resolved (dec/host round 2, NOTES-host.md R2.6): not a race.** DraftPolicy picks a suffix-draft window's width from
MEASURED round times; in the divergent run window 13 (p 161) went 8 wide instead of 5, the rows routed differently and round 7
followed. Timing-dependent in both EARLY modes (EARLY only changes what the rounds measure). With fixed draft costs
(STRATA_SPEC_FIXED=1 or --draft-costs) EARLY=1 gave one swap list and one window list in 10 of 10 runs (EARLY=0 2 of 2); the
host agent's audit of the worker's state (usage, host_res, pending, fetch_avail, in_flight, res_dirty) found no shared writer
during the overlap. **From here on every determinism test here runs with STRATA_SPEC_FIXED=1.** ADAPT_EARLY stays on.

### STOP mid-generation, then a request at once (bench-int/stop_test.py -> stoptest.txt)
Request 1 (struct, 2000) gets STOP after 300 streamed tokens; request 2 (prose, 300 greedy) follows immediately on the same
engine. EARLY=1 twice and EARLY=0 once: request 1 ends `cancel` at 305 tokens in all three, request 2's 300 ids are
identical in all three (09cfdf93b7), no error lines. The round that now still runs after a STOPped window is joined before
the break (generate.cpp) and its swaps land before the next request's prompt, as with EARLY=0.

### --pcie-mode dma: STRATA_FLAGB_DIRECT / STRATA_DMA_FLUSH (2 rounds, serve Auto)
struct 386.8 / 387.5 (defaults) vs 379.8 / 379.4 (both 0): +2.0%; prose 214.6 / 219.6 vs 212.9 / 216.9: +1.1%. No loss:
they stay on there. (dma mode is slower than the kernel mode here either way: struct ~387 vs ~417.)

### Serve A/B: previous int exe (bench-int/exe_prev2: the draft patch) vs this merge, 2 rounds, serve Auto
| fixture | prev tok/s | new tok/s | change | ms/window prev -> new |
|---|---|---|---|---|
| struct | 391.3 / 400.2 | 417.8 / 416.8 | +5.4% | 19.42 -> 18.59 |
| conv | 243.6 / 236.0 | 237.4 / 254.6 | +2.6% | 11.95 -> 11.83 |
| prose | 232.7 / 231.6 | 238.0 / 240.6 | +3.1% | 9.89 -> 9.37 |
| uk | 210.7 / 215.1 | 224.2 / 224.9 | +5.5% | 8.52 -> 8.23 |

Against Basalt's numbers of the same day (milestone B table): struct 417 vs 476, conv 246 vs 250, prose 239 vs 278,
uk 225 vs 240.

## Round 5 (2026-11-19 night): host round 2 merged, the x3p tier + dense_tc on the sm_120 table
Merged from strata-dec-host (NOTES-host.md R2.8; only src/program/generate.cpp, the serve `adapt` lambda): Swap.lout (the
victim's layer) in the fence / copy loop / srcp->prefetch / res_dirty_idx, STRATA_ADAPT_CAND_MIN (default 2.0),
STRATA_ADAPT_ACROSS (cross-layer ranking + its guards), STRATA_WIN_LOG. Added here (generate.cpp): STRATA_ADAPT_DECAY (the
tier's decay when no --adapt-* flag is given), STRATA_PCIE_AUTO_ADD (PcieAuto's share + N; default 0, unused by the table)
and STRATA_PCIE_FIXED=F (a native pack's start share, PcieAuto off; STRATA_PCIE_FIXED=0 gives PcieAuto back).

sm_120 table now 13 entries (nv_defaults.cpp; nv_defaults_test 13): + STRATA_ADAPT_FETCH=1, STRATA_ADAPT_ACROSS=1,
STRATA_ADAPT_CAND_MIN=1.5, STRATA_ADAPT_DECAY=0.96, STRATA_PCIE_FIXED=0.45 ("x3p"), STRATA_DENSE_TC=1 (every dense Q8_0
matrix of a 1..8-row window - never per width, so a column's value does not depend on the window width).

### PCIe share: PcieAuto + 0.10 vs fixed 0.45 (bench-int/serve_pf45.txt, 2 rounds, x3p tier)
The +0.10 rule settled at 0.50 on struct and 0.25 on prose (the pool reading moves with the share). Fixed 0.45:
struct 17.64 -> 17.22 ms/window (439.9 -> 449.8 tok/s, +2.2%), conv 11.45 -> 10.96 ms/window, prose 9.13 -> 9.11. Fixed wins.

### Identity, fixed draft costs (STRATA_SPEC_FIXED=1; exact5.txt = x3p build, exact6.txt = + dense_tc)
exact5: every md5 as in exact4. exact6 (dense_tc rounds differently from dp4a): conv 26c5e96e8e, struct 30b9f645ca, sampled
prose 10df564211 - one md5 per fixture over D / N / MTP proxy / HEAD_PLANES=0 / OVERLAP=0 / CSPLIT / M, i.e. drafting and
window width still do not change the output; STRATA_NVFP4_SM120=0 = base (293b92d3b9 / 30b9f645ca).
Swap lists, the tier ON (x3p, struct 2000, --pcie-frac 0, fixed placement + fixed costs): 1027e11904 in 3 of 3 (swaps3.txt).
Tests (tests12/13.txt): nv_defaults, q80p 121/0, draft_proxy 769/0, gr_hc8 0, gdn_rec 0, fp4g same bytes.

### Teacher-forced (corpus/set.json vs tf/strata_A.tsv; compare.py)
| arm | argmax = ref | KL | NLL - ref |
|---|---|---|---|
| B (noise floor: run A again) | 96.52% | 0.0118 | -0.0024 |
| d0 (previous int, round 3) | 96.08% | 0.0155 | -0.0015 |
| x3p (+ the tier, PcieAuto +0.10) | 96.03% | 0.0156 | -0.0029 |
| **def (every default: hc8 + x3p fixed 0.45 + dense_tc)** | 96.11% | 0.0150 | -0.0025 |
The placement change and dense_tc add nothing measurable over the previous int (whose +0.004 KL is hc8's).

### Serve (2 rounds, serve Auto, ctx 66661; final = the defaults build)
| fixture | base+DFlash2 | int round 4 (prev3) | int + x3p (+0.10 rule) | **int final** | Basalt |
|---|---|---|---|---|---|
| struct | 336.3 | 413.8 | 437.9 | **464.6** | 475.9 |
| conv* | 215.8 | 245.6 | 260.4 | **252.7** | 249.8 |
| prose | 205.1 | 241.2 | 242.1 | **242.6** | 277.9 |
| uk | 152.8 | 225.3 | 231.4 | **228.3** | 240.4 |
base / prev3 / x3p / Basalt: one interleaved hour (serve_final.txt); final: the next hour (serve_z.txt). *conv: placement
changes the greedy text (1000-1150 tokens), so it compares different texts. Struct ms/window 17.7 -> 16.65.
dense_tc alone at T=8 (x3p build, serve_dtc2.txt): struct 17.66 -> 17.29 ms/window (+2.0%), prose 9.17 -> 9.15.

Emulated budgets (--vram-reserve-mib, struct / conv tok/s): 24 GB prev3 275.7 / 177.1 -> x3p 288.5 / 186.3 -> final
304.3 / 192.1 (+10.4% struct); 16 GB 148.2 / 118.8 -> 149.5 / 125.9 -> 155.7 / 122.7 (+5.1% struct). Uncensored (PF_CVEC=1,
prev3 -> x3p +0.10 rule): struct 413.8 -> 441.9 (+6.8%), conv 240.8 -> 252.6, prose 243.2 -> 251.5, uk 221.2 -> 227.8.
No budget or profile loses, so the defaults are not scoped further.

## Round 6 (2026-11-20): slab (host R3), drafter round 2, QFUSE division check, the attention-layer front
### Host round 3: the dense slab
src/core/native_dense.cpp / .hpp (slab registry, dry pass + one allocation, dense_alloc at the five sites) and the
STRATA_TRACE lines of generate.cpp, merged three-way (git merge-file; my hc8 repack block was already in its snapshot).
dense_tc has no tensors of its own (it reads the Q8_0 weights in place), so nothing to route through the slab: the trace
says "native dense slab 3585.9 of 3585.9 MiB used". Slots 8856 -> 8952 (22.81 -> 23.05 GiB); struct misses per
layer-window 4.02 -> 3.89 (-3%), hit rate 96.4 -> 96.5..96.6%. Serve (serve_slab.txt, 2 rounds, Basalt same hour):
struct 460.3 -> 462.6 (16.78 -> 16.69 ms/window), prose 242.5 -> 250.8, uk 225.6 -> 229.7; Basalt 473.7 / 278.8 / 239.1.
The host tree's 454 -> 465 had another baseline (no dense_tc there); here the 3% fewer misses buy ~0.5% on struct.
exact7 = exact6 (fixed draft costs).

### Drafter round 2 (bench-draft/patch/round2-vs-int0214.patch)
Applied with git apply on LF copies, then CRLF again: mtp.cpp/.hpp, qsa_decode_attn.cu/.hpp (attn_merge_draft_gq),
draft_proxy.cu, new mtp_cond.* / mtp_route.*, cmake. Every file equals the drafter tree (cmake: plus my test lines).
draft_proxy_parity 769/0; the identity matrix (exact9/exact11) unchanged in every cell, MTP arms included.

### QFUSE's division (the drafter's question): no difference on this toolchain, and off by default
* STRATA_QFUSE is opt-in on CUDA (g_qfuse: env only; g_qfuse_hc: HIP gfx12 only), and neither tray profile sets it.
* src/kernels/q81_div_parity.cu (new): the fused quantizer (gdn_q8_1_store's code, STRATA_Q81_FASTMATH_DIV ->
  div.full.f32) against native_quantize_q8_1 (native_mmvq.cu, --use_fast_math) on 2^20 blocks, half of them built so
  x / d sits within 4e-7 of k + 0.5: 0 blocks differ (CUDA 13.3, sm_120a).
* gr_hc8_check now also compares the hc read's QFUSE q8_1 image byte for byte with native_quantize_q8_1: 0 of 640 blocks
  differ in each of the 4 cases.
So the released fork is not affected as shipped; with STRATA_QFUSE=1 the bytes also agree here.

### The attention-layer front (prose T~3, nsys, host agent's nsys_front_layer.py)
Base: attention front 97.2 us/layer (Basalt 64.0), GDN 81.9 (60.7). Kernel timeline of one attention front
(bench-int/nsys_front_list.py): hc8 read + q8_1 0-11 us; q|gate projection (dense_tc tile, 12288 rows) 11.7-35; k/v
24-37; norm-rope K, fwht K / V, append 37-44; scores / top-k 41-46; attention chunk 46-61 (14.6 us, Basalt 10.4); merge,
fwht, gate, q8_1 61-65; out-proj 65-78; FFN hc read 78-88; router mmvf, route, doorbell 89-97 (Basalt's front starts at its
q projection at 2.8 us: its attention-side hc read sits in the previous layer's combine; route + doorbell are inside its hc_fused).

Added (all exact, STRATA_KV_POST=0 restores the separate kernels):
* kv_rot_append_q8 (kv_q8.cu): fwht256 of K and V + kv_append_q8_steps in one launch (4 -> 2 kernels with norm-rope).
* qsa_gate_q8_1 (native_mmvq.cu): native_qsa_gate_apply + native_quantize_q8_1 in one launch.
* attn_merge_v2_kernel<U, true> (qsa_decode_attn.cu): the inverse rotation folded into the merge (the drafter's
  fwht-in-merge scheme). Merge, fwht, gate, quantize: 4 -> 2 kernels.
* qsa_post_parity (new test): 2560 byte-for-byte checks of all three against the kernels they replace.
Lesson (first attempt): a fused kernel must live in a translation unit with the SAME compile flags as the code it copies.
I first put fwht + append into native_rope.cu (--use_fast_math); the append's x / scale then used the fast division and
the identity matrix moved (conv 26c5e96e8e -> 71b000f364, defaults-off arm too) although a 240-check parity test passed.
Moved to kv_q8.cu (no fast-math), the gate + q8_1 to native_mmvq.cu (fast-math, like native_qsa.cu and the quantizer):
exact11 = exact7 in every cell.
* dense_tc_rows (dense_tc.cu): any subset of a Q8_0 matrix's 16-row-aligned rows, bitwise the whole call (parity
  checked). Used for STRATA_QPROJ_HALF=1 (opt-in): the query rows first, the gate rows later. Measured worse wherever the
  gate rows ran (they take the SMs of the kernels beside them): front 96.8 us whole vs 100.4 (after the K/V append),
  103.2 (beside the attention), 100.4 (after it, on the main stream). Off by default.
* Found and fixed on the way: qsa_decode_attn_batch_unrot first lacked its return (the window ran the attention twice).

Result (front_final.txt, front_var.txt): attention front 97.5 -> 96.7 us (-0.8); with the K/V fusion alone -1.5 us
(98.3 -> 96.8, q whole). Serve (serve_r6.txt; prev5 = the slab build, new = + drafter round 2 + these; 2 rounds,
Basalt same hour), tok/s (ms/window):
| fixture | prev5 | new | Basalt |
|---|---|---|---|
| struct (T~8) | 466.6 (16.52) | 466.7 (16.55) | 472.3 (16.39) |
| conv* | 251.7 (10.53) | 263.2 (11.06) | 248.3 |
| prose (T~3) | 250.8 (9.14) | 248.8 (9.28) | 275.2 (8.22) |
| uk | 229.2 (8.01) | 229.2 (7.99) | 239.5 (6.48) |
*conv: other texts. Net: no end-to-end change at T~3 or T~8 (inside the round-to-round noise of 1-2%); the drafter's
+2% on struct does not show here either (prev5 already at 466).

What is left of the 32 us (my reading of the timeline; not done in this round):
* the q|gate projection, 23 us on the critical path: the whole 33.5 MB; only a faster kernel helps (the half split does not);
* the attention chunk 14.6 vs 10.4 us (attn_chunk_kernel_pre75 = upstream's attn_chunk_kernel bitwise
  (qsa_merge_daily_parity), so switching between them changes nothing; Basalt's attn_chunk_v2 is another kernel);
* the attention-side hc read at the front (11 us; Basalt does it in the previous combine) and route + doorbell as two
  kernels (5.4 us a layer on ALL 48 layers, Basalt fuses them into hc_fused) - both cross into the host agent's
  per-layer flow (the combine, the doorbell), so I list them for the coordinator rather than change them.

## Round 7 (2026-11-20): staged INT8 attention chunks (Basalt's attn_chunk_v2) - DEFAULT on sm_120

`attn_chunk_staged_kernel` (qsa_decode_attn.cu, not fast-math, next to attn_chunk_kernel_pre75): Basalt's attn_chunk_v2
layout on our pools - the chunk's 64 K and V rows and their scales copied to shared memory by cp.async up front (one round
trip a block instead of ~20 dependent ones), the lane's 8 query dimensions straight into registers (no sq tile), 40.5 KB
static shared, `__launch_bounds__(256, 2)` (128 regs, 0 spill), carveout hint 100; values: 128 threads x 2 dimensions,
cells ascending, a masked cell staged as zeros. Same operands and order as pre75 (load8_q8 expression, reduce12, the
softmax partials, fmaf fold), so it is BITWISE the reference: no reduction-order change, no TF arm needed. Per-query
blocks: a row's result does not depend on the window's row count. Switch STRATA_ATTN_STAGED (=0 off; in the sm_120 table,
now 14 entries). INT8 only (kv_mode 1); other formats, HIP, slots/rows/step paths unchanged. No PDL (we do not use it).

Exactness:
* qsa_merge_daily_parity (+ staged arm, both mergers, int8): 1168 cases (n_q 1..6, caps 2051/2112/32768, widths -1..INT_MAX,
  mixed widths, page holes) 0 differ; also with STRATA_ATTN_STAGED=1 on the production entries: PASS.
* exact12 (fixed costs, STAGED=1): conv 26c5e96e8e, struct 30b9f645ca, prose 10df564211 in every arm = exact11; long:
  16K 8b542cd8cd both, 40K (prefill_64k's first 40000 tokens + template tail, bench-int/long/prefill_40k.txt) 7b402311b9
  both (106.2 / 106.6 tok/s). prefill_64k itself does not fit ctx 66661 with --max-new 400.
* defaults build (switch from the table): conv D 26c5e96e8e, struct M 30b9f645ca.

Speed:
* --bench (STRATA_BENCH_STAGED=1, chunk pass only, int8): cap 2051 T=3 width 2051 20.6 -> 10.7 us; T=1 18.5 -> 8.3;
  cap 32768 T=3 width 32768 114 -> 80 us (-30%).
* nsys prose (front_r7.txt): attention chunk 14.6 -> 7.5 us a layer (Basalt 10.4), attention front 96.6 -> 90.3 us.
* serve, same exe, env A/B, 2 rounds (serve_r7.txt), ms/window s0 -> s1: struct 16.41/16.39 -> 16.32/16.41 (flat),
  conv 11.07/11.04 -> 10.94/10.98, prose 9.29/9.21 -> 9.23/9.11, uk 7.98/7.97 -> 7.78/7.94 (-0.6..-1.3% at T 2-3.5).

Q|gate overlap in Basalt (record_qsa.cpp, qhalf): Q half on the main stream; the K, V, indexer-K and indexer-Q projections
on three side streams concurrently with it; the gate half on the indexer branch after its appends, overlapping q_post,
block scores, top-k, KV resolve and the attention chunks; the merge waits for it (merge_after) and applies the gate +
q8_1 itself (only without kv_rot). Ours (STRATA_QPROJ_HALF, round 6) is the same idea and measured slower (100-103 us);
our verify already has an indexer side branch (side_, STRATA_SIDE_MIXER); whether its K/V projections overlap
the Q projection like Basalt's three side streams was not checked this round. Not started.

## Round 8 (2026-11-20): attention-layer stream layout (Basalt's record_qsa.cpp qhalf) - in progress

What our layout overlaps today (nsys prose, layer 15 front, front_r7 g_s1 + nsys_front_list.py; STRATA_SIDE_MIXER=2 is the
sm_120 default, so it is already Basalt's split): graph branch A = indexer chain (idx-K bf16, tail snapshot, append,
idx-Q bf16 + norm/rope, block scores, top-k), then the attention on cs; branch B = q|gate dense_tc_tile + q norm/rope + fwht;
branch C = K, V dense_tc_one + K norm/rope + kv_rot_append. Times from the front start (us): hc read 0-10.5, q8_1 10.6,
q|gate tile 11.7-36.8 (25 us), K/V start only at 27.8, the indexer chain stalls 18.4 -> 30.1 (its 1-block kernels wait
for SMs behind the tile's queued blocks), scores 42.9, chunks 47.5-53.7, merge, gate+q8_1, o_proj 56.8-68.8, then the
host agent's hc/route/doorbell to 88. So the pieces are already off the main stream; the critical path is the q|gate
tile starving the two latency-bound chains, not missing concurrency.

Split placements (STRATA_QPROJ_HALF=1; STRATA_QPROJ_GATE_AT: 0 after the K/V store, 1 after the selection, 3 NEW = Basalt's:
on the queries' branch right after the q post, past its join; 4 NEW = 3 on the lowest-priority stream sh_cs_):
* exact13 (fixed costs): g1 and g3 conv 26c5e96e8e, struct 30b9f645ca, prose 10df564211 (D and P8 arms) = exact11.
* nsys front (front_r8.txt): whole 89.1, g0 93.2, g1 93.9, g3 92.3 us. g3 timeline: q half done 25.3, q post 29.1 (12 us
  earlier than whole), but the gate tile (29.2-48.9) starves idx-Q bf16 (29.7-43.4, 3 us alone) and K norm (41.2), so
  scores start 46.1 (whole 42.9).
* serve unprofiled, ms/window (2 rounds, serve_r8.txt):

| fixture | whole | g0 | g1 | g3 |
|---|---|---|---|---|
| struct | 16.47/16.49 | 16.49/16.64 | 16.40/16.42 | 16.38/16.63 |
| conv | 10.93/11.01 | 11.15/11.05 | 11.07/11.02 | 10.96/11.17 |
| prose | 9.19/9.14 | 9.22/9.27 | 9.12/9.23 | 9.16/9.25 |
| uk | 7.92/7.89 | 8.01/7.76 | 8.04/7.89 | 7.92/7.82 |

  No placement beats whole (all inside noise).

STRATA_GRAPH_PRIO=1 (existing opt-in: indexer branch -5, cs / side2 -2, shared expert 0; captured from the streams) and
GATE_AT=4 (gate rows on sh_cs_, priority 0): exact14 = exact11 set. nsys: whole+prio 87.2, g4+prio 90.8, g3+prio 91.4.
Priority only orders the dispatch of queued blocks: the gate tile is a persistent kernel (grid = occupancy x SMs, each
block loops over tiles), so once its blocks are resident the chains still wait; block_topk_cluster needs a whole cluster
free and waited 8 us behind it.

STRATA_QPROJ_GATE_GRID=N (new): dense_tc_rows / launch_tile cap the persistent grid at N blocks (a tile's arithmetic does
not depend on the block that takes it; qsa_post_parity runs a capped arm: PASS 2560). nsys front: c24 88.8, c48 85.4,
c80 88.6, c48 + prio 84.0 (timeline: gate tile 33.9-48.2 on 48 blocks ends with the merge, 47.4; scores at 30.3 instead
of 42.9), c48 at GATE_AT=1 86.3. exact15 (c48 + prio): conv 26c5e96e8e, struct 30b9f645ca, prose 10df564211.
Serve ms/window, 2 rounds (serve_r8c.txt), mean vs whole:

| fixture | whole | prio | c48 | c48+prio | c64+prio |
|---|---|---|---|---|---|
| struct | 16.39/16.50 | -0.4% | -0.9% | 16.22/16.22 -1.4% | +0.2% |
| conv | 10.98/10.91 | +0.2% | +0.9% | 10.91/10.97 0.0% | +0.8% |
| prose | 9.07/9.24 | -0.5% | 0.0% | 9.03/8.99 -1.6% | -1.2% |
| uk | 7.86/7.84 | -1.1% | -0.3% | 7.79/7.83 -0.5% | +0.1% |

Decision: c48 + prio as the sm_120 default (QPROJ_HALF=1, QPROJ_GATE_AT=3, QPROJ_GATE_GRID=48, GRAPH_PRIO=1; bitwise,
faster or equal everywhere). 48 is tuned on the 5090's 170 SMs. The front is 84 us against Basalt's 64.8: the rest is
the q half tile starving K/V (they start 15 us late) and the host agent's hc read / route / doorbell.

## Round 9 (2026-11-20): host's gdn_mid + round-7 step A merged, default on sm_120

Taken from D:/Projects/strata-dec-host (NOTES-host R6.8 / R7.1), hunk by hunk (not its round 5 GDN_REC / ROUTE_DB /
GDN_LAZY, not STRATA_HC_RT / hc8_read_rt_kernel and its device-function refactor):
* new src/kernels/cuda/gdn_mid.cu + include/strata/kernels/gdn_mid.hpp (copied; not fast-math); cmake entry;
* metal stubs (include + gdn_mid_supported / gdn_mid);
* gdn_parity.cpp section 8 (+ dense_tc / gdn_mid includes);
* verify.cpp: gdn_mid_on() (STRATA_GDN_MID=1 and STRATA_DENSE_TC=1), qcnt_ allocated with it, the first HC read of a
  GDN layer writes xq_'s q8_1 (gr_q8_tail), mid_on in the GDN branch (side forks skipped), the gdn_mid call;
* hc8_read.cuh: hc8_upq_kernel's launch_dependents before its wait (CRLF kept); gdn_mid.cu's pdl_trigger is in the copy;
* nv_defaults: STRATA_GDN_MID=1 (table 19 entries with round 8's four); STRATA_GDN_MID=0 is the way back.
* verify.cpp: STRATA_DF_PDL=0 now means off (it was any value set = on).
Checks: gdn_parity section 8 bitwise (32 cases), all other sections bitwise; nv_defaults_test ok; qsa_post_parity PASS;
qsa_merge_daily_parity PASS. Engine (fixed costs): defaults conv D 26c5e96e8e, GDN_MID=0 conv D 26c5e96e8e, prose Ds
10df564211; DF_PDL=1 conv D 26c5e96e8e, struct P8 30b9f645ca.
nsys prose (front_r9.txt), per-layer front attention / GDN: defaults 81.8 / 79.2, GDN_MID=0 82.1 / 82.4, + DF_PDL=1
79.0 / 76.8 (GDN back 80.7 -> 77.0).
* nv_defaults: + STRATA_DF_PDL=1 (table 20). Short A/B (serve_pdl.txt, 1 round): prose 8.99 -> 8.70 ms/window,
  struct 15.96 -> 16.03, struct with 16 GB reserved 48.98 -> 48.42; bitwise (conv D, struct P8 above).

Full set, once, on the final build (q26; rounds 7-9 defaults: ATTN_STAGED, QPROJ_HALF/GATE_AT 3/GATE_GRID 48/GRAPH_PRIO,
GDN_MID, DF_PDL):
* tests26.txt: nv_defaults_test ok, qsa_post_parity PASS, qsa_merge_daily_parity PASS, gdn_parity all bitwise (section 8:
  32 cases), q80p / draft_proxy / gr_hc8 / gdn_rec / fp4g / q81_div all 0 failures.
* exact17 = exact11 in every cell: conv 26c5e96e8e, struct 30b9f645ca, prose 10df564211, off arm 293b92d3b9.
  All four changes are bitwise, so no TF arm.
* serve, 2 rounds, same hour (serve_r9.txt), tok/s; p = exe_r6 (round 6), i = this build, B = Basalt:

| fixture | r6 | int | Basalt |
|---|---|---|---|
| struct | 468.0 / 465.4 | 485.3 / 479.9 | 474.7 / 476.4 |
| conv | 261.5 / 263.1 | 275.2 / 264.6 | 250.2 / 251.0 |
| prose | 251.2 / 249.2 | 262.1 / 268.0 | 278.9 / 273.6 |
| uk | 230.6 / 229.9 | 238.1 / 237.0 | 239.3 / 237.5 |
| struct, 16 GB reserved | 155.3 / 155.3 | 155.9 / 157.5 | - |

  ms/window r6 -> int: struct 16.56 -> 16.06 (-3.0%), conv 11.08 -> 10.61 (-4.3%), prose 9.22 -> 8.69 (-5.7%),
  uk 8.05 -> 7.47 (-7.2%). Int is now ahead of Basalt on struct and conv, level on uk, 3.5% behind on prose.

## Round 10 (2026-11-20 ~12:45): dec/front merged (sources only) - 3 more sm_120 defaults, all bit-exact
Applied D:\Projects\strata-dec-front\bench-front\front_vs_int.patch; the tree is now byte-identical to dec/front
(67 files compared, CRLF kept). New defaults (table 20 -> 23): STRATA_EARLY_READ=1 (layer l+1's first HC read launched
in post(l) on its own branch, gated on a combine counter; + STRATA_EARLY_SNAP), STRATA_GDN_MID_HEAD=2 (gdn_mid issues
only L2 prefetches before its grid wait, tile copies after - its 15 MB burst no longer stalls hc8_upq), STRATA_ROUTE_DB=2
(route + doorbell in one launch, rows to the host first). Way back: STRATA_EARLY_READ=0 STRATA_GDN_MID_HEAD=7
STRATA_ROUTE_DB=0. Front per layer: GDN 77.6 -> ~72, attention 79.1 -> ~75 us. Identity matrix unchanged (exactF.txt),
serve 2 rounds (serveF.txt): prose 259.4 -> 271.2 tok/s (ms/window -0.8%; part is window length), uk ms/w -2.3%,
conv -1.2%, struct flat; Basalt prose 276.1. 64K prompt 6163 ms (unchanged). Details: dec/front NOTES-front.md.
**build-dev here is NOT rebuilt yet** (the GPU-probe agent profiles prefill with this exe); the current exe for these
sources is D:\Projects\strata-dec-front\build-dev\strata.exe. Rebuild with build_dev.cmd when the probe is done.
