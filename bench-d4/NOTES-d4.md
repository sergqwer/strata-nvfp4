# Decode round 4 notes (bench-d4)

Tools: bench-d3's (gpu_free.sh, ident.sh, cc_check.cmd) plus:
- `ab.sh NAME A_EXE B_EXE`: bench-d3/ab.sh + A_ADD / B_ADD (extra engine args per arm through gpu-probe/r4/dec_client4.py's PF_ADD,
  e.g. "--spec 12"); each line also gets the slot count, the verifier's "window up to" line and the free VRAM at the request start.
- `ab_sum.py`: + ms/token (= ms/window / tokens/window: the figure to compare when the arms' windows differ), CPU, draft, slots.
- `build_w.cmd MAXT [target]`: build_dev.cmd's dev build with -DSTRATA_WINDOW_MAX_T=MAXT into build-wMAXT (CRLF; through gpu_lock.sh).

Absolute numbers: the CUPTI zombie (PID 22984, ~1005 MiB) and the desktop (~356 MiB) were on the 5090 during every run here, so
every arm has ~380 fewer cache slots than the tray will have (absolute tok/s ~4-5% struct / ~2% prose low). Relative results stand.

## E1: a wider verify window (DROPPED: slower on struct at every width past 8)

Power limit 600 W. NVML memory clock 14001 MHz nominal (the user's OC). 5090 headless. 05:31-05:40.

### The build
- The cap is ONE number: CMake cache STRATA_WINDOW_MAX_T (auto / 8 / 12 / 16; auto = 8 for CUDA, 16 for HIP) ->
  add_compile_definitions(STRATA_VERIFY_MAX_T=N) -> window_limits.hpp kWindowMaxT = kVerifyMaxT. cc_check.cmd's
  -DSTRATA_VERIFY_MAX_T=8 is that same macro, fixed to the release value.
- The wide build already existed upstream (HIP batching). Every T-templated part either has 9..16 instantiations behind
  `#if STRATA_VERIFY_MAX_T > 8` (native_bf16 router multi GEMV 9..16, native_mmvq MAX_NCOLS, s2_expert_grouped) or falls back:
  hc8 / fused_gr reads run in kFusedGrMaxT = 8 column slices (verify.cpp:2986), gdn_mid refuses T > 8 (gdn_mid_supported:
  MAXT = 8 -> the old GDN path), native_router's fused route+doorbell and the dense TC / interleaved Q8 paths are <= 8 only,
  the CPU pool takes jobs of <= cpu::MAXT = 8 tokens a blob, the QSA front / attention kernels are generic in T.
  Both builds (12 and 16) compiled and ran with no source change; exe 96.5 -> 103.6 MB.
- What did NOT exist: an MTP window past 8 rows. generate.cpp capped a speculative window at kVerifyNarrowT (spec_cap; the
  suffix rule's `min(spec + 2, 8)`), and the drafter's chain was max_t - 1 = 7 drafts (MtpDrafter: max_t <= 8 rows, step rows
  at max_t + j - 1, draft 0's row 2*max_t - 1). So `--spec 12/16` on any build stopped at 8, as the research saw.
- Built for the test (E1-dropped.patch, switch STRATA_WIDE_MTP, =0 the old cap; always off on the 8-row build):
  - mtp.hpp/.cpp: set_chain_cap(c) decouples the chain (C_ drafts, <= kWindowMaxT - 1) from the round's rows (max_t_ <= 8, the
    draft layer's kernels stay narrow); staging rows R2 = max_t + C_ + 1, draft 0's row = draft0_row() = max_t + C_ (both equal
    the old layout when C_ = max_t - 1); outputs / probs / top2 / coupled ring / q lists sized for C_ + 1; step_exec_ arrays
    [kWindowMaxT + 1]; draft_first rounds keep max_t rows. A window whose accepted row a >= 8 is caught up by the existing
    draft_long (prefill rows 0..a-1, then a one-row round).
  - generate.cpp: wide_mtp_on(); spec_cap and the suffix rule's cap = kWindowMaxT for --spec > 8; the drafter gets
    set_chain_cap(S_mtp - 1); the serve loop's drafts buffer holds the chain, dwin_s / tr_d hold kWindowMaxT (they were
    std::array<., 8>: cum_observe read past them at T_mtp > 9).
  - No kernel changed, so no cc_check was needed; the 8-row build is unchanged by construction.

### Slot cost (serve, tray GPTQ args, the auto cache)
| exe | slots | verify buffers | window graphs warm-up |
|---|---|---|---|
| d6 (8 rows) | 8323 | 91.2 MiB | 8 widths, 175 ms |
| w12, --spec 12 | 8320 (-3) | 115.6 MiB | 12 widths, 280 ms |
| w16, --spec 16 | 8316 (-7) | 140.1 MiB | 16 widths, 413 ms |
VRAM is not the problem: -7 slots = 18 MiB (~0.02 ms struct). Free at the request start 572 -> 516 (spec 12) / 462 MiB (spec 16) (the wider
verifier buffers and graphs come after the cache is sized).

### Serve A/B (ab.sh, tray GPTQ args, this tree's Auto = --mtp-cum auto, B_ADD = the spec; ab_smoke16/12/16s8, ab_struct10, ab_prose12)
| fixture | arm | n | ms/token | tok/s | ms/window | T | tok/window | reach | host (CPU) | draft | CPU experts / PCIe a layer-window | slots |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| struct | d6 spec 8 | 5 | 2.28-2.34 | 427-439 | 17.36-17.66 | 7.8-7.9 | 7.5-7.6 | 5.97-6.11 | 9.6-10.0 (9.1-9.5) | 1.18-1.23 | 3.25 / 1.74 | 8323 |
| struct | w16 spec 8 | 1 | 2.298 | 435.0 | 17.42 | 7.86 | 7.58 | 6.04 | 9.69 (9.26) | 1.19 | - | 8316 |
| struct | w12 spec 10 | 1 | 2.786 (+22%) | 359.1 (-18%) | 24.57 | 9.42 | 8.82 | 9.34 | 12.41 (11.83) | 1.90 | 3.98 / 2.33 | 8320 |
| struct | w12 spec 12 | 2 | 2.698 (+17%) | 370.7 (-15%) | 28.50 | 11.13 | 10.57 | 10.77 | 14.51 (13.85) | 2.23 | 4.6 / 2.9 | 8320 |
| struct | w16 spec 16 | 1 | 2.839 (+21%) | 352.4 (-17%) | 33.78 | 12.97 | 11.90 | 12.92 | 17.36 (16.62) | 2.45 | 5.59 / 3.65 | 8316 |
| prose | d6 spec 8 | 2 | 3.914 | 255.9 | 9.12 | 2.99 | 2.33 | 4.69 | 3.34 (3.19) | 0.65 | | 8323 |
| prose | w12 spec 12 | 2 | 3.956 (+1.1%) | 253.2 | 9.14 | 2.92 | 2.31 | 4.71 | 3.31 (3.16) | 0.66 | | 8320 |

- Every wide struct arm lost, 4 of 4 runs, by 15-18% tok/s; the narrow window on the wide build is equal (+1.6%, noise). Pairs
  past these were not run: the sign and size decide it.
- The cum rule does what it should: prose stays at T 2.9-3.0 with --spec 12 (the same windows, noise); struct goes to T 9.4 / 11.1 / 13.0.
- Acceptance holds (struct 1356/1455, 0.93 at spec 12; 0.91 at 16) - the drafts are not the problem.
- **Why it loses (measured):** a window row costs far more than the average token, at every width past 8.
  - CPU experts grow with the ROWS, rejected ones too: 3.25 -> 3.98 -> 4.6 -> 5.59 a layer-window (PCIe 1.74 -> 3.65) for
    T 7.8 -> 9.4 -> 11.1 -> 13.0. Neighbouring rows share few experts (entries per CPU expert stays ~1.03) and the tier's swaps
    per token fall (adapt is per window). The pool is DRAM-bound (D4), so CPU ms/token rises: 1.21 -> 1.34 -> 1.31 -> 1.40.
  - The GPU-reach wait steps up past 8 rows: 6.0 -> 9.3 ms for +1.5 rows (gdn_mid's fallback, hc8 / HC reads in two slices,
    the 8-row-only router / TC paths, twice the PCIe fetch).
  - The chain does not amortize: draft 0.16 -> 0.21 ms/token (one ~0.13 ms step per draft, plus draft_long's prefill path for a >= 8).
  - Even with GPU rows 9..16 as cheap as rows 1..8 (~0.9 ms a row, the research's fixed-T prose costs), spec 10 would be
    ~17.4 + 1.5 x (1.7 CPU + 0.9 GPU) + 0.4 draft = ~21.7 ms for 8.8 tokens = 2.46 ms/token against 2.28 (inferred): a row's
    CPU cost alone (~1.7 ms) is ~75% of today's whole per-token cost. So widening the T > 8 kernels (gdn_mid MAXT 16, 16-column
    HC reads) cannot rescue it on this card's cache size; it was not built.
- The research's +5..+12% assumed ~+1 CPU expert a layer for 4 more rows; measured it is ~+0.45 CPU and ~+0.4 PCIe a row.

### Exactness (CLI fixed placement, exact.sh arm M, struct 2000, tray MTP flags; cli_text.txt)
| exe / spec | rounds | accepted | text |
|---|---|---|---|
| w12 spec 8 | 258 of 8 | 1745 / 1777 | 30b9f645ca |
| w12 spec 12 | 181 of 12 | 1823 / 1880 | 30b9f645ca |
| w16 spec 16 | 149 of 16 | 1853 / 1965 | 30b9f645ca |
| w16 spec 8 | 258 of 8 | 1745 / 1777 | 30b9f645ca |
All = the reference struct hash (and d6's): the window width does not change the text, and the wide build at spec 8 has the 8-row
build's rounds and acceptance. (Fixed placement is pool-bound with ~43% VRAM hits, where the wider window IS faster - 86 -> 92 -> 96
tok/s - because there almost every expert is a CPU expert either way; serve is the case that counts.)

### Kept: nothing
- The source change is E1-dropped.patch (mtp.hpp, mtp.cpp, generate.cpp on top of round 3); the tree is restored
  (cumulative.patch = bench-d3's, byte-identical); E1.patch is empty.
- e1\ = d6\ byte copy (+ its DLLs): the comparison exe for E2. Its identity is D6's 7/7 (ident_d6.txt); no source change, so
  ident.sh was not re-run. strata_control/drafting.py unchanged (FORK_MTP_SPEC stays 8), so its tests were not touched.
- The release build stays 8 rows. (Both wide builds compile for 120-real; no kernel file changed, so the 4-target check is moot.)
- Files: E1-dropped.patch, ab.sh, ab_sum.py, build_w.cmd, ab_smoke16/12/16s8.txt, ab_struct10.txt, ab_prose12.txt, cli_text.txt,
  log/ (serve logs, build logs), w12\ and w16\ (the wide exes with the E1 change, + DLLs), e1w\ (the edit scripts, pre1\ = the
  three files before E1). build-w12 / build-w16 deleted.

## E2: elastic session tables under KV streaming (KEPT: the indexer pooled keys + the RoPE table; the K/V window = follow-up)

Power limit 600 W. NVML memory clock 14001 MHz nominal (the user's OC). 5090 headless. 05:50-06:11. The zombie (PID 22984,
~1005 MiB) and the desktop (~356 MiB) were on the card: absolute slots / tok/s are low, relative results stand.

### Design (layer.cpp, layer.hpp, rope_dynamic.cpp, generate.cpp; switch STRATA_SESSION_GROW=0 = the whole allocation)
- Under KV streaming (mode-1 main layers; one GPU, VMM cache, --batch <= 1) each layer's pooled keys (131074 rows x 512 B)
  and the primary's RoPE cos/sin table (524288 x 128 B each) are VMM ranges of the elastic K/V's kind (ElasticPool, the
  same g_pools): addresses for the whole context, physical chunks for the first 16384 cells (STRATA_KV_GROW_INIT).
  - ElasticPool + `extra` (the pooled keys' 2 spare rows) and `host` (a table's host copy: a fresh chunk gets its rows
    from it, else zeros as before). The RoPE host copy (128 MiB RAM) is built once at init, the same build_rope_table call.
  - qsa_state_bytes / init skip those arrays in the session buffer; qsa_state_zero zeroes the mapped part only (a chunk
    mapped later is zeroed then; no reader looks past the current blocks, so stale vs zero rows are never read).
  - --rope-scaling dynamic (the tray uses it): rope_table_rebuild stores the new table through qsa_elastic_table_store
    (host copy replaced, mapped rows uploaded); the re-rotation of pooled keys covers `cells` <= mapped.
  - generate: kvg_start takes `qsa_kv_elastic() || qsa_session_elastic()`, so the existing kvg_ensure / kvg_trim /
    kvg_release run unchanged: growth at the request start (n + 256), before a dialog-cache restore (incoming / image),
    and in the serve MTP loop at a window boundary (p + T + 64 > cells, after adapt_quiesce). The cache gives slots just
    below the prompt path's loan (kvg.top = lend_first), hot experts move into the coldest loan slots, D2D copies are
    synced before unmap; a short request trims back and the slots refill from the profile. Growth only fails if the cache
    is at its floor (128 slots) and the driver has no chunk: the same rule as the elastic K/V.
  - Chunks are 2 MiB per array: pooled keys map in 16384-cell steps per layer, RoPE in 16384 positions (kvg.step 8192).
  - Debug: STRATA_SESSION_GROW_CHECK=1 reads every mapped table row back after a grow and compares it with the host copy.
- No .cu file changed (no cc_check needed); generic host code, no nv_defaults line.

### Slots (serve, tray GPTQ args, auto cache; free at the request start 572 MiB in both)
| | d6 (e1) | E2 |
|---|---|---|
| session tables | 896 MiB in the session buffer | 52 of 920 MiB mapped (16384 cells) |
| slots at start (short context) | 8323 | **8643 (+320 = +0.82 GiB)** |
| after a 32K prompt (grown to 32768 cells, 84 MiB) | 8323 | **8632 (+309 = +0.80 GiB)** (11 slots given) |
| a 32K dialog restored after a short one (trim, regrow) | 8323 | 8622 (+299) (21 slots: the trim also returned the init chunks) |

The expected 1.2 / 0.85 GiB included the K/V window (408 MiB), which is not done here.

### Serve A/B (ab.sh e2 / e2cu, d6 vs E2, 05:59-06:08, 600 W; ab_e2.txt, ab_e2cu.txt)
| fixture | arm | n | ms/window | ms/token | tok/s | verify | reach | host (CPU) | draft | slots |
|---|---|---|---|---|---|---|---|---|---|---|
| struct | d6 | 3 | 17.51 (17.41..17.65) | 2.296 | 435.5 | 16.22 | 6.04 | 9.75 (9.32) | 1.21 | 8323 |
| struct | E2 | 3 | **16.91 (16.70..17.04)** | 2.255 (-1.8%) | 443.4 | 15.64 | 6.06 | 9.17 (8.71) | 1.19 | 8643 |
| prose | d6 | 3 | 9.11 (8.96..9.30) | 3.938 | 254.0 | 8.41 | 4.70 | 3.33 (3.18) | 0.64 | 8323 |
| prose | E2 | 3 | **8.71 (8.55..8.85)** | 3.824 (-2.9%) | 261.6 | 8.02 | 4.73 | 2.90 (2.76) | 0.62 | 8643 |
| conv | d6 | 2 | 10.86 | 3.825 | 261.4 | 10.12 | 4.62 | 5.12 (4.92) | 0.67 | 8323 |
| conv | E2 | 2 | 10.62 | 3.848 (+0.6%) | 260.0 | 9.89 | 4.63 | 4.88 (4.67) | 0.65 | 8643 |
| uk | d6 | 2 | 7.60 | 4.199 | 237.9 | 7.08 | 4.69 | 2.01 (1.91) | 0.46 | 8323 |
| uk | E2 | 2 | 7.58 | 4.174 (-0.6%) | 239.6 | 7.05 | 4.71 | 1.98 (1.85) | 0.46 | 8643 |

- Struct -0.60 ms/window (-3.4%) and prose -0.40 (-4.4%), better in every pair. The gain is all in the CPU pool (struct
  -0.61, prose -0.42 ms); the GPU reach is unchanged. That is what more slots should do.
- ms/token moves less on struct / conv because E2's arms happened to get slightly fewer tokens a window (7.50 vs 7.63;
  serve text varies). The slot price (~1.0 ms/GiB struct) predicted ~-0.8 ms. Prose came out above its prediction
  (-0.15), inside the prose noise (±0.2).
- Conv -0.24 ms/window (CPU -0.25) but ms/token +0.6% (tok/window 2.76 vs 2.84): noise. uk flat.

### Long context and the dialog cache (e2_runs.txt, log/e2_t*.log; e2_client.py: several requests, reply md5)
- The sequence: a 32K prompt (long32_prose.txt = long64_prose's first 32000 tokens with its question) + 300 tokens, a
  19-token prose request, the 32K dialog continued (restored from the dialog cache, 32299 tokens reused), the short one
  continued.
- Growth 16384 -> 32768 cells before the prompt, trim after, regrow before the restore; no error. 32K decode: d6 1534-1684
  vs E2 1341-1534 ms for 300 tokens (serve, different texts).
- **Same placement** (serve with exact.sh's fixed-placement switches, --expert-cache 8323, E2 with STRATA_KV_GROW_FLOOR=100000
  so its tables grow from new VRAM and the cache is untouched): all 4 replies identical d6 vs E2 (767128ce85 / 1fe403b13b /
  308b47de9b / 132db21c0c), the 32K decode after growth and the restored 32K dialog included (t4_*).
- Adaptive serve at --expert-cache 8323 without fixed placement: request 0 identical (32K + decode, d0085f2587), and both
  requests of t1 identical (prose 32f832fa48, struct fbe05a25c1). Later requests after the long one differ (the tier's
  timing), which is why t4 fixes placement.
- Grow / trim / regrow from the cache with STRATA_SESSION_GROW_CHECK=1: both grows "equal to its host copy" (t5).
- STRATA_SESSION_GROW=0: 8323 slots, no elastic tables (t6).

### Identity (ident_e2.txt, final e2\strata.exe): all 7 = the reference
- def16 93c4688100 / e9d907e164, def64 a13d259b7f / 5b1479d38d, conv D + P8 26c5e96e8e, off 293b92d3b9, struct 30b9f645ca,
  prose 10df564211.
- def16 / def64 run at ctx 140000 with KV streaming: "elastic session tables ... 52.0 of 252.0 MiB", grown to 32760 /
  81912 cells before the prompt. The decode set runs at exact.sh's 66661: no streaming there, so E2 is off.

### Not done (follow-up): the K/V window (408 MiB below a 32K context, ~155 slots)
- The clock sweep (kv_stream.cu resolve, step 2) takes free slots in hand order from 0 after a reset. So a context of C
  cells only touches its first ceil(C / page) slots, and mapping the window's pools as the context grows is possible.
- What makes it bigger than this item:
  - a trim with resume > 0 keeps blocks resident in any slot (the hand wraps past 32K), so a shrink must reset the map
    (kv_stream_reset: refetch from the host copy, exact) or keep the occupied slots mapped;
  - qsa_state_zero, the snapshot restore and the prompt path's slot writes (PF_KV_UNSTAGE etc.) must be checked to touch
    mapped slots only;
  - the window graphs capture n_slots, which stays.
- --rope-scaling dynamic past 262K (the table rebuild through qsa_elastic_table_store) is not exercised here: it needs a
  > 262K prompt (a big test).

### Files
- E2.patch (layer.hpp, layer.cpp, rope_dynamic.cpp, generate.cpp on top of round 3; e2w/pre = those files before E2).
  cumulative.patch = `git diff` of the tree + qsa_front_bench.cpp (27 files).
- e2\strata.exe (+ e1's DLLs; strata_kq.dll byte-identical to the build's).
- e2_client.py, e2_run.sh, e2_runs.txt, long32_prose.txt, ab_e2.txt, ab_e2cu.txt, ident_e2.txt, log/ (build_e2a-c, e2_t*, ab_*).

## E3: adaptive tier decay 0.96 -> 0.90 (DROPPED: no net gain over the four fixtures; the accuracy arm not run)

Power limit 600 W. NVML memory clock 14001 MHz nominal (the user's OC). 5090 headless. 06:14-06:22. The zombie (PID 22984,
~1005 MiB) and the desktop (~356 MiB) were on the card: absolute slots / tok/s are low, relative results stand.

- The switch: STRATA_ADAPT_DECAY (nv_defaults.cpp sm_120 table "0.96"; generate.cpp ~4443 reads it when no --adapt-* flag
  is given; the tray GPTQ profile passes none). The A/B is env only: one exe (e2\strata.exe), B_ENV=STRATA_ADAPT_DECAY=0.9.
- Placement only, not bitwise: B's texts differ from A's (prose tokens/window 2.35 -> 2.26), so ms/token is the figure.

### Serve A/B (ab.sh e3 / e3cu, e2 vs e2 + decay 0.90; e3/ab_e3.txt, e3/ab_e3cu.txt)
| fixture | arm | n | ms/token | tok/s | ms/window | tok/window | verify | reach | host (CPU) | draft | CPU experts / PCIe a layer-window | hit rate | swaps/window |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| struct | e2 | 3 | 2.240 (2.228..2.251) | 446.3 | 16.83 | 7.51 | 15.56 | 6.07 | 9.08 (8.63) | 1.18 | 3.06 / 1.59 | 95.9% | 14.8 |
| struct | 0.90 | 3 | **2.192 (2.175..2.213), -2.2%** | 456.3 | 16.50 | 7.53 | 15.18 | 6.17 | 8.60 (8.15) | 1.22 | 2.84 / 1.42 | 96.3% | 16.6 |
| prose | e2 | 3 | 3.771 (3.747..3.806) | 264.9 | 8.87 | 2.35 | 8.19 | 4.73 | 3.07 (2.90) | 0.63 | 1.14 / 0.32 | 95.8% | 11.3 |
| prose | 0.90 | 3 | **3.869 (3.839..3.892), +2.6%** | 258.8 | 8.74 | 2.26 | 8.04 | 4.74 | 2.93 (2.78) | 0.64 | 1.09 / 0.30 | 95.8% | 10.9 |
| conv | e2 | 2 | 3.843 | 260.3 | 10.62 | 2.76 | 9.91 | 4.61 | 4.92 (4.72) | 0.65 | 1.77 / 0.69 | 94.3% | 11.2 |
| conv | 0.90 | 2 | 3.793 (-1.3%) | 263.8 | 10.85 (+2.2%) | 2.86 | 10.09 | 4.69 | 5.01 (4.81) | 0.69 | 1.79 / 0.70 | 94.3% | 16.5 |
| uk | e2 | 2 | 4.234 | 236.3 | 7.54 | 1.78 | 7.01 | 4.71 | 1.94 (1.81) | 0.46 | 0.74 / 0.14 | 96.2% | 9.0 |
| uk | 0.90 | 2 | 4.265 (+0.7%) | 234.1 | 7.66 (+1.6%) | 1.79 | 7.12 | 4.68 | 2.07 (1.96) | 0.47 | 0.80 / 0.16 | 96.0% | 8.1 |

- Struct reproduces the research (-7% CPU experts 3.06 -> 2.84, CPU -0.48 ms, ms/window -2.0%, ms/token -2.2%, 3 of 3).
- Prose: CPU -0.12 ms and ms/window -1.5%, but ms/token +2.6% in 3 of 3 (tok/s 257-261 vs 263-267): the changed
  placement gave texts with shorter accepted windows (2.26 vs 2.35 tok/window). The tok/window drop could be text
  variance rather than the decay, but that is what a user would see, and it cancels struct.
- Conv: ms/token -1.3% only through more tok/window; ms/window +2.2%, CPU experts equal (n=2, noise).
- uk: worse, CPU experts +8% (0.74 -> 0.80), ms/token +0.7% (n=2).
- Net over the four fixtures: struct -2.2%, prose +2.6%, conv -1.3%, uk +0.7% = no clear win. The faster decay helps the
  content whose experts repeat (struct) and costs where it doesn't (uk). A per-content-type setting is not wanted (the coordinator).

### Accuracy: not run (the coordinator, 06:19)
- The rule: the one larger run only for a clear net speed win; there is none, so the teacher-forced arm was skipped.
- Prepared, not run: e3/tf.sh. It is the release's method: basalt-work/corpus/set.json, 19 items = 19 separate requests,
  19437 positions; score_strata.py = the tray GPTQ args + DFlash2 spec 6; compare.py with a 95% bootstrap over the items, so
  every answer is scored as its own request. The arms are e2_A, d90 and e2_B (the noise floor), ~4 min each.
- 0.92 / 0.85 single struct runs: not run either. 0.92 can only sit between the two arms. 0.85 is Basalt's setting and
  goes further the same way, which costs uk / prose more.

### Kept: nothing
- E3.patch empty; nv_defaults.cpp unchanged (STRATA_ADAPT_DECAY 0.96); cumulative.patch = E2's.
- No exe built; e3\ holds only tf.sh, ab_e3.txt and ab_e3cu.txt. The Logs are log/ab_e3_*.log and log/ab_e3cu_*.log.

## E4: three small exact items ((a) KEPT: EARLY_READ off by default; (b) DROPPED: CPU NVFP4 accumulators in registers; (c) no change: GDN_MID_HEAD stays 2)

Power limit 600 W. NVML memory clock 14001 MHz nominal (the user's OC). 5090 headless. 06:24-06:42. The zombie (PID 22984,
~1005 MiB) and the desktop (~356 MiB) were on the card: absolute slots / tok/s are low, relative results stand.

### (a) STRATA_EARLY_READ off by default (KEPT)
- Serve A/B, one exe (e2\strata.exe), env only: A = the default (ER on), B = STRATA_EARLY_READ=0; 5 ABAB pairs struct + prose
  (e4/ab_e4a.txt, log/ab_e4a_*):

| fixture | arm | n | ms/window (min..max) | ms/token | tok/s | verify | reach | host (CPU) | draft | tok/window |
|---|---|---|---|---|---|---|---|---|---|---|
| struct | ER on (e2) | 5 | 16.98 (16.86..17.09) | 2.257 | 443.1 | 15.70 | 6.08 | 9.20 (8.75) | 1.18 | 7.52 |
| struct | ER off | 5 | **16.73 (16.56..16.84), -1.5%** | **2.208 (-2.2%)** | 452.8 | 15.45 | 6.09 | 8.95 (8.49) | 1.20 | 7.58 |
| prose | ER on (e2) | 5 | 8.83 (8.79..8.90) | 3.813 | 262.0 | 8.14 | 4.73 | 3.02 (2.87) | 0.64 | 2.32 |
| prose | ER off | 5 | 8.90 (8.67..9.09), +0.8% | 3.841 (+0.7%) | 260.3 | 8.21 | 4.76 | 3.06 (2.90) | 0.63 | 2.32 |

- Struct: off is faster in 5 of 5 pairs (-0.25 ms/window). The GPU-reach wait is equal (6.08 / 6.09); the difference is in the
  per-layer host / CPU part (8.75 -> 8.49). Why the CPU part moves was not isolated.
- Prose: +0.07 ms/window, inside the prose noise (±0.2): off was better in 2 of 5 pairs. On the pairs with the same text
  (tok/window 2.37 / 2.32 in both arms, pairs 3-5) off was +0.15..+0.19 ms, at the edge of the noise; the research's
  1-2 pairs had off at -0.14. Not a loss beyond noise, so per the rule it goes.
- Kept: `{"STRATA_EARLY_READ", "1"}` removed from the sm_120 table (nv_defaults.cpp, a comment in its place), nv_defaults_test
  30 -> 29 (both checks; nv_defaults_test: ok). The code, the env switch (=1 turns it on) and the hotfix guard
  gated_read_fits stay. With it off by default the small-card co-residency hazard is gone from the default path.
- The startup line now says "29 speed switches" and lists no STRATA_EARLY_READ (log of def16).
- Exact: D3 had the gated read bitwise the ungated one; identity 7/7 on the final exe (below), run with ER at its new default (off).
- Docs not touched: docs/NVFP4.md:483 and docs/fork/ENVIRONMENT.md:82/687 still list STRATA_EARLY_READ=1 as a default.
  This is for the final docs pass (E6), like round 3's switches.

### (b) CPU pool NVFP4 kernel with the accumulators in registers (DROPPED: no serve gain, DRAM-bound)
- Port (E4-dropped.patch): nvfp4_avx512.cpp row_dot_reg<NT> = r4/cpu/nvfp4_fast.cpp's row_dot (named acc0..acc6, forced
  inline, the prefetch distance read once per call), gu_rows / dot_rows templated on it, chosen once per call by
  STRATA_NVFP4_REGACC (default 1, =0 the acc[] loop); nvfp4_512_set_regacc() for tests (+ portable.cpp stub).
  The engine's slices of <= 7 tokens were kept (the POC's 8-wide slice was not ported).
- Bitwise check, in tree: `nvfp4_avx512_parity --regacc pack/experts.bin 48` (new mode): random blocks + 48 real experts
  of the GPTQ pack (one per layer, random expert ids), gate/up through silu with each expert's own tail scales and the down
  rows with s_down, 1..9 tokens, old vs new memcmp: **7,056,000 output rows, 0 differing** (e4/regacc_parity.txt). The default
  parity run (vs ggml, 1e-5) is unchanged: parity OK (e4/parity_default.txt).
- One core (cached expert, best of 50, us an expert gate/up + down): 1 token 192.8 -> 104.0 (1.85x), 2 tokens 1.51x, 3 1.28x,
  4 1.13x, 7 1.08x, 8 (4+4) 1.14x. From DRAM (`--bw`, 4.2 GB of blobs; e4/regacc_bw.txt):

| threads | tokens | acc[] GB/s | regacc GB/s |
|---|---|---|---|
| 1 | 1 | 15.0 / 15.0 | 26.4 / 26.5 |
| 1 | 3 | 11.5 / 11.5 | 14.5 / 14.6 |
| 16 | 1 | 65.1 / 63.7 | 57.5 / 61.3 |
| 16 | 3 | 59.9 / 58.7 | 61.5 / 63.5 |

  At 16 threads both are at the DRAM ceiling (~60-65 GB/s), as the research inferred.
- Serve, one exe (the E4 build with (b), default ER off), A = STRATA_NVFP4_REGACC=0, B = default (regacc); 3 pairs
  (e4/ab_e4b.txt):

| fixture | arm | ms/window (min..max) | ms/token | reach | host (CPU) |
|---|---|---|---|---|---|
| struct | acc[] | 16.83 (16.71..17.03) | 2.213 | 6.11 | 8.98 (8.54) |
| struct | regacc | 17.03 (16.84..17.28) (+0.20) | 2.250 (+1.7%) | 6.13 | 9.21 (8.76) |
| prose | acc[] | 8.91 (8.85..8.96) | 3.785 | 4.76 | 3.07 (2.92) |
| prose | regacc | 8.96 (8.83..9.07) (+0.05) | 3.779 (-0.2%) | 4.76 | 3.11 (2.96) |

  - The pool time does not fall: struct CPU +0.22 ms (regacc worse in 2 of 3 pairs), prose +0.04. Both are noise, with no
    gain. The pool is DRAM-bound; a faster core only waits on memory sooner.
  - Dropped. The tree is restored (the 4 files = e4w/pre). The exe with (b) is e4w/regacc_exe/strata.exe, beside its parity
    tool. It may still help a CPU with fewer cores or less DRAM bandwidth, and the prompt path's CPU share; neither was
    measured.

### (c) STRATA_GDN_MID_HEAD after D2's bounded stream (no change: 2 stays)
- Under D2's stream (DEPTH 1 > 0) only HEAD bit 0 (the up-front piece, now 10.9 KB a CTA instead of a 43.5 KB tile) and
  bit 1 (the L2 prefetches) act; bit 2 (tile 1) is only used by the DEPTH=0 ring, so 7 = 3.
- Sweep with the in-tree gdn_parity (rebuilt on this tree: bitwise vs the chain + cold-L2 cudaEvent timing, REPS 50, heads
  interleaved, 4 rounds; e4/sweep_head.sh, e4/sweep_head.txt). All arms bitwise (32 cases, 0 `***` lines each):

| HEAD | T 1 | T 3 | T 8 (us, mean [min-max] of 4 rounds) |
|---|---|---|---|
| 0 | 46.73 [46.6-46.9] | 47.92 [47.7-48.1] | 55.42 [55.4-55.5] |
| 1 | 47.35 [47.2-47.5] | 47.85 [47.7-48.1] | 55.75 [55.6-55.9] |
| **2 (default)** | **46.60 [46.4-46.7]** | **47.25 [47.2-47.3]** | **55.20 [55.1-55.3]** |
| 3 | 46.95 [46.9-47.0] | 47.58 [47.4-47.8] | 55.77 [55.7-55.9] |
| 7 | 47.00 [46.9-47.1] | 47.62 [47.4-47.8] | 55.83 [55.7-55.9] |

- 2 is the fastest at every T, by 0.3-0.6 us. The bench has no kernel before gdn_mid, so it cannot show the head start's
  cost to the read before it (F3's reason for 2); that cost only adds to the other values. Nothing to change; no serve run.

### Identity (ident_e4.txt, final e4\strata.exe = E2's source + the (a) defaults change): all 7 = the reference
- def16 93c4688100 / e9d907e164, def64 a13d259b7f / 5b1479d38d, conv D + P8 26c5e96e8e, off 293b92d3b9, struct 30b9f645ca,
  prose 10df564211.
- No .cu file changed in what is kept (no cc_check needed). (b)'s files were CPU C++ only.

### Files
- E4.patch: nv_defaults.cpp, nv_defaults_test.cpp, on top of E2 (e4w/pre = the 8 files E4 touched or might touch,
  before E4). cumulative.patch = `git diff` of the tree + qsa_front_bench.cpp (26 entries); e4w/cumulative_e2.patch is E2's copy.
- E4-dropped.patch: (b), nvfp4_avx512.cpp, nvfp4_avx512.hpp, portable.cpp, nvfp4_avx512_parity.cpp (--regacc).
- e4\strata.exe (+ e2's DLLs; strata_kq.dll byte-identical to the build's); e4w/regacc_exe/ (exe + parity tool with (b)).
- e4/: ab_e4a.txt, ab_e4b.txt, regacc_parity.txt, parity_default.txt, regacc_bw.txt, sweep_head.sh, sweep_head.txt;
  ident_e4.txt (also in bench-d3/); log/ab_e4a_*, ab_e4b_*, build_e4a.log, build_e4b.log.

## E5: bug hunt, --mtp-window at 16.7K (NOT A BUG: a placement near-tie; the 15 ms draft is the lent-slot refill; no fix)

5090 headless, 600 W, 06:45-07:20. The zombie (PID 22984) on the card. Fixture basalt-work/fixtures/long16_prose (16685 tokens),
400 greedy tokens, tray GPTQ args + the MTP set (rt-tuned, proxy, cum auto, spec 8), --max-context 524288 (KV streaming on).
Scripts: e5/rep.sh (CLI fixed placement = exact.sh arm M + STRATA_SPEC_FIXED=1; e5/rep.txt), e5/srv.sh (serve through
e2_client.py, reply md5; e5/serve_runs.txt, log/e5_*.log), e5/top.py (top logits of a dump).

### 1. Fixed placement: the window never changes the text
| exe | placement | w0 | w4096 | w8192 |
|---|---|---|---|---|
| base (release-equivalent) | CLI fixed, 8000 slots | 123ce29e2d | 123ce29e2d | 123ce29e2d |
| e4 | CLI fixed, 8000 slots | 988ba45ff0 | 988ba45ff0 | 988ba45ff0 |
| e4, long64_prose (66.6K) | CLI fixed | c006575b07 | c006575b07 | - |
| base | serve, fixed switches, 8000 slots | 63a4cd81f1 (400) | 63a4cd81f1 (400) | - |
| base | serve, fixed switches, **8338 slots** | **ad35678b63 (25, stop)** | **ad35678b63 (25, stop)** | - |
| base | serve, fixed switches, 8643 slots | c1b669b25f (400) | c1b669b25f (400) | - |
All 400-token texts end by length. The drafter ring (window 4096 + 96 cells, wrapping 4x at 16.7K, ~4000x at 66K) is exact.

### 2. The cause of the 25-token stop: the slot count, not the drafter
- A smaller window makes the drafter's K/V smaller (draft layer 857 -> 817 MiB at 4096, 822 at 8192), so the serve's auto cache
  gets **+15 slots** (8323 -> 8338; +13 at 8192). Research logs: long16_w0 / d6 8323, w4096 8338, w8192 8336.
- With 8338 slots the reply is the same 24 tokens and then EOS (248046) where the 8000 / 8643-slot runs write "\n\n" (271):
  w0 at 8338 fixed slots stops too (table above). In adaptive serve w4096 stopped on base, d6 and e4+STRATA_SESSION_GROW=0
  (ad35678b63, all three: 8338 slots), and w0 stopped too in one run (STRATA_PF_REFILL_BG=0, 8323 slots).
- Position 24 is a near tie: fixed CLI, prompt + the 24 tokens, first logits (e5/fl_p24_base.bin): 271 = 20.232,
  EOS = 19.966 (margin 0.27), the third token 13.34. CPU vs VRAM rounding of a few experts moves that.
- e4 (E2: +320 slots, 8643/8658 at the start) wrote 400 tokens at both windows in adaptive serve (e4971e38b6 / 3e7e0b7347),
  which is why the stop was not seen there or at 66K.

### 3. The 15 ms draft a window: the lent-slot refill, also with w0
- A 25-token reply is 8 windows. STRATA_PF_REFILL_BG refills the prompt path's 2390 lent slots beside decode (~137 ms, "2
  windows ran meanwhile"); the draft waits behind those copies: ~115 ms over the first windows.
- base w0, 25 tokens: draft 15.34 ms/window (the same as w4096's 15.2); with STRATA_PF_REFILL_BG=0: **1.16 ms**.
- Over 400 tokens it is ~0.7 ms/window of the 16K run's draft 1.46 (so most of the research's "+0.8 ms draft at 16K" is this,
  not the drafter's attention). Wall time equal: refill BG 1581 ms vs off 1600 ms (the refill moves into the prompt: 1311 ->
  1439 ms). Speed note only (the overlap buys ~nothing at 16K / 25 tokens); not changed here.

### 4. Side check: base vs e4 differ in CLI fixed placement at ctx 524288 (placement, E2 exact)
- 123ce29e2d (base, d6) vs 988ba45ff0 (e4), first difference ~token 280. E2's table growth before the prompt takes 3 cache
  slots ("the expert cache gave 3 slots ... 7997 of 8000 slots hold experts"): fixed placement is 3 experts different.
- e4 + STRATA_SESSION_GROW=0 = 123ce29e2d; e4 + STRATA_KV_GROW_FLOOR=100000 (grows from new VRAM, 8000 of 8000) = 123ce29e2d.
  So E2 is exact; the identity set's decode runs (ctx 66661, no streaming) do not see this.

### 5. Default paths: none passes --mtp-window
- Engine: o.mtp_window = 32768 fixed (generate.cpp:1055), only --mtp-window changes it; no auto-pick by VRAM or context.
- strata_control: only drafting.py MTP_ONLY_FLAGS (strip list); serve/split.py names it as a session-file field; setup / Auto
  never set it. Tray profiles (both GPTQ .tray.json): no --mtp-window. Docs suggest it for Strix Halo (8192) and the 780M (4096):
  those users get a different slot count, i.e. a different but equally valid placement.

### Kept: nothing (no bug, no fix)
- E5.patch empty; cumulative.patch = E4's; e5\strata.exe = e4\strata.exe byte copy (+ DLLs). No source change, so the identity
  set was not re-run (E4's 7/7 stands).
