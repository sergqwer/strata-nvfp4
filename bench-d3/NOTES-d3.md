# Decode round 3 notes (bench-d3)

Tools for every item (D1 made them):
- `base\` = dev build of the clean tree (origin/main 8b14f7966, build_dev.cmd, 120-real) + strata_kq.dll + cuBLAS 13.3 DLLs.
- `ab.sh NAME A_EXE B_EXE` (env A_ENV / B_ENV, PAIRS, FIXTURES, GEN_<fx>): serve, GPTQ tray profile args via rel04/dec_client.py
  (the profile's own mtp folder with rt-tuned, this tree's serve Auto), pair i = AB / BA alternating; `ab_sum.py` prints the means.
  Struct 1500 tokens, the others 1200: ~25 s an arm with the engine start.
- `ident.sh TAG EXE [ENV...]`: def16 + def64 (first-token logits md5) and the decode set (conv D / P8, off, struct P8, prose sampled),
  rel04/ident.sh's method. Never inside gpu_lock (exact.sh takes it).
- `gpu_free.sh`: rel04's gpu_free, minus PID 22984. That process is the probe agent's CUPTI zombie (C:\LLM_models path, no VRAM, not
  killable), and it made rel04's gpu_free wait forever. GPU_FREE_IGNORE overrides the list.
- `cc_check.cmd FILE.cu...`: compiles kernel files for sm_75 / 86 / 89 / 120 (objects in cc\).
- `src/kernels/router_chain_bench.cpp` (in the patch): GEMV shapes at T 1..8, plus the route+doorbell kernel with rows / ids to
  mapped host memory (`route_db host`).

## D1: P4 router exact T = 7, 8 (kept), P5 doorbell rows over blocks (dropped)

Power limit 600 W. NVML memory clock 14001 MHz nominal, 14.0-14.5 GHz observed (the user's OC). 5090 headless.

**Base identity** (base\strata.exe, dev build) = the plan's hashes (taken on the bundle exe), all 7:
def16 93c4688100 / e9d907e164, def64 a13d259b7f / 5b1479d38d, conv D + P8 26c5e96e8e, off 293b92d3b9, struct 30b9f645ca,
prose 10df564211 (ident_base.txt). The dev build needs no reference of its own.

### P4 (kept): STRATA_MMVF_EXACT78 (default on, =0 = the old `<N, 8, false>` default case)
- native_bf16.cu `bf16_gemv_fp32_mmvf_multi`: case 7 / 8 get `<N, 7, true>` / `<N, 8, true>`.
- It is generic CUDA, so there is no nv_defaults line and nv_defaults_test is unchanged.
- The dispatcher serves every bf16 multi GEMV: the router (verify.cpp:2556), the MTP router, the indexer wiq/wik, PLE, gr.cu HC
  and the shared-expert gate. Wide builds (MAX_T > 8) already had 9..16 and now also have 7 / 8.
- The rows kernels (STRATA_MMVF_ROWS, opt-in) and `_multi_aux` are untouched.
- Bitwise: router_chain_bench PARITY gives 72 hashes (9 shapes x T 1..8), identical with EXACT78=0 and =1 (micro1.txt).
- Micro (router_chain_bench, 48 distinct weight copies in one graph, us per call, micro2.txt):

| shape (n_in 2560) | T 7 old -> new | T 8 old -> new |
|---|---|---|
| router 512 out | 7.32 -> 5.33 | 7.82 -> 6.08 |
| 256 | 4.91 -> 4.35 | 5.21 -> 5.12 |
| 64 | 3.20 -> 2.60 | 3.49 -> 2.99 |
| 8 | 2.94 -> 2.13 | 3.24 -> 2.42 |

  Router + route_db chain at T 8: 13.15 -> 11.35 us a layer. At T 7: 12.57 -> 10.54. T 1..6 unchanged.
  This matches router_poc (7.43 -> 5.50, 8.02 -> 6.02).

### P5 (dropped): the EARLY doorbell rows over N blocks (STRATA_ROUTE_DB_BLOCKS, diff kept as D1-P5-dropped.patch)
- Design: block 0 routes and publishes the ids, every block copies a slice of the rows and fences, and the last block in (a device
  counter keyed by the ring) rings.
- native_multi_parity passed bitwise at 1 / 4 / 8 / 16 blocks, and all rings and rows were correct in the micro.
- Micro, `route_db host` (the engine's kernel, rows to mapped host), us a layer:

| T | 1 block | 4 | 8 | 16 |
|---|---|---|---|---|
| 3 | 4.60 | 4.81 | 4.65 | 4.66 |
| 7 | 5.36 | | 5.11 | 4.98 |
| 8 | 5.44 | 5.32 | 5.15 | 5.02 |

- Gain: <= 0.4 us a layer at T 7-8 and 0 at T 3. That is ~0.02 ms of a 17-18 ms struct window (0.1%): inside the noise, so it is
  a measured dead end.
- Why bell1 promised more: it timed block 0 only (entry to after its own fence), not the last block, so its 11.7 -> 2.0 us is not
  the doorbell's latency.
- The kernel is PCIe-write / fence bound, not issue bound: T 3 -> 8 adds 50 KB for +0.8 us.

### A/B (serve, tray GPTQ args, base vs D1 = P4 only; ab_d1.txt, ab_d1p.txt; 01:21-01:30, 600 W)

| fixture | arm | n | ms/window (min..max) | tok/s | verify | reach wait | host |
|---|---|---|---|---|---|---|---|
| struct | base | 3 | 18.14 (17.77..18.73) | 415.0 | 16.85 | 6.89 | 9.53 |
| struct | D1 | 3 | 17.98 (17.84..18.18) | 421.3 | 16.53 | 6.50 | 9.60 |
| prose | base | 2 | 8.77 (8.73..8.81) | 261.4 | 8.08 | 4.78 | 2.93 |
| prose | D1 | 2 | 8.82 (8.73..8.91) | 256.6 | 8.14 | 4.74 | 3.01 |

- Struct D1 vs base: -0.16 ms/window (-0.9%). Pairs -0.81 / +0.25 / +0.07: the first base arm (18.73) is an outlier.
- Prose: +0.05 (noise; P4 only changes T 7-8 windows).
- The expected effect is ~0.1 ms struct: 1.8 us x 48 router layers + the smaller shapes at T 7/8. The run-to-run spread is ~0.8 ms,
  so the serve A/B cannot resolve it with 3 pairs. P4 is kept on the bitwise micro (-25% router time at T 7/8), not on the A/B.
- Serve text is not fixed: one D1 prose run stopped at 1164 tokens. Serve placement is adaptive, so this is not an identity signal
  (identity = ident.sh).

### Identity on D1 (ident_d1.txt): all 7 hashes = the reference.

### Compile check (cc_check.log)
native_bf16.cu (with P4) and native_router.cu (with P5, before it was dropped) compile for sm_75 / 86 / 89 / 120.

### Files
- D1.patch = cumulative.patch: native_bf16.cu + router_chain_bench.cpp.
- D1-P5-dropped.patch.
- micro1.sh / micro2.sh (+ .txt), parity_d1.txt, parity_mmvf_*.txt.
- ab_d1*.txt, ident_base.txt, ident_d1.txt.
- d1\strata.exe = the P4 build, the comparison exe for D2.

## D2: P1 gdn_mid pair stream with bounded depth (kept), the even tile split (dropped)

Power limit 600 W. NVML memory clock 14001 MHz nominal, 14.5 GHz observed (the user's OC). 5090 headless.

### What changed (gdn_mid.cu; STRATA_GDN_MID_DEPTH, default 1, =0 = the old two-tile ring)
- The pair ring keeps its two 43,520-byte slots and its arithmetic. Only the copies change: each tile is copied in
  STRATA_GDN_MID_PIECES pieces (default 4 x 10,880 B; 1 / 2 / 4 / 5 / 8 allowed). The CTA's pieces run in tile order, and
  piece n is issued only after piece n - DEPTH has landed. Default: 10.6 KiB in flight per CTA, vs ~85 KiB before.
- Each piece has its own mbarrier (OFF_BAR + 32 .. + 160), and each slot has an armed-piece count (OFF_ARM). The count stops
  a wait on the other group's slot from reading that barrier while it is still on the phase two tiles back.
- The group that consumes a tile issues its pieces at the top of its iteration (ltid 0), not after the first barrier of
  tile k - 2. The up-front pieces (min(DEPTH, all)) follow HEAD bit 0, as the first tile copy did.
- STRATA_GDN_MID_HEAD, _SPLIT and _EF are unchanged. mid_grid() is untouched, so hotfix5's sms_of line applies (checked
  with patch --dry-run).
- gdn_mid only runs where STRATA_GDN_MID=1 (the sm_120 nv_defaults line) and the card has >= 160 SMs, so there is no new
  nv_defaults line and nv_defaults_test is unchanged.
- gdn_parity.cpp: STRATA_GDN_PARITY_REPS sets the timed launches (default 10).

### Kernel level (the decider): gdn_parity section 8 = the real kernel on the engine's shapes, cold L2 per launch, HEAD=2 as in the engine
Bitwise against the chain in every arm: 32 cases (T 1..8 x 3 reps + relaunch), 0 `***` lines.
The numbers are cudaEvent us per launch, mean of 50 launches, 4 interleaved rounds (kernel_d2.txt). The spread across
rounds is <= 0.3 us.

| arm | T 1 | T 3 | T 8 |
|---|---|---|---|
| old (DEPTH=0) | 47.1 | 49.4 | 59.4 |
| **new default (1 x 10.9 KB)** | **46.8 (-0.3)** | **47.3 (-2.1, -4.3%)** | **55.2 (-4.2, -7.1%)** |
| split 0, old | 48.0 | 51.2 | 61.4 |
| split 0, new | 49.2 | 51.8 | 62.0 |

- Effective rate over 61.3 MB: 1.24 -> 1.30 TB/s at T 3, 1.03 -> 1.11 TB/s at T 8. The launch is counted, so this is a
  lower bound.
- Inferred for 48 GDN layers: ~0.10 ms at prose (T~3) and ~0.20 ms at struct (T~8). That is half the plan's 0.3 / 0.4 ms.

Depth sweep (sweep1-3.txt, 3 rounds each, us at T 1 / 3 / 8):

| in flight / CTA | config (depth x piece) | T1 | T3 | T8 |
|---|---|---|---|---|
| 85 KiB (old) | ring 2 x 43.5K | 47.1 | 49.5 | 59.4 |
| 85 / 42.5 KiB | 2x43.5K, 1x43.5K, 2x21.8K, 4x10.9K, 8x5.4K | 46.4-47.1 | 48.9-49.2 | 59.1-59.5 |
| 32-34 KiB | 3x10.9K, 6x5.4K, 4x8.7K | 45.3-46.0 | 48.0-48.7 | 58.0-59.4 |
| 21 KiB | 1x21.8K, 2x10.9K, 4x5.4K | 45.3-46.1 | 47.1-47.6 | 57.1-57.3 |
| 16-17 KiB | 3x5.4K, 2x8.7K | 45.8-46.4 | 46.8-47.3 | 55.4-56.3 |
| **10.6 KiB** | **1x10.9K**, 2x5.4K | 46.0-46.6 | 47.1-47.5 | 55.2-55.3 |
| 8.5 KiB | 1x8.7K | 47.3 | 48.5 | 55.2 |
| 5.3 KiB | 1x5.4K | 53.5 | 54.5 | 59.6 |

- This matches the probe: 8-16 KiB a CTA is the knee. Below it the stream starves; above it the recurrence slows.
- Phase stamps (T 3, us): with the bound, all qkv is released at ~18-22 instead of 25-29, and rec starts at 25-26
  instead of 33-35.
- The new critical path is the z stream (C_PZ at ~36-37), which the 122 non-rec CTAs carry alone while the rec CTAs
  compute.

Measured dead ends, code removed:
- a deeper bound for the z tiles only (STRATA_GDN_MID_DEPTHZ 2/4/6 over qkv 1-2 pieces): +1..+3 us;
- z tiles moved onto the rec CTAs (RZ 1/2/3 a rec CTA, after its qkv): +1..+10 us, because they delay the recurrence.

**The even tile split (dropped):**
- "1024 / 170 = 6.02, four CTAs do 7" is the split-0 count. The engine runs split 1: rec CTAs 4 qkv tiles, the other 122
  CTAs 832 tiles = 6.82 each, so 100 of them do 7.
- Half-tile (8-row) units would not lower that maximum: 1664 / 122 = 13.6, so the most loaded CTA still does 14 halves =
  7 tiles.
- Moving the remainder onto the rec CTAs is the RZ dead end above. Split 0 with the bound is slower than split 1.

### Window level (CLI fixed placement, exact.sh STRATA_SPEC_FIXED=1, MTP, `--expert-cache auto`; win_d2.txt; 3 ABBA pairs)

| fixture | d1 ms/round | d2 ms/round | reach wait d1 / d2 | pool d1 / d2 | text |
|---|---|---|---|---|---|
| prose 600 (277 rounds) | 26.44 (7295..7340 ms) | 26.74 (7269..7565 ms) | 4.06 / 4.02 | 21.06 / 21.39 | 53fde85c40 both |
| struct 600 (112 rounds) | 65.06 | 63.97 (-1.7%, all 3 pairs) | 4.83 / 4.73 | 58.10 / 57.02 | 84de0c4d7c both |

- Fixed placement only hits ~43% of experts in VRAM, so the CPU pool is 80-90% of each round. This layer cannot resolve
  0.1-0.2 ms.
- The struct difference is pool time, not GPU time, so it is not attributed to D2.
- The CLI prints no `decode timing` line: that line is serve-only.

### Serve sanity (ab.sh d2, tray GPTQ args, 3 pairs, 01:58-02:03, 600 W; ab_d2.txt)

| fixture | arm | ms/window (min..max) | tok/s | verify | reach | host |
|---|---|---|---|---|---|---|
| struct | d1 | 17.47 (17.28..17.81) | 433.6 | 16.17 | 6.30 | 9.46 |
| struct | D2 | 17.43 (17.27..17.53) | 435.3 | 16.12 | 6.24 | 9.47 |
| prose | d1 | 9.00 (8.99..9.01) | 259.1 | 8.29 | 4.77 | 3.14 |
| prose | D2 | 8.94 (8.90..8.99) | 259.4 | 8.23 | 4.76 | 3.09 |

- D2 vs d1: struct -0.04 ms (-0.2%), prose -0.06 ms (-0.6%). Both have the expected sign and are inside the noise.
- The window gains less than the kernel alone suggests. In the engine gdn_mid overlaps the HC reads around it (PDL), and
  part of the saved time lands on the reach wait / CPU side.
- Kept on the kernel-level result, as P4 was in D1: bitwise, -4..-7% at T 3-8.

### Identity on D2 (ident_d2.txt): all 7 hashes = the reference
def16 93c4688100 / e9d907e164, def64 a13d259b7f / 5b1479d38d, conv D + P8 26c5e96e8e, off 293b92d3b9, struct 30b9f645ca,
prose 10df564211. The tok/s in that run do not count: cc_check was compiling at the same time.

### Compile check (cc_check_d2.log)
gdn_mid.cu compiles for sm_75 / 86 / 89 / 120. The kernel body stays behind `__CUDA_ARCH__ >= 900`, as before.

### Files
- D2.patch: gdn_mid.cu + gdn_parity.cpp (REPS env).
- cumulative.patch: D1 + D2.
- sweep_d2.sh + sweep_sum.py (sweep1-3.txt), final_d2.sh (kernel_d2.txt), win_d2.sh (win_d2.txt), ab_d2.txt, ident_d2.txt.
- d2\strata.exe (+ d1's DLLs; strata_kq.dll is byte-identical to the build's): D2 at its default, the comparison exe for D3.
- Lesson: Git Bash `sed -i` strips CRLF. It turned gdn_parity.cpp to LF, which was caught and restored, so edit CRLF files
  with Edit or with Python in binary mode.

## D3: P2 HC reads (kept: lo staging + late gate pass; dropped: the down -> up flag hand-off = "one launch")

Power limit 600 W. NVML memory clock 14001 MHz nominal (the user's OC). 5090 headless.

### What STRATA_EARLY_READ already covered (verify.cpp er_issue ~1636)
- Read 1 (hc_attn, half 0) of every layer l + 1, GDN and attention alike, for windows T 2..8. Skipped: T = 1 (n < 2), layer 1
  with PLE, cvec layers (the uncensored profile), tap layers, LFUSE / FUSE_HITS (HIP), device plan, batch, remote,
  HC_PERSIST, profiling, and now cards where gated_read_fits() fails (84 SMs etc.).
- Read 2 (hc_ffn, half 1) is never early, by design: its bo_prev is the mixer's output (gdn_mid / o-proj) right before it.
  It already starts under the mixer through PDL: in the probe's trace (p2_cupti.csv) its down blocks start 6-10 us
  before gdn_mid ends, weights in registers.
- So the probe's "launch read 1 early, weights before the wait" is done for both reads. What is left on the critical path
  is the latency chain after the dependency, measured in the probe's trace (tl_read: us after the combine end, read 1 /
  after gdn_mid end, read 2): down ends +3.9..+6.3 (T 2..8), upq ends a further +3.0..+6.3; read 2 the same shape
  (down +4.4..+6.3, upq +6.6..+11.4). upq is resident 1-2 us after the gate (u_start), so neither launch nor weights.

### What changed (hc8_read.cuh, fused_gr.cu GrMulti.hc_flag, fused_gr.hpp test hooks; default on, =0 the old path)
- **STRATA_HC8_LOSTAGE** (every hc8 upq launch): upq staged lo (T x 320 floats) with a strided loop that SASS shows as a
  real loop, one L2 round trip per pass (T * 320 / 256 passes: 10 at T 8). Now all loads are issued, then all stores.
  Same values, same shared layout.
- **STRATA_ER_LATE_PASS** (the gated read only): er_gate's pass count (atomicAdd + last-passer reset) moved from right
  after the gate, where thread 0 held the block for its round trip, to after the dots (er_pass). The reset still
  happens only when every block has passed.
- Generic CUDA code in the hc8 read (sm_120-built, used where STRATA_HC_Q8 is on): no nv_defaults line, nv_defaults_test
  unchanged. No new spinning or gated launch, so co-residency is unchanged.

### Kernel level (gr_hc8_check, kernel_d3.txt; hc8check_7-9.txt = the same arms, 3 runs, spread <= 0.05 us)
Bitwise: selftest solo reads on the old path, window reads on the new one: 124 / 124 cases; the gated arms' outputs =
the ungated old read: all T, all arms.

Chain (--bench: 24 reads in a graph, each waits for the previous, DRAM-cold weights, QFUSE tail as read 1), us a read:

| T | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| old | 7.60 | 8.11 | 9.47 | 9.48 | 10.50 | 11.25 | 12.09 | 13.18 |
| LOSTAGE | 7.51 | 7.85 | 9.38 | 8.96 | 9.82 | 10.55 | 11.27 | 12.30 |
| diff | -0.09 | -0.26 | -0.09 | -0.52 | -0.68 | -0.70 | -0.81 | -0.88 |

Early read (--bench-gated, new: per layer a forked gated read, 4 small kernels for the CPU-row wait, a real
combine_k10_vec4_sig, join; us a layer, medians of 31):

| T | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| old | 16.16 | 17.08 | 16.55 | 17.70 | 18.47 | 19.30 | 19.99 |
| LOSTAGE | -0.12 | -0.14 | -0.44 | -0.44 | -0.58 | -0.69 | -0.82 |
| + LATE_PASS | -0.31 | -0.26 | -0.53 | -0.59 | -0.64 | -0.76 | -0.86 |

Inferred per window (96 reads, 48 of them early): prose (T~3) ~0.02-0.03 ms, struct (T~8) ~0.09 ms (0.5%). Far below
the plan's 0.3-0.8 ms: the plan's estimate assumed the launch / weights were on the path, and they are not.

### Dropped: the flag hand-off (= the "down + up as one launch" idea), measured
- STRATA_HC8_FLAG (code removed; kept as d3w/*.flagvariant): each down row-group tail raises a counter after its writes,
  upq waits on it (relaxed polls, own cache line, pass count off the path) instead of griddepcontrol.wait on down's
  grid end. Safe without co-residency (upq is down's programmatic dependent, so every down block is resident first).
- Chain: T 1 -1.2 us, but T 2..8 +0.3..+0.9 us (5 runs). Gated with LATE_PASS: +0.1..+0.5 vs LOSTAGE + LATE_PASS.
- CUPTI on the chain bench (d3w/t1.csv, tr_hc8.py): upq end - down end is 3.5-6.4 us in both arms at T >= 2. The grid-end
  release is not the cost; upq's own post-wait chain is (lo staging, fixed above, then compute and the QFUSE tail).
  A single launch would remove the same boundary and nothing else, so it was not built.
- Remaining chain per read (inferred from the trace): down = bo_prev load, partial stores + fence + arrival atomic, tail
  loads, lo store (4 L2 round trips); upq = loads, compute, QFUSE tail (fence + atomic + read-back). Each L2 trip ~0.5-1
  us under load.

### Serve check (ab.sh d3, d2 vs d3, 3 pairs, 02:53-02:58, 600 W; ab_d3.txt)

| fixture | arm | ms/window (min..max) | tok/s | verify | reach | host |
|---|---|---|---|---|---|---|
| struct | d2 | 17.50 (17.42..17.61) | 435.9 | 16.19 | 6.26 | 9.51 |
| struct | D3 | 17.35 (17.19..17.46) | 436.0 | 16.03 | 6.10 | 9.53 |
| prose | d2 | 8.95 (8.76..9.08) | 254.8 | 8.26 | 4.76 | 3.11 |
| prose | D3 | 8.96 (8.88..9.05) | 257.6 | 8.25 | 4.72 | 3.15 |

Struct -0.15 ms/window (-0.8%), GPU-reach wait -0.16. Prose flat, reach -0.04. The sign is right, but both are inside
the run-to-run spread. Kept on the bitwise kernel result, as in D1 and D2.

### Identity (ident_d3.txt): all 7 hashes = the reference
def16 93c4688100 / e9d907e164, def64 a13d259b7f / 5b1479d38d, conv D + P8 26c5e96e8e, off 293b92d3b9, struct 30b9f645ca,
prose 10df564211 (cc_check was compiling at the same time: its tok/s do not count).

### 84 SMs (sm84_d3.txt): STRATA_EMULATE_SMS=84, conv D 400, the identity env: "STRATA_EARLY_READ: off on this card",
no "gate timed out" line, out 26c5e96e8e (= the 170-SM text).

### Compile check (cc_check_d3.log): fused_gr.cu (with hc8_read.cuh) for sm_75 / 86 / 89 / 120: OK.

### Files
- D3.patch (LF, git style; on top of D1 + D2 + the hotfix): hc8_read.cuh, fused_gr.cu, fused_gr.hpp, gr_hc8_check.cpp.
  pre3\ = those four files before D3. cumulative.patch = `git diff` of the tree (D1 + D2 + hotfix + D3).
- d3\strata.exe (+ cuBLAS DLLs, the build's strata_kq.dll): the comparison exe for D4.
- kernel_d3.txt, hc8check_1-9.txt (the development arms; 1-6 include the dropped flag, 1-4's gated column is invalid: see
  the lesson), ab_d3.txt, ident_d3.txt, sm84_d3.txt, cc_check_d3.log, d3w\ (tracer script + analysis, the flag variant).
- Lessons:
  - A graph's first launch can run a gated read before the combine it waits for: the gate timed out, and the sticky
    abort then made every later gate fall through, so those numbers were timings of an ungated read. The bench now runs
    every graph once with its gates open.
  - The CUPTI tracer left gr_hc8_check as an unkillable zombie (PID 48764), which holds build-dev\gr_hc8_check.exe: it was
    renamed to gr_hc8_check_zombie.exe so the build could relink. Never trace a gated run.
  - Heredocs in Git Bash collapse a doubled backslash, so a Python edit script's escaped n became a real newline in generated C strings: edit scripts go through the Write tool.

## D4: P3 the PCIe share by copy engine (dropped: slower), scheduling out of the pool's rows phases (not built)

Power limit 600 W. NVML memory clock 14001 MHz nominal, 14.5 GHz observed (the user's OC). 5090 headless.

### What the engine already has
- `--pcie-mode dma` (pcie_mode 0) IS the copy-engine variant: the pool thread, right after publishing the plan, calls
  cudaMemcpyAsync per blob (STRATA_DMA_BATCH=1/2: cudaMemcpyBatchAsync) on the verifier's copy_ stream, flag B by
  cuStreamWriteValue32 behind them, cudaStreamQuery flush (STRATA_DMA_FLUSH). The default `auto` = kernel mode 2:
  fetch_blobs (zero-copy, 16 blocks) on the PCIe branch stream (STRATA_PCIE_BRANCH), flag B raised at once.
- A copy-engine copy cannot be driven from inside the graph (dynamic sources, no device API); it has to be a host call
  during the window. That is why `auto` left DMA (issue #31 driver-lock hang; Edge0's "verify: timed out at layer 1").
  Here, with DMA_FLUSH, DMA ran 9 serve arms without a hang.
- DMA mode switched STRATA_ADAPT_FETCH off (fetch_can wanted mode 2), so `--pcie-mode dma` as shipped also changes
  placement: 47 swaps/window, CPU experts 4.6 vs 3.2, 23.8 vs 17.6 ms/window (d4probe1). Not a transport comparison.

### Component timers, kernel mode (d3, struct, STRATA_VERIFY_PROFILE=1, d4prof2; profiled windows are slower)
- per layer-window: VRAM hits 72.6, CPU experts 3.24, PCIe 1.72 (x 2.76 MB = 4.75 MB a layer); CPU pool 9.15 ms/window
  = 191 us a layer; 14 swaps/window (39 MB of refill copies).
- GPU stages, GDN | QSA ms/window: VRAM hits 3.32 | 1.13, **waitB (PCIe fetch beyond the hits) 2.21 | 0.79**,
  PCIe grp 0.26 | 0.09, waitCPU 1.08 | 0.37. Fetch ~155 us a layer from the fork = ~31 GB/s effective.
- STRATA_PCIE_FIXED 0.30 instead of 0.45: waitB 1.61, waitCPU 4.33, 19.49 vs 18.09 ms/window: the tuned share balances.
- ADAPT_FETCH=0 (d4prof1): waitB 6.91 ms kernel vs **10.74 DMA**; 22.07 vs 23.85 ms/window (d4probe2).

### Microbench (d4w/zc_dma.cu, zc1.txt): one op = n blobs of 2.76 MB from scattered 16 B-aligned sources in a 2 GiB
pinned arena (fresh each rep), engine's fetch_blobs_kernel vs one cudaMemcpyAsync per blob; median us, GB/s; load = 15
AVX2 threads streaming DRAM, bg = a refill-like H2D stream of 2.76 MB copies; 2 rounds (ranges)

| case | zc16 (engine) | zc64 | zc170 | DMA |
|---|---|---|---|---|
| n=1 alone | 59 us 46.6 | 46.6 | 46.7 | 62 us 44.3 |
| n=2 alone | 112 us 49.5 | 49.5 | 49.4 | 118 us 46.9 |
| n=3 alone | 163 us 50.7 | 50.6 | 50.5 | 174 us 47.6 |
| n=2, bg | 187 us 29.5 | 31.8 | 32.0 | 115 us 48.1 |
| n=1, load | 97-141 us 20-29 | 24-29 | 16 | 76-111 us 25-37 |
| n=2, load | 177-225 us 25-31 | 22 | 22-30 | 146-206 us 27-38 |
| n=3, load | 266-313 us 27-31 | 24-25 | 24-28 | 286-287 us 29 |
| n=2, load + bg | 274-276 us 20 | 18-22 | 19-20 | 140-147 us 38-39 |

- Alone the zero-copy kernel is 5% faster than DMA. Under CPU load DMA is 0-40% faster at n=1-2 and equal at n=3; with a
  concurrent copy-engine stream zero-copy halves while DMA keeps ~48 / ~39 GB/s. More blocks do not help zero-copy.
- The CPU column of zc1.txt is not valid (rates above the DRAM ceiling); the engine's CPU timer answers that question.

### What was built (STRATA_DMA_ADMIT, default 1, kept as D4-dropped.patch)
- DMA mode with fetch admission: fetch_can accepts pcie_mode 0; fetch_dma lands an admitted blob in its cache slot
  (dst2) instead of staging, and the plan points that PCIe group at the slot (DMA has no rebase kernel); dst2 is zeroed
  for DMA plans; generate/serve no longer switch ADAPT_FETCH off for dma. Placement policy = kernel mode's.
- Placement now matches (14 swaps/window, CPU experts 3.2-3.3), so the A/B below compares transport only.

### Serve A/B (ab4.sh = ab.sh + extra args; d3 kernel mode vs D4 exe `--pcie-mode dma` STRATA_DMA_BATCH=1; 03:18-03:24, 600 W)

| fixture | arm | n | ms/window (min..max) | tok/s | verify | reach | host | plan | CPU |
|---|---|---|---|---|---|---|---|---|---|
| struct | d3 kernel | 5 | 17.61 (17.21..17.85) | 429.1 | 16.29 | 6.13 | 9.74 | 0.26 | 9.28 |
| struct | D4 DMA | 5 | 18.33 (17.90..18.61) | 410.6 | 17.02 | 6.45 | 10.15 | 0.61 | 9.35 |
| prose | d3 kernel | 3 | 8.94 (8.86..9.03) | 255.2 | 8.24 | 4.72 | 3.14 | 0.07 | 2.99 |
| prose | D4 DMA | 3 | 9.42 (9.18..9.56) | 248.1 | 8.70 | 4.99 | 3.33 | 0.18 | 3.06 |

- Struct +0.72 ms/window (+4.1%), every one of 5 pairs worse; prose +0.48 (+5.4%), every pair worse. Single-pair
  probes agree: DMA loop +0.96 (d4x2), DMA batch +0.63 (d4x3).
- Where it goes: the pool thread's submission (plan +0.35 ms/window = ~7 us a layer, on the CPU's critical path) and
  a later GPU (reach +0.32: the copies start after a WDDM host call instead of at once). The CPU pool does NOT get
  faster with the share off the SMs (9.28 vs 9.35 ms): the pool is DRAM-bound and a PCIe byte costs it the same
  whichever engine reads it (PcieModel's note: d ~ c on this 9950X3D). Only fewer RAM bytes help the back.

### Not built: the share out of the pool's rows phases
- Serializing the fetch and the rows: ~83 us (5.8 MB CPU alone) + ~97 us (4.75 MB zero-copy alone) = ~180 us a layer
  against the ~190 us concurrent now, before the GPU->host->CPU hand-off (~5 us) and the CPU's non-DRAM work, which
  then cannot overlap the fetch. Inferred gain <= 0.5 ms/window and likely 0: not worth a placement-neutral but
  timing-fragile change. The DRAM-bound finding above is the same reason.

### Identity / text: no source change is kept, so d4\ = d3\ (byte copy) and the identity is D3's 7/7. The DMA exe was not
an identity candidate; cc_check: no .cu file changed.

### Files
- D4.patch (empty marker), D4-dropped.patch (expert_source.cpp, verify.cpp, generate.cpp); cumulative.patch unchanged.
- d4w/: zc_dma.cu (+ build_zc.cmd, zc1.txt), ab4.sh + dec_client4.py (PF_ADD extra args), pre4/ (the files before D4),
  x1/ (the DMA-admit exe). ab_d4probe1-3.txt, ab_d4prof1-2.txt, ab_d4x1-3.txt, ab_d4.txt, ab_d4p.txt.
- Lesson: `grep -q $'\r'` in Git Bash reported these CRLF sources as LF; count bytes with Python before scripted edits.

## D5: P6 the attention front's indexer chain (kept: 4 changes, all bitwise; dropped: both GEMVs on parallel branches)

Power limit 600 W. NVML memory clock 14001 MHz nominal (the user's OC). 5090 headless.

### Where the chain sits (the probe's own trace, no new tracing: d5w/dump_front.py on dprof/log/p2_cupti.csv, s_cupti.csv)
- The indexer chain is the attention front's critical path. Attention starts right after copy_kernel, on the chain's stream;
  the query side (q tile, norm+rope, fwht) ends ~11 us earlier at T 3 and ~30 us earlier at T 8 (front_p2.txt, front_s.txt).
- The chain: upq (read 1) -> wik GEMV -> append (1 block) -> wiq GEMV -> norm+rope -> block scores (256 CTAs) -> top-k ->
  resolve (1 block) -> copy_kernel -> attention.
- The probe's per-kernel times (resolve 4.4-8.8, scores 5-6, append 7-11 us) are CUPTI durations, which inflate every kernel.
  The bench below has the real numbers.
- block_scores is not single-block (256 CTAs), but each warp walked its block's T queries one after another. That is a
  dependent 5-shuffle chain per query (~0.85 us a window row), plus a staging loop of T x 2 KB per CTA.

### Tool: src/kernels/qsa_front_bench.cpp (new, in the patch)
- The engine's launchers on the real shapes: 12 layers with their own buffers, KV map (5,120 slots, int8) and a pinned
  host copy.
- The L2 is scrubbed (256 MB memset) before each replay. Numbers are the median of 60 replays, in us per layer.
- `parity`: old vs new byte for byte:
  - append: T 1..8 at 4 cell offsets, cell-0 windows, a skipped step, a short context;
  - scores: T 1..8 x 4 offsets, the whole buffer;
  - resolve: all hits (every array equal), all misses and a mix (the same resident set, counters and slot bytes = host
    copy; the miss-list order was never fixed, it is an atomic).
- Result: 53/53, 32/32, 24/24 at ctx 1400 and at ctx 8192.

### What changed (each has an =0 switch; generic CUDA, so no nv_defaults line and nv_defaults_test is unchanged)
- **STRATA_SCORES_PAIRS** (qsa_select.cu `block_scores_pairs_kernel`; default on for CUDA, off on HIP):
  - the decode call's (block, query) pairs are spread over the warps, one pair per warp pass;
  - the query and key slices are read straight from L2: no staging, no barrier;
  - per pair, the multi kernel's arithmetic expression for expression.
- **STRATA_KV_RESOLVE_ILP** (kv_stream.cu `resolve_kernel<RQ>`):
  - step 1 runs RQ = 4 (3-4 queries) or 8 (5+) queries a round, with 2 passes;
  - all width / id loads are issued first, then all page-table loads, then the stamps and claims;
  - ids past 2,048 go through the plain loop afterwards;
  - T <= 2 keeps the old loop (0.3 us slower at T 2 otherwise).
  - The first version kept dependent branches between the loads (SASS); the loads now carry no use in between.
- **STRATA_IDX_APPEND_PAR** (native_qsa_indexer.cu `append_par`, from 4 steps; 2-3 are slower so):
  - 128 threads a step;
  - the tail a step reads = the latest earlier step's key in that slot, else the incoming tail, all read before any write;
  - the stores are replayed in step order by one thread per dim;
  - a window with a cell-0 step runs the old body.
- **STRATA_IDX_FORK** (verify.cpp; default on wherever the side mixer is on): the append runs on a new branch `side3_`, at
  the side's priority, after the key GEMV and the tail's snapshot. It runs beside the query GEMV and joins before the
  block scores.
- Dispatch notes for all four. A serve struct run's STRATA_DISPATCH_REPORT (d5w/dispatch_struct.json) shows every one of
  them taken (not dead code).

### Kernel level (time4.txt, ctx 1400 = the fixtures' context), us per layer

| piece | T 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| append old | 1.26 | 1.61 | 2.29 | 3.48 | 3.99 | 4.34 | 4.85 | 5.53 |
| append new | = | = | = | 2.80 | 2.80 | 2.80 | 2.97 | 2.97 |
| scores old | 2.97 | 3.99 | 4.85 | 5.70 | 6.55 | 7.41 | 8.26 | 9.11 |
| scores new | 1.95 | 2.12 | 2.46 | 2.63 | 2.80 | 3.13 | 3.48 | 3.31 |
| resolve+copy old | 4.34 | 5.36 | 6.55 | 7.58 | 8.60 | 9.80 | 10.82 | 11.85 |
| resolve+copy new | = | = | 4.68 | 5.02 | 6.55 | 6.90 | 7.24 | 7.58 |
| CHAIN old (wik..copy) | 14.57 | 16.62 | 18.67 | 21.74 | 24.30 | 26.69 | 29.42 | 32.33 |
| CHAIN new kernels | 13.89 | 15.60 | 16.28 | 18.67 | 21.23 | 22.77 | 24.65 | 26.18 |
| **CHAIN new + fork** | **13.55** | **15.09** | **15.43** | **16.79** | **19.35** | **20.89** | **22.77** | **24.13** |
| both GEMVs parallel (dropped) | 16.63 | 18.42 | 18.93 | 20.58 | 22.90 | 24.21 | 25.35 | 26.30 |

- At ctx 8192 the chain goes 34.7 -> 32.3 (T 3) and 50.1 -> 45.0 (T 8). There the top-k (15 us, selection > 2,051 cells) is
  the largest step: a round-4 candidate.
- Gain: -3.2 us a layer at T 3 and -8.2 at T 8, below the chain's margin over the query side (11 / 30 us), so it is on the
  path. Over 12 layers: ~0.04 ms a prose window and ~0.10 ms a struct window (0.4-0.6%), inferred. That is half the plan's
  0.15-0.25, whose estimate came from CUPTI durations.
- Dropped: wik and wiq on parallel branches. Two GEMVs at once cost more than in series (+1..+3 us).

### Serve check (ab.sh d5, d4 vs D5, 3 pairs, 03:49-03:54, 600 W; ab_d5.txt)

| fixture | arm | ms/window (min..max) | tok/s | verify | reach | host |
|---|---|---|---|---|---|---|
| struct | d4 | 17.39 (17.23..17.65) | 435.7 | 16.09 | 6.12 | 9.56 |
| struct | D5 | 17.59 (17.40..17.79) | 427.0 | 16.28 | 6.06 | 9.81 |
| prose | d4 | 9.18 (9.07..9.30) | 255.0 | 8.45 | 4.71 | 3.35 |
| prose | D5 | 9.11 (8.98..9.24) | 255.6 | 8.40 | 4.72 | 3.30 |

- Struct +0.20 ms/window comes from the CPU pool's host time (+0.25; one D5 arm 10.11). The GPU-reach wait is -0.06.
- Prose -0.07 ms.
- Both are inside the run-to-run spread. Kept on the bitwise kernel / chain result, as in D1-D3.

### Identity (ident_d5.txt): all 7 hashes = the reference
- def16 93c4688100 / e9d907e164, def64 a13d259b7f / 5b1479d38d, conv D + P8 26c5e96e8e, off 293b92d3b9, struct 30b9f645ca,
  prose 10df564211.
- The final d5\strata.exe adds only the host-side dispatch notes to the exe identity was run on (no kernel or graph
  change).

### Compile check (cc_check_d5.log): kv_stream.cu, native_qsa_indexer.cu, qsa_select.cu for sm_75 / 86 / 89 / 120: OK.
qsa_decode_attn.cu (its sm_80 guards) is untouched.

### Files
- D5.patch: cmake (bench target), verify.hpp / verify.cpp (fork), kv_stream.{hpp,cu}, native_qsa_indexer.{hpp,cu},
  qsa_select.{hpp,cu} (test hooks *_set_*), qsa_front_bench.cpp (new). cumulative.patch = `git diff` of the tree + the new
  file.
- d5\strata.exe (+ strata_kq.dll, cuBLAS DLLs).
- d5w/: dump_front.py, front_p2/front_s.txt, par1.txt, time1-4.txt (development rounds; time4 = final), dispatch_struct.json.
- ab_d5.txt, ab_d5disp.txt, ident_d5.txt, cc_check_d5.log.
- Lesson: the indexer launchers throw on a null stream (an uncaught throw showed up as a bare exit 127 in Git Bash).

## D6: P7 the MTP draft steps (kept: PDL edges in the draft graphs; dropped: head L2 prefetch, cell-list branch)

Power limit 600 W. NVML memory clock 14001 MHz nominal (the user's OC). 5090 headless.

### Where a step's time goes (engine timers, CLI fixed placement: d6w/dr.sh = int exact.sh + tray MTP flags + STRATA_MTP_DUMP)
- Struct 2400 (311 rounds, 5.95 chain steps a round), d5: round graph 0.310 ms + steps 0.791 (0.133 a step) + host gaps 0.012
  = 1.13 ms a round. STRATA_MTP_PROFILE stamps, us a step: embedding branch -> add 9, attn HC read 7, q-proj + norm/rope 24,
  attention 14, o-proj 7, MLP HC read 7, router 8, experts 13, combine/write 2, mixer read 6, head (2-bit scan + exact pass) 31,
  select 2. The round graph's stamps sum to 0.216 of its 0.310: the rest is its start behind the commit graph (dec/draft round 2).
- The probe's "32 q80_pf launches 278 us" = fc_embedding, k, v, o (pf) + q (stream) x 6 steps + the round's q/o: their bytes
  (q-proj 33 MB DRAM-bound, ~21 us), not launches. dense_tc_one x 28 = shared expert gate/up/down + the pre-next fc_hidden
  (off the path); hc reads x 21 = 3 a step.
- D1's T 7/8 router and D3's gated read do not apply (the draft rest is T = 1; the round's catch-up front only does the attn
  read). D3's LOSTAGE is already taken by every hc8 upq launch (T = 1: -0.09 us). STRATA_QFUSE_HC=1 on the draft: same
  dumps, same 0.133 ms a step (no gain).

### Kept: STRATA_MTP_PDL=1 in the sm_120 table (nv_defaults.cpp; =0 off; nv_defaults_test 29 -> 30)
- The switch existed (dec/draft round 1: 140.1 vs 139.7 us/step, "no gain"). On the current tree it is consistent:

| struct 2400, CLI fixed | runs | chain steps ms/round | per step | round graph | dump |
|---|---|---|---|---|---|
| PDL off (d5) | 6 | 0.787 / 0.788 / 0.790 / 0.791 / 0.801 / 0.788 (0.791) | 0.133 | 0.309 | bb3b04f0d2 x6 |
| PDL on (d5 env, d6 default) | 6 | 0.774 / 0.776 / 0.779 / 0.771 / 0.774 / 0.772 (0.774) | 0.130 | 0.305 | bb3b04f0d2 x6 |

  -0.017 ms of chain steps + -0.004 round graph = ~-0.02 ms a struct round (-1.9% of the drafter, ~0.1% of a window). Prose 800:
  steps 0.283 -> 0.279, round 0.248 -> 0.244 (-0.008 ms a round).
- **Accepted drafts / drafts identical** (d6w/check_d6.txt; d5 vs d6 default, same CLI fixed runs): struct 2400 2093 of
  2143 both, dump bb3b04f0d2 both (x2), out 603ec26167; prose 800 474 of 684 both, dump ebe6bff144, out 09ab5fafa0;
  conv 400 249 of 351 both, dump e659ea0e3b, out 4dee2b2da7. The dump holds every round's drafts and probability bits.

### Dropped (measured; diffs in D6-dropped.patch, sources in d6w/headl2, d6w/winfork)
- **STRATA_MTP_HEAD_L2**: the proxy head's 2-bit rows (32.8 MB) prefetched into L2 on a branch forked at 4 points of the
  step (after q-proj / attention / o-proj / MLP read), joined after the head. Dumps identical. Steps 0.133 / 0.133 / 0.142 /
  0.134 / 0.137 ms vs 0.133 off. Profile at fork 1: head 31 -> 28 us but q + rope 24 -> 27, experts / mixer read +1: the
  step's DRAM bytes only move (the middle of a step is not DRAM-idle enough, and the copy pushes the layer's weights out of
  L2, as STRATA_PROXY_L2 did in dec/draft).
- **STRATA_MTP_WIN_FORK**: window_ids (reads only the staged step row) on its own branch beside the q projection. Steps
  0.783 / 0.783 vs 0.791 / 0.771 off: noise, and one of two runs gave a DIFFERENT dump (d3b9f8557f, 2094 of 2142) with PDL
  on: a race this layout opens. Removed.
- Not built: kernel fusions on the step's path (embedding gather + rms + q8_1, combine + write). At ~0.5-0.7 us a graph
  boundary that is ~2-3 us a step (~0.015 ms a struct round, 0.1%) for kernels whose reductions must be reproduced bit for
  bit; the step is bound by its bytes (q-proj + head = 66 MB of DRAM, ~37 us) and the L2 latency chains of ~47 kernels,
  not by launches.

### Serve check (ab.sh d6, d5 vs d6, 3 pairs, 04:28-04:33, 600 W; ab_d6.txt)

| fixture | arm | ms/window (min..max) | tok/s | verify | reach | host | draft |
|---|---|---|---|---|---|---|---|
| struct | d5 | 18.30 (17.45..19.99) | 417.8 | 16.91 | 6.77 | 9.70 | 1.22 / 1.22 / 1.47 |
| struct | D6 | 17.41 (17.10..17.60) | 434.5 | 16.13 | 6.06 | 9.66 | 1.19 / 1.19 / 1.18 |
| prose | d5 | 9.04 (8.97..9.09) | 256.3 | 8.32 | 4.68 | 3.26 | 0.67 / 0.64 / 0.67 |
| prose | D6 | 9.21 (9.18..9.23) | 254.7 | 8.50 | 4.71 | 3.41 | 0.65 / 0.64 / 0.64 |

- d5's struct arm 3 is an outlier (19.99, reach 8.27, draft 1.47); without it struct is 17.46 vs 17.41: noise. The draft
  part moves as the timers said: struct -0.03 ms/window in every pair, prose -0.02.
- Prose +0.17 ms/window sits in verify / host (host +0.15), not in the draft: the run-to-run spread.
- Kept on the bitwise draft-timer result, as D1-D5 were on theirs.

### Identity on the final d6\strata.exe (ident_d6.txt): all 7 hashes = the reference
- def16 93c4688100 / e9d907e164, def64 a13d259b7f / 5b1479d38d, conv D + P8 26c5e96e8e, off 293b92d3b9, struct 30b9f645ca,
  prose 10df564211. (P8 struct: 1745 of 1777 drafts accepted, as dec/draft round 2 with hc8.)

### Compile check: no .cu file changed in the kept change (nv_defaults.cpp / its test only), so no cc_check.

### Files
- D6.patch: nv_defaults.cpp, nv_defaults_test.cpp. cumulative.patch = `git diff` of the tree + qsa_front_bench.cpp.
- D6-dropped.patch (HEAD_L2: draft_proxy.{cu,hpp}, mtp.{cpp,hpp}; WIN_FORK: mtp.{cpp,hpp}); d6w/pre6 = those files before D6.
- d6\strata.exe (+ d5's DLLs; strata_kq.dll byte-identical to the build's).
- d6w/dr.sh (CLI fixed draft run + dump md5), eol.py, edit1-3.py, check_d6.txt; ab_d6.txt, ident_d6.txt, build_d6a-c.log.
- Lesson: a draft-graph branch is only safe if the dumps match over repeated runs: the window fork matched once and differed
  once.
