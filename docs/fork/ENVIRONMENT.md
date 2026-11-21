# Switches and environment variables

Engine flags and environment variables you can set, and every variable the code reads. Back to the
[README](../../README.md).

## Most useful

- `STRATA_NVFP4_SM120=0`: the RTX 50 decode and prompt kernels of 0.2.15-nvfp4.4 off, all at once
  ([RTX 50](#rtx-50-0215-nvfp44)).
- A switch's own `=0` turns that one switch off. A switch you set yourself is kept: the RTX 50 table only fills in
  what you did not set.
- `STRATA_AUTO_NEW_MTP=0`: the old Auto drafting instead of the MTP with Basalt's drafting.
- `--no-kv-stream` (engine flag) or `STRATA_KV_STREAM=0`: the whole K/V stays in VRAM instead of KV streaming.
- `STRATA_UPDATE_CHECK=0`: no check for a newer release at start (`"update_check": false` in the run config does
  the same).
- `STRATA_GEMM_WARM_FILE`: where the GEMM warm file goes, or `0` for none.

## Common switches

Engine flags go into the config's `"args"`, environment variables into its `"env"` (`{"STRATA_KV_ROT": "0"}`).
The switches kept only for A/B runs (each gives the same bits or a measured alternative) are in
[docs/NVFP4.md](../NVFP4.md#switches-kept-for-ab-runs).

| | |
| --- | --- |
| `--prefill auto:8192\|16384` | cap `--prefill auto`'s chunk (default 32768 here, 8192 upstream) |
| `--low-ram` / `--no-low-ram` | the low-RAM mode on / off (default: on below 92 GiB installed, or where `experts.bin` + 16 GiB do not fit) |
| `--ram-budget GIB` | the low-RAM mode with at most GIB of pinned expert copies |
| `STRATA_RESIDENT_HEADROOM_GIB=6` | RAM the low-RAM mode leaves free (10 when the pack's size turned it on) |
| `--no-kv-grow`, `STRATA_KV_GROW=0` | the K/V allocated for the whole context at start |
| `STRATA_KV_GROW_INIT` / `_STEP` | the K/V's cells at start (16384) and its growth step (8192) |
| `--vram-reserve-mib N` | VRAM left unused (default 700); also a smaller card's budget on a bigger one |
| `--pcie-frac F` | a fixed share of decode's cache misses fetched over PCIe (default: upstream's, set from the CPU pool's speed in the first decode windows) |
| `STRATA_PCIE_FRAC_DEFAULT=old` | decode keeps the link probe's fixed PCIe share (upstream 0.1.41's rule) |
| `STRATA_ROUTE_TAIL_SKIP=7` / `=0` | decode skips a missed expert every token of a window routes only at rank 7-9 / computes every one (default: off with NVFP4 experts, KL 0.035 with it on; upstream's 7 for the GGUF models) |
| `STRATA_PREFILL_CPU_SHARE=x\|0` | a fixed CPU share of a small prompt chunk's streamed experts / none (default `auto`: measured, and only while sharing is faster; `STRATA_DBG_CPU_GATE=1` prints its readings) |
| `--adapt-every N`, `--adapt-swaps N`, `--adapt-decay F` | the adaptive VRAM tier: re-rank every N rounds, up to N swaps, counts x F after each (default 2 / 192 / 0.92 with a cache of 20-60% of the experts and every expert in RAM, else upstream's 4 / 96 / 0.7) |
| `STRATA_ADAPT_FETCH=0\|1\|2` | decode: the PCIe share's experts admitted into the VRAM tier (2, the default: instead of the tier's own swaps where an admission ran; 1: beside them; 0: off) |
| `STRATA_ADAPT_WAIT=1` | each decode window waits for the tier's copies, so greedy decode repeats exactly (upstream's default; ~11% slower with this fork's tier, ~2% with `STRATA_ADAPT_LAG=2`) |
| `STRATA_PREFILL_NVFP4=w4a4x2\|w4a8\|w4a4\|fp16` | the prompt path's precision for NVFP4 experts (default `w4a4x2` on RTX 50: gate/up on the FP4 tensor cores with two FP4 terms per activation, down w4a8; `w4a8` on other cards; `fp16` reads the prompt most exactly, 24% slower than `w4a8`) |
| `STRATA_PREFILL_BF16X2=2\|1\|0` | exact inputs to the prompt path's BF16 projections: all but the hyper-connection (default), all (~9% slower prompt reading), off |
| `STRATA_KV_ROT=0` | int8 K/V without the Hadamard rotation |
| `STRATA_ROPE_TABLE=0` | upstream's fast-math RoPE angles instead of the float64 table |
| `STRATA_PROMPT_ATTN_IMMA=0` | the prompt attention on FP16 MMA instead of INT8 tensor cores |
| `STRATA_GDN_CHUNKED=0` | the prompt path's DeltaNet recurrence token by token (default: in chunks for prompt chunks of 128+ tokens) |
| `STRATA_NO_NVFP4_512=1` / `STRATA_NO_NVFP4_256=1` | the CPU's NVFP4 rows on ggml-cpu's dot instead of the AVX-512 / AVX2 kernels |
| `STRATA_FORCE_AVX2=1`, `STRATA_FORCE_ISA=avx2\|avx\|sse` | tests: the engine's CPU dispatch as on a CPU without AVX-512 / one that stops at that level (ggml-cpu runs what it was built for) |
| `STRATA_NO_LARGEPAGES=1` | 4 KB pages even when large pages are allowed |
| `--image-max-tokens N`, `--image-min-tokens N` (`serve.server`) | image tokens a picture becomes at most / at least: a bigger picture is scaled down, a smaller one up (also `"max_tokens"` / `"min_tokens"` in the config's `"vision"`; setup writes 1024; the model reads up to 4096, and more tokens read smaller text and encode slower) |
| config `"anthropic_thinking": "on_request"\|"model"` | an Anthropic request that does not ask for thinking renders without it / thinks as the template does (the server's default) |
| `STRATA_SPEC_STATS=1` | decode: round time, misses and accepted drafts per window size, and the drafts' acceptance by their probability |
| `STRATA_REQUEST_LINES=1` | `serve.server` echoes one summary line per request to stdout |
| `STRATA_BANG_AUDIT=1` | the "!" audit ([Troubleshooting](TROUBLESHOOTING.md#if-a-reply-turns-into-)) |
| `STRATA_KVG_CHECK=1\|2\|3`, `STRATA_DBG_NAN_VERIFY=1` | the audit's checks at every K/V growth and trim and at each prompt's and reply's end (2: the bytes too; 3: the table after every window) / name the first decode window with non-finite logits |
| `STRATA_VERIFY_ARENA=1` | print a checksum of the loaded arena |
| `STRATA_DUMP_FIRST_LOGITS=file` | write the first generated token's logits (compare prompt paths) |
| `STRATA_DUMP_MOE_INPUT=prefix`, `STRATA_DUMP_MOE_LAYER=l\|all` | dump the real MoE input rows of one layer / every layer (the GPTQ calibration) |
| `STRATA_EMULATE_CC=75\|86\|89` | tests: answer as that generation (with an engine built as its PTX, e.g. `86-virtual`) |
| `STRATA_QSA_WARP=1\|select\|attn` | the pre-sm_80 QSA kernels on any card, as RTX 20 runs them |
| `STRATA_EMULATE_RAM_GIB=N` | tests: the engine reads N GiB as installed, which decides the low-RAM mode |

## RTX 50 (0.2.15-nvfp4.4)

On RTX 50 cards (compute capability 12.0) 0.2.15-nvfp4.4 turns on a set of decode and prompt kernels after Basalt and
of our own. Other cards run as before.

| | |
| --- | --- |
| `STRATA_NVFP4_SM120=0` | the whole RTX 50 set of decode and prompt kernels off |
| a switch's own `=0` | that one switch off; a switch you set yourself is kept, and the start prints the count |
| `STRATA_AUTO_NEW_MTP=0` | the old Auto: DFlash2 where it was chosen before, else the plain MTP (default: the MTP with Basalt's drafting, `--mtp-head proxy --mtp-cum auto`, a window of 8, on CUDA with one request at a time on one card's layers) |
| `STRATA_GEMM_WARM_FILE=0` | no GEMM warm file (default `%LOCALAPPDATA%\strata\gemm-warm-<GPU>.txt`: the prompt's GEMM shapes, replayed at the next start before the first request) |
| `--no-kv-stream` | KV streaming off (on by default since 0.2.15-nvfp4.3 from a context of 131,072 tokens on one GPU) |

On RTX 50 the set moves the defaults of `STRATA_ADAPT_FETCH` and `--pcie-frac` in the table above:
`STRATA_ADAPT_FETCH=1` with `ADAPT_ACROSS=1`, `ADAPT_CAND_MIN=1.5`, `ADAPT_DECAY=0.96` and `PCIE_FIXED=0.45` (the
serve tier admits the PCIe share's experts, ranks misses across all layers, and the PCIe share is fixed at 0.45). Every
switch of the set, what it does, where it came from and whether it changes rounding is in two tables in docs/NVFP4.md:

- [RTX 50: decode and prompt kernels after Basalt](../NVFP4.md#rtx-50-decode-and-prompt-kernels-after-basalt-2026-11-20-release-0215-nvfp44)
  (`STRATA_HC_Q8`, `STRATA_DENSE_TC`, `STRATA_ATTN_STAGED`, `STRATA_GDN_MID`, `STRATA_QPROJ_HALF`, `STRATA_DF_PDL`,
  `STRATA_ADAPT_FETCH`, `STRATA_PREFILL_EQUAL`, `STRATA_GDN_TC`, `STRATA_PF_GROUP_SCRATCH`, `STRATA_EARLY_READ`,
  `STRATA_GDN_MID_HEAD`, `STRATA_ROUTE_DB`, and outside the table `STRATA_HEAD_PLANES`, `STRATA_KV_POST`,
  `STRATA_PF_FP4G`, `STRATA_DENSE_SLAB`);
- [RTX 50: faster prompts](../NVFP4.md#rtx-50-faster-prompts-2026-11-20-release-0215-nvfp44) (`STRATA_QSA_I8`,
  `STRATA_DENSE_F16C`, `STRATA_PF_HC_INJ`, `STRATA_PF_DM16H`, `STRATA_PF_GRAPH`, `STRATA_PF_SHARED_LATE`,
  `STRATA_KV_PREFETCH`, `STRATA_PF_KV_UNSTAGE_SIDE`, `STRATA_PF_UPMIX_SKIP32`, `STRATA_PF_REFILL_BG`).

## RTX 50 in 0.2.15-nvfp4.5

0.2.15-nvfp4.5 adds decode switches, all exact. Each one's `=0` turns it off:

| switch | what it does | on by default |
| --- | --- | --- |
| `STRATA_SELECT_LONG`, `STRATA_SCORES_DX` | from `STRATA_SELECT_LONG_AT` (24576) cells the decode windows score the indexer's blocks with the DX kernel, which reads each pooled key once for all of the window's queries (the long-context speedup) | RTX 50 |
| `STRATA_MTP_ATTN_PIPE` | the drafter's attention over its 32,768-cell window in one wave of blocks, the next chunk copied during this one's softmax | RTX 50 |
| `STRATA_MTP_MERGE_FAST` | the drafter attention's partials merged in parallel, in the same order | every CUDA card |
| `STRATA_KV_RESIDENT_FAST` | a new dialog's K/V blocks start mapped to the VRAM window in order, so the resolve step ends at once below 32,768 cells | every CUDA card |
| `STRATA_SESSION_GROW` | under KV streaming the indexer's pooled keys and the RoPE table grow with the context (`=0`: sized for the whole context at the start, 320 fewer expert slots) | yes |
| `STRATA_EARLY_READ` | the next layer's first hyper-connection read starts before the combine and waits on its counter; skipped on a card where its waiting blocks cannot all fit | off since 0.2.15-nvfp4.5 (`=1` turns it on) |
| `STRATA_VISION_IN_ENGINE` | `=0` keeps the CPU picture reader instead of the engine's (the config's `vision.in_engine`) | with `in_engine` |
| `STRATA_EMULATE_SMS` | test mode: all but N multiprocessors held by an idle kernel, to try a smaller card on a large one | off |

The rounds' other switches (`STRATA_MTP_PDL`, `STRATA_MMVF_EXACT78`, `STRATA_GDN_MID_DEPTH`, `STRATA_HC8_LOSTAGE`,
`STRATA_ER_LATE_PASS`, `STRATA_SCORES_PAIRS`, `STRATA_KV_RESOLVE_ILP`, `STRATA_IDX_APPEND_PAR`, `STRATA_IDX_FORK`) and
the measurements:
[RTX 50: decode rounds 3-5](../NVFP4.md#rtx-50-decode-rounds-3-5-long-contexts-smaller-cards-2026-11-21-release-0215-nvfp45).

## All environment variables

Every variable the code of 0.2.15-nvfp4.7 reads, collected by a scan of `src/`, `include/`,
`metal/src/`, `sycl/src/` (the engine, with `tools/settings_register.py`'s own scan), `strata_control/`, `serve/`,
`setup.py`, `tools/` and `release/`, one table per area. Many are upstream's, and many apply only to one backend.

- **Default:** the value the code uses when the variable is not set, where it can be read from the call; `on` / `off` for
  a switch; `unset` where the code only tests whether the variable is set or what it says.
- **sm_120 table:** the value the strata-nvfp4 RTX 50 table (`src/core/nv_defaults.cpp`) sets when you have not set
  the variable yourself.
- **What it does:** from this fork's docs where they describe it, else the comment at the place it is read, marked
  (from the code comment). A comment can be about the code around the read rather than the variable.
- **Read in:** the files that read it (the first three).

<!-- generated -->

### Setup and update (33)

| Variable | Default | sm_120 table | What it does | Read in |
| --- | --- | --- | --- | --- |
| `CUDA_PATH` | unset |  | the CUDA toolkit folder (standard CUDA variable); `release/make_windows_bundle.py` takes cuBLAS for the Windows zip from there | `release/make_windows_bundle.py` |
| `HF_ENDPOINT` | varies by caller |  | the Hugging Face address downloads use (standard Hugging Face variable) | `setup.py`, `strata_control/pictures.py`, `tools/mtp_fetch.py` +3 |
| `HF_HOME` | unset |  | the Hugging Face cache root (standard Hugging Face variable); looked in for model files already downloaded | `strata_control/discovery.py`, `strata_control/drafting_fetch.py` |
| `HF_HUB_CACHE` | unset |  | the Hugging Face hub cache folder (standard Hugging Face variable); looked in for model files already downloaded | `strata_control/discovery.py` |
| `HF_TOKEN` | unset |  | a Hugging Face access token (standard Hugging Face variable); only `tools/trellis/hf_slice.py` reads it | `tools/trellis/hf_slice.py` |
| `HUGGINGFACE_HUB_CACHE` | unset |  | the older name of `HF_HUB_CACHE` (standard Hugging Face variable) | `strata_control/discovery.py` |
| `STRATA_ALLOW_NO_ENGINE` | unset |  | (no description in the code) | `setup.py` |
| `STRATA_APP_UPDATE_PUBKEY` | unset |  | (no description in the code) | `strata_control/updater.py` |
| `STRATA_ARENA_RELEASE` | unset |  | a file-backed arena (STRATA_ARENA_MMAP): the experts no GPU holds are the ones the CPU pool and the prompt path will read - start reading them now instead of faulting them in 4 KB at a time mid-request ... (from the code comment) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_ATTACH_URL` | unset |  | (no description in the code) | `release/windows/start-ui.cmd` |
| `STRATA_BUILD_CACHE` | unset |  | (no description in the code) | `release/make_windows_bundle.py` |
| `STRATA_CONTROL_INSTALL` | unset |  | The desktop app runs this installer under Strata Control (its bundled Python, STRATA_CONTROL_INSTALL=1). (from the code comment) | `setup.py`, `tools/strata_mcp.py` |
| `STRATA_CUDA` | unset |  | the CUDA 12 engine for this model, on any card | `setup.py` |
| `STRATA_DATA_DIR` | unset |  | (no description in the code) | `strata_control/discovery.py` |
| `STRATA_DF2_END_UPDATE` | on |  | =0: the pre-fix behaviour (diagnostics) (from the code comment) | `src/program/generate.cpp` |
| `STRATA_EXECV` | unset |  | (no description in the code) | `setup.py` |
| `STRATA_EXPERIMENTAL_GFX1103` | unset |  | STRATA_EXPERIMENTAL_GFX1103=1 (opt-in, unsupported): the Radeon 780M / 760M / 740M iGPU (Ryzen 7040 / 8040, gfx1103) is taken as a unified-memory AMD card like Strix Halo, with the native dot4 and gfx11 WMMA kernels,... (from the code comment) | `setup.py` |
| `STRATA_EXPERIMENTAL_GFX1150` | unset |  | STRATA_EXPERIMENTAL_GFX1150=1 (opt-in, unvalidated): the Radeon 890M / 880M iGPU (Ryzen AI 300, Strix Point, gfx1150) is taken as a unified-memory AMD card like Strix Halo. (from the code comment) | `setup.py` |
| `STRATA_EXPERIMENTAL_SM60` | unset |  | the older cards are listed as usable | `setup.py` |
| `STRATA_FEED_MODELS` | unset |  | "0": setup and the screens do not read or fetch the feed's model list (tests, a locked-down PC) (from the code comment) | `strata_control/feed_models.py` |
| `STRATA_FILE_RELEASE` | unset |  | (no description in the code) | `src/core/expert_source.cpp`, `sycl/src/core/expert_source.cpp` |
| `STRATA_GATE_SETUP_TIMEOUT` | `"300"` |  | seconds for one setup run (the 0.2 GB file takes about 6) (from the code comment) | `tools/feed_model_install_gate.py` |
| `STRATA_HOME` | unset |  | (no description in the code) | `release/windows/start-ui.cmd` |
| `STRATA_INSTALL_DIR` | unset |  | The source (tools/, data/, serve/) stays in ROOT, inside the app; everything the install writes (run configs, logs, third_party, the data folder beside it) goes to home(), a per-user folder (STRATA_INSTALL_DIR). (from the code comment) | `setup.py`, `strata_control/calibration.py`, `strata_control/discovery.py` +2 |
| `STRATA_ISA_FLOOR` | unset |  | (no description in the code) | `setup.py` |
| `STRATA_NEED_ENGINE` | unset |  | (no description in the code) | `setup.py` |
| `STRATA_NVCC` | unset |  | 601: STRATA_NVCC=<path to nvcc> is the only one considered (a newer toolkit beside it that cannot build on this (from the code comment) | `setup.py` |
| `STRATA_PREBUILT_URL` | `PREBUILT_URL` |  | (no description in the code) | `setup.py` |
| `STRATA_SKIP_SHA256` | unset |  | (no description in the code) | `setup.py`, `tools/bundle_update.py` |
| `STRATA_UPDATE_ALLOW_HTTP` | unset |  | (no description in the code) | `strata_control/transfer.py` |
| `STRATA_UPDATE_CHECK` | on |  | `0` / false / no / off: no daily check for a newer strata-nvfp4 release at start (also `"update_check": false`, `--offline`) | `strata_control/engine_update.py`, `strata_control/fork_release.py` |
| `STRATA_UPDATE_TRUSTED_KEYS` | unset |  | (no description in the code) | `strata_control/signing.py` |
| `STRATA_VCVARS` | unset |  | (no description in the code) | `setup.py` |

### Serve and drafting (126)

| Variable | Default | sm_120 table | What it does | Read in |
| --- | --- | --- | --- | --- |
| `STRATA_ADMIT_SHORT_FIRST` | unset |  | a new prompt of at most N tokens takes the control lines before longer ones (a longer one that waited `STRATA_ADMIT_MAX_WAIT_S`, default 30, counts as short) | `serve/server.py` |
| `STRATA_ALLOWED_HOSTS` | unset |  | Comma-separated names; ports are ignored. | `serve/server.py` |
| `STRATA_API_KEY` | varies by caller |  | (no description in the code) | `serve/server.py`, `strata_control/benchmarking.py`, `strata_control/readiness.py` +1 |
| `STRATA_APP_UI` | unset |  | (no description in the code) | `serve/app_ui.py` |
| `STRATA_APP_UI_OPEN` | varies by caller |  | (no description in the code) | `release/windows/start-ui.cmd`, `serve/app_ui.py` |
| `STRATA_APP_UI_PORT` | unset |  | (no description in the code) | `serve/app_ui.py` |
| `STRATA_AUTO_NEW_MTP` | on |  | `0`: the old Auto. Else Auto runs the MTP with Basalt's drafting (`--mtp-head proxy --mtp-cum auto`, window 8) on CUDA, one request, one card's layers | `strata_control/drafting.py` |
| `STRATA_BATCH_MTP` | unset |  | (no description in the code) | `src/program/generate.cpp`, `src/qwen35/q36_engine.cu`, `sycl/src/program/generate.cpp` +2 |
| `STRATA_BATCH_MTP_SERIAL` | unset |  | "split+batch default pipelined groups win ... (BATCH_MTP_SERIAL=1 old)"; restores old serial groups. (from the settings register) | `src/program/generate.cpp` |
| `STRATA_BLOCKING_CONSOLE` | unset |  | (no description in the code) | `serve/server.py` |
| `STRATA_COALESCE_BURST_MS` | unset |  | Adaptive: a read is shared (and the leader waits for its followers) only when this engine has seen at least COALESCE_MIN_REQS requests arrive within COALESCE_BURST_MS (the leader's own included). (from the code comment) | `serve/server.py` |
| `STRATA_COALESCE_MIN_REQS` | unset |  | Adaptive: a read is shared (and the leader waits for its followers) only when this engine has seen at least COALESCE_MIN_REQS requests arrive within COALESCE_BURST_MS (the leader's own included). (from the code comment) | `serve/server.py` |
| `STRATA_COALESCE_PREFILL` | unset |  | STRATA_COALESCE_PREFILL=N (research, PAR-H; needs the engine's BCO / BCOK lines): a request that takes the control lines for its admission also sends the prompts of up to N-1 requests still waiting for them (BCO <slot... (from the code comment) | `serve/server.py` |
| `STRATA_COALESCE_TOKENS` | unset |  | prompt tokens read together at most (from the code comment) | `serve/server.py` |
| `STRATA_COALESCE_WAIT_MS` | unset |  | a burst's later requests register within this long (from the code comment) | `serve/server.py` |
| `STRATA_COMPONENTS_DIR` | unset |  | (no description in the code) | `strata_control/engine_component.py` |
| `STRATA_DFLASH2_TUNED_URL` | unset |  | tests and mirrors: a direct address (https, file:) of the same pinned bytes, instead of the feed (from the code comment) | `strata_control/drafting.py` |
| `STRATA_DISPATCH_REPORT` | unset |  | (no description in the code) | `include/strata/core/dispatch_stats.hpp` |
| `STRATA_DOCUMENTS_DIR` | unset |  | (no description in the code) | `strata_control/discovery.py` |
| `STRATA_DRAFT_COSTS_DIR` | unset |  | (no description in the code) | `serve/server.py` |
| `STRATA_DRAFT_COSTS_LEARN` | off |  | Test-harness setting: =1 in the environment of ONE engine start makes a machine with no draft-cost table learn it in that start (forced filler windows, a burst of requests, about 1000 timed rounds). The server and the la (from the settings register) | `src/program/generate.cpp` |
| `STRATA_DRAFT_KEEP` | unset |  | Marked "diagnostic" in results; =0 switches draft upkeep off to measure it. (from the settings register) | `src/program/generate.cpp` |
| `STRATA_DROP_CACHE_ON_EXIT` | unset |  | (no description in the code) | `serve/cache_drop.py` |
| `STRATA_ENGINE_BACKEND` | unset |  | (no description in the code) | `strata_control/engine_component.py` |
| `STRATA_GPU_POLL_S` | `"1"` |  | (no description in the code) | `serve/telemetry.py` |
| `STRATA_HEAD_HOST` | off |  | "SHORT OF VRAM: the head in mapped pinned host memory"; off unless set, user-facing spill option. (from the settings register) | `src/core/native_head.cpp` |
| `STRATA_HEAD_HOST_OFF` | unset |  | Layer-split spill step 5 (head on host) is the behaviour; variable turns it off. (from the settings register) | `src/core/native_head.cpp`, `src/program/generate.cpp` |
| `STRATA_HOST_CORE` | unset |  | (no description in the code) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_HOST_TL` | unset |  | (no description in the code) | `src/core/verify.cpp` |
| `STRATA_HTTP_BACKLOG` | unset |  | socketserver listens with a backlog of 5: a burst of 30-40 clients at once got "connection reset by peer" on the first ones (measured on 4 x R9700, also with one request at a time); the requests wait in the server, no... (from the code comment) | `serve/server.py` |
| `STRATA_KEEP_EMPTY_TURNS` | unset |  | STRATA_KEEP_EMPTY_TURNS=1 renders them as before.) (from the code comment) | `serve/frontend.py` |
| `STRATA_KEEP_LEFTOVERS` | unset |  | (no description in the code) | `serve/server.py` |
| `STRATA_KVA_DRAFT_FLOOR` | off |  | Code calls it "The research option". (from the settings register) | `include/strata/prefill/kva.hpp` |
| `STRATA_KV_HOST_DMA` | off |  | [11-14] CUDA: on by default (include/strata/core/kv_host_dma.hpp kv_host_dma_on): on a streamed layer (--kv-resident) the prompt's new K/V cells go from the staging pool to the RAM copy by one DMA copy, and no kernel wri (from the settings register) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_KV_HOST_REGISTER` | unset |  | Documented override of pinning order (WSL2 auto-detected). (from the settings register) | `src/core/layer.cpp` |
| `STRATA_KV_WIDE_HOST` | on |  | (no description in the code) | `sycl/src/prefill/kernels.dp.cpp` |
| `STRATA_LIFELINE` | unset |  | "watcher" forces the portable path (tests) (from the code comment) | `serve/lifeline.py` |
| `STRATA_MAX_BODY_MIB` | `"0"` |  | (no description in the code) | `serve/server.py` |
| `STRATA_MOCK_GPUS` | unset |  | (no description in the code) | `serve/gpus.py` |
| `STRATA_MOVE_ALWAYS_COPY` | unset |  | (no description in the code) | `strata_control/models_move.py` |
| `STRATA_MTP_AHEAD` | unset |  | (no description in the code) | `src/core/mtp.cpp` |
| `STRATA_MTP_ATTN_GQ` | unset |  | STRATA_MTP_ATTN_GQ=0 keeps the gate and the quantizer apart (from the code comment) | `src/kernels/cuda/qsa_decode_attn.cu` |
| `STRATA_MTP_ATTN_GRID` | 0 |  | STRATA_MTP_ATTN_GRID=N caps the blocks a KV head (each then loops over chunks: one wave of 2 x 170 measured slower on struct, 13 vs 9 us, the chunks' latency is serial) (from the code comment) | `src/kernels/cuda/qsa_decode_attn.cu` |
| `STRATA_MTP_ATTN_PIPE` | unset | `1` | the drafter's attention over its 32,768-cell window in one wave of blocks, the next chunk copied during this one's softmax | `src/kernels/cuda/qsa_decode_attn.cu` |
| `STRATA_MTP_ATTN_STAGED` | unset |  | STRATA_MTP_ATTN_STAGED=0 keeps qsa_decode_attn_batch; (from the code comment) | `src/kernels/cuda/qsa_decode_attn.cu` |
| `STRATA_MTP_BATCH` | unset |  | Only the _F16 sibling "(the A/B)" shows in evidence; this base switch unclear. (from the settings register) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_MTP_BATCH_F16` | on |  | CUDA / HIP: on unless set to 0 (src/prefill/prefill.cpp:2319, the MTP drafter's prompt pass on FP16 GEMMs instead of Q8_0 MMQ; =0 is MMQ's form until MMQ leaves the build); SYCL: off unless set to 1 (no MMQ there). Not d (from the settings register) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_MTP_BATCH_RING` | on |  | STRATA_MTP_BATCH_RING=0: the drafter's own pass for a ring (the A/B). (from the code comment) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_MTP_CATCHUP_ALL` | unset |  | 783 PR-i: STRATA_MTP_CATCHUP_ALL=1 catches the drafter's K/V up for the whole verified window, rejected rows included (from the code comment) | `src/core/mtp.cpp`, `sycl/src/core/mtp.cpp` |
| `STRATA_MTP_COND` | unset |  | (no description in the code) | `src/core/mtp.cpp` |
| `STRATA_MTP_FULL_HEAD` | off |  | STRATA_MTP_FULL_HEAD=1 (a diagnostic): draft over the whole vocabulary (the main head itself) instead of rt/draft_vocab.bin's subset - the reference head, to measure what the subset costs in acceptance (from the code comment) | `src/core/mtp.cpp`, `src/program/generate.cpp`, `sycl/src/core/mtp.cpp` |
| `STRATA_MTP_GEMV_PF` | unset |  | STRATA_MTP_GEMV_PF=0 keeps native_mmvq. (from the code comment) | `src/core/mtp.cpp` |
| `STRATA_MTP_HC8` | unset |  | (no description in the code) | `src/core/mtp.cpp` |
| `STRATA_MTP_HIGH_PRIO` | unset |  | STRATA_MTP_HIGH_PRIO=1 (CUDA): the draft streams at the highest priority in every mode, as STRATA_MTP_HIP_PRIO on HIP - the round no longer waits behind the commit graph the verifier launched just before it (its 6912-... (from the code comment) | `src/core/mtp.cpp` |
| `STRATA_MTP_KEEP` | unset |  | Not with DFlash2 (its own block decides), a split, or STRATA_MTP_KEEP=1. (from the code comment) | `src/program/generate.cpp` |
| `STRATA_MTP_MERGE_FAST` | unset |  | the drafter attention's partials merged in parallel, in the same order | `src/kernels/cuda/qsa_decode_attn.cu` |
| `STRATA_MTP_MULTI_COPY` | on |  | (no description in the code) | `src/kernels/cuda/elementwise.cu`, `src/kernels/verify_parity.cpp` |
| `STRATA_MTP_OVERLAP` | unset |  | (no description in the code) | `src/core/mtp.cpp` |
| `STRATA_MTP_PDL` | off | `1` | programmatic dependent launch in the drafter's graphs (RTX 50 table) | `src/core/mtp.cpp` |
| `STRATA_MTP_PREFILL_SYNC` | unset |  | STRATA_MTP_PREFILL_SYNC: the prompt pass's per-group loop (see MtpDrafter::prefill); (from the code comment) | `src/core/mtp.cpp`, `sycl/src/core/mtp.cpp` |
| `STRATA_MTP_PRE_NEXT` | unset |  | (no description in the code) | `src/core/mtp.cpp` |
| `STRATA_MTP_PRIORITY` | unset |  | --pipeline-windows 2 (the forcing graphs): the chain shares its card with stage 1's windows and the next launch on stage 0 waits for it, so its stream gets the highest priority there (STRATA_MTP_PRIORITY=0: the default) (from the code comment) | `src/core/mtp.cpp`, `sycl/src/core/mtp.cpp` |
| `STRATA_MTP_QSTREAM` | unset |  | STRATA_MTP_QSTREAM=0 reads it the plain way (from the code comment) | `src/core/mtp.cpp` |
| `STRATA_MTP_REVISION` | unset |  | STRATA_MTP_REVISION overrides it (e.g. main). (from the code comment) | `tools/mtp_fetch.py` |
| `STRATA_MTP_ROOM_MIN` | unset |  | CUDA, `--serve` with `--mtp`, no `--batch` | `src/program/generate.cpp` |
| `STRATA_MTP_ROUTE_GROUP` | unset |  | STRATA_MTP_ROUTE_GROUP=0 keeps the router, the grouping and the q8_0 input as three launches on the routed path (from the code comment) | `src/core/mtp.cpp` |
| `STRATA_MTP_SELECT_WIDE` | unset |  | (no description in the code) | `src/core/mtp.cpp` |
| `STRATA_MTP_SHARED_BRANCH` | unset |  | (no description in the code) | `src/core/mtp.cpp`, `sycl/src/core/mtp.cpp` |
| `STRATA_MTP_SIDE_STREAM` | 0 |  | Off on HIP, on for gfx103x | `src/core/mtp.cpp` |
| `STRATA_MTP_STAGE0` | unset |  | Overlap option reported slower on Turing; no engine verdict. (from the settings register) | `src/program/generate.cpp` |
| `STRATA_MTP_STEP_COPY` | unset |  | STRATA_MTP_STEP_COPY=1 keeps the copy per step (from the code comment) | `src/core/mtp.cpp` |
| `STRATA_MTP_TOP2` | off |  | Runner-up hit counter for MTP statistics; diagnostic. (from the settings register) | `src/core/mtp.cpp`, `sycl/src/core/mtp.cpp` |
| `STRATA_MTP_TUNED_DIR` | unset |  | (no description in the code) | `strata_control/drafting.py` |
| `STRATA_NAN_GUARD` | on |  | `0`: upstream's non-finite-logits guard (#879) off; on by default | `src/core/verify.cpp` |
| `STRATA_NODE` | unset |  | (no description in the code) | `serve/app_ui.py` |
| `STRATA_NO_ACQUIRE` | unset |  | (no description in the code) | `strata_control/acquisition.py` |
| `STRATA_NO_PRESET` | unset |  | (no description in the code) | `serve/server.py` |
| `STRATA_NO_VISION` | unset |  | (no description in the code) | `tools/hip/build_windows.bat` |
| `STRATA_PARALLEL_SOLO` | unset |  | (no description in the code) | `serve/server.py` |
| `STRATA_PF_PEER_HOST` | unset |  | Without P2P the activations and the result rows take the mapped host route of the layer split's helper (STRATA_PF_PEER_HOST=0: refuse as before, the prompt path then stays on the primary) (from the code comment) | `src/prefill/prefill.cpp` |
| `STRATA_PIN_RESERVE_GIB` | unset |  | STRATA_PIN_RESERVE_GIB is the RAM that must stay above every step (default: the resident headroom, at least 2 GiB) STRATA_NUMA_SPLIT placement forces the drop and the paced path whatever the guard (STRATA_PIN_GUARD=0... (from the code comment) | `src/core/expert_source.cpp` |
| `STRATA_PIN_RESERVE_MIB` | unset |  | Tuning knob for pinned-arena reserve; no on/off story. (from the settings register) | `include/strata/core/pin_budget.hpp` |
| `STRATA_PIPED_MTP` | unset |  | Shipped as opt-in in 0.2.7 with --batch-mtp. (from the settings register) | `src/program/generate.cpp` |
| `STRATA_PIPELINE_SPEC_DEPTH` | unset |  | On each stage A is committed there (its snapshot taken first) right before B launches, as stage 0 alone did with two stages STRATA_PIPELINE_SPEC_DEPTH=k (debug): B runs speculatively on the first k front stages only (from the code comment) | `src/program/generate.cpp` |
| `STRATA_PORT_WAIT_S` | `PORT_WAIT_S` |  | (no description in the code) | `serve/server.py` |
| `STRATA_PREFILL_HOST_PLAN` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_PRESET_DIR` | unset |  | (no description in the code) | `strata_control/presets.py` |
| `STRATA_PRIVATE_MTP_DEMAND` | off |  | Private opt-in (=1), off by default: demand-driven MTP drafting state; debug disagreement or forced-miss modes keep the normal path (sycl/src/program/generate.cpp:11735). Read from the code only; the integration lane mea (from the settings register) | `sycl/src/program/generate.cpp` |
| `STRATA_PRIVATE_MTP_DEMAND_MICRO` | unset |  | Private micro-benchmark switch of the demand-driven MTP path (a number; replays the last chain's staging, no model or device allocation; sycl/src/program/generate.cpp:12153). Read from the code only; the integration lane (from the settings register) | `sycl/src/program/generate.cpp` |
| `STRATA_Q36_DRAFT_VOCAB` | unset |  | (no description in the code) | `src/qwen35/q36_engine.cu` |
| `STRATA_Q36_NO_MTP_CPU` | on |  | (no description in the code) | `src/qwen35/q36_engine.cu` |
| `STRATA_REPLICA_SHARED_ARENA` | unset |  | (no description in the code) | `serve/replicas.py` |
| `STRATA_REPLICA_START` | unset |  | (no description in the code) | `serve/server.py` |
| `STRATA_REQUEST_LINES` | unset |  | `serve.server` echoes one summary line per request to stdout | `serve/server.py` |
| `STRATA_REQUIRE_SIGNED_COMPONENTS` | unset |  | (no description in the code) | `strata_control/engine_component.py` |
| `STRATA_ROUTE_RESIDENT_MTP` | unset |  | Documented routing change, off by default; lab says not faster ("not recommended"). (from the settings register) | `src/program/generate.cpp` |
| `STRATA_RUNTIME_DIR` | unset |  | (no description in the code) | `strata_control/runtime.py` |
| `STRATA_SERVE_STOP_BOUND` | unset |  | STRATA_SERVE_STOP_BOUND=0 turns it off (from the code comment) | `src/program/generate.cpp` |
| `STRATA_SERVE_STOP_SKIP` | unset |  | on unless set to 0: a decode window that finishes after the server's STOP is dropped whole (not committed, not printed), so the session ends at the rows the client was sent (lane AGENT-CAP; serial and pipelined loop; src (from the settings register) | `src/program/generate.cpp` |
| `STRATA_SPEC_ADAPT_EVERY` | unset |  | For the first STRATA_SPEC_ADAPT_WARM rounds it swaps every STRATA_SPEC_ADAPT_EVERY rounds (set per card by nv_defaults, where measured); later the normal interval. (from the code comment) | `src/program/generate.cpp` |
| `STRATA_SPEC_ADAPT_WARM` | 0 |  | For the first STRATA_SPEC_ADAPT_WARM rounds it swaps every STRATA_SPEC_ADAPT_EVERY rounds (set per card by nv_defaults, where measured); later the normal interval. (from the code comment) | `src/program/generate.cpp` |
| `STRATA_SPEC_COST` | off |  | the draft window follows what a draft costs | `src/program/generate.cpp` |
| `STRATA_SPEC_COST_TABLE` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_SPEC_COUPLED` | unset |  | Ryzen AI Max+ 395 (Radeon 8060S, Linux, ROCm 7.14.1), UD-Q4_K_XL, `--spec 4 --mtp`, temperature 1.0, 12-13 requests per arm | `include/strata/core/coupled_draft.hpp` |
| `STRATA_SPEC_DEPTH` | off |  | Draft depth stats switch (counters pass). (from the settings register) | `src/program/generate.cpp` |
| `STRATA_SPEC_FIXED` | off |  | repeatable output: =1 fixes the draft policy's window costs (no measured round time changes them), as --draft-costs does with this machine's learned table; an expert override for tests and A/Bs, off by default (from the settings register) | `src/spec/draft_policy.cpp` |
| `STRATA_SPEC_FIXED_TABLE` | unset |  | with STRATA_SPEC_FIXED=1: the round ms of windows of 1, 2, ... rows, comma separated, for every context bucket (unset: fixed where the costs stand) (from the settings register) | `src/spec/draft_policy.cpp` |
| `STRATA_SPEC_GUMBEL` | off |  | Ryzen AI Max+ 395 (Radeon 8060S, Linux, ROCm 7.14.1), UD-Q4_K_XL, `--spec 4 --mtp`, temperature 1.0, 12-13 requests per arm | `src/kernels/cuda/sampler.cu` |
| `STRATA_SPEC_MIN_TEMP` | 0.0 |  | STRATA_SPEC_MIN_TEMP=x: coupled / probabilistic drafting only for requests with temperature >= x (default 0: every sampled request). (from the code comment) | `include/strata/core/spec_prob.hpp` |
| `STRATA_SPEC_PROB` | unset |  | RTX 3060 IQ3_XXS, 200-token answers, 10 pairs | `include/strata/core/coupled_draft.hpp` |
| `STRATA_SPEC_PROB_DT` | 1.0 |  | STRATA_SPEC_PROB_DT=x: the drafter's chain runs at x * the request's temperature (default 1). (from the code comment) | `include/strata/core/spec_prob.hpp` |
| `STRATA_SPEC_PROB_GATE` | unset |  | (no description in the code) | `include/strata/core/spec_prob.hpp` |
| `STRATA_SPEC_T_CYCLE` | unset |  | Window cost measurement switch used with --spec-oracle. (from the settings register) | `src/program/generate.cpp` |
| `STRATA_SPEC_WATCH` | unset |  | STRATA_SPEC_WATCH=<seconds> (diagnostics, off by default): the serve watchdog for this loop. (from the code comment) | `src/program/generate.cpp` |
| `STRATA_SPLIT_ENGINE` | unset |  | (no description in the code) | `strata_control/split_engine.py` |
| `STRATA_SPLIT_MTP_BATCH` | off |  | The last card fills the draft layer's prompt K/V in batches, as on one card. | `src/program/generate.cpp` |
| `STRATA_START_RETRIES` | `"3"` |  | Such a start is tried again after a short wait (STRATA_START_RETRIES, default 3; 0: never). (from the code comment) | `serve/server.py` |
| `STRATA_STREAM_WRITE_TIMEOUT_S` | `"60"` |  | (no description in the code) | `serve/server.py` |
| `STRATA_TELEMETRY_KEY` | unset |  | (no description in the code) | `serve/server.py`, `strata_control/benchmarking.py` |
| `STRATA_UMA_RESERVE_GIB` | unset |  | Reserve for Strix Halo UMA; "0 = the old sizing", shipped 0.2.2. (from the settings register) | `include/strata/core/uma_cache.hpp` |
| `STRATA_UNDECLARED_TOOL_CALLS` | unset |  | (no description in the code) | `serve/server.py` |
| `STRATA_VISION_GPU_BLAS_KEEP` | unset |  | the cuBLAS handle goes with the room and is made again at the next picture: 138 -> 74 MiB of VRAM left outside the slots after the first picture, no time measured (STRATA_VISION_GPU_BLAS_KEEP=1 keeps it) (from the code comment) | `tools/vision/own/vit_gpu.cu` |
| `STRATA_VISION_GPU_REFILL_BG` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_VISION_IN_ENGINE` | unset |  | `=0` keeps the CPU picture reader instead of the engine's (the config's `vision.in_engine`) | `serve/server.py` |
| `STRATA_VISION_TOWER` | unset |  | Without --tower the environment's STRATA_VISION_TOWER decides (the model's run config carries STRATA_VISION_TOWER=mtmd when someone picks llama.cpp's reader in the settings; an older strata-vision ignores it). (from the code comment) | `tools/vision/strata_vision.cpp` |
| `STRATA_WDDM_SHARED_RESERVE_MIB` | 2048 |  | with two or more CUDA cards the pinned experts are cut to the Windows shared segment, less this reserve (upstream #3725) | `src/core/expert_source.cpp` |

### Engine, general (398)

| Variable | Default | sm_120 table | What it does | Read in |
| --- | --- | --- | --- | --- |
| `CUDA_DEVICE_ORDER` | varies by caller |  | the order CUDA numbers the cards in (standard CUDA variable); `PCI_BUS_ID` makes it match nvidia-smi | `serve/gpus.py`, `serve/server.py`, `src/program/generate.cpp` +1 |
| `CUDA_LAUNCH_BLOCKING` | unset |  | `1`: every kernel launch waits for the kernel (standard CUDA variable, for debugging) | `src/program/generate.cpp` |
| `CUDA_MODULE_LOADING` | varies by caller |  | when CUDA loads kernels (standard CUDA variable); the engine sets `EAGER` unless you set it, so every kernel loads before the expert cache takes the free VRAM | `src/program/generate.cpp`, `src/qwen35/q36_engine.cu` |
| `CUDA_VISIBLE_DEVICES` | varies by caller |  | which NVIDIA cards CUDA shows (standard CUDA variable); `strata_control/gpu_roles.py` maps card numbers through it, the image encoder tools hide the cards with it | `strata_control/gpu_roles.py`, `tools/vision/own/own_main.cpp`, `tools/vision/strata_vision.cpp` |
| `STRATA_ADMIT_MERGE_FIRST` | off |  | an admission reads its last prompt tokens and its first token in one window (all experts resident only) | `src/program/generate.cpp` |
| `STRATA_AUTO_NOBORROW` | unset |  | Not with an explicit --prefill N, a split, helper caches, STRATA_PREFILL_BORROW=1 or STRATA_AUTO_NOBORROW=0. (from the code comment) | `src/program/generate.cpp` |
| `STRATA_AUX_CPUS` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_B6_MMVQ_ROWS` | 0 |  | Exact multi-column MMVQ reusing the activation fragment across rows | `src/kernels/cuda/native_mmvq.cu` |
| `STRATA_BATCH_CONCURRENT_READ` | unset |  | STRATA_BATCH_CONCURRENT_READ=1 (opt-in, one GPU, --batch with --no-prefill-borrow): the prompt is read on a thread of its own while this thread keeps the slots' batch windows going, so the decoding slots do not stop f... (from the code comment) | `src/program/generate.cpp` |
| `STRATA_BATCH_DF2` | off |  | STRATA_BATCH_DF2=0 turns them off): every batch slot gets its own drafter (the pack, embedding and draft head shared, its own K/V ring and buffers). (from the code comment) | `src/program/generate.cpp`, `strata_control/drafting.py` |
| `STRATA_BATCH_DF2_CONF` | 0.35 |  | STRATA_BATCH_DF2_DEFER=0: every round drafts for every slot, spare rows or not (the first measurement's behaviour, for the A/B) STRATA_BATCH_DF2_CONF=P (default: --draft-conf, else 0.35; 0 off): a slot's draft rows st... (from the code comment) | `src/program/generate.cpp` |
| `STRATA_BATCH_DF2_DEFER` | unset |  | STRATA_BATCH_DF2_DEFER=0: every round drafts for every slot, spare rows or not (the first measurement's behaviour, for the A/B) STRATA_BATCH_DF2_CONF=P (default: --draft-conf, else 0.35; 0 off): a slot's draft rows st... (from the code comment) | `src/program/generate.cpp` |
| `STRATA_BATCH_DF2_MAXX` | 99 |  | STRATA_BATCH_DF2_MAXX=N: at most N draft rows a slot and window (default kDrafts) (from the code comment) | `src/program/generate.cpp` |
| `STRATA_BATCH_GDN_GROUPS` | unset |  | the GDN recurrence in the window, and the commit of 1-row groups, for all slots in one launch. The conv commits stay per slot. | `src/core/verify.cpp`, `sycl/src/core/verify.cpp` |
| `STRATA_BATCH_GROUPS_AUTO` | unset |  | Two B-series cards with 8+ slots run 2 pipelined groups by default since 0.2.9; variable turns it off. (from the settings register) | `sycl/src/program/generate.cpp` |
| `STRATA_BATCH_GROUPS_NV_SERIAL` | unset |  | Pipelined batch groups became default in 0.2.7; this restores the old one-group serial mode. (from the settings register) | `src/program/generate.cpp` |
| `STRATA_BATCH_TELEMETRY` | off |  | (no description in the code) | `src/program/generate.cpp`, `strata_cli/cli.py` |
| `STRATA_BF16_TC` | -1 |  | "=1 / =0 turns it on / off on any 7.x card"; on by default for 7.x. (from the settings register) | `src/prefill/gemm.cu`, `sycl/src/prefill/gemm.dp.cpp` |
| `STRATA_CKPT_ASYNC` | off |  | A turn checkpoint's state is copied card-to-card into a VRAM stage and moved to the host by a background thread. A reader waits only for the copy it needs. The stage is one buffer the size of the state, allocated at the first save; if the allocation fails the save copies as before. | `src/core/conversation_state.cpp` |
| `STRATA_CKPT_GATHER` | off |  | A checkpoint's 37 small card-to-host copies are gathered into one buffer and fetched with one copy. Not tested with a layer split across cards. | `src/core/conversation_state.cpp` |
| `STRATA_CKPT_POOL` | unset |  | The checkpoint pool keeps N ready checkpoints, sized at start and topped up in the background, so a save never faults in a fresh vector. It holds N x 118 MB of host memory from start-up. `STRATA_CKPT_POOL=1` alone is unchanged: it reuses dropped checkpoints and allocates nothing ahead of time. | `src/core/arch_defaults_test.cpp`, `src/program/generate.cpp` |
| `STRATA_CKPT_POOL_READY` | -1 |  | The checkpoint pool keeps N ready checkpoints, sized at start and topped up in the background, so a save never faults in a fresh vector. It holds N x 118 MB of host memory from start-up. `STRATA_CKPT_POOL=1` alone is unchanged: it reuses dropped checkpoints and allocates nothing ahead of time. | `src/program/generate.cpp` |
| `STRATA_CKPT_REREAD` | off |  | (no description in the code) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_CKPT_THREADS` | off |  | a checkpoint is saved into the vectors of a dropped one; a split's stage parts are saved at once | `src/program/generate.cpp` |
| `STRATA_CMAKE_EXTRA` | unset |  | (no description in the code) | `tools/hip/build_windows.bat` |
| `STRATA_COLD_OVERLAP` | off (0) |  | (no description in the code) | `src/core/expert_source.cpp`, `src/program/generate.cpp` |
| `STRATA_COPY_ENGINE` | off |  | (no description in the code) | `sycl/src/core/copy_queue.cpp` |
| `STRATA_CPU_IQ_ALL_EXACT1` | off |  | AVX2 CPUs without AVX-512 | `src/kernels/cpu/native_expert.cpp` |
| `STRATA_CPU_YMM` | on |  | (no description in the code) | `src/kernels/cpu/expert.cpp` |
| `STRATA_CVEC_FUSE` | off |  | the prompt's gated-delta-net recurrence and the steered layer's fused write | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_DEC_BATCH` | unset |  | (no description in the code) | `src/core/expert_source.cpp`, `src/core/verify.cpp`, `src/kernels/cuda/shared_expert.cu` +3 |
| `STRATA_DEFAULTS` | unset |  | STRATA_DEFAULTS=026 (read once). (from the code comment) | `include/strata/core/defaults_026.hpp`, `setup.py` |
| `STRATA_DEFERRED_REGISTER` | off |  | register the arena per layer on a thread ahead of its readers (upstream's opt-in) / load the arena after the dense weights | `src/core/expert_source.cpp`, `sycl/src/core/expert_source.cpp` |
| `STRATA_DENSE_SLAB` | on |  | dec/host round 3, STRATA_DENSE_SLAB (default 1; 0 = one cudaMalloc per matrix as before): `load` places the matrices and their 8-bit hyper-connection copies in one allocation sized by a dry pass over the same rules (B... (from the code comment) | `src/core/native_dense.cpp` |
| `STRATA_DENSE_SLAB_MIB` | 0 |  | (no description in the code) | `sycl/src/core/native_dense.cpp` |
| `STRATA_DEQUANT_WIDE` | -1 |  | "on by default for sm_80 (off: STRATA_DEQUANT_WIDE=0)"; bitwise identical, shipped 0.2.7. (from the settings register) | `src/kernels/cuda/dequant_bf16.cu` |
| `STRATA_DF2_FORK` | unset |  | (no description in the code) | `src/core/dflash2.cpp` |
| `STRATA_DF2_FUSE_H` | on |  | (no description in the code) | `metal/src/engine/dflash2.inc` |
| `STRATA_DF2_GATE` | off |  | Retired in 0.2.13 (cleanup 4.5, batch D): code removed, the setting is no longer read. Evidence: [checked] Calibrated DFlash2 window gate (AdaptiveGate) lost to the fixed confidence 0.5: -3.4% on UD-IQ4_XS-v31, -4.9% on  (from the settings register) | `include/strata/spec/window_rule.hpp` |
| `STRATA_DF2_GEMM` | 1 |  | 1 (default): rows a warp by matrix size, bitwise the same results; 0: 8 rows a warp as before (from the code comment) | `src/kernels/cuda/dflash2.cu` |
| `STRATA_DF2_GEMM4` | 0 |  | 0: the table; 1: 8 rows a warp as before; 10 and up: that variant for every matrix (bench / A/B) (from the code comment) | `src/kernels/cuda/dflash2.cu`, `sycl/src/kernels/cuda/dflash2.dp.cpp` |
| `STRATA_DF2_HEAD` | unset |  | Draft head format selector (native keeps Q6_K, 2x slower on Metal); tuning switch, not a shipped-vs-lost verdict. (from the settings register) | `include/strata/core/dflash2.hpp`, `metal/src/engine/dflash2.inc` |
| `STRATA_DF2_HEAD_IL` | on |  | (no description in the code) | `src/core/dflash2.cpp` |
| `STRATA_DF2_LOOKUP_SKIP` | unset |  | STRATA_DF2_LOOKUP_SKIP=N (match length in tokens; 0 = off). (from the code comment) | `include/strata/spec/window_rule.hpp` |
| `STRATA_DF2_MM` | unset |  | rows per simdgroup, simdgroups per threadgroup (from the code comment) | `metal/src/engine/dflash2.inc` |
| `STRATA_DF2_PRIO` | off |  | the draft round runs while the verifier's async commit still has work on the card: a high-priority stream keeps it from queueing behind that (CUDA by default; on HIP only with STRATA_DF2_PRIO=1, where the engine's oth... (from the code comment) | `src/core/dflash2.cpp` |
| `STRATA_DF2_ROW_MS` | 3.0 |  | Retired in 0.2.13 (cleanup 4.5, batch D): code removed, the setting is no longer read. Evidence: [checked] Row-cost input of STRATA_DF2_GATE only (include/strata/spec/window_rule.hpp:123); retires with it. Evidence for t (from the settings register) | `include/strata/spec/window_rule.hpp` |
| `STRATA_DF2_SHARE_HEAD_OFF` | unset |  | (no description in the code) | `src/core/dflash2.cpp`, `src/program/generate.cpp` |
| `STRATA_DF2_SKIP` | unset |  | STRATA_DF2_SKIP=attn,head,ctx,mlp,qkvo (measurement only): leaves those parts out of the round, so the graph's GPU time without them can be read (from the code comment) | `src/core/dflash2.cpp` |
| `STRATA_DF2_STAGE` | unset |  | DFlash2 on a layer split: its card is the LAST stage's by default (the head rows, the embedding path and the Main role are there; only the earlier stages' taps cross, a few KB a window), or the FIRST stage's with STRA... (from the code comment) | `src/program/generate.cpp` |
| `STRATA_DISK_PREFER` | unset |  | STRATA_DISK_PREFER=N (tests): for the next N restores the disk tier also wins a tie with the RAM cache, so one engine restores a conversation from the disk and later the same state from RAM, and the two answers compar... (from the code comment) | `src/program/generate.cpp` |
| `STRATA_DL_CONNECTIONS` | unset |  | (no description in the code) | `setup.py`, `tools/mtp_fetch.py` |
| `STRATA_DMA_BATCH` | unset |  | Documented "--pcie-mode dma, STRATA_DMA_BATCH=1" user option. (from the settings register) | `include/strata/core/dma_batch.hpp` |
| `STRATA_DMA_FLUSH` | on |  | STRATA_DMA_FLUSH (default 1; =0 off): a cudaStreamQuery on the copy stream after a step's uploads and their flag-B raise, so WDDM submits them now rather than at the next driver call (#2188 did the same for the adapti... (from the code comment) | `src/core/verify.cpp` |
| `STRATA_DOORBELL_STORE` | off |  | (no description in the code) | `src/core/verify.cpp`, `sycl/src/core/verify.cpp` |
| `STRATA_DOWN_LANES` | unset |  | Lane-count tuning knob; "16/16 and 16/8 better", no ship/drop verdict. (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_DOWN_WS` | off |  | STRATA_DOWN_WS=1 on a compute capability 6.0 device: the IQ4_NL / Q2_0 down rows at n_ff 640 take native_down_ws_kernel (whatever the expert mode: it is bitwise mode 0's, and mode 8's matches mode 0) (from the code comment) | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_DUMMY_NODES` | 0 |  | (no description in the code) | `src/core/verify.cpp` |
| `STRATA_EARLY_REMOTE_CONTEXTS` | off |  | STRATA_EARLY_REMOTE_CONTEXTS=1: create EVERY secondary context here, like CUDA1's. (from the code comment) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_EARLY_SNAP` | on |  | (no description in the code) | `src/core/verify.cpp` |
| `STRATA_ESIMD_LV` | 2 |  | (no description in the code) | `sycl/src/kernels/cuda/qsa_prompt_attn_esimd.dp.cpp` |
| `STRATA_EXCHANGE_ROTATE` | unset |  | "Documented opt-in" (#2042) with known warnings. (from the settings register) | `src/core/expert_source.cpp`, `src/kernels/cpu/pool.cpp`, `sycl/src/core/expert_source.cpp` |
| `STRATA_EXIT_WITH_PARENT` | unset |  | (no description in the code) | `tools/mtp_fetch.py` |
| `STRATA_EXPDEC` | off |  | Gate/up of the window's resident experts with the q8_1 rows in shared memory | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_EXPDEC_D20` | 0 |  | Kernel option for IQ4_NL down; one bench showed no gain but docs mention it and role is a sub-option. (from the settings register) | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_EXPDEC_DOWN` | 0 |  | Defaults to 1 on gfx12, 0 elsewhere; per-arch default with documented off switch. (from the settings register) | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_EXPDEC_DR` | 4 |  | down rows per lane: 1, 2, 4 (Q2_0: R=4 wins on both cards) (from the code comment) | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_EXPDEC_DRB` | 0 |  | down rows per block (0 = auto) (from the code comment) | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_EXPDEC_MINB` | STRATA_EXPDEC_MINB |  | gate/up __launch_bounds__ min blocks: 2 (<= 128 regs) or 4 (<= 64) (from the code comment) | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_EXPDEC_MINBLK` | 800 |  | auto: the most rows per block that keep this many blocks (from the code comment) | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_EXPDEC_R` | 1 |  | gate/up rows per lane: 1, 2 (bench: R=1 wins on the 2080 Ti) (from the code comment) | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_EXPDEC_RB` | 0 |  | gate/up rows per block (0 = auto) (from the code comment) | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_EXPERT_I8` | unset |  | CUDA / HIP, not a user setting. Unset: the HIP cards gfx1030 / gfx1031 / gfx1032 (RX 6000 series RDNA2, no matrix cores) run a prompt layer's expert groups on our own int8 dot-product kernels (native_expert_i8) for the g (from the settings register) | `include/strata/kernels/q36_prompt_route.hpp`, `src/kernels/cuda/iq_kernels.cu`, `src/qwen35/q36_engine.cu` |
| `STRATA_EXPERT_NIBBLE` | unset |  | (no description in the code) | `include/strata/core/expert_nibble.hpp` |
| `STRATA_EXPERT_NIBBLE_FMTS` | unset |  | (no description in the code) | `include/strata/core/expert_nibble.hpp` |
| `STRATA_EXPERT_NIBBLE_RECHECK` | unset |  | (no description in the code) | `include/strata/core/expert_nibble.hpp` |
| `STRATA_EXPERT_NIBBLE_SKIP` | unset |  | (no description in the code) | `include/strata/core/expert_nibble.hpp` |
| `STRATA_EXPERT_SPLIT` | off |  | Test-compared multi kernel path, off by default. (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_EXPERT_V2` | off |  | grouped decode experts, butterfly warp sums, fewer launches, the GDN step over four blocks per head, producer-written activation images, batched PLE projections, the shared expert on its own stream | `src/kernels/cuda/iq_kernels.cu`, `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_EXPERT_V2K` | off |  | Documented opt-in for UD-Q4_K_XL decode, bitwise, 8-31% faster layer experts. (from the settings register) | `src/kernels/cuda/iq_kernels.cu`, `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_EXP_MODE` | off (-1) |  | Users run STRATA_EXP_MODE=8 on sm_75; mode 4 measured slower; documented tuning mode. (from the settings register) | `src/kernels/cuda/iq_kernels.cu`, `src/kernels/native_grouped_parity.cpp`, `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_FETCH_PVM` | unset |  | (no description in the code) | `src/core/expert_source.cpp`, `sycl/src/core/expert_source.cpp` |
| `STRATA_FETCH_THREADS` | unset |  | Threads that read the experts from the GGUF while it answers (default 8). The prompt path has its own: `STRATA_STAGER_THREADS` (default 32 here) and `STRATA_STAGER_RING` (128). | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_FILL_AHEAD` | unset |  | STRATA_FILL_AHEAD=0 turns it off (from the code comment) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_FILL_SERIAL` | on |  | STRATA_FILL_SERIAL=1: the serial form. (from the code comment) | `sycl/src/program/generate.cpp` |
| `STRATA_FLAGB_DIRECT` | 2 |  | (no description in the code) | `src/core/verify.cpp` |
| `STRATA_FOLLOW` | unset |  | Described as a "STRATA_FOLLOW test path"; test-only switch. (from the settings register) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_FOLLOW_LOGPOS` | off |  | (no description in the code) | `src/core/verify.cpp`, `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_FS_ADMIT` | unset |  | (no description in the code) | `src/core/expert_source.cpp` |
| `STRATA_FS_AHEAD` | unset |  | 1348, one prefetch design: with the Foresight swap space on (STRATA_FS_SLOTS) the look-ahead also runs on the pinned-RAM tiers, where there are no pages to warm, and feeds its predictions to the swap space (STRATA_FS_... (from the code comment) | `src/program/generate.cpp` |
| `STRATA_FS_BUDGET` | unset |  | (no description in the code) | `src/core/expert_source.cpp` |
| `STRATA_FS_DEPTH` | unset |  | (no description in the code) | `src/core/expert_source.cpp` |
| `STRATA_FUSED_SWIGLU_Q81` | unset |  | (no description in the code) | `src/kernels/cuda/shared_expert.cu`, `sycl/src/kernels/cuda/shared_expert.dp.cpp` |
| `STRATA_FUSE_CONVAB` | unset |  | 143-amdk: STRATA_FUSE_CONVAB=1 - the GDN conv and the a/b projections in one launch. (from the code comment) | `src/core/verify.cpp` |
| `STRATA_FUSE_HEAD_GR` | off |  | the step graphs start from the hidden input the graph before them made (needs that every round and step graph makes it: one-row rounds and steps under STRATA_MTP_OVERLAP, and no STRATA_FUSE_HEAD_GR); (from the code comment) | `src/core/mtp.cpp`, `src/core/verify.cpp`, `sycl/src/core/mtp.cpp` +1 |
| `STRATA_FUSE_HITS` | -1 |  | (no description in the code) | `src/core/verify.cpp` |
| `STRATA_FUSE_WAIT` | -1 |  | (no description in the code) | `src/core/verify.cpp` |
| `STRATA_GDN_HEAD` | unset |  | the prompt's gated-delta-net recurrence and the steered layer's fused write | `src/prefill/kernels.cu`, `sycl/src/prefill/kernels.dp.cpp` |
| `STRATA_GDN_KEYHEAD` | unset |  | "8% slower prompt: off" but stays opt-in; listed in PR table, not a user feature. (from the settings register) | `src/prefill/kernels.cu`, `sycl/src/prefill/kernels.dp.cpp` |
| `STRATA_GDN_NOY` | unset |  | the prompt's gated-delta-net recurrence and the steered layer's fused write | `src/prefill/kernels.cu`, `sycl/src/prefill/kernels.dp.cpp` |
| `STRATA_GDN_PIPELINE` | unset |  | (no description in the code) | `src/prefill/kernels.cu`, `sycl/src/prefill/kernels.dp.cpp` |
| `STRATA_GDN_PP` | 0 |  | the prompt's gated-delta-net recurrence and the steered layer's fused write | `src/prefill/kernels.cu`, `sycl/src/prefill/kernels.dp.cpp` |
| `STRATA_GDN_REC_HEADS` | off |  | the one-block-per-head kernel (A/B) (from the code comment) | `src/prefill/kernels.cu`, `sycl/src/prefill/kernels.dp.cpp` |
| `STRATA_GDN_SPLIT` | off |  | grouped decode experts, butterfly warp sums, fewer launches, the GDN step over four blocks per head, producer-written activation images, batched PLE projections, the shared expert on its own stream | `src/kernels/cuda/verify_kernels.cu`, `sycl/src/kernels/cuda/verify_kernels.dp.cpp` |
| `STRATA_GDN_XE` | unset |  | 1,251 to 1,207 (-3.5%) | `sycl/src/prefill/kernels.dp.cpp` |
| `STRATA_GGML_DIR` | unset |  | (no description in the code) | `setup.py`, `tools/hip/build_windows.bat` |
| `STRATA_GGUF_PLAN_READ` | off |  | (no description in the code) | `sycl/src/prefill/prefill.cpp` |
| `STRATA_GGUF_PY` | unset |  | (no description in the code) | `setup.py`, `tools/_paths.py` |
| `STRATA_GROUPED_PAIR_MIN_HITS` | 0 |  | (no description in the code) | `src/kernels/cuda/s2_expert_grouped.cu`, `sycl/src/kernels/cuda/s2_expert_grouped.dp.cpp` |
| `STRATA_GROUPED_V1` | off |  | (no description in the code) | `src/kernels/cuda/iq_kernels.cu`, `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_GROUP_COPY` | off |  | (no description in the code) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_GRP_KEEP_GOING` | off |  | (STRATA_GRP_REPS=N more repetitions, STRATA_GRP_KEEP_GOING=1 counts every failure instead of stopping at the first: to measure a rare one) (from the code comment) | `src/kernels/gr_parity.cpp` |
| `STRATA_GRP_REPS` | unset |  | (STRATA_GRP_REPS=N more repetitions, STRATA_GRP_KEEP_GOING=1 counts every failure instead of stopping at the first: to measure a rare one) (from the code comment) | `src/kernels/gr_parity.cpp` |
| `STRATA_GU_LANES` | unset |  | Lane-count tuning knob for SYCL quants; measurement note only. (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_GU_SWIGLU` | -1 |  | 147-verify: STRATA_GU_SWIGLU=1 runs gate/up, SwiGLU and the q8_1 blocks as native_gu_swiglu_kernel, where launch_gu would run native_gu_multi_kernel's 16-lane path (n_embd 2560, the split formats); bitwise the same hq. (from the code comment) | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_HC8_DOWN` | 0 |  | (no description in the code) | `src/kernels/cuda/hc8_read.cuh` |
| `STRATA_HC8_NV` | unset |  | (no description in the code) | `src/kernels/cuda/fused_gr.cu` |
| `STRATA_HC8_PDL` | 1 |  | (no description in the code) | `src/kernels/cuda/fused_gr.cu` |
| `STRATA_HC8_UP` | 1 |  | (no description in the code) | `src/kernels/cuda/hc8_read.cuh` |
| `STRATA_HCD_EXACT` | off |  | The hyper-connection down projection (N 320, K 10240) in a WMMA kernel that reproduces hipBLASLt solution 1176 / 1177's summation order (`hip_prefill_hcd_exact_parity` checks it bit for bit). It runs only where the table makes hipBLASLt pick one of those two, and says once if it cannot. | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_HEAD_MIX_MULTI` | on |  | Part of PR off-switch set measured on AMD; no explicit verdict for this switch. (from the settings register) | `src/core/gfx12_extra_test.cpp`, `src/core/mtp.cpp`, `src/core/verify.cpp` +2 |
| `STRATA_HEAD_PLANES` | on |  | STRATA_HEAD_PLANES (default on; =0 keeps the GGUF layout): a Q8_0 head of 2560-wide rows rewritten in place as bit planes (q80_vhead.hpp), the same bytes per row. (from the code comment) | `src/core/native_head.cpp` |
| `STRATA_HEAD_SCREEN` | on |  | (no description in the code) | `src/core/verify.cpp` |
| `STRATA_I8_BISECT` | unset |  | Debug only: STRATA_EXPERT_I8's kernels return at point n, uniform over the block (a fault hunt on HIP). 0: off. (from the settings register) | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_I8_FEWROWS` | unset |  | Research: the few-rows route of STRATA_EXPERT_I8's prompt path: a layer whose experts average fewer routed rows than the limit runs on the grouped kernels instead of the int8 kernel. Not set: 6 on a Tesla P100, 7 elsewhe (from the settings register) | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_I8_FORCE` | on |  | (no description in the code) | `sycl/src/prefill/esimd_moe.dp.cpp` |
| `STRATA_I8_GRF` | 256 |  | (no description in the code) | `sycl/src/prefill/esimd_moe.dp.cpp` |
| `STRATA_I8_MT` | 0 |  | (no description in the code) | `sycl/src/prefill/esimd_moe.dp.cpp` |
| `STRATA_I8_N8` | unset |  | The 8-entry slices of the tile-2..4 int8 expert kernel (STRATA_EXPERT_I8's prompt path; two weight rows and one entry of each 8-entry slice per thread; output bit-identical). Not set: on for tile 2 (up to about 1,950 tok (from the settings register) | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_I8_TM` | unset |  | Research: the tile of STRATA_EXPERT_I8's kernels, 2, 3 or 4 rows of 16 entries per thread (32, 48 or 64 entries a block), 8 = the 128-entry tile. Unset: chosen by the entries per expert. (from the settings register) | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_I8_WG` | 1 |  | (no description in the code) | `sycl/src/prefill/esimd_moe.dp.cpp` |
| `STRATA_IDX_APPEND_PAR` | unset |  | the indexer chain: block scores one (block, query) pair a warp, batched resolve lookups, the window append in parallel and on its own branch beside the query projection | `src/kernels/cuda/native_qsa_indexer.cu` |
| `STRATA_IDX_FORK` | unset |  | the indexer chain: block scores one (block, query) pair a warp, batched resolve lookups, the window append in parallel and on its own branch beside the query projection | `src/core/verify.cpp` |
| `STRATA_IGPU_DEFAULTS` | on |  | iGPU defaults on gfx1103/1150/1151; =0 turns them all off. (from the settings register) | `src/core/arch_defaults.cpp` |
| `STRATA_IL_PAD` | 1 |  | "bit 0, default on, =0 off" for 6-column padded IQ4_XS kernel (v0.2.10). (from the settings register) | `src/kernels/cuda/native_mmvq.cu` |
| `STRATA_INDEXER_PER_TOKEN` | off |  | STRATA_INDEXER_PER_TOKEN=1: the old (from the code comment) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_IO_PREFETCH` | unset |  | STRATA_IO_PREFETCH=0 / STRATA_IO_PF_GATE=0 / STRATA_COLD_OVERLAP=0 turn them off. io prefetch + the gate: the GGUF read in place and (0.2.1) an experts.bin pack; zero-copy only for the GGUF in place (from the code comment) | `src/program/generate.cpp` |
| `STRATA_IO_PREFETCH_DEPTH` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_IO_THREADS` | unset |  | Queue depth tuning knob; larger values give no gain. (from the settings register) | `src/platform/direct_file.cpp`, `sycl/src/platform/direct_file.cpp` |
| `STRATA_IQ256_GATHER` | 0 |  | (no description in the code) | `src/kernels/cpu/expert_layout.cpp`, `src/kernels/cpu/iq_avx2_parity.cpp` |
| `STRATA_IQ3S_MT1` | unset |  | Listed among speed opt-in switches in 0.1.40 notes. (from the settings register) | `src/kernels/cpu/native_expert.cpp` |
| `STRATA_IQ3S_WIDE` | unset |  | STRATA_IQ3S_WIDE=1: the wide kernel; unset or 0: the multi kernel (default). (from the code comment) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_IQ4XS512_MIN` | 1 |  | STRATA_IQ4XS512_MIN=N: from N tokens on. (from the code comment) | `src/kernels/cpu/native_expert.cpp` |
| `STRATA_IQ512_FUSED_MAX` | 0 |  | "REJECT as a default; opt-in kept"; code says OPT-IN (N). (from the settings register) | `src/kernels/cpu/native_expert.cpp` |
| `STRATA_IQ512_FUSED_TYPES` | unset |  | (no description in the code) | `src/kernels/cpu/native_expert.cpp` |
| `STRATA_IQ_GATHER` | unset |  | CPU gather switch left off because Zen 4 gathers are microcoded; hardware tuning. (from the settings register) | `src/kernels/cpu/iq_avx512.cpp` |
| `STRATA_IQ_MT_MIN` | on (2) |  | PR #1818 "makes STRATA_IQ_MT_MIN=1 behaviour the DEFAULT"; variable keeps old rounding. (from the settings register) | `src/kernels/cpu/iq_avx2_parity.cpp`, `src/kernels/cpu/native_expert.cpp`, `src/kernels/native_expert_parity.cpp` |
| `STRATA_IQ_PREFETCH` | 2048 |  | E-2 prefetching, the same knob as the AVX-2/512 kernels: STRATA_IQ_PREFETCH is the distance in bytes. (from the code comment) | `src/kernels/cpu/iq_avx1.cpp`, `src/kernels/cpu/iq_avx2.cpp`, `src/kernels/cpu/iq_avx2_parity.cpp` +1 |
| `STRATA_IQ_STAGE_GRID` | unset |  | STRATA_IQ_STAGE_GRID=0 disables staging 64-bit i-quant codebook tables into shared memory (from the code comment) | `src/kernels/cuda/iq_kernels.cu`, `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_IQ_STAGE_GRID_MMVQ` | off |  | So mmvq stages only with STRATA_IQ_STAGE_GRID_MMVQ=1 (bitwise the same either way). (from the code comment) | `src/kernels/cuda/iq_kernels.cu`, `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_IQ_STAGE_TABLES` | unset |  | (no description in the code) | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_KEEP_IO_PRIORITY` | off |  | (no description in the code) | `src/platform/io_priority.cpp` |
| `STRATA_KEY` | unset |  | (no description in the code) | `tools/early_close_test.py`, `tools/par_load.py`, `tools/serve_load.py` +1 |
| `STRATA_KFD_QUIET` | unset |  | =0 turns off the engine's KFD-quiet changes on an integrated gfx11 GPU under Linux (gfx1103, gfx1150, gfx1151: the load order, the drain on a stop by signal, the staged pageable copies; on gfx1103 also the unregistered e (from the settings register) | `include/strata/core/kfd_quiet.hpp`, `src/program/generate.cpp` |
| `STRATA_KFD_QUIET_LEND` | unset |  | =1 makes the prompt path keep its own buffers instead of lending expert cache slots on an integrated gfx11 GPU under Linux, for chunks up to 16,384 tokens where the buffers fit with 3 GiB to spare (off by default; =0 or  (from the settings register) | `src/program/generate.cpp` |
| `STRATA_KFD_QUIET_NOREG` | unset |  | =0 registers the expert arena with the GPU driver on the Radeon 780M class (gfx1103) under Linux (by default it is not registered and not locked with mlock there; STRATA_ARENA_PIN_GIB or STRATA_UMA_REGISTER=1 set by the  (from the settings register) | `src/program/generate.cpp` |
| `STRATA_KFD_QUIET_XFER` | unset |  | =0 leaves GPU_PINNED_MIN_XFER_SIZE unset on an integrated gfx11 GPU under Linux (the engine sets it to 1000000 before HIP starts, so pageable copies go through the HIP runtime's staging buffers; a value the caller set wi (from the settings register) | `src/program/generate.cpp` |
| `STRATA_KQ256` | unset |  | Multi-token AVX2 kernels for the Q4_K / Q5_1 / Q8_0 experts. Bit-exact with ggml's, but measured no faster, so off. | `src/kernels/cpu/native_expert.cpp` |
| `STRATA_KQ_GU_LANES` | unset |  | Opt-in lane split tuning for K-quant gate/up; reduction order differs. (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_KQ_SPLIT` | off |  | Measured as KqSplit 0 vs 1 for a no-regression check; no verdict. (from the settings register) | `src/kernels/cuda/iq_kernels.cu`, `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_L2PF` | 0 |  | STRATA_L2PF=MB (0 = off): while a layer's window waits for the CPU pool's rows, a branch reads the next layer's weights (in the order its kernels use them, up to MB megabytes) into L2. (from the code comment) | `src/core/verify.cpp` |
| `STRATA_L2PF_BLOCKS` | 32 |  | (no description in the code) | `src/core/verify.cpp` |
| `STRATA_LAYER_HASH_FILE` | unset |  | Debug (Intel Arc verify path): after each CPU-expert service writes a line of FNV hashes of the activation rows, routed ids, weights and CPU-expert output rows; compare two runs to find the first layer that differs (CLEA (from the settings register) | `sycl/src/core/verify.cpp` |
| `STRATA_LAYER_SPLIT_GATE` | off |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_LEND_MIRROR` | unset |  | STRATA_LEND_MIRROR=0 turns it off; =N: the last N slots (default 10% of the cache with a loan, at most 1,200). (from the code comment) | `sycl/src/program/generate.cpp` |
| `STRATA_LFUSE` | off |  | grouped decode experts, butterfly warp sums, fewer launches, the GDN step over four blocks per head, producer-written activation images, batched PLE projections, the shared expert on its own stream | `src/core/verify.cpp`, `sycl/src/core/verify.cpp` |
| `STRATA_LFUSE_GATE` | on |  | (no description in the code) | `src/core/verify.cpp`, `sycl/src/core/verify.cpp` |
| `STRATA_LFUSE_PAIR` | on |  | (no description in the code) | `src/core/verify.cpp`, `sycl/src/core/verify.cpp` |
| `STRATA_LOGPOS` | off |  | Quant yardstick / KL harness switch, reads all window logits. (from the settings register) | `metal/src/engine/window.cpp`, `src/core/verify.cpp`, `src/program/generate.cpp` +1 |
| `STRATA_LOGPOS_EXTRA` | unset |  | (no description in the code) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_LOOKAHEAD` | unset |  | Turns off the routing-aware prefetch (on by default in this mode). While the CPU works on a layer, a thread applies the next layer's router to this layer's input and asks the OS to read the predicted experts' pages. Pages only; the answers are the same. `STRATA_LOOKAHEAD_K` sets the experts per token (default 10). | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_LOOKAHEAD_K` | 10 |  | (no description in the code) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_LOOKUP_CHAIN_FIXED` | off |  | no policy gate (from the code comment) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_LOOKUP_MAX_T` | unset |  | strata-nvfp4: off unless set (see fork_env_or) (from the code comment) | `src/program/generate.cpp` |
| `STRATA_MANIFEST` | unset |  | [checked] Run manifest file for measurement (include/strata/core/run_manifest.hpp:5, 'OFF by default'); strata_control/split_engine.py:506 sets it for the split engine's own runs. (from the settings register) | `include/strata/core/run_manifest.hpp` |
| `STRATA_MIRROR_MIB` | 0 |  | Free RAM less 4 GiB | `sycl/src/program/generate.cpp` |
| `STRATA_MMVF_EXACT78` | unset |  | exact router kernels for windows of 7 and 8 tokens | `src/kernels/cuda/native_bf16.cu` |
| `STRATA_MMVF_ROWS` | off |  | decode layouts and kernels (packed Q8_0 / Q6_K weights, 4-row BF16 GEMV, one-cell-per-thread attention scores) | `src/kernels/cuda/native_bf16.cu`, `src/kernels/mmvf_rows_parity.cpp`, `sycl/src/kernels/cuda/native_bf16.dp.cpp` |
| `STRATA_MMVQ_A2` | unset |  | (no description in the code) | `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_MMVQ_CSPLIT` | off |  | STRATA_MMVQ_CSPLIT (strata-nvfp4): Q8_0 windows of 5..8 columns as two column groups of neighbouring blocks (the kernel's CSPLIT, bitwise the same outputs). (from the code comment) | `src/kernels/cuda/native_mmvq.cu` |
| `STRATA_MMVQ_IL` | on |  | STRATA_MMVQ_IL=0 keeps native_mmvq's multi-column kernels (from the code comment) | `src/core/verify.cpp` |
| `STRATA_MMVQ_IL_120` | on |  | 147-verify: =0 gives a big 12.0 card the shared table again (from the code comment) | `src/kernels/cuda/native_mmvq.cu` |
| `STRATA_MMVQ_IL_89` | unset |  | (no description in the code) | `src/kernels/cuda/native_mmvq.cu`, `src/kernels/mmvq_il_parity.cpp` |
| `STRATA_MMVQ_IL_PF` | 0 |  | Hoisted-load, exact-integer kernels for the interleaved small-batch matvec where the sm_80 table picks them (Q6_K only so far, [MMVQ_IL_TABLE.md](../MMVQ_IL_TABLE.md)). On a card that is not sm_80 the value 1 does nothing. `=2` to `=6` force one variant everywhere (A/B only). | `src/kernels/cuda/native_mmvq.cu` |
| `STRATA_MMVQ_IL_Q3K` | on |  | (no description in the code) | `src/kernels/cuda/native_mmvq.cu` |
| `STRATA_MMVQ_IL_ROWS` | unset |  | (no description in the code) | `src/kernels/cuda/native_mmvq.cu` |
| `STRATA_MMVQ_IL_T1` | on |  | (no description in the code) | `src/kernels/cuda/native_mmvq.cu` |
| `STRATA_MMVQ_IL_WIDE` | on |  | (no description in the code) | `src/kernels/cuda/native_mmvq.cu` |
| `STRATA_MMVQ_LARGE_GRF` | unset |  | On for Arc A-series | `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_MMVQ_PF` | unset |  | Retired in 0.2.13 (cleanup 4.5, batch B): code removed, the setting is no longer read. Evidence: [checked] Aligned weight-load prefetch in the wide Q6_K kernel (SYCL only): STRATA_MMVQ_PF=2/3/4/6 slower everywhere (qkv 7 (from the settings register) | `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_MMVQ_RPW` | unset |  | default: the shared kernel (measured equal at warm clocks) (from the code comment) | `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_MMVQ_SG` | unset |  | Retired in 0.2.13 (cleanup 4.5, batch B): code removed, the setting is no longer read. Evidence: [checked] SIMD16 builds of the multi-column mmvq and the wide Q6_K kernel (SYCL only): 'Tried, no gain', no setting beats S (from the settings register) | `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_MMVQ_V2` | off |  | decode: attention k+v+q and the drafter's k+v in one Q8_0 launch (the same bits; under 1%) | `src/kernels/cuda/native_mmvq.cu` |
| `STRATA_MMVQ_WAVE` | unset |  | Test control for multi_exact parity; test-only. (from the settings register) | `src/kernels/cuda/native_mmvq.cu`, `src/kernels/mmvq_multi_parity.cpp`, `sycl/src/kernels/cuda/native_mmvq.dp.cpp` +1 |
| `STRATA_MMVQ_WIDE` | unset |  | (no description in the code) | `sycl/src/kernels/cuda/native_mmvq.dp.cpp`, `sycl/src/kernels/mmvq_bench.cpp` |
| `STRATA_MMVQ_WIDE_32` | unset |  | (no description in the code) | `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_MMVQ_WIDE_K` | unset |  | (no description in the code) | `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_MMVQ_WIDE_K_RPW` | unset |  | (no description in the code) | `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_MMVQ_WIDE_MAXC` | unset |  | (no description in the code) | `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_MMVQ_WIDE_MIN` | unset |  | (no description in the code) | `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_MMVQ_WIDE_RPW` | unset |  | (no description in the code) | `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_NAN_INJECT` | unset |  | Test injection for the NaN guard. (from the settings register) | `src/core/verify.cpp` |
| `STRATA_NIB_GU_PROTO` | 0 |  | STRATA_NIB_GU_PROTO: unset / 0 = off (the S / P skeleton per format); 1 (or 31) = gu-proto 31 (the activations staged once per two row blocks with 16-byte loads, the production row dot). (from the code comment) | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_NO_AVXVNNI` | unset |  | (no description in the code) | `src/kernels/cpu/expert_layout.cpp`, `src/kernels/cpu/iq_avx2_parity.cpp` |
| `STRATA_NO_IQ128` | on |  | (no description in the code) | `src/kernels/cpu/native_expert.cpp`, `src/program/generate.cpp` |
| `STRATA_NO_IQ256` | on |  | Turns off the IQ256 gather path that is used by default. (from the settings register) | `src/kernels/cpu/native_expert.cpp`, `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_NO_IQ4NL` | on |  | STRATA_NO_IQ4NL falls back to it. (from the code comment) | `src/kernels/cpu/native_expert.cpp` |
| `STRATA_NO_IQ4XS512` | on |  | STRATA_NO_IQ4XS512=1 keeps ggml's dot; (from the code comment) | `src/kernels/cpu/native_expert.cpp` |
| `STRATA_NO_IQ512` | on |  | Kill-switch for the AVX-512 IQ multi-token kernels that are on by default. (from the settings register) | `src/kernels/cpu/native_expert.cpp` |
| `STRATA_NO_IQ512X1` | on |  | STRATA_NO_IQ512X1=1 turns this off; the first use checks the bits against this ggml. (from the code comment) | `src/kernels/cpu/native_expert.cpp` |
| `STRATA_NO_KFD_SETTLE` | unset |  | turns off the wait before READY on an integrated gfx11 GPU under Linux (the engine waits until its own KFD eviction counter was quiet for 2 s, at most 10 s; automatic on that class, nothing on other devices); for A/B run (from the settings register) | `include/strata/core/kfd_settle.hpp` |
| `STRATA_NO_MMVQ_ROWS2` | unset |  | [checked] per-card rule picks one or two rows; =1 forces one row everywhere, =0 two everywhere (src/kernels/cuda/native_mmvq.cu:1346); tunable override of a built-in choice, identical text per notes-quants.md:219. (from the settings register) | `src/kernels/cuda/native_mmvq.cu`, `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_NO_MULTI_GR` | off |  | (no description in the code) | `src/core/mtp.cpp`, `src/core/verify.cpp`, `src/kernels/cuda/fused_gr.cu` +4 |
| `STRATA_NO_NVFP4_256` | on |  | the CPU's NVFP4 rows on ggml-cpu's dot instead of the AVX-512 / AVX2 kernels | `src/kernels/cpu/native_expert.cpp` |
| `STRATA_NO_NVFP4_512` | on |  | the CPU's NVFP4 rows on ggml-cpu's dot instead of the AVX-512 / AVX2 kernels | `src/kernels/cpu/native_expert.cpp` |
| `STRATA_NO_Q50_512` | on |  | STRATA_NO_Q50_512=1 keeps ggml's dot; (from the code comment) | `src/kernels/cpu/native_expert.cpp` |
| `STRATA_NO_Q8K_AVX1` | on |  | STRATA_NO_Q8K_AVX1=1 keeps ggml's scalar reference on those CPUs). (from the code comment) | `src/kernels/cpu/native_expert.cpp` |
| `STRATA_NO_Q8K_AVX2` | on |  | STRATA_NO_Q8K_AVX2=1: ggml's on any CPU. (from the code comment) | `src/kernels/cpu/native_expert.cpp` |
| `STRATA_NO_SUB16_GU` | off |  | (no description in the code) | `src/kernels/cuda/iq_kernels.cu`, `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_NO_TQ_KERNELS` | unset |  | (no description in the code) | `src/kernels/cpu/tq_cpu.cpp` |
| `STRATA_NO_VBMI` | unset |  | STRATA_NO_VBMI=1 answers no (a test knob: it runs the substitute q2 unpack on a CPU that has VBMI) (from the code comment) | `src/kernels/cpu/expert_layout.cpp` |
| `STRATA_NUMA_CARD` | unset |  | On by default on Linux hosts with more than one memory node: the engine keeps its threads on the graphics card's node, so the pinned buffers it allocates land there too (only when that node has the CPUs and the free memo (from the settings register) | `src/program/generate.cpp` |
| `STRATA_NUMA_POOL` | unset |  | NUMA split turned on by default in 0.2.2; =0 turns it off. (from the settings register) | `src/kernels/cpu/pool.cpp`, `src/program/generate.cpp` |
| `STRATA_NUMA_SPLIT` | unset |  | 0.2.2 auto-splits the CPU pool per NUMA node; variables turn it off. (from the settings register) | `src/kernels/cpu/pool.cpp`, `src/program/generate.cpp` |
| `STRATA_NVFP4_PREFETCH` | 2048 |  | software prefetch distance in bytes (STRATA_NVFP4_PREFETCH, 0 = off), as the AVX-512 rows (read on first use, behind the CPU check: a static initialiser here would run on every CPU, see above) (from the code comment) | `src/kernels/cpu/nvfp4_avx2.cpp`, `src/kernels/cpu/nvfp4_avx512.cpp`, `src/kernels/cpu/nvfp4_avx512_parity.cpp` |
| `STRATA_NV_KQ_DEFAULT` | unset |  | (no description in the code) | `src/core/nv_defaults.cpp` |
| `STRATA_OLD_GROUPED` | off |  | (no description in the code) | `src/kernels/cuda/s2_expert_grouped.cu`, `sycl/src/kernels/cuda/s2_expert_grouped.dp.cpp` |
| `STRATA_OLD_IQ_MMVQ` | off |  | [checked] Per-column kernels 'bitwise equal to the new ones; kept for A/B timing' (src/kernels/cuda/iq_kernels.cu:3419); also the parity reference of iq_multi_parity.cpp:146 and native_expert_parity.cpp:445 (iq_set_old_k (from the settings register) | `src/kernels/cuda/iq_kernels.cu`, `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_OLD_Q80_SCALED` | off |  | (no description in the code) | `src/kernels/cuda/quantize_act.cu` |
| `STRATA_OWNED_PRICE` | unset |  | STRATA_OWNED_PRICE=exact prices the real allocation instead (Prefill::bytes_needed_owned: 2 MiB pages, the ring in one piece, as PR #796 measured) and gives that cache the difference. (from the code comment) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_PARK_ASYNC` | off |  | A park copies the K/V card-to-card into a VRAM stage (only while 1.5 GB stay free) and moves it to the host in the background. The checkpoints the request drops move into the parked image instead of being copied. | `src/core/conversation_snapshot.cpp`, `src/program/generate.cpp` |
| `STRATA_PARK_REWIND_MAX` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_PARK_SEGPOOL_MIB` | 0 |  | on by default on Linux (lane PARK-POOL): unset = an arena of min(2 GiB, a quarter of the conversation cache) in 64 MiB steps (none under 128 MiB), reserved once and made resident in the background; the K/V buffers of par (from the settings register) | `src/program/generate.cpp` |
| `STRATA_PAR_COUNTERS` | off |  | Release notes call it a research switch for parallel work. (from the settings register) | `include/strata/program/par_counters.hpp`, `serve/server.py` |
| `STRATA_PA_FAST` | off |  | Prompt attention with single FP16 q and p | `src/kernels/cuda/qsa_prompt_attn.cu`, `sycl/src/kernels/cuda/qsa_prompt_attn.dp.cpp` |
| `STRATA_PA_RPF` | unset |  | Attention-kernel variant tried in a strix prefill experiment. (from the settings register) | `src/kernels/cuda/qsa_prompt_attn.cu` |
| `STRATA_PA_TRIM` | unset |  | Defaults in the code: trim 2 + rpf on; PA_TRIM=0\|1\|2 selects variants. (from the settings register) | `src/kernels/cuda/qsa_prompt_attn.cu` |
| `STRATA_PA_V1_PREFETCH` | -1 |  | Neutral prefill, decode slower on R9700; experiment, no shipped verdict. (from the settings register) | `src/kernels/cuda/qsa_prompt_attn.cu` |
| `STRATA_PA_V2T` | off |  | Turing: v2 with register staging instead of cp.async, opt-in (STRATA_PA_V2T=1) until measured (from the code comment) | `src/kernels/cuda/qsa_prompt_attn.cu` |
| `STRATA_PEER_DIRECT` | unset |  | (no description in the code) | `src/core/peer_experts.cpp` |
| `STRATA_PEER_HOT` | 0.45 |  | Multi-GPU: STRATA_PEER_HOT=<f> gives the peer card a share f of the HOT pairs, so both cards compute routed experts every layer (the primary alone did ~25 of ~30 per layer-window). (from the code comment) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_PEER_HOT_AT` | 8700 |  | (no description in the code) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_PEER_NO_P2P` | unset |  | Workaround for opt-in peer mode first-request fault (26/26 clean); hardware workaround. (from the settings register) | `src/core/peer_experts.cpp` |
| `STRATA_PIPED_CONCURRENT` | unset |  | (no description in the code) | `src/core/verify.cpp`, `src/program/generate.cpp` |
| `STRATA_PIPED_CONCURRENT_MIN` | unset |  | STRATA_PIPED_CONCURRENT_MIN=N (opt-in, with STRATA_PIPED_CONCURRENT=1): only a read of at least N tokens runs beside the decoding slots on its own thread; a shorter one takes the usual path (drained windows, no priori... (from the code comment) | `src/program/generate.cpp` |
| `STRATA_PIPED_CONCURRENT_PRIO` | unset |  | (no description in the code) | `src/core/verify.cpp` |
| `STRATA_PIPED_READ_PIECE` | unset |  | Opt-in (#1906): on the pipelined slot groups a read goes in pieces of STRATA_PIPED_READ_PIECE tokens (whole chunks). (from the code comment) | `src/program/generate.cpp` |
| `STRATA_PIPELINE_AGREE` | unset |  | (no description in the code) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_PIPELINE_AUTO` | unset |  | (no description in the code) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_PIPELINE_AUTO_MIN_STAGES` | 3 |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_PIPELINE_DOOM_SKIP` | unset |  | STRATA_PIPELINE_DOOM_SKIP=0: served in full. (from the code comment) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_PIPELINE_LOOKUP_PON` | unset |  | STRATA_PIPELINE_LOOKUP_PON: B's estimate (default 0.75; 0 = no lookup B) (from the code comment) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_PIPELINE_PRESTAGE` | unset |  | (no description in the code) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_PIPELINE_SNAP_OVERLAP` | unset |  | (no description in the code) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_PIPELINE_SWITCH` | unset |  | (no description in the code) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_PIPELINE_THETA` | unset |  | (no description in the code) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_PIPE_BYIELD_NODRAIN` | off |  | a BYIELD that reaches a pipelined batch is dropped without draining the groups | `src/program/generate.cpp` |
| `STRATA_PIPE_HELPER_SIBLING` | off |  | (no description in the code) | `sycl/src/core/verify.cpp` |
| `STRATA_POOL_CORES` | unset |  | Experimental, off by default --pool-cores option, documented. (from the settings register) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_POOL_UMWAIT` | unset |  | STRATA_POOL_UMWAIT=<TSC cycles> (experimental, x86 WAITPKG only): a parked worker waits with UMONITOR/UMWAIT on the epoch's cache line instead of spinning on _mm_pause. (from the code comment) | `src/kernels/cpu/pool.cpp` |
| `STRATA_POOL_UMWAIT_C02` | unset |  | (no description in the code) | `src/kernels/cpu/pool.cpp` |
| `STRATA_POOL_UMWAIT_ESC` | unset |  | STRATA_POOL_UMWAIT_ESC=<TSC cycles> (experimental): an escalating wait - C0.1 for the first cycles of a park, C0.2 after (a wait that has lasted that long is a GPU phase, a draft or a prompt read, where a slower wake... (from the code comment) | `src/kernels/cpu/pool.cpp` |
| `STRATA_PRIMARY_DEVICE` | 0 |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_PRIVATE_B60_NEXT_POOL` | off |  | Private opt-in (=1), off by default: the 'B60 next pool' path of the fused GR and IQ kernels (the name is the code's); read at sycl/src/kernels/cuda/fused_gr.dp.cpp:2895 and iq_kernels.dp.cpp:5191 and only effective on t (from the settings register) | `sycl/src/kernels/cuda/fused_gr.dp.cpp`, `sycl/src/kernels/cuda/iq_kernels.dp.cpp`, `sycl/src/kernels/cuda/native_mmvq.dp.cpp` +2 |
| `STRATA_PRIVATE_DOWN_S4` | off |  | Private opt-in (=1), off by default: the S4 down-projection variant of the IQ2 kernels (p4_enabled, sycl/src/kernels/cuda/iq_kernels.dp.cpp:4721). Read from the code only; the integration lane measured nothing on it. (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_PRIVATE_GU_GRID64` | off |  | Private opt-in (=1), off by default: the 64-wide grid variant of the IQ gate/up kernel (grid64_enabled, sycl/src/kernels/cuda/iq_kernels.dp.cpp:4816). Read from the code only; the integration lane measured nothing on it. (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_PRIVATE_GU_GRID_S4` | off |  | Private opt-in (=1), off by default: the S4 grid table variant of the IQ gate/up kernel (grid_s4_enabled, sycl/src/kernels/cuda/iq_kernels.dp.cpp:4913). Read from the code only; the integration lane measured nothing on i (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_PRIVATE_GU_IQ1_PACKED` | off |  | Private opt-in (=1), off by default: packed-weight IQ1_M gate/up kernel, Battlemage cards with packed weights only (i1_enabled, sycl/src/kernels/cuda/iq_kernels.dp.cpp:5090). Read from the code only; the integration lane (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp`, `sycl/src/kernels/cuda/weight_permute.dp.cpp` |
| `STRATA_PRIVATE_GU_PREPARED` | off |  | Private opt-in (=1), off by default: prepared row reuse of the IQ2 gate/up kernel (gu22_row_reuse_enabled, sycl/src/kernels/cuda/iq_kernels.dp.cpp:4227). Read from the code only; the integration lane measured nothing on  (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_PRIVATE_GU_S4_BASIS9` | off |  | Private opt-in (=1), off by default: basis-9 variant of the S4 grid gate/up kernel (grid_s4_basis9_enabled, sycl/src/kernels/cuda/iq_kernels.dp.cpp:5310). Read from the code only; the integration lane measured nothing on (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_PRIVATE_GU_S4_SLM` | off |  | Private opt-in (=1), off by default: shared-local-memory variant of the S4 grid gate/up kernel (grid_s4_slm_enabled, sycl/src/kernels/cuda/iq_kernels.dp.cpp:5192). Read from the code only; the integration lane measured n (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_PRIVATE_GU_S4_WG16` | off |  | Private opt-in (=1), off by default: work-group-16 variant of the S4 grid gate/up kernel (grid_s4_wg16_enabled, sycl/src/kernels/cuda/iq_kernels.dp.cpp:5427). Read from the code only; the integration lane measured nothin (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_PRIVATE_GU_XXS_PACKED` | off |  | Private opt-in (=1), off by default: packed-weight IQ2_XXS gate/up kernel, Battlemage cards with packed weights only (p16_enabled, sycl/src/kernels/cuda/iq_kernels.dp.cpp:4623). Read from the code only; the integration l (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp`, `sycl/src/kernels/cuda/weight_permute.dp.cpp` |
| `STRATA_PRIVATE_GU_XXS_S4` | off |  | Private opt-in (=1), off by default: sixteen-row S4 variant of the IQ2_XXS gate/up kernel (xxs_s4_enabled, sycl/src/kernels/cuda/iq_kernels.dp.cpp:5480). Read from the code only; the integration lane measured nothing on  (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_PRIVATE_IQ4_INT4_DPAS` | off |  | Private opt-in (=1), off by default: exact int4 DPAS path of the IQ4 matrix-vector kernel when the weights are packed (sycl/src/kernels/cuda/native_mmvq.dp.cpp:3717). Read from the code only; the integration lane measure (from the settings register) | `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_PRIVATE_PACKED_REGISTER` | off |  | Private opt-in (=1), off by default: register-resident packed-weight kernel, Battlemage cards with packed weights only (ps_enabled, sycl/src/kernels/cuda/iq_kernels.dp.cpp:4467). Read from the code only; the integration  (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_PROXY_FINAL2` | 0 |  | (no description in the code) | `src/kernels/cuda/draft_proxy.cu` |
| `STRATA_PROXY_L2` | unset |  | (no description in the code) | `src/kernels/cuda/draft_proxy.cu` |
| `STRATA_PW_WITH_BATCH` | off |  | CUDA/HIP layer split with `--pipeline-windows 2` and `--batch` slots | `src/program/generate.cpp` |
| `STRATA_Q2_BITPLANE` | off |  | AVX2 CPUs without AVX-512 | `src/kernels/cpu/q2_avx1.cpp`, `src/kernels/cpu/q2_avx2.cpp` |
| `STRATA_Q2_WARP` | unset |  | (no description in the code) | `src/kernels/cuda/native_mmvq.cu` |
| `STRATA_Q36_BLAS_F32` | off |  | the hipBLAS / cuBLAS SGEMM (A/B) (from the code comment) | `src/qwen35/q36_engine.cu` |
| `STRATA_Q36_KEEP_PACKED` | on |  | (no description in the code) | `src/qwen35/q36_engine.cu` |
| `STRATA_Q36_NO_EXT` | unset |  | (no description in the code) | `src/qwen35/q36_engine.cu` |
| `STRATA_Q36_OLD_KERNELS` | on |  | (no description in the code) | `src/qwen35/q36_engine.cu` |
| `STRATA_Q36_READ_WINDOWS` | unset |  | Documented tunable of read-time windows for Qwen3.6. (from the settings register) | `src/qwen35/q36_engine.cu` |
| `STRATA_Q36_STREAM_ALL` | off |  | STRATA_Q36_STREAM_ALL=1: the whole layer from host memory (the first version: PCIe-bound) (from the code comment) | `src/qwen35/q36_engine.cu` |
| `STRATA_Q50_512_MIN` | 2 |  | STRATA_Q50_512_MIN=N: from N tokens on (default 2). (from the code comment) | `src/kernels/cpu/native_expert.cpp` |
| `STRATA_Q6P_GRID` | 0 |  | (no description in the code) | `src/kernels/cuda/native_mmvq.cu`, `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_Q6P_SELFTEST` | unset |  | STRATA_Q6P_SELFTEST=1: bitwise check against the GGUF-layout kernels (1..4 columns) and paired timings (from the code comment) | `src/kernels/cuda/native_mmvq.cu`, `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_Q6_PACKED` | off |  | decode layouts and kernels (packed Q8_0 / Q6_K weights, 4-row BF16 GEMV, one-cell-per-thread attention scores) | `src/kernels/cuda/native_mmvq.cu`, `src/program/generate.cpp`, `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_Q6_PLANAR` | unset |  | (no description in the code) | `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_Q8_GATHER_WIDE` | unset |  | Turns off the 8-wide int8 KV gather (#1095) that the experimental build uses below compute capability 7.5. | `src/kernels/cuda/kv_q8.cu` |
| `STRATA_Q8_PACKED` | off |  | decode layouts and kernels (packed Q8_0 / Q6_K weights, 4-row BF16 GEMV, one-cell-per-thread attention scores) | `src/kernels/cuda/native_mmvq.cu`, `src/qwen35/q36_engine.cu`, `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_QFUSE` | unset |  | grouped decode experts, butterfly warp sums, fewer launches, the GDN step over four blocks per head, producer-written activation images, batched PLE projections, the shared expert on its own stream | `src/core/arch_defaults_test.cpp`, `src/core/verify.cpp`, `sycl/src/core/verify.cpp` |
| `STRATA_QFUSE_HC` | 0 |  | HC up fused-quantize kernel variant with sub-switch; tuning experiment. (from the settings register) | `src/core/mtp.cpp`, `src/core/verify.cpp` |
| `STRATA_QFUSE_HC_LOCAL` | unset |  | =0 is the atomic tail of the previous HC kernel; local block-owned kernel is new path (0.1.43). (from the settings register) | `src/kernels/cuda/fused_gr.cu`, `src/kernels/gr_parity.cpp` |
| `STRATA_QIL` | -1 |  | Neutral prefill, decode slower on R9700; experiment without shipped verdict. (from the settings register) | `src/core/verify.cpp` |
| `STRATA_QI_FAST_LN` | unset |  | measured: no gain on the 3060 (2.10 vs 2.10 s/step) (from the code comment) | `tools/image/sdcpp_native/qwen_image_2_1_native.hpp` |
| `STRATA_QI_QSPLIT` | unset |  | measured: no accuracy gain, +7% step time (from the code comment) | `tools/image/sdcpp_native/qwen_image_2_1_native.hpp` |
| `STRATA_READ_AHEAD` | unset |  | READ_AHEAD=0 turns read-ahead off; on by default. (from the settings register) | `src/platform/memory.cpp`, `tools/vision/strata_vision.cpp` |
| `STRATA_REFILL_BLOCKING` | off |  | (no description in the code) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_REFILL_PARTS` | unset |  | Refills the expert-cache slots a prompt borrowed straight from the GGUF mapping instead of through an assembled 4 MB copy per expert. Only gfx1151 has measured it. | `src/program/generate.cpp` |
| `STRATA_REFILL_SERIAL` | off |  | STRATA_REFILL_SERIAL=1: stage after stage, as 0.1.30/0.1.31. (from the code comment) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_REMOTE_ZEROCOPY` | on |  | STRATA_REMOTE_ZEROCOPY=0 copies. (from the code comment) | `src/core/remote_experts.cpp`, `sycl/src/core/remote_experts.cpp` |
| `STRATA_RING_BYTES` | on |  | (no description in the code) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_RING_TIMEOUT_S` | 20 |  | Ring wait timeout for hang diagnosis. (from the settings register) | `sycl/src/core/verify.cpp` |
| `STRATA_S2_SWIGLU_Q8` | on |  | (no description in the code) | `src/kernels/cuda/s2_expert_grouped.cu`, `sycl/src/kernels/cuda/s2_expert_grouped.dp.cpp` |
| `STRATA_SCORES_DX` | unset | `1` | from `STRATA_SELECT_LONG_AT` (24576) cells the decode windows score the indexer's blocks with the DX kernel, which reads each pooled key once for all of the window's queries (the long-context speedup) | `src/kernels/cuda/qsa_select.cu`, `src/kernels/qsa_select_long_parity.cpp` |
| `STRATA_SCORES_HALVE` | -1 |  | (no description in the code) | `src/kernels/cuda/qsa_select.cu` |
| `STRATA_SCORES_MULTI` | unset |  | (no description in the code) | `src/kernels/cuda/qsa_scores_gfx1100_test.cu`, `src/kernels/cuda/qsa_select.cu`, `src/kernels/qsa_slots_parity.cpp` +1 |
| `STRATA_SCORES_PAIRS` | unset |  | the indexer chain: block scores one (block, query) pair a warp, batched resolve lookups, the window append in parallel and on its own branch beside the query projection | `src/kernels/cuda/qsa_select.cu` |
| `STRATA_SCORES_THREAD` | -1 |  | On by default for gfx12; =0 turns it off. (from the settings register) | `src/kernels/cuda/qsa_select.cu`, `src/kernels/qsa_scores_gfx1100_parity.cpp` |
| `STRATA_SELECT_ALT_MIN_TOKENS` | on |  | "no gain below 32K. A gfx12 default therefore ..." threshold tuning. (from the settings register) | `src/core/gfx12_extra_test.cpp`, `src/kernels/cuda/qsa_select.cu` |
| `STRATA_SELECT_GEMM` | -1 |  | STRATA_SELECT_GEMM=0: the warp kernel always, =1: the GEMM always (on every card). (from the code comment) | `sycl/src/prefill/prefill.cpp` |
| `STRATA_SELECT_GEMM_MIN` | unset |  | blocks (4 cells each) (from the code comment) | `sycl/src/prefill/prefill.cpp` |
| `STRATA_SELECT_LONG` | unset | `1` | from `STRATA_SELECT_LONG_AT` (24576) cells the decode windows score the indexer's blocks with the DX kernel, which reads each pooled key once for all of the window's queries (the long-context speedup) | `src/core/verify.cpp` |
| `STRATA_SELECT_LONG_AT` | unset | `24576` | from 24,576 cells the decode windows score the indexer's blocks with the DX kernel, which reads each pooled key once for all of the window's queries; the same scores bit for bit, so the same blocks (RTX 50 table; the kernels now build for every CUDA card) | `src/core/verify.cpp` |
| `STRATA_SELECT_OLD` | off |  | Old warp kernel kept via =1 behind new tensor-core selection; used to perturb baselines. (from the settings register) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_SELECT_SIMT` | unset |  | QSA block scores on the FP32 tiled scorer instead of the warp kernel; helps long prompts (128K: +7.4%). Not bitwise the warp kernel. [Data](../../bench/results/2026-10-07-rdna2-qsa-select-simt/README.md). | `src/kernels/cuda/qsa_select.cu` |
| `STRATA_SELECT_TWO_PASS` | on |  | (no description in the code) | `src/kernels/cuda/qsa_select.cu`, `src/kernels/qsa_two_pass_parity.cpp`, `src/prefill/prefill.cpp` +1 |
| `STRATA_SELECT_TWO_PASS_MIN` | unset |  | (no description in the code) | `src/kernels/cuda/qsa_select.cu` |
| `STRATA_SELECT_WMMA` | on |  | The block scorer on the matrix cores | `src/core/arch_defaults.cpp`, `src/core/gfx12_extra_test.cpp`, `src/kernels/cuda/qsa_select.cu` +1 |
| `STRATA_SEL_OVERLAP` | off |  | STRATA_SEL_OVERLAP (debug, D-1's question): how much do neighbouring queries' selections share? Per tile of 16 queries: the union of their selected cells against the sum of their widths. (from the code comment) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_SESSION_BUFFERED` | off |  | (no description in the code) | `src/core/conversation_file.cpp`, `src/core/conversation_file_test.cpp` |
| `STRATA_SESSION_GROW` | on |  | under KV streaming the indexer's pooled keys and the RoPE table grow with the context (`=0`: sized for the whole context at the start, 320 fewer expert slots) | `src/program/generate.cpp` |
| `STRATA_SETTINGS_FILE` | unset |  | (no description in the code) | `setup.py`, `strata_control/discovery.py`, `tools/strata_mcp.py` |
| `STRATA_SHARD1` | unset |  | default: $STRATA_SHARD1, else the development layout (<Strata>/Q2_0 beside <Strata>/Public/Engine) (from the code comment) | `tools/canonical_xcheck.py` |
| `STRATA_SH_FORK_LATE` | on |  | STRATA_SH_FORK_LATE=0 forks at the top as before (from the code comment) | `src/core/verify.cpp`, `sycl/src/core/verify.cpp` |
| `STRATA_SH_STREAM` | on |  | Off on HIP, except the gfx1151 table | `src/core/arch_defaults_test.cpp`, `src/core/verify.cpp`, `sycl/src/core/verify.cpp` +1 |
| `STRATA_SIMT2_TILE` | 1 |  | Tile 2 measured slower (3.00 vs 2.82 ms); tile selector, no ship verdict on the switch. (from the settings register) | `src/kernels/cuda/qsa_select.cu` |
| `STRATA_SMALL_UNR` | unset |  | (no description in the code) | `src/core/arch_defaults_test.cpp`, `src/kernels/cuda/native_bf16.cu`, `src/kernels/cuda/verify_kernels.cu` |
| `STRATA_SNAPSHOT_FULL_CAPTURE` | on |  | (no description in the code) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_SOURCE` | unset |  | (no description in the code) | `setup.py`, `tools/mtp_fetch.py` |
| `STRATA_SPILL_OFF` | on |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_SPLIT_COMPUTE_POW` | 0.0 |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_SPLIT_COVER_B` | off |  | STRATA_SPLIT_COVER_B overrides b for a model whose routing is sharper or flatter, and STRATA_SPLIT_MISS_MS still scales the miss cost it feeds. (from the code comment) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_SPLIT_CPU_SHARE` | on |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_SPLIT_EARLY_OFF` | unset |  | STRATA_STAGE_TRIM=0, STRATA_SPLIT_EARLY_OFF=1 or STRATA_SPILL_OFF=1 keeps the old path. (from the code comment) | `src/program/generate.cpp` |
| `STRATA_SPLIT_LINK_ABS` | off |  | STRATA_SPLIT_LINK_ABS=1: 0.1.41's 20 GB/s rule. (from the code comment) | `src/program/generate.cpp` |
| `STRATA_SPLIT_MISS_MS` | unset |  | Tunes the split miss-cost model. (from the settings register) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_SPLIT_NEAR_TIE_PCT` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_SPLIT_OVERLAP` | unset |  | Overlapped hand-off between stages on different cards. | `src/program/generate.cpp` |
| `STRATA_SPLIT_OWN` | off |  | Gives every card its own prompt buffers. `auto` does it only where the buffers are at most 12% of each card's VRAM. | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_SPLIT_OWN_BUFFERS` | on |  | (no description in the code) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_SPLIT_OWN_PLACE` | unset |  | (no description in the code) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_SPLIT_RING` | 0 |  | Sets the prompt path's streamed ring to N slots (`0`: the pinned-share rule). A split uses 96 when at least 75% of the (layer, expert) pairs sit in some card's cache. | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_SPLIT_SMALL_MAX` | 0 |  | (no description in the code) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_SPLIT_SMALL_OWN` | 0 |  | Code: chunk geometry changes rounding so it is off by default; no verdict. (from the settings register) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_SPLIT_SPEED_SCALE` | 0.0 |  | Test seam for unequal card pairs; unset changes nothing. (from the settings register) | `src/program/generate.cpp` |
| `STRATA_SPLIT_UNIFORM` | off |  | "#2418 DONE (opt-in): STRATA_SPLIT_UNIFORM=1 cherry-picked ... documented". (from the settings register) | `src/program/generate.cpp` |
| `STRATA_SPLIT_WAIT_MS` | 30000 |  | The overlapped split's hand-off wait on the GPU is bounded: STRATA_SPLIT_WAIT_MS (default 30000, 0 = no bound). (from the code comment) | `src/core/verify.cpp` |
| `STRATA_SSD_GBPS` | unset |  | STRATA_SSD_GBPS overrides. (from the code comment) | `src/program/generate.cpp` |
| `STRATA_SSD_KEEPALIVE` | unset |  | STRATA_SSD_KEEPALIVE = ms without a read before one page is read anyway (default 100, 0 = off), STRATA_SSD_KEEPALIVE_WINDOW = seconds after the last row request that this goes on (default 60; then the SSD may sleep un... (from the code comment) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_SSD_KEEPALIVE_WINDOW` | 100.0 |  | STRATA_SSD_KEEPALIVE = ms without a read before one page is read anyway (default 100, 0 = off), STRATA_SSD_KEEPALIVE_WINDOW = seconds after the last row request that this goes on (default 60; then the SSD may sleep un... (from the code comment) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_STACK_ISOLATE` | on |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_STAGER_RING` | on |  | Stager ring size tuning; evidence is a deadlock investigation, no verdict. (from the settings register) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_STAGER_THREADS` | unset |  | 8/16 threads tried, -0.3%, nothing there; tuning knob, not a tested-and-dropped feature. (from the settings register) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_STAGE_TRIM` | unset |  | Every card of a layer split keeps the dense weights of all the layers, as before. By default each card, explicit split or `auto`, loads only its own layers' dense weights and the freed VRAM goes to its expert cache. | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_STALL_STACKS` | off |  | Stall stack diagnostic, off unless set. (from the settings register) | `src/program/generate.cpp` |
| `STRATA_STARTUP_UNBUFFERED` | unset |  | (no description in the code) | `src/core/expert_source.cpp` |
| `STRATA_STATE_HASH` | off |  | (no description in the code) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_STATE_HASH_FILE` | unset |  | Debug (Intel Arc verify path): before every window writes a hash of each GDN layer's recurrent state and of the int8 KV pages already filled. (from the settings register) | `sycl/src/core/verify.cpp` |
| `STRATA_STATE_HASH_GDN` | off |  | debug: the GDN states as the prompt path leaves them (from the code comment) | `src/prefill/prefill.cpp`, `src/program/generate.cpp`, `sycl/src/prefill/prefill.cpp` +1 |
| `STRATA_TAP_FILE` | unset |  | "the research dump"; off, nothing captured. (from the settings register) | `src/core/verify.cpp` |
| `STRATA_TAP_LAYERS` | unset |  | (no description in the code) | `src/core/verify.cpp` |
| `STRATA_TG_FLUSH_US` | 2000 |  | (no description in the code) | `src/core/session.cpp`, `sycl/src/core/session.cpp` |
| `STRATA_TQ_ISA` | unset |  | (no description in the code) | `src/kernels/cpu/tq_cpu.cpp` |
| `STRATA_TQ_ROWCH` | 4 |  | Kernel tuning experiment (row chunks) on Pascal TQ, 239 to 217; no ship verdict. (from the settings register) | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_TQ_STAGE` | on |  | Staged codebook on by default for cc 6.0; =0 turns off, =1 forces elsewhere. (from the settings register) | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_TQ_SUB8` | unset |  | Gated by tq_sub8_on(); no result verdict in evidence. (from the settings register) | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_TSUM` | off |  | grouped decode experts, butterfly warp sums, fewer launches, the GDN step over four blocks per head, producer-written activation images, batched PLE projections, the shared expert on its own stream | `src/core/arch_defaults_test.cpp`, `src/kernels/cuda/fused_gr.cu`, `src/kernels/cuda/iq_kernels.cu` +4 |
| `STRATA_UMA_REGISTER` | off |  | Strix UMA memory registration experiment; no verdict. (from the settings register) | `src/core/arch_defaults.cpp`, `src/program/generate.cpp` |
| `STRATA_UNBUFFERED_LOAD` | unset |  | force the expert reads unbuffered / through the file cache (default: unbuffered only when the cache cannot keep the files) | `src/core/expert_source.cpp`, `src/core/pinned.cu`, `src/program/generate.cpp` +3 |
| `STRATA_VARGS` | `os.environ.get("VARGS", ""` |  | (no description in the code) | `tools/qwen35/q36_venc.py` |
| `STRATA_VIT_BF16_AS_F16` | unset |  | (no description in the code) | `tools/vision/own/vit_gpu.cu` |
| `STRATA_VIT_CPU` | unset |  | STRATA_VIT_CPU=ref selects the float32 reference (VitRef: the oracle, for tests) (from the code comment) | `tools/vision/own/own_reader.cpp` |
| `STRATA_VIT_CPU_QB` | unset |  | STRATA_VIT_CPU_VPACK=0 / STRATA_VIT_CPU_QB=<multiple of 48>: diagnostics. (from the code comment) | `tools/vision/own/vit_cpu.cpp` |
| `STRATA_VIT_CPU_TAIL1` | unset |  | (no description in the code) | `tools/vision/own/cpu_ops.cpp` |
| `STRATA_VIT_CPU_VPACK` | unset |  | STRATA_VIT_CPU_VPACK=0 / STRATA_VIT_CPU_QB=<multiple of 48>: diagnostics. (from the code comment) | `tools/vision/own/vit_cpu.cpp` |
| `STRATA_VIT_FA` | unset |  | (no description in the code) | `tools/vision/own/vit_gpu.cu` |
| `STRATA_VIT_FA_UNITS` | unset |  | (no description in the code) | `tools/vision/own/vit_gpu.cu` |
| `STRATA_VIT_HPA16` | unset |  | (no description in the code) | `tools/vision/own/vit_gpu.cu` |
| `STRATA_VIT_HPA16_ATT` | unset |  | (no description in the code) | `tools/vision/own/vit_gpu.cu` |
| `STRATA_VIT_LT_SOL` | unset |  | (no description in the code) | `tools/vision/own/vit_gpu.cu` |
| `STRATA_VIT_NPAD` | unset |  | (no description in the code) | `tools/vision/own/vit_gpu.cu` |
| `STRATA_VIT_QALIGN` | unset |  | (no description in the code) | `tools/vision/own/vit_gpu.cu` |
| `STRATA_VIT_REF_ROUND` | unset |  | (no description in the code) | `tools/vision/own/vit_cpu.cpp`, `tools/vision/own/vit_gpu.cu`, `tools/vision/own/vit_ref.cpp` +1 |
| `STRATA_VIT_ROWS` | unset |  | (no description in the code) | `tools/vision/own/vit_gpu.cu`, `tools/vision/own/vit_sycl.cpp` |
| `STRATA_VIT_SCORES` | unset |  | (no description in the code) | `tools/vision/own/vit_gpu.cu`, `tools/vision/own/vit_sycl.cpp` |
| `STRATA_VIT_SCORES_LARGE` | unset |  | STRATA_VIT_SCORES_LARGE (M elements, a diagnostic): the chunk above 16,384 patches; a size the GPU cannot hold tests the fallback to the default chunk (from the code comment) | `tools/vision/own/vit_gpu.cu`, `tools/vision/own/vit_sycl.cpp` |
| `STRATA_WATCHDOG_IO_S` | unset |  | Tunes hung-engine watchdog (on by default). (from the settings register) | `src/program/generate.cpp` |
| `STRATA_WATCHDOG_S` | 60 |  | User-tunable stall watchdog seconds. (from the settings register) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_WDDM_BUDGET` | unset |  | (no description in the code) | `src/core/device.cu`, `sycl/src/core/device.dp.cpp` |
| `STRATA_WINDOW_SYNC` | 0 |  | "default is 1 on an i915 card, 0 elsewhere"; arch default, 0/2 tune. (from the settings register) | `sycl/src/core/verify.cpp` |
| `STRATA_WINDOW_WARM` | unset |  | Research switch: =0 turns off the window warm-up (one dummy verify window per allowed width, once, before the first request is read, so a width the policy picks inside a long request does not pay its first-run graph or p (from the settings register) | `src/program/generate.cpp` |
| `STRATA_WMMA_BF16` | off |  | STRATA_WMMA_BF16=0 excludes bf16): RDNA3 / RDNA3.5 WMMA dense GEMM, before hipBLASLt; it returns false for shapes it does not take (and off gfx11) and this falls through (from the code comment) | `src/prefill/gemm.cu`, `sycl/src/prefill/gemm.dp.cpp` |
| `STRATA_WMMA_GEMM` | off |  | 313 (opt-in STRATA_WMMA_GEMM=1; (from the code comment) | `src/prefill/gemm.cu`, `sycl/src/prefill/gemm.dp.cpp` |

### Decode (97)

| Variable | Default | sm_120 table | What it does | Read in |
| --- | --- | --- | --- | --- |
| `STRATA_ARGMAX_MULTI` | unset |  | STRATA_ARGMAX_MULTI=0: always the one-block kernel. (from the code comment) | `src/kernels/cuda/sampler.cu`, `sycl/src/kernels/cuda/sampler.dp.cpp` |
| `STRATA_ARGMAX_ROWS` | -1 |  | (no description in the code) | `src/kernels/cuda/verify_kernels.cu` |
| `STRATA_ATTN_BATCH` | 32 |  | SYCL port: how many queries the prompt path hands the FP32 attention kernel at once (STRATA_ATTN_BATCH, default 32; the tensor-core kernel is opt-in and slower here). (from the code comment) | `sycl/src/prefill/prefill.cpp` |
| `STRATA_ATTN_DAILY` | unset |  | Decode attention, two cells per step | `src/kernels/cuda/qsa_decode_attn.cu` |
| `STRATA_ATTN_LANECELL` | off (-1) |  | decode layouts and kernels (packed Q8_0 / Q6_K weights, 4-row BF16 GEMV, one-cell-per-thread attention scores) | `src/kernels/cuda/qsa_decode_attn.cu`, `src/kernels/qsa_slots_parity.cpp`, `sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp` |
| `STRATA_ATTN_MERGE_V2` | -1 |  | STRATA_ATTN_MERGE_V2: 1 merges with attn_merge_v2_kernel, 0 with attn_merge_kernel. (from the code comment) | `src/kernels/cuda/qsa_decode_attn.cu` |
| `STRATA_ATTN_PERCELL` | on |  | STRATA_ATTN_PERCELL=0: the warp kernel. (from the code comment) | `sycl/src/prefill/prefill.cpp` |
| `STRATA_ATTN_RS` | 2 |  | 1,254 to 1,265 (+0.9%) | `sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp` |
| `STRATA_ATTN_STAGED` | off | `1` | INT8 attention chunks staged by `cp.async`, two blocks per SM | `src/kernels/cuda/qsa_decode_attn.cu` |
| `STRATA_ATTN_V3` | unset |  | (no description in the code) | `src/kernels/cuda/qsa_decode_attn.cu` |
| `STRATA_BATCH_ATTN_ROWS` | unset |  | block scores, top-k and decode attention of all the window's rows (up to 8) in one launch each. Not with KV streaming (a slot with `kv_mode` 1 keeps the per-row launches). | `src/core/verify.cpp`, `sycl/src/core/verify.cpp` |
| `STRATA_BATCH_DECODE_SHARE` | 0.5 |  | --batch: a prompt read while slots are decoding goes one prompt chunk at a time (the same chunks as one run: each run starts where the last ended, at a multiple of the chunk), and between two chunks the slots decode f... (from the code comment) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_DENSE_TC` | off | `1` | an int8 tensor-core GEMV for every Q8_0 dense matrix of a 1-8 token window | `src/core/verify.cpp`, `src/kernels/cuda/native_mmvq.cu` |
| `STRATA_DF2_ATTN` | 0 |  | (no description in the code) | `sycl/src/kernels/cuda/dflash2.dp.cpp` |
| `STRATA_DF_BRANCH` | unset |  | (no description in the code) | `src/core/verify.cpp` |
| `STRATA_DF_PDL` | off (0) | `1` | programmatic dependent launch in the doorbell graphs too | `src/core/verify.cpp`, `src/kernels/cuda/verify_kernels.cu`, `src/kernels/pdl_parity.cpp` |
| `STRATA_DF_QB_Q4` | unset |  | the per-token GEMVs / norms / RoPEs / copies of this layer as one launch over the window's rows each - row-wise identical arithmetic (STRATA_DEC_BATCH=0: token by token) STRATA_DF_QB_Q4 (default on since 0.1.45; STRAT... (from the code comment) | `src/core/verify.cpp` |
| `STRATA_EARLY_READ` | unset |  | the next layer's first hyper-connection read starts before the combine and waits on its counter; skipped on a card where its waiting blocks cannot all fit | `src/core/verify.cpp`, `src/kernels/cuda/fused_gr.cu` |
| `STRATA_EARLY_READ_FIT` | unset |  | =0: skip this check (tests: the failure it prevents) (from the code comment) | `src/kernels/cuda/fused_gr.cu` |
| `STRATA_FUSE_DECODE` | off |  | The decode GEMV / MoE fusions of #1097. Bitwise the unfused kernels. | `include/strata/kernels/fuse_decode.hpp` |
| `STRATA_GDN_MID` | unset | `1` | a DeltaNet layer's middle (q/k/v and z projections, gates, recurrence, norm, output projection) in one persistent kernel | `src/core/verify.cpp` |
| `STRATA_GDN_MID_DEPTH` | 1 |  | the DeltaNet kernel keeps 10.6 KiB in flight a block instead of ~85 KiB (a bounded stream); `=0` the old ring | `src/kernels/cuda/gdn_mid.cu` |
| `STRATA_GDN_MID_EF` | 1 |  | (no description in the code) | `src/kernels/cuda/gdn_mid.cu` |
| `STRATA_GDN_MID_HEAD` | 7 | `2` | the DeltaNet kernel only prefetches into L2 before its wait, so its weight burst does not stall that read | `src/kernels/cuda/gdn_mid.cu` |
| `STRATA_GDN_MID_PIECES` | 4 |  | the DeltaNet kernel keeps 10.6 KiB in flight a block instead of ~85 KiB (a bounded stream); `=0` the old ring | `src/kernels/cuda/gdn_mid.cu` |
| `STRATA_GDN_MID_SPLIT` | 1 |  | (no description in the code) | `src/kernels/cuda/gdn_mid.cu` |
| `STRATA_GRAPH_PRIO` | unset | `1` | the attention layer's query rows first, the gate rows later on 48 blocks, stream priorities kept in the graphs | `src/core/verify.cpp` |
| `STRATA_GR_DOWN_DIRECT` | unset |  | (no description in the code) | `sycl/src/kernels/cuda/fused_gr.dp.cpp` |
| `STRATA_GR_DOWN_MAX4` | unset |  | Code says opt-in, bitwise; only a combined-flags slowdown, no explicit not-shipped verdict. (from the settings register) | `src/kernels/cuda/fused_gr.cu`, `sycl/src/kernels/cuda/fused_gr.dp.cpp` |
| `STRATA_GR_DOWN_SLICED` | unset |  | (no description in the code) | `sycl/src/kernels/cuda/fused_gr.dp.cpp` |
| `STRATA_GR_DOWN_TG` | -1 |  | STRATA_GR_DOWN_TG=2 is an existing measured +2.4% tuning option. (from the settings register) | `src/kernels/cuda/fused_gr.cu` |
| `STRATA_GR_FAST` | -1 |  | "slower on the P40 below 5 tokens", "Measured and not changed"; per-arch kernel toggle, no ship verdict. (from the settings register) | `src/kernels/cuda/fused_gr.cu` |
| `STRATA_GR_LARGE_GRF` | unset |  | "default on Alchemist"; variable only tunes/turns off. (from the settings register) | `sycl/src/kernels/cuda/fused_gr.dp.cpp` |
| `STRATA_GR_MORE_GROUPS` | unset |  | STRATA_GR_MORE_GROUPS=1 forces it on, =0 off. (from the code comment) | `sycl/src/kernels/cuda/fused_gr.dp.cpp` |
| `STRATA_GR_NORM_FAST` | -1 |  | The register-split hyper-connection norm (#2358). | `src/kernels/cuda/fused_gr.cu` |
| `STRATA_GR_NORM_SPLIT` | unset |  | (no description in the code) | `sycl/src/kernels/cuda/fused_gr.dp.cpp` |
| `STRATA_GR_Q32_G` | 1 |  | (no description in the code) | `src/core/arch_defaults_test.cpp`, `src/kernels/cuda/fused_gr.cu` |
| `STRATA_GR_Q32_PL` | unset |  | Retired in 0.2.13 (cleanup 4.5, batch C): code removed, the setting is no longer read. Evidence: [checked] Software-pipelined group loads for the HC up read: no gain over G=2, 'stays behind its env var, off ... can be dr (from the settings register) | `src/kernels/cuda/fused_gr.cu` |
| `STRATA_GR_Q8_PFS` | unset |  | (no description in the code) | `src/core/arch_defaults_test.cpp`, `src/kernels/cuda/fused_gr.cu` |
| `STRATA_GR_R2` | 2 |  | STRATA_GR_R2 (v3 read, 5..8-token launches; default 2): 0 = the generic kernels as before, 1 / 2 / 4 = the exact-T kernels with that many down rows per warp (the same dots in the same order, so the same bits; fewer bl... (from the code comment) | `src/kernels/cuda/fused_gr.cu` |
| `STRATA_GR_SPLIT` | unset |  | Code: "gfx906, opt-in STRATA_GR_SPLIT=1", ~3% faster per verify window. (from the settings register) | `src/kernels/cuda/fused_gr.cu` |
| `STRATA_GR_UNFUSED` | off |  | F-1: STRATA_GR_UNFUSED=1 keeps the FP32 copy of the normalized rows (gr_norm + gr_mix), the A/B arm (from the code comment) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_GR_UP75` | -1 |  | (no description in the code) | `src/kernels/cuda/fused_gr.cu` |
| `STRATA_GR_UP_COLS` | unset |  | STRATA_GR_UP_COLS=8 / 4 run 320 / 640 blocks with 4 / 2 rows per warp; every row's dot and butterfly and every column's stream sum are the same. (from the code comment) | `sycl/src/kernels/cuda/fused_gr.dp.cpp` |
| `STRATA_GR_UP_ILP` | off |  | (no description in the code) | `sycl/src/kernels/cuda/fused_gr.dp.cpp` |
| `STRATA_GR_UP_WIDE` | -1 |  | (no description in the code) | `src/kernels/cuda/fused_gr.cu` |
| `STRATA_GR_V3` | on (500) |  | upstream's default hyper-connection read instead of the split V3 kernels | `src/kernels/cuda/fused_gr.cu`, `src/kernels/fused_gr_bench.cpp`, `src/kernels/gr_parity.cpp` +2 |
| `STRATA_GR_V3_PF` | -1 |  | BW-DENSE STRATA_GR_V3_PF: the v3 read's load-ordering variants, bitwise the kernels without them (gr_v3_screen: 0 differing, T 1-8). STRATA_GR_V3_PF=0 restores the plain kernels, any value forces that mask. (from the code comment) | `src/kernels/cuda/fused_gr.cu` |
| `STRATA_GR_V4` | unset |  | the hyper-connection read: the split V3 kernels / V4 without PDL (default: V4 with programmatic dependent launch on sm_90+; the same bits as V3) | `src/kernels/cuda/fused_gr.cu` |
| `STRATA_GR_V4_R2` | 2 |  | (no description in the code) | `src/kernels/cuda/fused_gr.cu` |
| `STRATA_GR_V5` | 0 |  | (no description in the code) | `src/kernels/cuda/fused_gr.cu` |
| `STRATA_HC_PERSIST` | off |  | The verifier's hyper-connection read as one persistent kernel (HIP, wave32). Measured on 2x RX 6900 XT: -2.8% to -4.2% ms per verify window. Not offered on gfx11 and gfx12. Opt-in for measurement on gfx103x. [Data](../../bench/results/2026-10-12-hc-persist-gfx1030/README.md). | `src/core/verify.cpp`, `src/kernels/cuda/fused_gr.cu` |
| `STRATA_HC_Q8` | off | `1` | the 8-bit hyper-connection read (int8 rows, fp16 scales) | `metal/src/engine/window.cpp`, `src/core/mtp.cpp`, `src/core/native_dense.cpp` +1 |
| `STRATA_HC_Q8_ALL_STAGES` | off |  | Stage-local HC_Q8 retained default-on; =1 restores all-stage Q8 behaviour. (from the settings register) | `src/core/native_dense.cpp` |
| `STRATA_HC_Q8_INJECT` | off |  | STRATA_HC_Q8_INJECT=1: the F32 inject rows get a Q8_0 copy too (S25) (from the code comment) | `src/core/native_dense.cpp`, `sycl/src/core/native_dense.cpp` |
| `STRATA_HC_REQ8` | unset |  | Hyper-connection projections as int8 plus a scale per 32 (more expert-cache slots) | `src/program/generate.cpp` |
| `STRATA_HC_SPLIT` | unset |  | STRATA_HC_SPLIT: unset or 2 = the newest the check accepts (staged), 1 = at most split, 0 = the plain read (from the code comment) | `src/kernels/cuda/fused_gr.cu`, `sycl/src/kernels/cuda/fused_gr.dp.cpp` |
| `STRATA_HC_UPMIX` | off |  | the prompt path's hyper-connection up projection with the mix fused: default on compute capability 12.x (bitwise the cuBLAS + mix pair there at chunks of 33+ tokens) / opt-in elsewhere / off | `src/core/arch_defaults_test.cpp`, `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_LOGPOS_TOPK` | 0 |  | STRATA_LOGPOS_TOPK=K: the K most likely tokens and their log-probabilities too (`id:logprob`), for a top-k comparison with another engine on the same tokens (docs/UNSLOTH_Q4.md) (from the code comment) | `src/core/verify.cpp`, `sycl/src/core/verify.cpp` |
| `STRATA_MULTI_BLOCK_ARGMAX` | -1 |  | (no description in the code) | `src/kernels/cuda/verify_kernels.cu`, `sycl/src/kernels/cuda/verify_kernels.dp.cpp` |
| `STRATA_NORM_ROPE` | unset |  | AMD: opt in until rope_parity check 6 passes there (from the code comment) | `src/kernels/cuda/native_rope.cu`, `sycl/src/kernels/cuda/native_rope.dp.cpp` |
| `STRATA_NO_NORM_ROPE` | unset |  | (no description in the code) | `src/kernels/cuda/native_rope.cu`, `sycl/src/kernels/cuda/native_rope.dp.cpp` |
| `STRATA_PIPED_DECODE_MAX_MS` | 0.0 |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_PIPED_DECODE_MIN_MS` | 0.0 |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_PRIVATE_GR_2D_PAIR` | off |  | Private opt-in (=1), off by default: the 2-D pair layout of the fused GR kernel, main actor only (gr_2d_pair_enabled, sycl/src/kernels/cuda/fused_gr.dp.cpp:2764). Read from the code only; the integration lane measured no (from the settings register) | `sycl/src/kernels/cuda/fused_gr.dp.cpp` |
| `STRATA_PRIVATE_GR_F32_PHASE` | unset |  | Private selector for STRATA_PRIVATE_GR_F32_REG: which phase takes the FP32 register path; unset or 'both' = both phases, otherwise the named phase only (gf_phase, sycl/src/kernels/cuda/fused_gr.dp.cpp:2691). Read from th (from the settings register) | `sycl/src/kernels/cuda/fused_gr.dp.cpp` |
| `STRATA_PRIVATE_GR_F32_REG` | unset |  | Private switch, off by default (any non-zero number turns it on): the FP32 register path of the fused GR kernel for 1 to 4 tokens when the down projection is sliced (gf_enabled, sycl/src/kernels/cuda/fused_gr.dp.cpp:2686 (from the settings register) | `sycl/src/kernels/cuda/fused_gr.dp.cpp` |
| `STRATA_PRIVATE_GR_LO_VEC4` | unset |  | Private switch, off by default (non-zero turns it on): four-wide loads in the GR up-projection low-order part (gr_lo_vec4_enabled, sycl/src/kernels/cuda/fused_gr.dp.cpp:2367). Read from the code only; the integration lan (from the settings register) | `sycl/src/kernels/cuda/fused_gr.dp.cpp` |
| `STRATA_PRIVATE_GR_Q8_UP32` | off |  | Private opt-in (=1), off by default: the Q8 up-projection kernel with 32-row blocks (more blocks, fewer serial rows per warp, same bits), Battlemage layout only (b60_gr_q8_up32_enabled, sycl/src/kernels/cuda/fused_gr.dp. (from the settings register) | `sycl/src/kernels/cuda/fused_gr.dp.cpp` |
| `STRATA_PRIVATE_GR_SILU_FROM_LO` | unset |  | Private switch, off by default (non-zero turns it on): the combined kernel takes SiLU from the low-order part (gr_combo_silu_enabled, sycl/src/kernels/cuda/fused_gr.dp.cpp:2530). Read from the code only; the integration  (from the settings register) | `sycl/src/kernels/cuda/fused_gr.dp.cpp` |
| `STRATA_PRIVATE_GR_UP_ROWS16` | off |  | Private opt-in (=1), off by default: 16-row up-projection blocks of the fused GR kernel, Battlemage layout only (gr_up_rows16_enabled, sycl/src/kernels/cuda/fused_gr.dp.cpp:2517). Read from the code only; the integration (from the settings register) | `sycl/src/kernels/cuda/fused_gr.dp.cpp` |
| `STRATA_QI_EXACT_ATTN` | unset |  | (no description in the code) | `tools/image/sdcpp_native/qwen_image_2_1_native.hpp` |
| `STRATA_QI_GGML_ATTN` | unset |  | (no description in the code) | `tools/image/sdcpp_native/qwen_image_2_1_native.hpp` |
| `STRATA_QPROJ_GATE_AT` | 1 | `3` | the attention layer's query rows first, the gate rows later on 48 blocks, stream priorities kept in the graphs | `src/core/verify.cpp` |
| `STRATA_QPROJ_GATE_GRID` | 0 | `48` | the attention layer's query rows first, the gate rows later on 48 blocks, stream priorities kept in the graphs | `src/core/verify.cpp` |
| `STRATA_QPROJ_HALF` | off | `1` | the attention layer's query rows first, the gate rows later on 48 blocks, stream priorities kept in the graphs | `src/core/verify.cpp` |
| `STRATA_ROPE_DYNAMIC` | off |  | dynamic: yarn's knobs, applied by the dialog's length (STRATA_ROPE_DYNAMIC=1 turns yarn or the absent flag into it) (from the code comment) | `src/program/generate.cpp` |
| `STRATA_ROPE_DYNAMIC_MSCALE` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_ROPE_DYNAMIC_REREAD` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_ROPE_DYNAMIC_STEP` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_ROPE_TABLE` | on |  | upstream's fast-math RoPE angles instead of the float64 table | `src/kernels/cuda/native_rope.cu`, `src/program/generate.cpp`, `sycl/src/kernels/cuda/native_rope.dp.cpp` +1 |
| `STRATA_ROUTER_BLOCK` | unset |  | (no description in the code) | `src/kernels/cuda/router_top10.cu`, `sycl/src/kernels/cuda/router_top10.dp.cpp` |
| `STRATA_ROUTE_DB` | unset | `2` | the router and the doorbell in one kernel | `src/core/verify.cpp`, `src/kernels/cuda/native_router.cu`, `src/kernels/native_multi_parity.cpp` |
| `STRATA_ROUTE_PER_TOKEN` | unset |  | (no description in the code) | `src/core/layer.cpp`, `sycl/src/core/layer.cpp` |
| `STRATA_ROUTE_PRIOR` | unset |  | STRATA_ROUTE_PRIOR=lambda: residency-biased routing (the top-10 picked again with a logit bonus for the resident experts; route_prior.cu). (from the code comment) | `src/core/verify.cpp` |
| `STRATA_ROUTE_PROJ_PER_TOKEN` | unset |  | (no description in the code) | `src/core/layer.cpp`, `sycl/src/core/layer.cpp` |
| `STRATA_ROUTE_TAIL_SKIP` | on (0) |  | decode skips a missed expert every token of a window routes only at rank 7-9 / computes every one (default: off with NVFP4 experts, KL 0.035 with it on; upstream's 7 for the GGUF models) | `src/core/expert_source.cpp`, `src/program/generate.cpp`, `tools/force_check.py` |
| `STRATA_SIDE_MIXER` | -1 |  | The mixer's input-only work on graph branches | `src/core/verify.cpp` |
| `STRATA_TOPK_ACTIVE_ANY` | off |  | STRATA_TOPK_ACTIVE_ANY=1 (tests): the Turing dispatch on any CUDA card, so qsa_topk_active_parity checks it on whatever card runs the tests (the kernels are the same on every architecture) (from the code comment) | `src/kernels/cuda/qsa_select.cu`, `sycl/src/kernels/cuda/qsa_select.dp.cpp` |
| `STRATA_TOPK_CAPACITY_GUARD` | off |  | Turing stride guard; unclear if default fix or opt-out. (from the settings register) | `src/kernels/cuda/qsa_select.cu`, `sycl/src/kernels/cuda/qsa_select.dp.cpp` |
| `STRATA_TOPK_MULTI` | 0 |  | (no description in the code) | `src/kernels/cuda/qsa_select.cu`, `src/kernels/qsa_select_long_parity.cpp` |
| `STRATA_TOPK_MULTI_N` | 0 |  | (no description in the code) | `src/kernels/cuda/qsa_select.cu`, `src/kernels/qsa_select_long_parity.cpp` |
| `STRATA_TOPK_OLD` | off |  | STRATA_TOPK_OLD=1: the original kernel (from the code comment) | `src/kernels/cuda/qsa_select.cu`, `sycl/src/kernels/cuda/qsa_select.dp.cpp` |
| `STRATA_TOPK_REG_MIN_BLOCKS` | unset |  | The prompt's QSA top-k uses the register kernel from 16K cells of context instead of 28K. The selected cells are identical (`qsa_topk_parity`). | `src/kernels/cuda/qsa_select.cu` |
| `STRATA_TOPK_SPLIT` | -1 |  | (no description in the code) | `src/kernels/cuda/qsa_select.cu` |
| `STRATA_TOPK_STREAM` | on |  | STRATA_TOPK_STREAM=0 restores the register kernel there. campaign3: on HIP too (was CUDA-only; gfx11 has no sm_90 cluster path and fell to the ref radix past the register kernel's reach - the deep-decode top-k cost). (from the code comment) | `src/kernels/cuda/qsa_select.cu`, `sycl/src/kernels/cuda/qsa_select.dp.cpp` |
| `STRATA_WARM_GRAPHS` | on |  | SYCL port: every window graph and the drafter's graphs are captured here, once, instead of on first use in the loop below (STRATA_WARM_GRAPHS=0: as before). (from the code comment) | `sycl/src/program/generate.cpp` |

### Prompt (prefill) (223)

| Variable | Default | sm_120 table | What it does | Read in |
| --- | --- | --- | --- | --- |
| `STRATA_AUX_STAGER` | unset |  | STRATA_AUX_STAGER=0 leaves them where they were created (a small spare set would then hold all of them: it limits the copy rate of a long read) (from the code comment) | `src/prefill/prefill.cpp` |
| `STRATA_BATCH_YIELD_CHUNKS` | unset |  | (a layer split reads its stages as a pipeline over one run's chunks: beside slots a chunk at a time, with none decoding in pieces of STRATA_BATCH_YIELD_CHUNKS chunks (default 8; 0 = one run) so a BYIELD is taken) (from the code comment) | `src/program/generate.cpp` |
| `STRATA_COMPLEMENT_PIN_GIB` | unset |  | path 2's cap, STRATA_COMPLEMENT_PIN_GIB: unset or "auto" = below the GPU's shared-memory budget (Windows; see the slices below), "all" = slices until the driver refuses one, N = at most N GiB, 0 = no slices (path 3) (from the code comment) | `src/core/expert_source.cpp` |
| `STRATA_DENSE_F16C` | off | `1` | the prompt's dense FP16 GEMMs with f16 accumulation in chains of 64, then f32 | `src/prefill/gemm.cu` |
| `STRATA_DENSE_F16C_CHAIN` | 64 |  | STRATA_DENSE_F16C_CHAIN=16\|32\|64 (k per f16 chain, default 64) (from the code comment) | `src/prefill/gemm_f16c.cu` |
| `STRATA_DENSE_F16C_MIN_T` | unset |  | (no description in the code) | `src/prefill/gemm.cu` |
| `STRATA_DENSE_F16C_MIN_TILES` | unset |  | STRATA_DENSE_F16C_MIN_TILES (default 400): fewer 128 x 128 tiles than that leave the card half idle in the kernel's two-CTA waves, where cuBLAS picks a smaller tile (gemm_f16c_bench: 512 x 2560 at T 1024 0.55x, at T 1... (from the code comment) | `src/prefill/gemm.cu` |
| `STRATA_DENSE_F16C_SKIP` | unset |  | (no description in the code) | `src/prefill/gemm.cu` |
| `STRATA_DENSE_F32` | unset |  | 8.4-9.8 TFLOPS at every size from 512 tokens; qkv at 7,041 tokens 85 -> 45 ms | `src/prefill/gemm.cu` |
| `STRATA_DENSE_MMQ` | off |  | (no description in the code) | `src/prefill/gemm.cu`, `sycl/src/prefill/gemm.dp.cpp` |
| `STRATA_DEQUANT_SIDE` | off |  | The PLE rows go up raw and are dequantized on the card, on their own stream; the token-embedding table in VRAM; the dense dequant on a side stream (CUDA only). All bitwise. | `src/prefill/gemm.cu` |
| `STRATA_DF2_ATTN_CHUNK` | 0 |  | (no description in the code) | `sycl/src/kernels/cuda/dflash2.dp.cpp` |
| `STRATA_DF2_NO_PROMPT_CTX` | off |  | STRATA_DF2_NO_PROMPT_CTX=1 (measurement only): the prompt's rows are not given to the drafter (its first drafts are poor), which attributes the drafter's share of the prompt read time (from the code comment) | `sycl/src/program/generate.cpp` |
| `STRATA_DF_PLE` | unset |  | STRATA_DF_PLE (opt-in: STRATA_DF_PLE=1): the window's post-projection steps in one pass, bitwise the per-token loop below (the norms divide at run time as its norms do; the history advances token by token), which also... (from the code comment) | `src/core/verify.cpp` |
| `STRATA_EMB_REUSE_ACCOUNT` | off |  | VRAM-planner accounting for embedding buffer reuse; evidence does not say if on or retired. (from the settings register) | `src/prefill/prefill.cpp` |
| `STRATA_F16_RANGE` | off |  | what reaches FP16's range (from the code comment) | `src/prefill/prefill.cpp` |
| `STRATA_GDN_CHUNK` | -1 |  | "The opt-in chunked recurrence on gfx11 and gfx12"; user-visible opt-in. (from the settings register) | `src/prefill/gdn_chunk.cu`, `src/prefill/kernels.cu` |
| `STRATA_GDN_CHUNKED` | on |  | the prompt path's DeltaNet recurrence token by token (default: in chunks for prompt chunks of 128+ tokens) | `src/core/arch_defaults.cpp`, `src/core/arch_defaults_test.cpp`, `src/prefill/gdn_rec_parity.cu` +1 |
| `STRATA_GDN_CHUNKED_MIN` | unset |  | (no description in the code) | `src/prefill/kernels.cu` |
| `STRATA_GDN_CONVL2` | 0 | `2` | the prompt's DeltaNet scan on tensor cores | `src/core/arch_defaults_test.cpp`, `src/prefill/kernels.cu`, `sycl/src/prefill/kernels.dp.cpp` |
| `STRATA_GDN_CONV_SERIAL` | off |  | the old walk (A/B) (from the code comment) | `src/prefill/kernels.cu`, `sycl/src/prefill/kernels.dp.cpp` |
| `STRATA_GDN_FUSE` | unset |  | (no description in the code) | `src/prefill/kernels.cu` |
| `STRATA_GDN_KH3` | unset |  | (no description in the code) | `src/prefill/kernels.cu` |
| `STRATA_GDN_REC_WARP` | 0 |  | The chain-split DeltaNet recurrence of the prompt (#1098). Changes the FP32 summation order, so the default stays the pipelined kernel. `1` is the bit-exact sync-free variant; `STRATA_GDN_CHUNK=1` the chunked path. | `src/prefill/kernels.cu` |
| `STRATA_GDN_TC` | on | `1` | the prompt's DeltaNet scan on tensor cores | `src/core/arch_defaults_test.cpp`, `src/prefill/kernels.cu` |
| `STRATA_GDN_TC_FUSE` | unset |  | the split tensor-core scan can write y16 itself (STRATA_GDN_TC_FUSE=0 off): then the norm kernel below is skipped (from the code comment) | `src/prefill/kernels.cu` |
| `STRATA_GDN_TC_NP` | 3 | `1` | the prompt's DeltaNet scan on tensor cores | `src/prefill/kernels.cu` |
| `STRATA_GDN_TC_TILES` | 0 |  | (no description in the code) | `src/prefill/kernels.cu` |
| `STRATA_GDN_W12_LB` | unset |  | (no description in the code) | `src/prefill/kernels.cu` |
| `STRATA_GEMM_WARM` | on |  | cuBLASLt's GEMM kernels load on the first prompt again (default: on the prewarm thread at start) | `src/prefill/gemm.cu` |
| `STRATA_GEMM_WARM_FILE` | unset (the default path) |  | the GEMM warm file: a path, or `0` for none (unset: `%LOCALAPPDATA%\strata\gemm-warm-<GPU>.txt`); the prompt's GEMM shapes, replayed at the next start | `src/prefill/gemm.cu` |
| `STRATA_GEMM_WARM_STEP` | 512 |  | cuBLAS picks the kernel by the exact T (a 66617-token prompt's last chunk of 22072 rows runs another kernel than its 22272-row chunks): each recorded shape also runs at every STRATA_GEMM_WARM_STEP-th T (default 512) o... (from the code comment) | `src/prefill/gemm.cu` |
| `STRATA_HANDOFF_PIPE` | 0 |  | a layer split hands a prompt chunk to the next card in P pieces (2..16), gated by events, on a persistent thread | `src/prefill/prefill.cpp` |
| `STRATA_HANDOFF_PIPE_WORKER` | unset |  | (STRATA_HANDOFF_PIPE_WORKER=0: a thread per chunk again, for an A/B of the two halves.) (from the code comment) | `src/prefill/prefill.cpp` |
| `STRATA_I8_COUNT` | unset |  | Debug only: prints one line per prompt chunk with the routed entries that ran on native_expert_i8 and the ones that stayed on the group walk's own kernels. 0: off. (from the settings register) | `src/prefill/prefill.cpp` |
| `STRATA_IGPU_PROMPT_SET` | unset |  | gfx1151+Windows default prompt set; "STRATA_IGPU_PROMPT_SET=0 turns it off"; flipped 0.2.1. (from the settings register) | `src/core/arch_defaults.cpp` |
| `STRATA_IO_PF_AHEAD` | unset |  | (no description in the code) | `src/core/expert_source.cpp` |
| `STRATA_IO_PF_BUDGET_FRAC` | unset |  | (no description in the code) | `src/core/expert_source.cpp` |
| `STRATA_IO_PF_BUDGET_MB` | unset |  | (no description in the code) | `src/core/expert_source.cpp` |
| `STRATA_IO_PF_GATE` | unset |  | (no description in the code) | `src/core/expert_source.cpp` |
| `STRATA_IO_PF_GATE_HI` | unset |  | (no description in the code) | `src/core/expert_source.cpp` |
| `STRATA_IO_PF_GATE_LO` | 0 |  | (no description in the code) | `src/core/expert_source.cpp` |
| `STRATA_IO_PF_GATE_USEFUL` | unset |  | (no description in the code) | `src/core/expert_source.cpp` |
| `STRATA_IO_PF_MINCONF` | 0 |  | (no description in the code) | `src/core/expert_source.cpp` |
| `STRATA_IO_PF_SCHED` | unset |  | (no description in the code) | `src/core/expert_source.cpp` |
| `STRATA_IO_PF_STAGE` | unset |  | "measured +25% on an RTX 3060 under a 32 GB limit" but slower elsewhere; documented opt-in. (from the settings register) | `src/core/expert_source.cpp`, `src/program/generate.cpp` |
| `STRATA_IO_PF_TARGET` | unset |  | (no description in the code) | `src/core/expert_source.cpp` |
| `STRATA_IO_PF_THREADS` | 0 |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_IQ4_W4A8` | 0 |  | CUDA sm_80..sm_88 (Ampere) and sm_120 (Blackwell, only where the prompt experts run on MMQ: the fused int8 kernels are its default and are faster), IQ4_XS / IQ4_NL experts (`=2`: the IQ4_NL down products only) | `src/prefill/moe_iq4_w4a8.cu` |
| `STRATA_KVA_GDN_NORM` | unset |  | KVA: the gate projection feeds only the output normalization, which a state-only layer skips (STRATA_KVA_GDN_NORM=1 keeps the full recurrence as the A/B control for the state-only one) (from the code comment) | `src/prefill/prefill.cpp` |
| `STRATA_KVA_MIN_PROMPT` | 4096 |  | (no description in the code) | `include/strata/prefill/kva.hpp` |
| `STRATA_LM_PREFETCH` | on |  | (no description in the code) | `src/prefill/prefill.cpp` |
| `STRATA_MMQ_DIRECT` | 0 |  | CUDA, native packs; needs a build with `-DSTRATA_MMQ_TAB=ON` (off by default because it costs 36-38 MiB of VRAM per GPU) | `src/prefill/moe_mmq.cu` |
| `STRATA_MMQ_GROUP_SORT` | unset |  | CUDA, needs the build option `-DSTRATA_MMQ_TAB=ON`: the prompt MMQ reads resident experts in their cache slots; full MMQ groups sorted by row count (docs/DETAILS.md) | `src/prefill/mmq_resident_sort.hpp` |
| `STRATA_MMQ_KQ` | on |  | Multiplies the Q4_K / Q5_K / Q5_1 experts of a prompt with llama.cpp's MMQ kernels instead of dequantizing them to FP16. It rounds differently from the FP16 path, so it is opt-in for this file. On CUDA the kernels are a library next to the engine (`libstrata_kq.so` / `strata_kq.dll`), opened only then, so they take no VRAM from other models ([QUANTS.md](../QUANTS.md)). | `src/core/gfx12_extra_test.cpp`, `src/prefill/mmq_kq_loader.cpp`, `src/prefill/moe_mmq.cu` +1 |
| `STRATA_MMQ_RESIDENT_SORT_NE` | unset |  | (no description in the code) | `src/prefill/mmq_resident_sort.hpp` |
| `STRATA_MMQ_WS` | unset |  | opt-in (from the code comment) | `src/prefill/moe_ws.cu` |
| `STRATA_NV_CHUNK` | unset |  | Auto chunk 32768 on cc>=8.0 became default; STRATA_NV_CHUNK=0/16/1 are escapes (v0.2.11). (from the settings register) | `src/program/generate.cpp`, `setup.py` |
| `STRATA_OLD_SAMPLER` | off |  | (no description in the code) | `src/kernels/cuda/sampler.cu`, `sycl/src/kernels/cuda/sampler.dp.cpp` |
| `STRATA_PENALTY_PROMPT` | unset |  | STRATA_PENALTY_PROMPT=1 brings the old window back (the last penalty_last_n tokens of prompt + reply). (from the code comment) | `src/program/generate.cpp`, `src/qwen35/q36_engine.cu`, `sycl/src/program/generate.cpp` |
| `STRATA_PF_BATCH` | 0 |  | Up to N experts per launch (above 128 only with the 256-thread kernels; bitwise). | `src/prefill/prefill.cpp` |
| `STRATA_PF_BK64` | on |  | STRATA_PF_BK64=0: the BK 32 kernel for the wide shapes too (the A/B; the same bits) (from the code comment) | `src/prefill/wmma_gemm.cu` |
| `STRATA_PF_CHAIN_TIME` | off |  | (no description in the code) | `src/prefill/prefill.cpp` |
| `STRATA_PF_DM16` | off |  | STRATA_PF_DM16 (gfx11, the F16 fused expert kernels): the down rows go to the combine as FP16 instead of FP32 (half the bytes written and read) (from the code comment) | `src/prefill/prefill.cpp` |
| `STRATA_PF_DM16H` | off | `1` | the HC `Dm` rows in FP16 (the largest value seen is 23) | `src/prefill/prefill.cpp` |
| `STRATA_PF_DMA_PAGEABLE` | off |  | STRATA_PF_DMA_PAGEABLE=1 restores the direct copy from pageable memory. (from the code comment) | `sycl/src/prefill/prefill.cpp` |
| `STRATA_PF_DQ_FUSED` | on |  | STRATA_PF_DQ_FUSED=0: two launches) (from the code comment) | `sycl/src/prefill/prefill.cpp` |
| `STRATA_PF_F16` | off |  | Runs the covered prompt experts on the F16 matrix cores instead of the int8 ones. It rounds differently from the int8 kernels. | `src/prefill/moe_f16_common.cuh` |
| `STRATA_PF_F16_BN` | 0 |  | STRATA_PF_F16_BN=64\|128 forces one. (from the code comment) | `src/prefill/moe_f16_common.cuh` |
| `STRATA_PF_FEWROWS` | unset |  | Research switch, off unless set ("0" off, anything else on; src/prefill/fewrows.hpp), except that the Tesla P100 prompt default turns it on when gp100_prompt_default.hpp's kGp100Default is true. On the FP16 prompt path a (from the settings register) | `src/prefill/prefill.cpp` |
| `STRATA_PF_FP4G` | 2 |  | STRATA_PF_FP4G=0 is the way back to MMQ (from the code comment) | `src/prefill/moe_mmq.cu` |
| `STRATA_PF_FP4G_BULK` | off |  | (no description in the code) | `src/prefill/mmq_nvfp4_fp4g.cu` |
| `STRATA_PF_FP4G_FLAT` | on |  | STRATA_PF_FP4G_FLAT=0: grid y = the largest expert's tiles for every expert (round 2); else one list of tiles (from the code comment) | `src/prefill/mmq_nvfp4_fp4g.cu` |
| `STRATA_PF_FP4G_MIN_ROWS` | unset |  | strata-nvfp4 (pfx): STRATA_PF_FP4G takes a group (or a GROUP_SCRATCH range) only from STRATA_PF_FP4G_MIN_ROWS rows an expert on average, default 96 (fp4g_parity, 16 experts, MMQ / FP4G ms 12 rows 0.041 / 0.075, 64 0.0... (from the code comment) | `src/prefill/prefill.cpp` |
| `STRATA_PF_FP4G_PERSIST` | 1 |  | STRATA_PF_FP4G_PERSIST (with the flat list): bit 1 the down product, bit 2 mode 2's gate/up on the persistent kernel; default 1 (64K nsys: down 364 -> 326 ms; gate/up 519 -> 542, so it stays one block per tile) (from the code comment) | `src/prefill/mmq_nvfp4_fp4g.cu` |
| `STRATA_PF_FP4G_QOLD` | on (0) |  | (no description in the code) | `src/prefill/fp4g_parity.cpp`, `src/prefill/mmq_nvfp4_a44.cu`, `src/prefill/mmq_nvfp4_fp4g.cu` |
| `STRATA_PF_FP4G_R144` | off |  | mode 2 writes H in MMQ's layout, or with STRATA_PF_FP4G_R144=1 in the 144-byte K-step layout: each sub-block's 8 code bytes and its scale byte of each term, wherever the layout puts them (from the code comment) | `src/prefill/fp4g_parity.cpp`, `src/prefill/mmq_nvfp4_fp4g.cu` |
| `STRATA_PF_FUSED` | off |  | gfx12 native IQ packs; integrated gfx11 | `src/core/arch_defaults.cpp`, `src/core/arch_defaults_test.cpp`, `src/core/nv_defaults.cpp` +7 |
| `STRATA_PF_FUSED_F16` | off |  | f16 mma instead of the int8 products in the 256-thread kernels (rounding level). | `src/prefill/moe_fused_iq.cu` |
| `STRATA_PF_FUSED_IQ2S` | unset |  | "OPT-IN (STRATA_PF_FUSED_IQ2S=1)" fused IQ2_S/IQ4_XS gate/up on gfx12. (from the settings register) | `src/prefill/moe_fused_iq.cu` |
| `STRATA_PF_FUSED_IQ2X` | unset |  | [OWN-4A 11-10] IQ2_XXS / IQ2_XS gate/up (the IQ3_XXS pack, the IQ2_XS sets) on the fused prompt kernels: default ON on gfx12 under Linux (0.2.15, owner 11-10); =0 keeps those layers on MMQ (two releases); Windows gfx12:  (from the settings register) | `src/prefill/moe_fused_iq.cu` |
| `STRATA_PF_FUSED_KQ` | off |  | the same fused prompt kernels for the K-quant experts of Unsloth's UD-Q4_K_XL files (Q4_K or Q5_K gate/up, Q5_1 or Q8_0 down), instead of llama.cpp's MMQ; `STRATA_PF_FUSED_KQ=0` gives them back | `src/core/nv_defaults.cpp`, `src/prefill/moe_fused_iq.cu`, `sycl/src/prefill/moe_fused_iq.dp.cpp` |
| `STRATA_PF_FUSED_NATIVE` | unset |  | (no description in the code) | `src/prefill/moe_fused_iq.cu`, `sycl/src/prefill/moe_fused_iq.dp.cpp` |
| `STRATA_PF_FUSED_NT` | unset |  | The fused prompt-expert kernels with 256 threads and two blocks a multiprocessor (bitwise). Adds device code to every CUDA build. | `src/prefill/moe_fused_iq.cu` |
| `STRATA_PF_FUSED_Q2N` | off |  | STRATA_PF_FUSED_Q2N=1: the native Q2_0 pack's Q2_0 gate/up on these kernels here too (what #2145 took with STRATA_PF_FUSED=1 on every CUDA card until b6443f3a kept it to Turing). (from the code comment) | `src/prefill/moe_fused_iq.cu` |
| `STRATA_PF_FUSED_Q50` | off |  | [OWN-4A 11-10] Q5_0 down layers (UD-IQ4_XS-v31) on the fused prompt kernels: default ON on gfx12 under Linux (0.2.15, owner 11-10) and on gfx1151 under Windows (round 0217); =0 keeps the dequantize + FP16 GEMM route (two (from the settings register) | `src/prefill/moe_fused_iq.cu` |
| `STRATA_PF_FUSED_Q6K` | off |  | [DRAFT] CUDA and HIP gfx12 (phase 4, opt-in, default off): a Q6_K gate/up with a Q8_0 down (UD-Q6_K_XL's 47 layers, UD-Q5_K_XL's one Q6_K layer) on the fused native kernels instead of MMQ, on any sm_80+ card and on gfx12 (from the settings register) | `src/prefill/moe_fused_iq.cu` |
| `STRATA_PF_FUSED_Q8D` | off |  | [DRAFT] CUDA and HIP gfx12 (phase 4, opt-in, default off): a Q8_0 down behind an i-quant gate/up (the pinned UD-IQ4_XS's five Q8_0-down layers) on the fused native kernels instead of MMQ, on any sm_80+ card and on gfx120 (from the settings register) | `src/prefill/moe_fused_iq.cu` |
| `STRATA_PF_FUSED_Q8GU` | off |  | [DRAFT] CUDA and HIP gfx12 (phase 4, opt-in, default off): a Q8_0 gate/up with a Q8_0 down (the Q8_0 set, UD-Q6_K_XL's one Q8_0 layer) on the fused native kernels instead of MMQ, on any sm_80+ card and on gfx1200 / gfx12 (from the settings register) | `src/prefill/moe_fused_iq.cu` |
| `STRATA_PF_FUSED_TILE` | 0 |  | Tile tuning: nothing on sm_86, auto stays. (from the settings register) | `src/prefill/moe_fused_iq.cu`, `sycl/src/prefill/moe_fused_iq.dp.cpp` |
| `STRATA_PF_FUSED_TQ` | off |  | [DRAFT] CUDA and HIP gfx12 (phase 4): the Gyro trellis experts (TQK6 / TQK7, TQ2_T / TQ2_T), Hadamard-folded layers included, on the fused native kernels instead of MMQ. CUDA, unset: Gyro-S (TQK6 / TQK7) on compute capab (from the settings register) | `include/strata/prefill/fused_tq_default.hpp`, `src/prefill/moe_fused_iq.cu` |
| `STRATA_PF_FUSED_WS` | off |  | Warp-specialized kernel with the same F16 products; needs about 101 KB of shared memory, so sm_80 and sm_90 only (it says "not available" elsewhere). | `src/prefill/moe_fused_iq.cu` |
| `STRATA_PF_GEMM` | off |  | The prompt's FP16 projections on the 128 x 256 WMMA GEMM | `src/prefill/gemm.cu`, `src/prefill/prefill.cpp`, `sycl/src/prefill/gemm.dp.cpp` +1 |
| `STRATA_PF_GRAPH` | 0 | `7` | the QSA indexer loop of a layer and each expert group's kernels as CUDA graphs (updated in place, not rebuilt) | `src/prefill/prefill.cpp` |
| `STRATA_PF_GROUP_CAP` | 0 |  | STRATA_PF_GROUP_CAP=<rows> (a test knob, at least T): a smaller cap, so that most groups run as sub-ranges (from the code comment) | `src/prefill/prefill.cpp` |
| `STRATA_PF_GROUP_SCRATCH` | unset | `1` | smaller prompt buffers, fewer expert slots borrowed | `src/prefill/prefill.cpp` |
| `STRATA_PF_HASH_FILE` | unset |  | Debug (Intel Arc prompt path): for the first GDN layer of every prompt chunk, one line per row with the hash of the layer input, projections, gates, conv output and recurrence output, plus the recurrent state. (from the settings register) | `sycl/src/prefill/prefill.cpp` |
| `STRATA_PF_HCDOWN` | off |  | S23 (opt-in STRATA_CVEC_FUSE=1): a steered layer's FFN write + control vector + the next half's norm in one pass over R (gr_write_cvec_norm_rs; bitwise the gr_write + cvec_apply + gr_norm_rs it replaces) S23 (opt-in S... (from the code comment) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_PF_HC_INJ` | off | `1` | the hyper-connection inject folded into the HC write kernels | `src/prefill/prefill.cpp` |
| `STRATA_PF_HC_REGION` | unset | `2` | smaller prompt buffers, fewer expert slots borrowed | `src/prefill/prefill.cpp` |
| `STRATA_PF_I8` | unset |  | Int8 prompt path on B70, +20% at 1K prompt, identical output. (from the settings register) | `sycl/src/prefill/prefill.cpp` |
| `STRATA_PF_KQ_OCC` | off |  | trial switch (OWN-4A, 11-10): the K-quant fused prompt kernels on the occupancy launch, 3 blocks per WGP; HIP only, off by default (from the settings register) | `src/prefill/moe_fused_iq.cu` |
| `STRATA_PF_KV_UNSTAGE_SIDE` | on |  | KV streaming's copies on their own stream, ordered by events | `src/prefill/prefill.cpp` |
| `STRATA_PF_OCC` | 0 |  | STRATA_PF_OCC=N sets the blocks per WGP (an experiment knob). (from the code comment) | `src/prefill/moe_fused.cu`, `src/prefill/moe_fused_iq.cu`, `sycl/src/prefill/moe_fused.dp.cpp` +1 |
| `STRATA_PF_OCC_W` | 0 |  | STRATA_PF_OCC_W=N sets it for every occupancy variant (an experiment knob). (from the code comment) | `src/prefill/moe_fused_iq.cu` |
| `STRATA_PF_P1A` | off |  | on by default for some GPU architectures only (gfx1151 (Windows) src/core/arch_defaults.cpp:212=1); opt-in elsewhere; the variable overrides it (from the settings register) | `src/prefill/moe_fused_iq.cu` |
| `STRATA_PF_P1A_L2` | off |  | with STRATA_PF_P1A on gfx1151: also the Q5_K gate/up and Q8_0 down layers on the P1A kernels; not measured as a default (from the settings register) | `src/prefill/moe_fused_iq.cu` |
| `STRATA_PF_PAD` | off |  | 4K prompt 1,247 vs 1,243 tok/s, 8K 1,330 vs 1,321 (noise); an earlier crash with padded GEMMs on RDNA4 was fixed in 0.1.46 | `src/prefill/gemm.cu`, `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_PF_PEER_COMPACT` | unset |  | (no description in the code) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_PF_PEER_CPU` | unset |  | Release note: stays off; owner approved as opt-in. (from the settings register) | `src/prefill/prefill.cpp` |
| `STRATA_PF_PEER_F16` | unset |  | (no description in the code) | `src/prefill/prefill.cpp` |
| `STRATA_PF_PEER_GATHER` | unset |  | (no description in the code) | `src/prefill/prefill.cpp` |
| `STRATA_PF_PEER_OUT_PIPE` | unset |  | =0: one copy after the last group (the A/B) (from the code comment) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_PF_PEER_RING` | unset |  | (no description in the code) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_PF_PEER_STREAM` | 0.35 |  | (no description in the code) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_PF_PEER_SUMS` | unset |  | (no description in the code) | `src/prefill/prefill.cpp` |
| `STRATA_PF_PEER_XSTREAM` | unset |  | (no description in the code) | `src/prefill/prefill.cpp` |
| `STRATA_PF_PLAIN_FLOOR` | unset |  | In a build without llama.cpp's MMQ that loads the plain Q2_0 pack (not a native pack) on a discrete CUDA card, prompt chunks of 1,536 tokens or more are read by the streamed walk on the fused kernels, with or without the (from the settings register) | `src/prefill/prefill.cpp` |
| `STRATA_PF_Q8D_F16` | unset |  | (no description in the code) | `src/prefill/moe_fused_iq.cu` |
| `STRATA_PF_REFILL_BG` | 1 |  | serve refills the borrowed expert slots beside decode, hottest first, in batched copies | `src/program/generate.cpp` |
| `STRATA_PF_REFILL_BG_API` | on |  | STRATA_PF_REFILL_BG_API=1 (default): one cudaMemcpyBatchAsync a batch (one call: a single cudaMemcpyAsync of a 2.7 MB blob costs ~40 us of host time under WDDM); 0: a cudaMemcpyAsync each (from the code comment) | `src/program/generate.cpp` |
| `STRATA_PF_REFILL_BG_BATCH` | 32 |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_PF_SHARED_LATE` | on |  | a streamed chunk's shared expert runs on the GPU while the host groups the routed ones | `src/prefill/prefill.cpp` |
| `STRATA_PF_SHARE_OWN` | unset |  | A prompt layer on which the CPU prefill share took experts keeps the staged fused entry and the int8 kernels for the GPU's own experts, in a build without llama.cpp's MMQ (before, such a layer ran the GPU's experts on th (from the settings register) | `src/prefill/prefill.cpp` |
| `STRATA_PF_SMALL` | unset |  | STRATA_PF_SMALL=0 turns the rule off. (from the code comment) | `src/program/generate.cpp` |
| `STRATA_PF_SMALL_FUSED` | unset |  | Way back for the small-card prompt path (CUDA, one card, native pack, default chunk under 8,192 tokens): unset, the native fused expert kernels run inside the path's MMQ buffer layout (the layer-major walk's groups and t (from the settings register) | `src/program/generate.cpp` |
| `STRATA_PF_STAGED_FO` | unset |  | The staged walk's fused group entry for a layer that has no MMQ tile but that the fused kernels cover (UD-Q4_K_XL on gfx1151 without STRATA_MMQ_KQ; every layer of a build without llama.cpp's MMQ): a prompt chunk below th (from the settings register) | `src/prefill/prefill.cpp` |
| `STRATA_PF_STEP_SYNC` | off |  | Opt-in diagnosis switch, slower by design. (from the settings register) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_PF_SWITCH_MIN_T` | unset |  | STRIX_HALO.md section 5 sets it for users to keep default numerics on short prompts. (from the settings register) | `src/kernels/cuda/qsa_prompt_attn.cu`, `src/prefill/gemm.cu`, `src/prefill/prefill.cpp` +3 |
| `STRATA_PF_TQ_G1151` | unset |  | on by default on Windows gfx1151 in a build without llama.cpp's MMQ (the fused prompt kernels take Gyro-S's trellis experts; include/strata/prefill/fused_tq_hip.hpp kG1151DefaultOn); =0 (or STRATA_GFX1151_DEFAULTS=0) tur (from the settings register) | `src/prefill/moe_fused_iq.cu` |
| `STRATA_PF_UPMIX_SKIP32` | on |  | the attention half skips an FP32 write nobody reads | `src/prefill/prefill.cpp` |
| `STRATA_PF_W16_CACHE_MIB` | unset |  | Default on for batch slots on Linux CUDA (62853e78d, 0.2.9); =0 turns it off. (from the settings register) | `src/prefill/gemm.cu` |
| `STRATA_PF_WMMA` | unset |  | The FP16 tensor-core prompt experts; also takes IQ4_NL down projections. | `src/prefill/prefill.cpp` |
| `STRATA_PF_WMMA2` | off |  | A second prompt-expert kernel for IQ2_S gate/up and IQ4_NL down (#2356). | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_PLE_AUTO_MMAP` | unset |  | =1 keeps the old auto-mmap switch; 0.1.43.1 keeps direct by default. (from the settings register) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_PLE_AUTO_RAM` | unset |  | Automatic conversation-cache RAM sizing; variable tunes/turns off. (from the settings register) | `src/program/generate.cpp` |
| `STRATA_PLE_BATCH` | on |  | grouped decode experts, butterfly warp sums, fewer launches, the GDN step over four blocks per head, producer-written activation images, batched PLE projections, the shared expert on its own stream | `src/core/verify.cpp`, `src/kernels/verify_batch_parity.cpp`, `src/prefill/prefill.cpp` +2 |
| `STRATA_PLE_COMPACT` | off |  | The PLE rows go up raw and are dequantized on the card, on their own stream; the token-embedding table in VRAM; the dense dequant on a side stream (CUDA only). All bitwise. | `src/prefill/prefill.cpp` |
| `STRATA_PLE_COPY_STREAM` | off |  | The PLE rows go up raw and are dequantized on the card, on their own stream; the token-embedding table in VRAM; the dense dequant on a side stream (CUDA only). All bitwise. | `src/prefill/prefill.cpp` |
| `STRATA_PLE_GGUF` | unset |  | Build/test cmake variable for PLE GGUF; test setup. (from the settings register) | `src/kernels/ple_parity.cpp`, `sycl/src/kernels/ple_parity.cpp` |
| `STRATA_PLE_READERS` | 8 |  | the prompt's PLE rows through N reader threads (default 8 on Windows; 0 = one thread reaps every read) | `src/kernels/ngram.cpp` |
| `STRATA_PREFILL_AUTO_MAX` | 32768 |  | Documented chunk cap; nothing sets it by default on Strix. (from the settings register) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_PREFILL_BF16X2` | 2 |  | exact inputs to the prompt path's BF16 projections: all but the hyper-connection (default), all (~9% slower prompt reading), off | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_PREFILL_BORROW` | off |  | STRATA_PREFILL_BORROW=1 keeps 0.2.7's lending, STRATA_GFX1151_DEFAULTS=0 turns the whole gfx1151 block off. (from the code comment) | `src/program/generate.cpp` |
| `STRATA_PREFILL_COMBINE_WRITE` | unset |  | the prompt path's earlier kernels: activations quantized per expert row / swiglu and quantize apart / combine then the hyper-connection write (the same bits) | `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_COPY_EVENT_EVERY` | 1 |  | Tuning of copy event frequency; >1 keeps a switch off. (from the settings register) | `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_COPY_WAIT_DEDUPE` | unset |  | Experimental copy-wait variant that keeps a switch off; no verdict. (from the settings register) | `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_CPU_SHARE` | unset |  | a fixed CPU share of a small prompt chunk's streamed experts / none (default `auto`: measured, and only while sharing is faster; `STRATA_DBG_CPU_GATE=1` prints its readings) | `src/prefill/prefill.cpp`, `src/program/generate.cpp` |
| `STRATA_PREFILL_CPU_SHARE_BATCH` | off |  | "inconclusive / no gain on GB10; keep as an opt-in". (from the settings register) | `src/program/generate.cpp` |
| `STRATA_PREFILL_CPU_SHARE_MAX` | off |  | CUDA, one GPU, no `--batch`, prompt chunks under 1,024 tokens | `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_EQUAL` | off | `2` | A layer split keeps today's cut, or with STRATA_PREFILL_EQUAL=1 the older opt-in rule below (#693); STRATA_PREFILL_EQUAL=0 keeps today's cut everywhere. (from the code comment) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_PREFILL_FIRST` | 0 |  | 256 if the cache holds every expert, else 0 | `sycl/src/prefill/prefill.cpp` |
| `STRATA_PREFILL_GROUP_GATHER` | unset |  | the prompt path gathers, waits and releases one expert at a time | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_PREFILL_HELP` | off |  | The idle card computes a share of the streamed experts of a one-chunk prompt. `STRATA_PREFILL_HELP_FRAC=f` fixes the share. | `src/prefill/prefill.cpp`, `src/program/generate.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_PREFILL_HELPER_SOURCE` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_PREFILL_HELP_FRAC` | -1.0 |  | STRATA_PREFILL_HELP_FRAC sets the share for every T. (from the code comment) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_PREFILL_IN_PLACE` | unset |  | the prompt's MMQ reads experts from a gathered group buffer again (default: in place, from their ring / cache slots; the same bits) | `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_ISSUER` | unset |  | STRATA_PREFILL_ISSUER=0: inline. (from the code comment) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_PREFILL_LAYER_MAJOR` | unset |  | PREFILL_LAYER_MAJOR=1 opt-in with gains on small cards. (from the settings register) | `src/prefill/prefill.cpp`, `src/program/generate.cpp` |
| `STRATA_PREFILL_LEND_PCT` | 85 |  | (no description in the code) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_PREFILL_LM_EVEN` | on |  | (no description in the code) | `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_LM_FORCE` | off |  | (no description in the code) | `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_LM_MARGIN_MIB` | 96 |  | (no description in the code) | `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_LM_R` | unset |  | (no description in the code) | `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_LM_TOKENS` | unset |  | (no description in the code) | `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_LOAN_WALK` | unset |  | "STRATA_PREFILL_LOAN_WALK=0 keeps the rounded chunk"; a prompt piece below the walk floor lends buffers for the staged walk (default on). (from the settings register) | `src/program/generate.cpp` |
| `STRATA_PREFILL_MMQ` | unset |  | mmq_plan(): no plan, no fused-only layers (from the code comment) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_PREFILL_NVFP4` | off |  | the prompt path's precision for NVFP4 experts (default `w4a4x2` on RTX 50: gate/up on the FP4 tensor cores with two FP4 terms per activation, down w4a8; `w4a8` on other cards; `fp16` reads the prompt most exactly, 24% slower than `w4a8`) | `src/prefill/mmq_expert_bench.cpp`, `src/prefill/mmq_nvfp4_parity.cpp`, `src/prefill/moe_fused_iq.cu` +1 |
| `STRATA_PREFILL_PIPE` | 0.0 |  | Reads each prompt in the chunk count with the least pipeline time for b/a = R tokens. `1` evens out today's chunk count. | `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_PLAN_TOKS` | 0 |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_PREFILL_RING` | unset |  | Ring size tuning; default ring in place, variable sets size (=96 way back). (from the settings register) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_PREFILL_ROUTED_STREAM` | unset |  | Documented "(opt-in; ...)"; round0211 saw 0.0% on one card but doc lists it. (from the settings register) | `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_ROUTED_STREAM_DEPTH` | unset |  | STRATA_PREFILL_ROUTED_STREAM=1 (opt-in, routed_stream.hpp): the issuer keeps at most a few copies queued ahead of the link (STRATA_PREFILL_ROUTED_STREAM_DEPTH, default 16), each layer's routing is published once its r... (from the code comment) | `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_STREAM_AHEAD` | unset |  | Stream-ahead is on; =0 gave no change in a regression check. (from the settings register) | `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_STREAM_ALL` | unset |  | On when VRAM holds more than 90% of the (layer, expert) pairs | `sycl/src/prefill/prefill.cpp` |
| `STRATA_PREFILL_STREAM_MIN` | unset |  | `STRATA_PF_FUSED=1` reaches only chunks of N tokens or more (1,024 by default, the floor measured on a discrete card). On Windows a UD-Q4_K_XL set reads pieces below 1,024 tokens on the fused kernels' staged entry by itself (pieces of 96 to 1,000 tokens read 3.8 to 5.2 times sooner than on the old path; `STRATA_PF_STAGED_FO=0` gives the long-prompt path from 64 tokens instead, 3.6 to 4.5 times sooner); other sets keep 1,024, and a smaller N is opt-in. | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_PREFILL_SWIGLU_QUANT` | unset |  | the prompt path's earlier kernels: activations quantized per expert row / swiglu and quantize apart / combine then the hyper-connection write (the same bits) | `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_SYNC` | off |  | STRATA_PREFILL_SYNC=1 (debug): wait for the GPU at every mark and log the phase that just finished, so a hang names the stage that never completes (from the code comment) | `sycl/src/prefill/prefill.cpp` |
| `STRATA_PREFILL_TAIL_LOAN` | unset |  | "STRATA_PREFILL_TAIL_LOAN=0 keeps the stream floor's tail"; the lent fused MoE buffers are sized for the segment's own short last chunk, not for stream floor - 1 tokens (default on; one GPU). (from the settings register) | `src/program/generate.cpp` |
| `STRATA_PREFILL_TOKEN_ROWS` | unset |  | the prompt path's earlier kernels: activations quantized per expert row / swiglu and quantize apart / combine then the hyper-connection write (the same bits) | `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_WARM` | unset |  | Warm-up tunable; mixed-arch HIP skips it by default since 0.2.1. (from the settings register) | `src/prefill/prefill.cpp` |
| `STRATA_PROMPT_ATTN_ESIMD` | 1 |  | "on by default for Battlemage ... Set =0 to turn it off" (0.2.9). (from the settings register) | `sycl/src/kernels/cuda/qsa_prompt_attn_esimd.dp.cpp` |
| `STRATA_PROMPT_ATTN_IMMA` | unset |  | the prompt attention on FP16 MMA instead of INT8 tensor cores | `src/kernels/cuda/qsa_prompt_attn.cu` |
| `STRATA_PROMPT_ATTN_OLD` | off |  | Bit-identical new prompt attention path shipped in 0.1.22; =1 restores old behaviour. (from the settings register) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_PROMPT_ATTN_Q4` | unset |  | (no description in the code) | `src/kernels/cuda/qsa_prompt_attn.cu`, `sycl/src/kernels/cuda/qsa_prompt_attn.dp.cpp` |
| `STRATA_PROMPT_ATTN_Q4_TC` | off |  | Turing runs mode 4 as pairs of m16n8k8 MMAs (another summation order than the FP32 kernel it replaces): manual opt-in STRATA_PROMPT_ATTN_Q4_TC=1 (#1581, c3c: qsa_prompt_attn_parity at 4096/512 and 1500/512 on an RTX 2... (from the code comment) | `src/kernels/cuda/qsa_prompt_attn.cu` |
| `STRATA_PROMPT_ATTN_V1` | off |  | STRATA_PROMPT_ATTN_V1=1 (debug): the first version, same accuracy, another summation order - the control for how far the model amplifies an FP32-level change. (from the code comment) | `src/kernels/cuda/qsa_prompt_attn.cu`, `sycl/src/kernels/cuda/qsa_prompt_attn.dp.cpp` |
| `STRATA_PROMPT_DIRECT` | off |  | Stays opt-in (Linux). (from the settings register) | `src/program/generate.cpp` |
| `STRATA_PROMPT_DN` | unset |  | (no description in the code) | `metal/src/engine/prompt_chain.inc` |
| `STRATA_PROMPT_GU` | unset |  | STRATA_PROMPT_GU / STRATA_PROMPT_DN = TM,TN,TK,SG override them (test tooling; (from the code comment) | `metal/src/engine/prompt_chain.inc` |
| `STRATA_PROMPT_KPROF` | off |  | (no description in the code) | `metal/src/engine/prompt_chain.inc` |
| `STRATA_PROMPT_MERGEH` | unset |  | (no description in the code) | `metal/src/engine/prompt_chain.inc` |
| `STRATA_PROMPT_V6` | 4 |  | (no description in the code) | `metal/src/engine/prompt_chain.inc` |
| `STRATA_Q36_CHUNK` | unset |  | STRATA_Q36_CHUNK=512 gives the old size after bigger default chunk (0.1.45). (from the settings register) | `src/qwen35/q36_engine.cu` |
| `STRATA_Q36_NO_MMQ` | on |  | (no description in the code) | `src/qwen35/q36_engine.cu` |
| `STRATA_QSA_CLUSTER` | unset |  | STRATA_QSA_CLUSTER=0: the one-CTA kernels below (from the code comment) | `src/kernels/cuda/qsa_select.cu`, `sycl/src/kernels/cuda/qsa_select.dp.cpp` |
| `STRATA_QSA_EARLY_EXIT` | unset |  | STRATA_QSA_EARLY_EXIT=0 keeps the old staging (every CTA loads the queries first); the scores are the same bits (from the code comment) | `src/kernels/cuda/qsa_scores_gfx1100_test.cu`, `src/kernels/cuda/qsa_select.cu`, `sycl/src/kernels/cuda/qsa_select.dp.cpp` |
| `STRATA_QSA_FUSE_GATE` | unset |  | Attention gate epilogue fusion experiment on strix branch. (from the settings register) | `src/prefill/prefill.cpp` |
| `STRATA_QSA_FUSE_NR` | unset |  | STRATA_QSA_FUSE_NR default on; exact fusion with parity test. (from the settings register) | `src/prefill/prefill.cpp` |
| `STRATA_QSA_I8` | unset | `1` | QSA block scores on int8 tensor cores (six int8 products per score) | `src/kernels/cuda/qsa_select.cu` |
| `STRATA_QSA_ROWS` | off |  | Block scores of a verify window with queries in registers | `src/kernels/cuda/qsa_select.cu`, `src/kernels/qsa_rows_bench.cpp` |
| `STRATA_QSA_SLOTS` | on |  | a window's QSA block scores, top-k and attention run as one launch over its rows | `src/core/arch_defaults_test.cpp`, `src/core/verify.cpp` |
| `STRATA_QSA_SPLIT_ATTN` | unset |  | Decode attention split across blocks on the int8 mma path (sm_80+, int8 KV) | `src/core/mtp.cpp`, `src/core/verify.cpp` |
| `STRATA_QSA_STAT` | off |  | (no description in the code) | `src/kernels/cuda/qsa_prompt_attn.cu` |
| `STRATA_QSA_WARP` | unset |  | the pre-sm_80 QSA kernels on any card, as RTX 20 runs them | `src/kernels/cuda/qsa_prompt_attn.cu`, `src/kernels/cuda/qsa_select.cu`, `sycl/src/kernels/cuda/qsa_prompt_attn.dp.cpp` +1 |
| `STRATA_QUANT_GATHER` | unset |  | the prompt path quantizes a token's row once per expert again (default: once per token, scattered; the same bytes) | `src/prefill/moe_mmq.cu` |
| `STRATA_RAM_ADAPT_PROMPT_W` | 0 |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_SAMPLER_ONE_BLOCK` | off |  | (no description in the code) | `src/kernels/cuda/sampler.cu`, `sycl/src/kernels/cuda/sampler.dp.cpp` |
| `STRATA_SPLIT_CHUNK16K` | unset |  | Keeps the 16384-token auto chunk on AMD gfx12 splits (default 8192). | `src/program/generate.cpp` |
| `STRATA_SPLIT_PF_CHUNKS` | unset |  | extras/split-prefill (opt-in): STRATA_SPLIT_PF_CHUNKS=k - a layer split reads a request segment in about k equal chunks of at least STRATA_SPLIT_PF_MIN tokens (default 2048), so its stages overlap on different chunks... (from the code comment) | `src/program/generate.cpp` |
| `STRATA_SPLIT_PF_MIN` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_SPLIT_PF_RES` | 0.5 |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_SPLIT_SLOW_CHUNK` | unset |  | the piece of each hand-off the slow stage reads at a time (its own buffers; serve's STRATA_SPLIT_SLOW_CHUNK) (from the code comment) | `src/program/generate.cpp` |
| `STRATA_STAGER_BATCH` | 1 |  | Stager batch-size tuning knob. (from the settings register) | `src/prefill/prefill.cpp` |
| `STRATA_STAGER_SLEEP` | unset |  | Ryzen 9 7940HS laptop, RTX 4070, IQ3_S, 64K (issue #1101) | `src/prefill/prefill.cpp` |
| `STRATA_STAGER_SPREAD` | on |  | STRATA_STAGER_SPREAD=0: placed as before (the aux set while --aux-cpus is on, else where they were created). (from the code comment) | `src/prefill/prefill.cpp` |
| `STRATA_STAGER_SSD` | unset |  | Documented: =1/0 forces stager profile. (from the settings register) | `src/prefill/prefill.cpp` |
| `STRATA_STAGE_RESIDENT` | unset |  | STRATA_STAGE_RESIDENT=0 keeps the old sizing). (from the code comment) | `src/prefill/prefill.cpp` |
| `STRATA_SWITCH_POINT` | unset |  | Default: on for one class of card, off for every other (CUDA builds only; HIP never): NVIDIA discrete on PCIe, one GPU, link probe ran, compute capability 8.6 (RTX A6000 / A4000 class). Values: 0 = off, 1 = on for any CU (from the settings register) | `src/prefill/prefill.cpp` |
| `STRATA_TOPK_SAMPLE` | -1 |  | The prompt path's one-read top-k: STRATA_TOPK_SAMPLE=0 / 1 forces it off / on for any card; unset it is a Turing kernel of the series (STRATA_SM75_PERF=1). (from the code comment) | `src/kernels/cuda/qsa_select.cu` |
| `STRATA_W4A8_CFG` | 0 |  | (no description in the code) | `src/prefill/moe_iq4_w4a8.cu` |
| `STRATA_WS_FAST` | 0 |  | opt-in (from the code comment) | `src/prefill/moe_ws.cu` |
| `STRATA_XN_F16` | on |  | the hyper-connection's xn16 straight as the FP16 the tensor-core GEMM reads (its two readers then skip the conversion pass; the same values): STRATA_XN_F16=0 keeps the BF16 image (from the code comment) | `src/prefill/prefill.cpp` |

### K/V, cache and memory (144)

| Variable | Default | sm_120 table | What it does | Read in |
| --- | --- | --- | --- | --- |
| `STRATA_ADAPT_ACROSS` | off | `1` | the serve tier admits the PCIe share's experts, ranks misses across all layers, PCIe share fixed at 0.45 | `src/program/generate.cpp` |
| `STRATA_ADAPT_ASYNC` | unset |  | STRATA_ADAPT_ASYNC as the process sees it (-1 unset). (from the code comment) | `include/strata/core/adapt_async_default.hpp` |
| `STRATA_ADAPT_ASYNC_BATCH` | 0 |  | arena only (the other modes keep their steps as they were): the choice on this thread (STRATA_ADAPT_PICK_INLINE), the wait for the copies (STRATA_ADAPT_ASYNC_WAIT), the copies in flight per tick (STRATA_ADAPT_ASYNC_BA... (from the code comment) | `src/program/generate.cpp` |
| `STRATA_ADAPT_ASYNC_WAIT` | 1 |  | arena only (the other modes keep their steps as they were): the choice on this thread (STRATA_ADAPT_PICK_INLINE), the wait for the copies (STRATA_ADAPT_ASYNC_WAIT), the copies in flight per tick (STRATA_ADAPT_ASYNC_BA... (from the code comment) | `src/program/generate.cpp` |
| `STRATA_ADAPT_CALLBACK_WARMUP` | on |  | 0.1.43.1: on by default (STRATA_ADAPT_CALLBACK_WARMUP=0 turns it off). (from the code comment) | `src/program/generate.cpp` |
| `STRATA_ADAPT_CAND_MIN` | 2.0 | `1.5` | the serve tier admits the PCIe share's experts, ranks misses across all layers, PCIe share fixed at 0.45 | `src/program/generate.cpp` |
| `STRATA_ADAPT_DECAY` | unset | `0.96` | the serve tier admits the PCIe share's experts, ranks misses across all layers, PCIe share fixed at 0.45 | `src/program/generate.cpp` |
| `STRATA_ADAPT_EARLY` | on |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_ADAPT_EVICT_SYNC` | unset |  | the tier's evictions reach the device's table only when its copies land (the old behaviour: with the tier not waiting, a stale activation could be computed) | `src/program/generate.cpp` |
| `STRATA_ADAPT_FETCH` | 2 | `1` | decode: the PCIe share's experts admitted into the VRAM tier (2, the default: instead of the tier's own swaps where an admission ran; 1: beside them; 0: off) | `src/program/generate.cpp` |
| `STRATA_ADAPT_FETCH_MARGIN` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_ADAPT_FETCH_MIN` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_ADAPT_JOB_CPU` | unset |  | Pinning is on by default; the variable only picks a CPU or disables pinning (-1). (from the settings register) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_ADAPT_LAG` | 1 |  | Default 1 = #463 and 0.1.39 (the very next window waits for the copies: ~4.6 ms per window on an RTX 2060 SUPER over PCIe 3.0 x8); lag 2 is opt-in (STRATA_ADAPT_LAG=2). (from the code comment) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_ADAPT_LINK_BUDGET` | off |  | OPT-IN (STRATA_ADAPT_LINK_BUDGET=1): measured on the 4x 4090 rental, CUDA1 at 2 GB/s, 16 GB cards, the tier HELPS decode even through that link - auto K=41: 66.8 tok/s on, 43.8 off; --layer-split 24: 84.6 on, 55.2 off... (from the code comment) | `src/program/generate.cpp` |
| `STRATA_ADAPT_NOWAIT` | on |  | (no description in the code) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_ADAPT_OVERLAP` | off |  | One GPU, the whole arena in RAM, no `--adapt-async`, `--pipeline-windows`, `STRATA_ADAPT_LAG`, `STRATA_ADAPT_NOWAIT`, `STRATA_DMA_BATCH`; never on a layer split | `src/program/generate.cpp` |
| `STRATA_ADAPT_OVERLAP_RESIDENT` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_ADAPT_OVERLAP_SLOW_US` | 0 |  | test hook (STRATA_ADAPT_OVERLAP_SLOW_US=N): a slow link, to make the windows wait for the copies (from the code comment) | `src/program/generate.cpp` |
| `STRATA_ADAPT_PICK_INLINE` | 1 |  | arena only (the other modes keep their steps as they were): the choice on this thread (STRATA_ADAPT_PICK_INLINE), the wait for the copies (STRATA_ADAPT_ASYNC_WAIT), the copies in flight per tick (STRATA_ADAPT_ASYNC_BA... (from the code comment) | `src/program/generate.cpp` |
| `STRATA_ADAPT_SIGNAL` | unset |  | "STRATA_ADAPT_SIGNAL=host brings the callback back"; v0.1.43.1 stream-order copy is the new default. (from the settings register) | `src/program/generate.cpp` |
| `STRATA_ADAPT_WAIT` | off |  | each decode window waits for the tier's copies, so greedy decode repeats exactly (upstream's default; ~11% slower with this fork's tier, ~2% with `STRATA_ADAPT_LAG=2`) | `src/program/generate.cpp` |
| `STRATA_ALLOW_UNPINNED` | unset |  | (no description in the code) | `setup.py`, `tools/mtp_fetch.py` |
| `STRATA_ARENA_FALLBACK_RESIDENT` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_ARENA_FILE_CACHE` | unset |  | #5327: set to 0 to refuse an arena shortfall that free pages, droppable file cache or a reserved hugetlb pool could serve (the 0.1.56 behaviour: the run reads the experts from the file) (from the settings register) | `src/core/expert_source.cpp` |
| `STRATA_ARENA_LOCK` | unset |  | (no description in the code) | `src/core/pinned.cu`, `sycl/src/core/pinned.dp.cpp` |
| `STRATA_ARENA_MMAP` | off |  | Writes experts.bin for mmap experts; user-facing storage option, off unless set. (from the settings register) | `src/core/expert_source.cpp`, `src/program/generate.cpp`, `sycl/src/core/expert_source.cpp` +1 |
| `STRATA_ARENA_PIN_GIB` | unset |  | (no description in the code) | `src/core/pinned.cu`, `sycl/src/core/pinned.dp.cpp` |
| `STRATA_ARENA_PIN_PER_STAGE` | unset |  | (no description in the code) | `src/core/pinned.cu` |
| `STRATA_ARENA_PIN_SPREAD` | on |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_ARENA_RAM_FALLBACK` | unset |  | A hand-set `--ple-io ram` and a shared arena file are kept as asked (warn only), STRATA_ARENA_RAM_FALLBACK=0 is the old behaviour. (from the code comment) | `src/program/generate.cpp` |
| `STRATA_ARENA_RAM_WAIT_S` | 20.0 |  | STRATA_ARENA_RAM_WAIT_S changes the seconds (0 loads at once, as before this wait). (from the code comment) | `src/core/expert_source.cpp` |
| `STRATA_ARENA_SYNC` | on |  | register the arena per layer on a thread ahead of its readers (upstream's opt-in) / load the arena after the dense weights | `src/program/generate.cpp` |
| `STRATA_ARENA_THP` | unset |  | (no description in the code) | `src/core/pinned.cu` |
| `STRATA_BATCH_KV_BUDGET` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_BATCH_KV_FRESH` | off |  | STRATA_BATCH_KV_FRESH=1 takes this path on one GPU too (tests). (from the code comment) | `src/program/generate.cpp` |
| `STRATA_BATCH_KV_INIT` | unset |  | batch-vram: with the elastic K/V (one GPU) each slot's K/V is elastic too, its own owner (1 + slot): it maps STRATA_BATCH_KV_INIT cells (8192) now and grows from the expert cache as its context grows (slot_kv_ensure). (from the code comment) | `src/program/generate.cpp` |
| `STRATA_BATCH_KV_WHOLE` | off |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_BATCH_SELF_COMMIT` | off |  | a one-row-per-slot window advances the GDN state itself, no commit graph | `src/core/verify.cpp`, `src/program/generate.cpp` |
| `STRATA_CACHE_DIR` | unset |  | Directory for the crossover cache file; user-set path, no experiment verdict. (from the settings register) | `sycl/src/prefill/prefill.cpp` |
| `STRATA_CACHE_MESSAGE_BOUNDARY` | unset |  | (no description in the code) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_CACHE_VMM` | on |  | STRATA_CACHE_VMM=0 = the old arena (from the code comment) | `src/program/generate.cpp` |
| `STRATA_CKPT_VRAM_SLOTS` | 0 |  | The last N saved states also stay in VRAM (N x 118 MB, only while 1.5 GB stay free). A restore from one copies card-to-card. | `src/core/conversation_state.cpp` |
| `STRATA_COMMIT_BATCH` | on |  | (no description in the code) | `src/core/verify.cpp` |
| `STRATA_COMMIT_GROWTH` | unset |  | STRATA_COMMIT_GROWTH=0: 0 (as v1.1.17 counted). (from the code comment) | `include/strata/platform/commit.hpp` |
| `STRATA_COMMIT_SYNC` | on |  | the verify commit waits for its graph again | `src/core/verify.cpp`, `sycl/src/core/verify.cpp` |
| `STRATA_CONV_PRESSURE_HYST_MIB` | 1024 |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_CONV_PRESSURE_MS` | unset |  | Before a command, every STRATA_CONV_PRESSURE_MS (2000; 0 = off) while idle and between batch windows: below the floor the retained K/V, then the parked conversations oldest first (to the disk tier as any eviction, the... (from the code comment) | `src/program/generate.cpp` |
| `STRATA_DF2_WAITCOMMIT` | off |  | Draft-flash experiment switch; results only mention sibling switches reverted, nothing explicit for this one. (from the settings register) | `src/program/generate.cpp` |
| `STRATA_DISJOINT_ADAPT` | unset |  | 731 (opt-in, STRATA_DISJOINT_ADAPT=1): the adaptive tiers leave an expert a helper GPU holds out of the primary's promotion candidates (it would sit in both caches). (from the code comment) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_EMBED_VRAM` | off |  | The PLE rows go up raw and are dequantized on the card, on their own stream; the token-embedding table in VRAM; the dense dequant on a side stream (CUDA only). All bitwise. | `src/core/native_head.cpp` |
| `STRATA_FILE_CACHE_OLD_RULE` | unset |  | gfx906 keeps the old file-tier cache rule; `=1` forces it on any Linux build, `=0` the current one | `src/core/expert_source.cpp` |
| `STRATA_FILE_TIER_HONEST` | on |  | STRATA_FILE_TIER_HONEST=0 turns it off; an I/O-path change only). available_memory_bytes credits the group's whole `file` counter as reclaimable (cgroup_available_bytes), and on a RAM-budget config that credit include... (from the code comment) | `src/core/expert_source.cpp` |
| `STRATA_FILE_TIER_IDLE` | unset |  | P1.8 default on; "off STRATA_FILE_TIER_IDLE=0" restores old behaviour (v0.2.12). (from the settings register) | `src/program/generate.cpp` |
| `STRATA_FS_SLOTS` | off (0) |  | 4 slots, 6 pairs | `src/core/expert_source.cpp`, `src/program/generate.cpp` |
| `STRATA_GDN_COMMIT_SPLIT` | on |  | (no description in the code) | `src/kernels/cuda/verify_kernels.cu`, `sycl/src/kernels/cuda/verify_kernels.dp.cpp` |
| `STRATA_I8_SMEM_LIMIT` | unset |  | Test only: pretends the card has this many bytes of shared memory per block, so STRATA_EXPERT_I8's kernels fall to a smaller tile or answer 'not supported' (native_i8_parity uses it). (from the settings register) | `src/kernels/cuda/i8_expert.cuh` |
| `STRATA_KEEP_MAPPING` | on |  | STRATA_KEEP_MAPPING=1 keeps it (A/B). (from the code comment) | `src/program/generate.cpp` |
| `STRATA_KVA` | off |  | (no description in the code) | `include/strata/prefill/kva.hpp` |
| `STRATA_KVA_ANY_MODEL` | off |  | (no description in the code) | `include/strata/prefill/kva.hpp` |
| `STRATA_KVA_CAPTURE` | unset |  | (no description in the code) | `include/strata/prefill/kva.hpp` |
| `STRATA_KVA_CAPTURE_FP32` | off |  | (no description in the code) | `include/strata/prefill/kva.hpp` |
| `STRATA_KVA_MODE` | unset |  | (no description in the code) | `include/strata/prefill/kva.hpp` |
| `STRATA_KVA_ONLY` | unset |  | (no description in the code) | `include/strata/prefill/kva.hpp` |
| `STRATA_KVA_ORACLE` | unset |  | (no description in the code) | `include/strata/prefill/kva.hpp` |
| `STRATA_KVA_ORACLE_FP32` | off |  | (no description in the code) | `include/strata/prefill/kva.hpp` |
| `STRATA_KVA_PROJECTOR` | unset |  | (no description in the code) | `include/strata/prefill/kva.hpp` |
| `STRATA_KVA_SKIP` | unset |  | Documented moe mode of KV approximation, opt-in. (from the settings register) | `include/strata/prefill/kva.hpp` |
| `STRATA_KVA_SPLIT` | kKvaSplitDefault |  | Documented default 36; tunes first approximated layer. (from the settings register) | `include/strata/prefill/kva.hpp` |
| `STRATA_KVA_SUFFIX` | 0 |  | (no description in the code) | `include/strata/prefill/kva.hpp` |
| `STRATA_KV_GROW` | unset |  | the K/V allocated for the whole context at start | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_KV_GROW_FLOOR` | unset |  | (no description in the code) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_KV_GROW_HOLD` | 0 |  | [checked] opt-in: keeps the K/V as long as the longest parked conversation; default trims to the request as 0.1.40 (src/program/generate.cpp, comment above the read at line 16551). (from the settings register) | `src/program/generate.cpp` |
| `STRATA_KV_GROW_INIT` | 16384 |  | the K/V's cells at start (16384) and its growth step (8192) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_KV_GROW_STEP` | unset |  | (no description in the code) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_KV_MAIN_OFF` | off |  | A/B only: STRATA_KV_RING_OFF keeps the drafter fully resident, STRATA_KV_MAIN_OFF the main layers (from the code comment) | `src/core/layer.cpp`, `sycl/src/core/layer.cpp` |
| `STRATA_KV_POST` | unset |  | (no description in the code) | `src/kernels/cuda/kv_q8.cu` |
| `STRATA_KV_PREFETCH` | unset | `1` | KV streaming's copies on their own stream, ordered by events | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_KV_RESIDENT_FAST` | 1 |  | a new dialog's K/V blocks start mapped to the VRAM window in order, so the resolve step ends at once below 32,768 cells | `src/kernels/cuda/kv_stream.cu` |
| `STRATA_KV_RESIDENT_MIN` | 0 |  | STRATA_KV_RESIDENT_MIN=N overrides (the selection reads at most idx_top_k + idx_block - 1 cells per row, so a window of a few thousand cells holds it; the pool must hold the blocks of every row of a verify window). (from the code comment) | `src/core/layer.cpp` |
| `STRATA_KV_RESOLVE_ILP` | unset |  | the indexer chain: block scores one (block, query) pair a warp, batched resolve lookups, the window append in parallel and on its own branch beside the query projection | `src/kernels/cuda/kv_stream.cu` |
| `STRATA_KV_RING_OFF` | off |  | A/B only: STRATA_KV_RING_OFF keeps the drafter fully resident, STRATA_KV_MAIN_OFF the main layers (from the code comment) | `src/core/layer.cpp`, `sycl/src/core/layer.cpp` |
| `STRATA_KV_ROT` | off |  | int8 K/V without the Hadamard rotation | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_KV_STAGE_OWN` | off |  | the staging pool of a streamed session: every page of one layer (same sequence in init and bytes_needed) STRATA_KV_STAGE_OWN (A/B only): the staging pool gets its own allocation instead of borrowed expert slots, so a... (from the code comment) | `src/prefill/prefill.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_KV_STREAM` | on (131,072+ context) |  | `0`: no KV streaming by default. Else, from a 131,072-token context on one GPU with the RAM for it, 32,768 K/V cells stay in VRAM (`--no-kv-stream` does the same as `0`) | `src/program/generate.cpp` |
| `STRATA_MEMORY_LIMIT_MIB` | unset |  | Total RAM this engine's container may use, for when a container cannot see its own limit. Config `args`: `"--memory-limit-mib", "102400"`. `0` = none. | `src/program/generate.cpp` |
| `STRATA_MEM_CREDIT_INACTIVE` | unset |  | under a container memory limit the memory guard and the expert arena's RAM check credit active and inactive file cache minus the engine's own mapped pages; set to 1 brings back the old inactive-only credit. (from the settings register) | `src/core/conversation_memory.cpp`, `src/core/expert_source.cpp` |
| `STRATA_NO_ARENA_CACHE_DROP` | unset |  | turns off the file-cache give-back during and after the expert load on an integrated gfx11 GPU under Linux (posix_fadvise DONTNEED behind the arena read and once per model file; automatic on that class, nothing on other  (from the settings register) | `include/strata/core/page_cache_drop.hpp` |
| `STRATA_NO_ARENA_THP` | off |  | unset: the engine decides (no huge-page request for the expert arena on an integrated gfx11 GPU under Linux: gfx1103, gfx1150, gfx1151; a request everywhere else, as before); 1 (any value but 0): never request them, as b (from the settings register) | `src/core/pinned.cu`, `sycl/src/core/pinned.dp.cpp` |
| `STRATA_NO_BATCH_KV_STEP` | off |  | STRATA_NO_BATCH_KV_STEP=1 appends per token. (from the code comment) | `src/core/mtp.cpp`, `src/core/verify.cpp`, `sycl/src/core/mtp.cpp` +1 |
| `STRATA_NO_LARGEPAGES` | off |  | 4 KB pages even when large pages are allowed | `src/core/expert_source.cpp`, `src/core/pinned.cu`, `sycl/src/core/pinned.dp.cpp` |
| `STRATA_ONE_TOKEN_COMMIT` | unset |  | HIP: opt-in (=1) (from the code comment) | `src/core/verify.cpp`, `sycl/src/core/verify.cpp` |
| `STRATA_PARK_VRAM_MIB` | 0 |  | A park also keeps the K/V in VRAM slots (64 MiB granules, up to N MiB, only while 1.5 GB stay free). Switching back restores it card-to-card. | `src/core/conversation_snapshot.cpp` |
| `STRATA_PARTIAL_PIN` | unset |  | Registers the hottest part of the RAM budget (up to `STRATA_PARTIAL_PIN_GIB`, default 24) with the GPU driver, so the GPU computes a share of the misses over PCIe (`--pcie-frac`). Measured no faster, and it changes the numerics of those experts, so off. | `src/core/expert_source.cpp`, `sycl/src/core/expert_source.cpp` |
| `STRATA_PARTIAL_PIN_GIB` | 24.0 |  | at most STRATA_PARTIAL_PIN_GIB (default 24): registering 30 GiB of a 40 GiB arena left the driver unable to page-lock the prompt path's small buffers afterwards (RTX 5070, WDDM) (from the code comment) | `src/core/expert_source.cpp`, `sycl/src/core/expert_source.cpp` |
| `STRATA_PCIE_AUTO_ADD` | 0.0 |  | STRATA_PCIE_AUTO_ADD (dec/int; 0.10 in the sm_120 table, else 0): added to the balance share. (from the code comment) | `src/program/generate.cpp` |
| `STRATA_PCIE_AUTO_CANON` | off |  | "Windows: opt-in (=1); RTX 5070 decoded 3% slower with it". (from the settings register) | `src/program/generate.cpp` |
| `STRATA_PCIE_AUTO_SPLIT` | unset |  | Only a neighbouring opt-out (PCIE_FRAC_DEFAULT=old) is described; role of this variable unclear. (from the settings register) | `src/program/generate.cpp` |
| `STRATA_PCIE_BALANCE` | unset |  | CUDA, one GPU, no `--batch`, no `--pcie-frac` | `src/program/generate.cpp` |
| `STRATA_PCIE_BRANCH` | 1 |  | dec/host STRATA_PCIE_BRANCH (see record_window's pcb_branch): default on 1: fork at the doorbell (beside the shared expert and the hits), 2: after the shared expert (beside the hits), 0 off; STRATA_PCIE_BRANCH_BLOCKS:... (from the code comment) | `src/core/verify.cpp` |
| `STRATA_PCIE_BRANCH_BLOCKS` | 16 |  | (no description in the code) | `src/core/verify.cpp` |
| `STRATA_PCIE_FIXED` | unset | `0.45` | the serve tier admits the PCIe share's experts, ranks misses across all layers, PCIe share fixed at 0.45 | `src/program/generate.cpp` |
| `STRATA_PCIE_FRAC_DEFAULT` | unset |  | decode keeps the link probe's fixed PCIe share (upstream 0.1.41's rule) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_PCIE_K_RULE` | unset |  | Flag with "defaults unchanged"; +102% on 2-worker pool, off unless set. (from the settings register) | `src/program/generate.cpp` |
| `STRATA_PCIE_MIN1` | unset |  | Stays opt-in (RubenTorney opt-ins). (from the settings register) | `src/core/expert_source.cpp` |
| `STRATA_PCIE_SHARE_MAX` | unset |  | Hybrid pool share ceiling flag; defaults unchanged. (from the settings register) | `src/program/generate.cpp` |
| `STRATA_PCIE_START` | unset |  | strata-nvfp4: the link-only start (see fork_env_or) (from the code comment) | `src/program/generate.cpp` |
| `STRATA_PIN_BY_NEED` | unset |  | Nor with upstream's STRATA_PIN_BY_NEED=1 (#1905, opt-in): its pinned set is chosen before the arena opens, from the GPU cache's size estimated once the weights, session and drafter are in - the arena then opens where... (from the code comment) | `src/program/generate.cpp` |
| `STRATA_PIN_BY_NEED_SKIP` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_PIN_GUARD` | unset |  | (no description in the code) | `src/core/expert_source.cpp` |
| `STRATA_PIN_LIMIT_FILE` | unset |  | fix/mgpu-pin-limit: where a layer split under Windows (WDDM) writes the pinned memory at which the driver refused, per RAM size, GPUs and driver; the next start pins 2 GiB less. Unset: `%LOCALAPPDATA%/strata/wddm-pin-limit.txt`; `0`: nowhere (a refusal then hands slices back as in 0.2.15-nvfp4.6) | `src/program/generate.cpp` |
| `STRATA_PIN_MARGIN_GIB` | unset |  | extras/wddm-pin: STRATA_PIN_MARGIN_GIB, what stays free of the shared-memory budget, and what a layer split hands back after the driver's refusal (default 4 GiB, #243's) (from the code comment) | `src/core/pinned.cu` |
| `STRATA_PIN_TOTAL_GIB` | unset |  | fix/mgpu-pin-limit: what a layer split under Windows (WDDM) pins of the expert arena on all its GPUs together, in GiB. Unset: 62 (one driver limit near 64 GiB); `0`: no such limit. Each GPU's own cap still holds | `src/program/generate.cpp` |
| `STRATA_PIPED_CONCURRENT_UNPIN` | unset |  | STRATA_PIPED_CONCURRENT_UNPIN=0 keeps the inherited CPUs (an A/B); (from the code comment) | `src/program/generate.cpp` |
| `STRATA_PIPELINE_ADAPT_ASYNC` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_PIPE_ADAPT_EVERY` | unset |  | "(opt-in; unset or 0: off, as before)". (from the settings register) | `src/program/generate.cpp` |
| `STRATA_POOL_SPIN_US` | unset |  | a test knob; see kSpinBeforeSleep (from the code comment) | `src/kernels/cpu/pool.cpp` |
| `STRATA_PRIVATE_GR_SOA_CACHE` | off |  | Private opt-in (=1), off by default: structure-of-arrays cache of the fused GR kernel, main actor only (gr_soa_enabled, sycl/src/kernels/cuda/fused_gr.dp.cpp:2969). Read from the code only; the integration lane measured  (from the settings register) | `sycl/src/kernels/cuda/fused_gr.dp.cpp` |
| `STRATA_Q36_FIXED_SLOTS` | off |  | the slots as placed (tests) (from the code comment) | `src/qwen35/q36_engine.cu` |
| `STRATA_QI_NO_QKV_FUSE` | unset |  | (no description in the code) | `tools/image/sdcpp_native/qwen_image_2_1_native.hpp` |
| `STRATA_QI_OLD_QKV` | unset |  | (no description in the code) | `tools/image/sdcpp_native/qi21-native.cu` |
| `STRATA_RAM_ADAPT` | 0 |  | Documented option; =8 slower pp (0.838) in A/B so not default. (from the settings register) | `src/program/generate.cpp` |
| `STRATA_RAM_ADAPT2` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_RAM_ADAPT_CACHED_W` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_RAM_ADAPT_PER_LAYER` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_RAM_ADAPT_QUEUE` | unset |  | (no description in the code) | `src/core/expert_source.cpp` |
| `STRATA_RAM_ADAPT_SKIP_TOP` | unset |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_RAM_KEEP_GIB` | unset |  | Documented; app passes it for the work-profile RAM reserve. (from the settings register) | `src/program/generate.cpp` |
| `STRATA_REMOTE_SPIN` | on |  | STRATA_REMOTE_SPIN=0 keeps the driver's default. (from the code comment) | `src/core/remote_experts.cpp`, `sycl/src/core/remote_experts.cpp` |
| `STRATA_RESIDENT_HEADROOM_GIB` | unset |  | RAM the low-RAM mode leaves free (10 when the pack's size turned it on) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_RESIDENT_PIN` | unset |  | Ordinary pageable memory for the RAM budget, also for the RAM copy the engine makes on its own (see above). | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_ROUTE_RESIDENT` | unset |  | "shipped, opt-in": stays opt-in. (from the settings register) | `src/core/verify.cpp`, `src/program/generate.cpp` |
| `STRATA_ROUTE_RESIDENT_RANKS` | unset |  | (no description in the code) | `src/core/verify.cpp`, `src/program/generate.cpp` |
| `STRATA_SLOT_CKPT_MOVE` | off |  | the admission's turn checkpoint moves into the slot (no 112 MB copy); it also keeps the dropped checkpoints' storage, as `STRATA_CKPT_POOL` does; a session SAVE or parked conversation taken right after an admission lacks the moved checkpoint (the slot holds it) | `src/program/generate.cpp` |
| `STRATA_SLOT_COPY_BATCHSYNC` | off |  | with the above: queue the ~116 copies and synchronize each device once | `src/program/generate.cpp` |
| `STRATA_SLOT_COPY_DIRECT` | unset |  | an admission (and a slot's resume) copies the running state and the QSA K/V device to device, instead of through host images | `src/program/generate.cpp` |
| `STRATA_SPLIT_HEADROOM` | unset |  | 1440: of the placements within STRATA_SPLIT_HEADROOM_TOL (default 0.2%) of the best predicted time, the one whose tightest card keeps the most VRAM beyond all its layers' experts (split_rules::pick_with_headroom). (from the code comment) | `sycl/src/program/generate.cpp` |
| `STRATA_SPLIT_HEADROOM_TOL` | unset |  | (no description in the code) | `sycl/src/program/generate.cpp` |
| `STRATA_SPLIT_SLOW_RESIDENT` | on |  | (no description in the code) | `src/program/generate.cpp` |
| `STRATA_STAGE_PIN` | 0 |  | GGUF read in place | `src/core/expert_source.cpp` |
| `STRATA_TAIL_SKIP_MAX_CACHED` | unset |  | Moves the cached-percent line of the default tail-skip rule. (from the settings register) | `src/program/generate.cpp` |
| `STRATA_UMA_DROP_PAGES` | on |  | The A/B switch of the shared-memory page drop (lane SPARK, gate on the DGX Spark): a GPU that shares the system RAM (cudaDevAttrIntegrated) and holds every expert gives the GGUF's cached expert pages back after the promp (from the settings register) | `src/program/generate.cpp` |
| `STRATA_UMA_HEADROOM_GIB` | unset |  | STRATA_UMA_HEADROOM_GIB: a whole number of GiB, 0..1024; anything else keeps the default 6 (said once) (from the code comment) | `src/core/expert_cache.cpp`, `sycl/src/core/expert_cache.cpp` |
| `STRATA_UMA_PIN_CAP` | unset |  | STRATA_UMA_PIN_CAP=0 keeps the RAM budget as given: on a Strix Halo (Windows HIP) with --batch 2 or more the engine cuts the pinned RAM copy of experts so that it plus the slots' pinned K/V stays at 22.75 GiB (15.0 GiB o (from the settings register) | `src/program/generate.cpp` |
| `STRATA_VNNI_TIER` | off |  | The tier below VBMI is OPT-IN (STRATA_VNNI_TIER=1): on such a CPU the 512-bit i-quant and Q2_0 rows round differently from the AVX-2 rows it runs today (measured under SDE Cascade Lake: every multi-token gate/up line... (from the code comment) | `src/kernels/cpu/expert_layout.cpp` |
| `STRATA_VRAM_FREE` | unset |  | RTX 5090 (WDDM), Core i7-13700K, 64 GB, 128K context, image encoder loaded, `--expert-cache auto` | `src/core/expert_cache.cpp` |

### GPU-specific (149)

| Variable | Default | sm_120 table | What it does | Read in |
| --- | --- | --- | --- | --- |
| `STRATA_AMD_COMMUNITY_TUNING` | unset |  | STRATA_AMD_COMMUNITY_TUNING: on by default; (from the code comment) | `include/strata/core/amd_community.hpp` |
| `STRATA_ARGMAX_ROWS_SM70` | unset |  | The multi-block greedy pick on sm_7x (#2357), bitwise the one-block pick. | `src/kernels/cuda/verify_kernels.cu` |
| `STRATA_ATTN_PRE75` | unset |  | STRATA_ATTN_PRE75=0 turns PR #540's kernel off (A/B); on HIP =1 also runs it on another wave32 card. (from the code comment) | `src/kernels/cuda/qsa_decode_attn.cu`, `sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp` |
| `STRATA_COPY_HIP_PRIO` | unset |  | Review states the dma copy priority switch stays opt-in. (from the settings register) | `src/core/verify.cpp` |
| `STRATA_GFX1151_CHUNK` | unset |  | "=0 keeps 6,144": variable restores the old chunk after the new default (v0.2.3). (from the settings register) | `src/program/generate.cpp` |
| `STRATA_GFX1151_DEFAULTS` | on |  | (no description in the code) | `src/core/arch_defaults.cpp`, `src/prefill/moe_fused_iq.cu`, `src/program/generate.cpp` |
| `STRATA_GFX1151_EXP8` | unset |  | (no description in the code) | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_GFX12_CHUNK` | unset |  | STRATA_GFX12_DEFAULTS=0 or STRATA_GFX12_CHUNK=0 keeps 8192. 0.2.10.1 (owner 11-03, "do all 5"): the ceiling is 32768, not 16384. STRATA_GFX12_CHUNK=16 keeps 16384. (from the code comment) | `src/program/generate.cpp` |
| `STRATA_GFX12_DEFAULTS` | on |  | Turns off the gfx1200/1201 default switch set (0 restores previous behaviour). (from the settings register) | `include/strata/core/gfx12_extra.hpp`, `src/core/arch_defaults.cpp`, `src/core/native_dense.cpp` +2 |
| `STRATA_GFX12_RING` | unset |  | STRATA_PREFILL_RING=96 restores the old value; so does STRATA_GFX12_RING=0. gfx12 only: the ring was measured on an R9700. (from the code comment) | `src/prefill/prefill.cpp` |
| `STRATA_GFX906_ATTN_QUERY_SWIZZLE` | off |  | Documented gfx906 attention switch set by hand in AMD_HIP.md. (from the settings register) | `src/kernels/cuda/qsa_decode_attn.cu` |
| `STRATA_GFX906_ATTN_REDUCE12` | off |  | Documented gfx906 attention switch, opt-in =1. (from the settings register) | `src/kernels/cuda/qsa_decode_attn.cu` |
| `STRATA_HIPBLASLT_TABLE` | unset |  | Use the packaged table when it matches | `src/prefill/gemm.cu` |
| `STRATA_HIPBLASLT_TUNING` | unset |  | Use this table file; always kept | `src/prefill/gemm.cu`, `sycl/src/prefill/gemm.dp.cpp` |
| `STRATA_HIPBLASLT_VERBOSE` | off |  | (no description in the code) | `src/prefill/gemm.cu`, `sycl/src/prefill/gemm.dp.cpp` |
| `STRATA_HIP_ADAPT_KERNEL_COPY` | off |  | "Opt-in: does the adaptive swaps copies with a kernel instead of SDMA (#884, 2x gfx1030 hang)". (from the settings register) | `src/program/generate.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_HIP_ARCHS` | unset |  | (no description in the code) | `tools/hip/build_windows.bat` |
| `STRATA_HIP_PROMPT_F16` | off |  | Runs the 16-bit prompt GEMMs FP16 in and out (the engine prints a tip on gfx103x). On by default on a Radeon 780M (gfx1103) under Linux; `=0` turns it off there; every other card needs `=1`. Radeon 780M, Q2_0, prompt x1.11 / x1.15 / x1.19 / x1.25 at 300 / 1,000 / 2,048 / 8,192 tokens (x1.11 / x1.15 / x1.63 / x1.45 beside `STRATA_PF_FUSED=1`); forced-token KL against the FP16 path 0.00593 against 0.00543 for MMQ (24 prompts, 64 tokens each, ratio 1.09, 95% CI 0.97 to 1.23). On a Radeon AI PRO R9700 (gfx1201) the dense GEMMs are slower with it (x0.70 to x0.72), so it stays opt-in there. `STRATA_F16_RANGE=1` prints, per prompt, the largest value that entered or left FP16 (FP16 ends at 65504); `STRATA_DBG_NAN=1` counts non-finite FP16 outputs. Checks: [2026-10-04-rdna2-fp16-prompt](../../bench/results/2026-10-04-rdna2-fp16-prompt/README.md). | `src/prefill/gemm.cu`, `sycl/src/prefill/gemm.dp.cpp` |
| `STRATA_HIP_QUEUE_PRIME` | unset |  | HIP queue priming became default on R9700 Linux in 0.2.1; variable forces or sets it. (from the settings register) | `src/program/generate.cpp` |
| `STRATA_HIP_ROUTER_FAST` | on |  | gfx906 (wave64): the one-wavefront kernel below stays the default; the S6 kernel is wave32-shaped and only runs here on request (STRATA_HIP_ROUTER_FAST=1) until it is measured on this card (from the code comment) | `src/kernels/cuda/router_top10.cu`, `sycl/src/kernels/cuda/router_top10.dp.cpp` |
| `STRATA_HIP_ROUTER_OLD` | off |  | (no description in the code) | `src/kernels/cuda/router_top10.cu`, `sycl/src/kernels/cuda/router_top10.dp.cpp` |
| `STRATA_HIP_SWIGLU_FUSED` | off |  | gfx906: the separate SwiGLU and q8_1 passes stay the default (the A/B baseline); the RDNA one-pass kernel (bitwise the same) with STRATA_HIP_SWIGLU_FUSED=1 (from the code comment) | `src/kernels/cuda/iq_kernels.cu`, `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_HIP_VMM` | on |  | Used where device supports VMM; "STRATA_HIP_VMM=0 turns it off". (from the settings register) | `src/core/vmm.cpp` |
| `STRATA_HIP_WMMA` | off |  | gfx12; gfx11 on request | `src/kernels/cuda/qsa_prompt_attn.cu`, `sycl/src/kernels/cuda/qsa_prompt_attn.dp.cpp`, `sycl/src/prefill/prefill.cpp` |
| `STRATA_HIP_WMMA_MIN_CAP` | unset |  | (no description in the code) | `src/kernels/cuda/qsa_prompt_attn.cu` |
| `STRATA_HIP_WMMA_MIN_Q` | unset |  | (no description in the code) | `src/kernels/cuda/qsa_prompt_attn.cu` |
| `STRATA_LM_HIP` | off |  | Commit "STRATA_LM_HIP opt-in"; no verdict against it. (from the settings register) | `src/prefill/prefill.cpp` |
| `STRATA_METAL_ARENA_CHUNK_GB` | unset |  | (no description in the code) | `metal/src/engine/stream.cpp` |
| `STRATA_METAL_ARENA_GB` | unset |  | Streaming: the arena's size, or the memory left to macOS | `metal/src/engine/window.cpp` |
| `STRATA_METAL_ARENA_STEP_GB` | unset |  | (no description in the code) | `metal/src/engine/stream.cpp` |
| `STRATA_METAL_BATCH_ROWS` | unset |  | (no description in the code) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_CLASS_WASTE` | 0.15 |  | The layers' slot sizes (9 in the IQ3_XXS mix) are merged into a few classes, each as big as its largest member, so that no member wastes more than STRATA_METAL_CLASS_WASTE (default 0.15) of its slot: every class must... (from the code comment) | `metal/src/engine/stream.cpp` |
| `STRATA_METAL_CONTIG` | 1 |  | Streaming: read the model file even when the experts' copy is there | `metal/src/engine/stream.cpp` |
| `STRATA_METAL_CTAB` | unset |  | STRATA_METAL_CTAB=0 turns it off (from the code comment) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_DRAFT_VOCAB` | unset |  | (no description in the code) | `metal/src/engine/dflash2.inc`, `metal/src/engine/window.cpp` |
| `STRATA_METAL_F16` | unset |  | Decode windows of a resident model: the GEMVs of Q2_0, Q3_K-Q6_K and IQ4_XS and the Q2_0 routed experts in fp16. On in the config setup writes for a resident IQ3_S or Q2_0 pack (M5 Max: decode +9 to +15%, top-1 96-97% and mean KL 0.009-0.016 against fp32; the 61-task quality suite passes the same tasks as with it off); `=0` in the config's `env` turns it off. Other sets: off unless set. M1 Max: a 2-token window's GPU time 56-60 -> 40-41 ms, mean KL 0.009 against fp32 | `metal/src/engine/window.cpp` |
| `STRATA_METAL_FUSE` | unset |  | Turn off two decode fusions (same tokens, slower) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_GDN_FUSE` | unset |  | STRATA_METAL_GDN_FUSE=0 keeps the separate ones. (from the code comment) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_GDN_SCAN` | unset |  | Measured by metal/tests/gdn_chunk_bench: Apple M4 (the base chip): gdn_scanr_4_8, 1.16x on the scan, 1.11x on the GDN prompt stage, T 512..32K Apple M5 Max: gdn_scanr_4_4 from T >= 4096 (1.04x at 4K, 1.10x at 16K); be... (from the code comment) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_GEMM_TILES` | unset |  | STRATA_METAL_GEMM_TILES=0 keeps the old tiles everywhere. (from the code comment) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_GEMV_NR` | 1 |  | NR=2 pays only at exactly two rows; tuning experiment, no ship/not-ship verdict. (from the settings register) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_GYRO_TILES` | unset |  | off: the A/B arm (from the code comment) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_HCLD` | unset |  | Metal HC load experiment; evidence concerns related rejected variants, not clearly this switch. (from the settings register) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_IGNORE_REPACK` | unset |  | Start although a model repack or setup's preparation is running (the engine refuses by default) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_IO_THREADS` | 6 |  | (no description in the code) | `metal/src/engine/stream.cpp` |
| `STRATA_METAL_KEEP_FREE_GB` | unset |  | Streaming: the arena's size, or the memory left to macOS | `metal/src/engine/window.cpp` |
| `STRATA_METAL_KEEP_HOST_WEIGHTS` | unset |  | Streaming: keep the host code's own copies of the weights (2 to 3 GB) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_KPROF` | unset |  | Per-window traces, a time breakdown, per-kernel GPU times | `metal/src/engine/window.cpp` |
| `STRATA_METAL_LAZY_FOLD` | unset |  | Turn off two decode fusions (same tokens, slower) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_LDB` | unset |  | Batched loads kept (+2.0-2.9%, bit-exact) with STRATA_METAL_LDB=0 as the way back. (from the settings register) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_LEAVE_SHARE` | unset |  | Streaming, 64 GB or more: the share of the RAM left to macOS when the footprint cap is set | `metal/src/engine/window.cpp` |
| `STRATA_METAL_LOGPOS` | unset |  | (no description in the code) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_MATH` | unset |  | (no description in the code) | `metal/src/mtl_rt.mm` |
| `STRATA_METAL_MAX_THREADS_CAP` | unset |  | test hook, see require_fit (from the code comment) | `metal/src/mtl_rt.mm` |
| `STRATA_METAL_MTP_CHAIN` | unset |  | One GPU submission per draft step instead of one per round | `metal/src/engine/window.cpp` |
| `STRATA_METAL_MTP_GGUF` | unset |  | the MTP model in GGUF form (tools/mtp_pack.py) beside the rt directory: STRATA_METAL_MTP_GGUF, else <rt>/../mtp-q2_0.gguf (from the code comment) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_MTP_OVERLAP` | unset |  | MTP-layer K/V rows "now go to a second command queue"; switch turns the overlap off. (from the settings register) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_MTP_Q8` | unset |  | The draft layer's matrices in BF16 instead of Q8_0 | `metal/src/engine/window.cpp` |
| `STRATA_METAL_NFCHECK` | off |  | Check switch, listed as off unless set. (from the settings register) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_PARTIAL` | unset |  | Retired in 0.2.13 (cleanup 4.5, batch A): code removed, the setting is no longer read. Evidence: [checked] Opt-in partial commit in gdn_step: 0.3-0.8% slower at --spec 2 on the M5 Max (metal/src/engine/window.cpp:434-438 (from the settings register) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_PREFETCH` | big ? 6 : 10 |  | (no description in the code) | `metal/src/engine/stream.cpp`, `metal/src/engine/window.cpp` |
| `STRATA_METAL_PREFETCH2` | 0 |  | (no description in the code) | `metal/src/engine/stream.cpp` |
| `STRATA_METAL_PREFETCH_MRU` | big ? 0 : 1 |  | (no description in the code) | `metal/src/engine/stream.cpp` |
| `STRATA_METAL_PROMPT` | unset |  | Read the prompt through decode windows (slow, exact fallback; about 190 tok/s) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_PROMPT_CHUNK` | 2048 |  | Prompt chunk size (default 2048 resident; see streaming above) | `metal/src/engine/prompt_chain.inc`, `metal/src/engine/window.cpp` |
| `STRATA_METAL_PROMPT_EXPERTS` | unset |  | The M5 layout for the prompt's expert GEMMs. Fails on an M1 Max, so it is off there (default v2) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_PROMPT_GEMM` | unset |  | (no description in the code) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_PROMPT_GEMM_MIN` | 64 |  | (no description in the code) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_PROMPT_HOSTPIPE` | unset |  | (no description in the code) | `metal/src/engine/prompt_chain.inc` |
| `STRATA_METAL_PROMPT_MIN` | unset |  | (no description in the code) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_PROMPT_QSA` | unset |  | v1 \| flash (default: flash with the M5's layout) (from the code comment) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_READ_SPLIT` | 1 |  | (no description in the code) | `metal/src/engine/stream.cpp` |
| `STRATA_METAL_ROUTE` | unset |  | Retired in 0.2.13 (cleanup 4.5, batch A): code removed, the setting is no longer read. Evidence: [checked] Fused router_route dispatch: no faster (T2 GPU 13.035 vs 13.027 ms) and 0.3-0.5% slower end to end on the M5 Max  (from the settings register) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_SKIP` | unset |  | Leave-one-class-out profiling switch on Metal. (from the settings register) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_SPIN` | unset |  | STRATA_METAL_SPIN=0: the blocking wait. (from the code comment) | `metal/src/mtl_rt.mm` |
| `STRATA_METAL_STEPPROF` | unset |  | (no description in the code) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_STEPPROF_AT` | unset |  | (no description in the code) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_STEPPROF_OUT` | unset |  | (no description in the code) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_STEPPROF_REPS` | unset |  | (no description in the code) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_STREAM` | unset |  | Stream the experts from the SSD even when the pack fits / never (the default decides by size) | `metal/src/engine/stream.cpp` |
| `STRATA_METAL_STREAM_FAKE` | 0 |  | "timing only (WRONG answers): every missing expert is pointed at slot 0"; benchmark diagnostic. (from the settings register) | `metal/src/engine/stream.cpp` |
| `STRATA_METAL_STUBS` | unset |  | (no description in the code) | `metal/src/engine/missing.cpp` |
| `STRATA_METAL_WIRE_ARENA` | unset |  | Streaming, 64 GB or more: leave the filled arena pageable (default: wired) | `metal/src/engine/stream.cpp` |
| `STRATA_METAL_WPF_MIB` | unset |  | Retired in 0.2.13 (cleanup 4.5, batch A): code removed, the setting is no longer read. Evidence: [checked] Weight prefetch beside gr_up: -5.9% on UD-IQ4_XS-v31, 0 of 4 rounds faster on every kind, 'Dead' (phaseA-tests/ro (from the settings register) | `metal/src/engine/window.cpp` |
| `STRATA_METAL_WRAP_SPLIT` | unset |  | (no description in the code) | `metal/src/mtl_rt.mm` |
| `STRATA_MMVQ_IL_SM80` | unset |  | On by default for sm_80 in 0.2.7; =0 turns it off. (from the settings register) | `src/kernels/cuda/native_mmvq.cu` |
| `STRATA_MTP_HIP_PRIO` | unset |  | engine-latency, STRATA_MTP_HIP_PRIO (default on gfx1200/gfx1201, =1 elsewhere opt-in, =0 off): the draft stream at the highest priority. (from the code comment) | `src/core/mtp.cpp` |
| `STRATA_MTP_Q8_SM60` | off |  | Off-by-default Pascal MTP GEMV subset option. (from the settings register) | `src/core/mtp.cpp`, `src/program/generate.cpp` |
| `STRATA_NVFP4_SM120` | on (RTX 50) |  | `0`: the whole strata-nvfp4 RTX 50 table off (src/core/nv_defaults.cpp); each switch's own `=0` turns that one off | `src/core/nv_defaults.cpp` |
| `STRATA_PATTN_SM60` | 0 |  | STRATA_PATTN_SM60=1 turns it on (a tester switch, read once). (from the code comment) | `src/kernels/cuda/qsa_pattn.cu` |
| `STRATA_PF_FUSED_SM40` | unset |  | CUDA diagnostic for bug reports, not a user setting: =0 puts compute capability 8.0, 8.6, 8.9 and 12.0 cards with 40-79 SMs back on MMQ for the IQ3_XXS / IQ3_S / IQ4_XS gate/up layers with an IQ4_NL or Q2_0 down (the fus (from the settings register) | `src/core/nv_defaults.cpp`, `src/prefill/moe_fused_iq.cu`, `src/prefill/moe_fused_iq_q36.cu` |
| `STRATA_PF_FUSED_SM75` | off |  | Turing (sm_75): the same kernels with plain copies and m8n8k16 MMAs, opt-in (STRATA_PF_FUSED_SM75=1) (from the code comment) | `src/prefill/moe_fused.cu`, `src/prefill/moe_fused_iq.cu` |
| `STRATA_PF_FUSED_SM75_IQ` | unset |  | On a Turing card (sm_75) the i-quant gate/up pairs run on the fused kernels when STRATA_PF_FUSED_SM75 is on; 0 here puts them back on the int8 dp4a kernels (Q2_0 / Q2_0 stays fused). (from the settings register) | `src/prefill/moe_fused_iq.cu` |
| `STRATA_PF_FUSED_SM86` | unset |  | CUDA diagnostic for bug reports, not a user setting: =0 puts compute capability 8.0, 8.6 and 8.9 cards (the sm_86 class) back on MMQ for the default list's gate/up formats (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS; (from the settings register) | `src/prefill/moe_fused_iq.cu` |
| `STRATA_PF_XMX` | unset |  | XMX on A-series became default (xmxdef); =0 gives the old logits. (from the settings register) | `sycl/src/prefill/xmx_moe.dp.cpp` |
| `STRATA_PF_XMX_GU_MAX` | 250 |  | Sub-knob of PF_XMX experiments; XMX variants measured slower. (from the settings register) | `sycl/src/prefill/xmx_moe.dp.cpp` |
| `STRATA_PF_XMX_ROWS` | 200 |  | STRATA_PF_XMX_ROWS moves it (from the code comment) | `sycl/src/prefill/prefill.cpp` |
| `STRATA_PROMPT_ATTN_XMX` | 0 |  | 1,258 to 952 / 913 | `sycl/src/kernels/cuda/qsa_prompt_attn_xmx.dp.cpp` |
| `STRATA_Q8_SM60` | off |  | (no description in the code) | `src/core/native_head.cpp`, `src/kernels/cuda/native_mmvq.cu` |
| `STRATA_Q8_SM60_BPS` | 0 |  | One workgroup per SM would leave too few rows in flight on a 48-SM die; four per SM is what the shapes of Qwen3.8-Flash-Next measured best (STRATA_Q8_SM60_BPS overrides it). (from the code comment) | `src/kernels/cuda/q8_sm60.cuh` |
| `STRATA_QI_GFX12_ATTN` | unset |  | [DRAFT] Switch of Strata's own WMMA attention in the image DiT code (strata-dit-step, strata-image) on gfx1200 / gfx1201 (RDNA4): on by default there, 0 turns it off. Measured on an R9700 at 1024 x 1024, 8 steps: 6.93 s  (from the settings register) | `src/image/dit/qi21_kernels.cu`, `tools/image/sdcpp_native/qi21-native.cu` |
| `STRATA_RDNA2_SGEMM` | 0 |  | Widens both GEMM inputs to FP32 (exact) and runs SGEMM, no 65504 limit. `=all` forces it on any AMD card. With `STRATA_HIP_PROMPT_F16=1` the FP16 route runs first and SGEMM takes the products it leaves. Slower than the FP16 route. | `src/prefill/gemm.cu` |
| `STRATA_ROCBLAS_TUNING` | unset |  | A rocBLAS solution table for the FP16 route; setup points the engine at the directory when a table for the architecture exists. The log says `rocBLAS tuning enabled (... rows ...)` or why not. See [rocBLAS table](../AMD_HIP.md#rocblas-table-gfx103x). | `src/prefill/gemm.cu` |
| `STRATA_ROCBLAS_VERBOSE` | unset |  | (no description in the code) | `src/prefill/gemm.cu` |
| `STRATA_ROCM_INDEX` | unset |  | (no description in the code) | `setup.py`, `tools/hip/build_windows.bat` |
| `STRATA_ROCM_KEEP` | unset |  | (no description in the code) | `tools/hip/build_windows.bat` |
| `STRATA_ROCM_VERSION` | unset |  | (no description in the code) | `setup.py`, `tools/hip/build_windows.bat` |
| `STRATA_SM120_PERF` | unset |  | Umbrella for sm_120 bit-exact kernels, +0.3..0.5%; documented. (from the settings register) | `include/strata/core/sm120_series.hpp` |
| `STRATA_SM121_PERF` | unset |  | (no description in the code) | `src/kernels/cuda/iq_kernels.cu` |
| `STRATA_SM60_PERF` | unset |  | (no description in the code) | `src/core/nv_defaults.cpp` |
| `STRATA_SM60_TABLE` | unset |  | Mode 8 on by default for GP100; SM60_TABLE=0 keeps the CUDA layout (0.1.43). (from the settings register) | `src/kernels/cuda/iq_kernels.cu`, `src/kernels/cuda/native_mmvq.cu` |
| `STRATA_SM61_TABLE` | unset |  | Pascal table switch; reports show no change; hardware workaround. (from the settings register) | `src/kernels/cuda/iq_kernels.cu`, `src/kernels/cuda/native_mmvq.cu` |
| `STRATA_SM70_TABLE` | unset |  | Turns on the three decode kernels in the second table above together. Bitwise the default kernels' output. Individual switches `STRATA_EXP_MODE`, `STRATA_MMVQ_IL` and `STRATA_GR_FAST` still apply on top. | `src/kernels/cuda/fused_gr.cu`, `src/kernels/cuda/iq_kernels.cu`, `src/kernels/cuda/native_mmvq.cu` |
| `STRATA_SM75_PERF` | unset |  | Umbrella for 22 ported sm_75 commits; users run =1. (from the settings register) | `include/strata/core/sm75_series.hpp` |
| `STRATA_SM86_PERF` | unset |  | (no description in the code) | `src/core/nv_defaults.cpp` |
| `STRATA_SPEC_ADAPT_SM86` | unset |  | sm_86 only: the P100 (sm_60) gained 4-16% on JSON/code/chat/tool calls but lost 3% on the 4K agent transcript (5/5 pairs), so there it stays opt-in Owner round 9 (2026-10-22): on the z440 A4000s the 5-pair A/B against... (from the code comment) | `src/core/nv_defaults.cpp` |
| `STRATA_SYCL_ADAPT_WINDOW_PRESENCE` | off |  | Opt-in (=1), off by default: the mirrored adaptive route histogram counts presence of an expert in a window instead of rows, to match the CPU usage counter's increment order (sycl/src/core/verify.cpp:2673). Read from the (from the settings register) | `sycl/src/core/verify.cpp` |
| `STRATA_SYCL_BF16_DET` | unset |  | Intel Arc: the prompt path's bf16 GEMMs with 64 or fewer outputs (ssm alpha/beta, hyper-connection inject) run on a fixed-order kernel because oneMKL's result changes from launch to launch for those shapes. 0 gives the o (from the settings register) | `sycl/src/prefill/gemm.dp.cpp` |
| `STRATA_SYCL_BIN` | unset |  | (no description in the code) | `serve/server.py` |
| `STRATA_SYCL_DEQUANT_GU_BOUNDED` | off |  | Opt-in (=1), off by default: bounded-launch IQ gate/up dequantization to FP16 (sycl/src/kernels/cuda/iq_kernels.dp.cpp:6643). Read from the code only; the integration lane measured nothing on it. (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_SYCL_DEQUANT_GU_SG` | 1 |  | Number of subgroups of the IQ gate/up dequantization; unset = 1 (the single-subgroup path); other values take the tiled kernel (dequant_gu_subgroups, sycl/src/kernels/cuda/iq_kernels.dp.cpp:6613). Read from the code only (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_SYCL_DEQUANT_GU_STATIC_ROW` | off |  | Opt-in (=1), off by default: static row indexing in the tiled IQ gate/up dequantization; only read when STRATA_SYCL_DEQUANT_GU_SG is not 1 (sycl/src/kernels/cuda/iq_kernels.dp.cpp:6647). Read from the code only; the inte (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_SYCL_DEQUANT_GU_STATIC_TYPE` | off |  | Opt-in (=1), off by default: static type dispatch in the tiled IQ gate/up dequantization; only read when STRATA_SYCL_DEQUANT_GU_SG is not 1 (sycl/src/kernels/cuda/iq_kernels.dp.cpp:6646). Read from the code only; the int (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_SYCL_EXPERT_MULTI16` | unset |  | Switch, off by default (non-zero turns it on): row reuse of the IQ2_XXS expert kernel (g_iq2xxs_reuse, sycl/src/kernels/cuda/iq_kernels.dp.cpp:4251). Read from the code only; the integration lane measured nothing on it. (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_SYCL_EXPERT_STAGE` | off |  | Opt-in (=1), off by default: staged expert reads in the verify path; needs device planning with STRATA_VERIFY_NO_HOST and not the Battlemage layout (sycl/src/core/verify.cpp:1659). Read from the code only; the integratio (from the settings register) | `sycl/src/core/verify.cpp` |
| `STRATA_SYCL_EXPERT_STAGE_COPY` | 8 |  | Tuning number of the staged expert copy of STRATA_SYCL_EXPERT_STAGE; the code's default is 8 (sycl/src/core/verify.cpp:1664). Read from the code only; the integration lane measured nothing on it. (from the settings register) | `sycl/src/core/verify.cpp` |
| `STRATA_SYCL_EXPERT_STAGE_GROUPS` | unset |  | Tuning string of the staged expert reads of STRATA_SYCL_EXPERT_STAGE: the group batching (sycl/src/core/verify.cpp:1663). Read from the code only; the integration lane measured nothing on it. (from the settings register) | `sycl/src/core/verify.cpp` |
| `STRATA_SYCL_F16_EXPERT_PAIR` | off |  | Opt-in (=1), off by default: FP16 expert pair kernel in the prompt path; needs a native layout, no MMQ and XMX mode 0 (sycl/src/prefill/prefill.cpp:4309). Read from the code only; the integration lane measured nothing on (from the settings register) | `sycl/src/prefill/prefill.cpp` |
| `STRATA_SYCL_GDN_PREFILL_REG_QK` | off |  | Opt-in (=1), off by default: register-resident QK variant of the GDN prompt kernel (sycl/src/prefill/kernels.dp.cpp:2762). Read from the code only; the integration lane measured nothing on it. (from the settings register) | `sycl/src/prefill/kernels.dp.cpp` |
| `STRATA_SYCL_GDN_QK_GRF256` | off |  | Opt-in (=1), off by default: 256-register-file variant of the GDN QK kernel for the 16 / 48 head shape with 10,240 conv channels (sycl/src/kernels/cuda/verify_kernels.dp.cpp:2244). Read from the code only; the integratio (from the settings register) | `sycl/src/kernels/cuda/verify_kernels.dp.cpp` |
| `STRATA_SYCL_GDN_STATE_COMMIT` | off |  | Opt-in (=1), off by default: GDN state-only commit in the verify path (sycl/src/core/verify.cpp:2129). Read from the code only; the integration lane measured nothing on it. (from the settings register) | `sycl/src/core/verify.cpp` |
| `STRATA_SYCL_GEMM_NO_STAGE` | off |  | "the tests negative control". (from the settings register) | `sycl/src/prefill/gemm.dp.cpp` |
| `STRATA_SYCL_GEMM_STAGE` | unset |  | Documented, off on Battlemage; =1 forces staging (hardware workaround). (from the settings register) | `sycl/src/prefill/gemm.dp.cpp` |
| `STRATA_SYCL_GR_CACHED_NORM` | 0 |  | Mode number, default 0 (off); only the value 3 selects the cached-norm mode (sycl/src/kernels/cuda/fused_gr.dp.cpp:77). Read from the code only; the integration lane measured nothing on it. (from the settings register) | `sycl/src/kernels/cuda/fused_gr.dp.cpp` |
| `STRATA_SYCL_GR_LO_ONCE` | unset |  | Switch, off by default (non-zero turns it on): the GR low-order part is computed once (g_gr_lo_once, sycl/src/kernels/cuda/fused_gr.dp.cpp:1150). Read from the code only; the integration lane measured nothing on it. (from the settings register) | `sycl/src/kernels/cuda/fused_gr.dp.cpp` |
| `STRATA_SYCL_GR_NORM_BOUNDS` | off |  | Opt-in (=1), off by default: direct bounds in the GR norm (g_gr_norm_direct_bounds, sycl/src/kernels/cuda/fused_gr.dp.cpp:570). Read from the code only; the integration lane measured nothing on it. (from the settings register) | `sycl/src/kernels/cuda/fused_gr.dp.cpp` |
| `STRATA_SYCL_GU16_METADATA_QUARTET` | off |  | Opt-in (=1), off by default: metadata-quartet gate/up dequantization for type 16 with n_ff 640 and n_embd 2560 (sycl/src/kernels/cuda/iq_kernels.dp.cpp:6547). Read from the code only; the integration lane measured nothin (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_SYCL_IQ4_LOOKUP` | unset |  | ON by default; =0 turns it off: the IQ4 matrix-vector kernel selects its table values by bit select, not a lookup, unless set to 0 (use_bit_select, sycl/src/kernels/cuda/native_mmvq.dp.cpp:3745). Read from the code only; (from the settings register) | `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_SYCL_IQ4_PAIR_REUSE` | unset |  | ON by default; =0 turns it off: pair reuse in the IQ4 matrix-vector kernel (sycl/src/kernels/cuda/native_mmvq.dp.cpp:3737). Read from the code only; the integration lane measured nothing on it. (from the settings register) | `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_SYCL_IQ4_Q8_SPLIT` | unset |  | ON by default; =0 turns it off: split Q8 activation handling in the IQ4 matrix-vector kernel (sycl/src/kernels/cuda/native_mmvq.dp.cpp:3741). Read from the code only; the integration lane measured nothing on it. (from the settings register) | `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_SYCL_IQ4_T4_ESIMD` | off |  | Opt-in (=1), off by default: ESIMD IQ4 kernel for 4 columns on Battlemage with n_in 2560 or 6144 (sycl/src/kernels/cuda/native_mmvq.dp.cpp:3758). Read from the code only; the integration lane measured nothing on it. (from the settings register) | `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_SYCL_IQ4_T4_XMX` | off |  | Opt-in (=1), off by default: XMX IQ4 kernel for 4 columns on the shapes 2560 to 6144 / 10240 / 12288 / 248320 and 6144 to 2560 (sycl/src/kernels/cuda/native_mmvq.dp.cpp:3764). Read from the code only; the integration lan (from the settings register) | `sycl/src/kernels/cuda/native_mmvq.dp.cpp` |
| `STRATA_SYCL_MIRROR_ADAPT` | off |  | Opt-in (=1), off by default: adaptive routing mirror on the SYCL verify path (sycl_mirror_adapt, sycl/src/core/verify.cpp:75). Read from the code only; the integration lane measured nothing on it. (from the settings register) | `sycl/src/core/verify.cpp`, `sycl/src/program/generate.cpp` |
| `STRATA_SYCL_PACKED_SIGNS` | off |  | Opt-in (=1), off by default: packed-sign IQ gate/up row-reuse kernel for the 2560 / 640 shape, only with the old kernels, split-multi and IQ2_XXS reuse off (sycl/src/kernels/cuda/iq_kernels.dp.cpp:5641). Read from the co (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_SYCL_Q2_WORD_EXPAND` | on |  | ON by default; =0 turns it off: the Q2 word-expand path of the IQ kernels is taken unless the variable starts with 0 (q2_word_expand_enabled, sycl/src/kernels/cuda/iq_kernels.dp.cpp:3706-3709: `return !(v && v[0] == '0') (from the settings register) | `sycl/src/kernels/cuda/iq_kernels.dp.cpp` |
| `STRATA_SYCL_QSA_CHUNK_DOWN` | unset |  | ON by default; =0 turns it off: chunked down projection of the QSA decode attention (sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp:963). Read from the code only; the integration lane measured nothing on it. (from the settings register) | `sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp` |
| `STRATA_SYCL_QSA_VALUE8_K8_VEC` | unset |  | ON by default; =0 turns it off: the vector variant of the QSA value kernel with 8 keys (sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp:2569). Read from the code only; the integration lane measured nothing on it. (from the settings register) | `sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp` |
| `STRATA_SYCL_TLA_ATTN` | off |  | Opt-in (=1), off by default, and compiled only in builds that define the macro of the same name: the TLA attention kernel for batch 32 with a 2,051 cap (sycl/src/prefill/prefill.cpp:3638). Read from the code only; the in (from the settings register) | `sycl/src/prefill/prefill.cpp` |
| `STRATA_VISION_HIP_ARCHS` | unset |  | (no description in the code) | `tools/hip/build_windows.bat` |

## Debug and test probes (195)

Not for normal use: dumps, parity checks, emulation, checks, traces and timing lines. Name and where it is read.

| Variable | Read in |
| --- | --- |
| `STRATA_ACT_DUMP` | `src/kernels/cpu/native_expert.cpp` |
| `STRATA_ACT_DUMP_COV` | `src/kernels/cpu/native_expert.cpp` |
| `STRATA_ACT_DUMP_LAYERS` | `src/kernels/cpu/native_expert.cpp` |
| `STRATA_ADAPT_CHECKRES` | `src/program/generate.cpp` |
| `STRATA_ADAPT_FETCH_CHECKRES` | `src/program/generate.cpp` |
| `STRATA_ADAPT_FETCH_VERIFY` | `src/program/generate.cpp` |
| `STRATA_ADAPT_LOG` | `src/program/generate.cpp` |
| `STRATA_BANG_AUDIT` | `src/program/generate.cpp` |
| `STRATA_BATCH_TRACE` | `serve/server.py` |
| `STRATA_BENCH_IN_PLACE` | `src/prefill/mmq_expert_bench.cpp` |
| `STRATA_BENCH_PLANAR` | `sycl/src/kernels/mmvq_bench.cpp` |
| `STRATA_BENCH_QMIXED` | `src/kernels/fused_gr_bench.cpp` |
| `STRATA_BENCH_STAGED` | `src/kernels/qsa_merge_daily_parity.cpp` |
| `STRATA_CHECK_SWAPS` | `sycl/src/program/generate.cpp` |
| `STRATA_COLD_TRACE` | `src/prefill/prefill.cpp` |
| `STRATA_COMMIT_TIMING` | `src/core/verify.cpp` |
| `STRATA_CRASH_BT` | `src/program/generate.cpp` |
| `STRATA_CVEC_FUSE_CHECK` | `src/prefill/prefill.cpp` |
| `STRATA_DBG_CPU_GATE` | `src/prefill/prefill.cpp` |
| `STRATA_DBG_DRAFT` | `sycl/src/program/generate.cpp` |
| `STRATA_DBG_FEAT` | `src/prefill/prefill.cpp` |
| `STRATA_DBG_FEAT_LAYERS` | `src/prefill/prefill.cpp` |
| `STRATA_DBG_FORCE_SLOW_LAYER` | `src/prefill/prefill.cpp` |
| `STRATA_DBG_GDN` | `src/kernels/cuda/verify_kernels.cu` |
| `STRATA_DBG_NAN` | `src/core/verify.cpp` |
| `STRATA_DBG_NAN_VERIFY` | `src/core/verify.cpp` |
| `STRATA_DBG_ROUTE_OVERRIDE` | `src/prefill/prefill.cpp` |
| `STRATA_DBG_TAIL_NOREUSE` | `src/program/generate.cpp` |
| `STRATA_DEBUG` | `serve/mcp.py` |
| `STRATA_DECODE_TIMING` | `src/program/generate.cpp` |
| `STRATA_DENSE_F16C_CHECK` | `src/prefill/gemm.cu` |
| `STRATA_DENSE_F16C_LOG` | `src/prefill/gemm.cu` |
| `STRATA_DF2_CHECK_REBUILD` | `metal/src/engine/dflash2.inc` |
| `STRATA_DF2_DUMP` | `metal/src/engine/dflash2.inc` |
| `STRATA_DF2_DUMP_ROUND` | `metal/src/engine/dflash2.inc` |
| `STRATA_DF2_LOG_MIB` | `metal/src/engine/dflash2.inc` |
| `STRATA_DF2_PROF` | `metal/src/engine/dflash2.inc` |
| `STRATA_DF2_PROFILE` | `src/core/dflash2.cpp` |
| `STRATA_DF2_RING_LOG` | `src/core/dflash2.cpp` |
| `STRATA_DF2_RING_STATS` | `src/core/dflash2.cpp` |
| `STRATA_DF2_STATS` | `metal/src/engine/dflash2.inc` |
| `STRATA_DF2_TAP_DUMP` | `src/program/generate.cpp` |
| `STRATA_DF2_TIMING` | `src/core/dflash2.cpp` |
| `STRATA_DISPATCH_STATS` | `include/strata/core/dispatch_stats.hpp` |
| `STRATA_DOORBELL_CHECK` | `sycl/src/kernels/cuda/elementwise.dp.cpp` |
| `STRATA_DRAFT_LOG` | `src/program/generate.cpp` |
| `STRATA_DRAFT_TIMING` | `src/core/mtp.cpp` |
| `STRATA_DUMP_CTX` | `tools/qwen35/q36_llama_dump.cpp` |
| `STRATA_DUMP_FIRST_LOGITS` | `src/program/generate.cpp` |
| `STRATA_DUMP_MOE_INPUT` | `src/prefill/prefill.cpp` |
| `STRATA_DUMP_MOE_LAYER` | `src/prefill/prefill.cpp` |
| `STRATA_DUMP_NGL` | `tools/qwen35/q36_llama_dump.cpp` |
| `STRATA_DUMP_SEL` | `sycl/src/prefill/prefill.cpp` |
| `STRATA_DUMP_WINDOW_TOP2` | `sycl/src/program/generate.cpp` |
| `STRATA_EMULATE_CC` | `include/strata/core/emulate.hpp` |
| `STRATA_EMULATE_RAM_GIB` | `src/platform/memory.cpp` |
| `STRATA_EMULATE_SMS` | `include/strata/core/emulate.hpp` |
| `STRATA_ESIMD_DBG` | `sycl/src/kernels/cuda/qsa_prompt_attn_esimd.dp.cpp` |
| `STRATA_FILE_TIER_PROBE` | `src/core/expert_source.cpp` |
| `STRATA_FORCE_AVX2` | `src/kernels/cpu/expert_layout.cpp` |
| `STRATA_FORCE_FAULT` | `src/core/verify.cpp` |
| `STRATA_FORCE_ISA` | `src/kernels/cpu/expert_layout.cpp` |
| `STRATA_FOUR_CARD_PROFILE` | `strata_control/amd_four_card.py` |
| `STRATA_FS_VERIFY` | `src/core/expert_source.cpp` |
| `STRATA_GDN_CHUNK_PROF` | `src/prefill/gdn_chunk.cu` |
| `STRATA_GDN_PARITY_REPS` | `src/kernels/gdn_parity.cpp` |
| `STRATA_GDN_PARITY_TRACE` | `src/kernels/gdn_parity.cpp` |
| `STRATA_GDN_VERIFY_BENCH` | `sycl/src/kernels/gdn_parity.cpp` |
| `STRATA_GFX906_ATTN_QUERY_SWIZZLE_TRACE` | `src/kernels/cuda/qsa_decode_attn.cu` |
| `STRATA_GFX906_ATTN_REDUCE12_TRACE` | `src/kernels/cuda/qsa_decode_attn.cu` |
| `STRATA_HC_CHECK` | `sycl/src/kernels/cuda/fused_gr.dp.cpp` |
| `STRATA_HC_UPMIX_CHECK` | `src/prefill/prefill.cpp` |
| `STRATA_HEAD_IN_DUMP` | `src/core/verify.cpp` |
| `STRATA_HEAD_SCREEN_CHECK` | `src/core/verify.cpp` |
| `STRATA_I8_DBG` | `sycl/src/prefill/esimd_moe.dp.cpp` |
| `STRATA_IDX_FP16_CHECK` | `src/prefill/prefill.cpp` |
| `STRATA_INJECT_FATAL` | `src/program/generate.cpp` |
| `STRATA_INJECT_FATAL_ONCE` | `src/program/generate.cpp` |
| `STRATA_IO_STATS` | `src/program/generate.cpp` |
| `STRATA_KVA_PARITY` | `include/strata/prefill/kva.hpp` |
| `STRATA_KVG_CHECK` | `src/program/generate.cpp` |
| `STRATA_KV_RESOLVE_PROF` | `src/kernels/cuda/kv_stream.cu` |
| `STRATA_LAUNCH_PROBE` | `src/core/verify.cpp` |
| `STRATA_LAYER_SPLIT_PROBE` | `src/program/generate.cpp` |
| `STRATA_LOGITS_DUMP` | `src/core/verify.cpp` |
| `STRATA_LOOKAHEAD_STATS` | `src/program/generate.cpp` |
| `STRATA_METAL_ALLOC_LOG` | `metal/src/gpu/runtime.cpp` |
| `STRATA_METAL_EXPERT_PROFILE` | `metal/src/engine/window.cpp` |
| `STRATA_METAL_MSL_DUMP` | `metal/src/mtl_rt.mm` |
| `STRATA_METAL_PROFILE` | `metal/src/engine/prompt_chain.inc` |
| `STRATA_METAL_SAMPLE_STATS` | `metal/src/engine/window.cpp` |
| `STRATA_METAL_TRACE` | `metal/src/engine/prompt_chain.inc` |
| `STRATA_MMQ_DIRECT_LOG` | `src/prefill/prefill.cpp` |
| `STRATA_MMVF_CHECK` | `src/kernels/cuda/native_bf16.cu` |
| `STRATA_MTP_DUMP` | `src/core/mtp.cpp` |
| `STRATA_MTP_PROFILE` | `src/core/mtp.cpp` |
| `STRATA_P4_TIMING` | `include/strata/program/pipe_probe.hpp` |
| `STRATA_P4_TRACE` | `include/strata/program/pipe_probe.hpp` |
| `STRATA_PARITY_DUMP` | `src/kernels/route_resident_parity.cu` |
| `STRATA_PARK_TRACE` | `src/core/conversation_snapshot.cpp` |
| `STRATA_PF_ALLOC_TRACE` | `src/prefill/prefill.cpp` |
| `STRATA_PF_DM_PROBE` | `src/prefill/prefill.cpp` |
| `STRATA_PF_OWN_CHECK` | `src/program/generate.cpp` |
| `STRATA_PF_OWN_PROBE_OFF` | `src/program/generate.cpp` |
| `STRATA_PF_REFILL_BG_LOG` | `src/program/generate.cpp` |
| `STRATA_PF_TILE_DEBUG` | `src/prefill/moe_fused_iq.cu` |
| `STRATA_PIPELINE_DEBUG` | `src/program/generate.cpp` |
| `STRATA_PIPELINE_DUMP` | `sycl/src/core/verify.cpp` |
| `STRATA_PIPELINE_FORCE_MISS` | `src/program/generate.cpp` |
| `STRATA_PIPELINE_LOG` | `src/program/generate.cpp` |
| `STRATA_PIPELINE_TRACE` | `src/program/generate.cpp` |
| `STRATA_PLE_CHECK` | `src/kernels/ngram.cpp` |
| `STRATA_PLE_PROBE` | `src/program/generate.cpp` |
| `STRATA_PLE_TRACE` | `src/ngram/ple_reader.cpp` |
| `STRATA_POOL_TRACE` | `src/core/expert_source.cpp` |
| `STRATA_PREFILL_DUMP_R` | `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_DUMP_R_ALL` | `src/prefill/prefill.cpp` |
| `STRATA_PREFILL_TIMING` | `src/prefill/prefill.cpp` |
| `STRATA_PRIVATE_MTP_DEMAND_LOG` | `sycl/src/program/generate.cpp` |
| `STRATA_PROFILE_DIR` | `strata_control/profiles.py` |
| `STRATA_PROFILE_PRIOR_ENTRIES` | `sycl/src/program/generate.cpp` |
| `STRATA_Q36_PLACE_DEBUG` | `src/qwen35/q36_engine.cu` |
| `STRATA_Q36_VDEBUG` | `src/qwen35/q36_engine.cu` |
| `STRATA_Q36_VERIFY` | `src/qwen35/q36_engine.cu` |
| `STRATA_QI_TIMING` | `tools/image/sdcpp_native/qwen_image_2_1_native.hpp` |
| `STRATA_QSA_DUMP` | `src/prefill/prefill.cpp` |
| `STRATA_QSA_DUMP_LAST` | `src/prefill/prefill.cpp` |
| `STRATA_QSA_DUMP_LAYER` | `src/prefill/prefill.cpp` |
| `STRATA_QSA_DUMP_STEPS` | `src/prefill/prefill.cpp` |
| `STRATA_QSA_I8_CHECK` | `src/kernels/cuda/qsa_select.cu` |
| `STRATA_REPLICA_PROFILE` | `strata_control/replica_profile.py` |
| `STRATA_ROUTE_DUMP` | `src/core/expert_source.cpp` |
| `STRATA_ROUTE_STATS` | `src/core/verify.cpp` |
| `STRATA_RSS_TRACE` | `src/program/generate.cpp` |
| `STRATA_RUNTIME_PROFILE` | `serve/lifeline.py` |
| `STRATA_SCORES_THREAD_TRACE` | `src/kernels/cuda/qsa_select.cu` |
| `STRATA_SELECT_TWO_PASS_CHECK` | `src/kernels/cuda/qsa_select.cu` |
| `STRATA_SESSION_GOLDEN_PRINT` | `src/core/conversation_file_test.cpp` |
| `STRATA_SESSION_GROW_CHECK` | `src/core/layer.cpp` |
| `STRATA_SNAPSHOT_VERIFY` | `src/program/generate.cpp` |
| `STRATA_SPEC_TRACE` | `src/program/generate.cpp` |
| `STRATA_SPEC_TRACE_NOPICK` | `src/program/generate.cpp` |
| `STRATA_SPLIT_COMPUTE_PROBE` | `src/program/generate.cpp` |
| `STRATA_SPLIT_TIMING` | `src/program/generate.cpp` |
| `STRATA_STATE_DUMP` | `sycl/src/core/verify.cpp` |
| `STRATA_SYCL_PREFILL_COPY_PROFILING` | `sycl/src/prefill/prefill.cpp` |
| `STRATA_SYCL_ROUTING_AUDIT_FILE` | `sycl/src/core/verify.cpp` |
| `STRATA_TEST_BELL_SLEEP_US` | `src/program/generate.cpp` |
| `STRATA_TEST_BIG_GDN` | `src/core/conversation_snapshot_test.cpp` |
| `STRATA_TEST_CACHE_FAIL` | `src/program/generate.cpp` |
| `STRATA_TEST_COMPLEMENT_REFUSE` | `src/core/expert_source.cpp` |
| `STRATA_TEST_CONC_ON_HOST_CPU` | `src/program/generate.cpp` |
| `STRATA_TEST_COPY_FAIL_AT` | `src/core/conversation_copy.hpp` |
| `STRATA_TEST_GRAPH_OOM` | `src/core/verify.cpp` |
| `STRATA_TEST_HANDOFF_DROP` | `src/core/verify.cpp` |
| `STRATA_TEST_PAGEABLE` | `include/strata/core/pinned_ring.hpp` |
| `STRATA_TEST_PINNED_SHARE` | `src/program/generate.cpp` |
| `STRATA_TEST_PIN_BUDGET_MIB` | `include/strata/core/pin_budget.hpp` |
| `STRATA_TEST_PIN_LEFT_KIB` | `include/strata/core/pin_budget.hpp` |
| `STRATA_TEST_SLOW_COPY_US` | `src/core/conversation_copy.hpp` |
| `STRATA_TEST_SYSFS_ROOT` | `src/kernels/cpu/pool.cpp` |
| `STRATA_TEST_UNMAP_GGUF` | `src/core/expert_source.cpp` |
| `STRATA_TEST_VERIFY_STALL` | `src/core/verify.cpp` |
| `STRATA_TRACE` | `src/core/native_dense.cpp` |
| `STRATA_TRACE_ADAPT` | `src/program/generate.cpp` |
| `STRATA_USAGE_DUMP` | `sycl/src/program/generate.cpp` |
| `STRATA_VERIFY_ALL_RESIDENT` | `src/core/verify.cpp` |
| `STRATA_VERIFY_ALL_SLOTS` | `sycl/src/program/generate.cpp` |
| `STRATA_VERIFY_ARENA` | `src/core/expert_source.cpp` |
| `STRATA_VERIFY_ASYNC_LAUNCH` | `src/core/verify.cpp` |
| `STRATA_VERIFY_COHERENT` | `src/core/verify.cpp` |
| `STRATA_VERIFY_COMPLEMENT` | `src/core/expert_source.cpp` |
| `STRATA_VERIFY_DEBUG` | `src/core/verify.cpp` |
| `STRATA_VERIFY_DEVICE_PLAN` | `src/core/expert_source.cpp` |
| `STRATA_VERIFY_DUMP` | `sycl/src/program/generate.cpp` |
| `STRATA_VERIFY_EAGER` | `sycl/src/core/verify.cpp` |
| `STRATA_VERIFY_EARLY_PLE_FLAG` | `src/core/verify.cpp` |
| `STRATA_VERIFY_FIND` | `sycl/src/program/generate.cpp` |
| `STRATA_VERIFY_HOST_UNCACHED` | `sycl/src/core/verify.cpp` |
| `STRATA_VERIFY_LEAN` | `sycl/src/core/verify.cpp` |
| `STRATA_VERIFY_NODES` | `src/core/mtp.cpp` |
| `STRATA_VERIFY_NO_HOST` | `sycl/src/core/verify.cpp` |
| `STRATA_VERIFY_PIPE` | `sycl/src/core/verify.cpp` |
| `STRATA_VERIFY_PROFILE` | `src/core/verify.cpp` |
| `STRATA_VERIFY_PROFILE_EVERY` | `sycl/src/core/verify.cpp` |
| `STRATA_VERIFY_QDEDUP` | `src/core/verify.cpp` |
| `STRATA_VERIFY_SPLIT` | `sycl/src/core/verify.cpp` |
| `STRATA_VERIFY_STEPPED` | `sycl/src/core/verify.cpp` |
| `STRATA_VERIFY_TRACE` | `src/core/verify.cpp` |
| `STRATA_VISION_ARCH_CHECK` | `tools/vision/strata_vision.cpp` |
| `STRATA_VISION_GPU_LOG` | `src/program/generate.cpp` |
| `STRATA_VISION_GPU_TEST_FAIL` | `src/program/generate.cpp` |
| `STRATA_VIT_PROF` | `serve/server.py` |
| `STRATA_VIT_TEST_FAIL` | `tools/vision/own/own_reader.cpp` |
| `STRATA_WIN_LOG` | `src/program/generate.cpp` |
