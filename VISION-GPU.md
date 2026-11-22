# The picture reader on the GPU, only while it reads (vision/gpu-on-demand)

Base: main 61c2451dd (= 0.2.15-nvfp4.4). The worktree is D:\Projects\strata-vision-gpu, branch vision/gpu-on-demand. Nothing is
committed. The patch is bench-vision\vision-gpu.patch (14 files). Every run was on an RTX 5090 under WDDM, headless, at a
**power limit of 600 W** (recorded beside each run in bench-vision\out\*.txt and ident_*.txt). The VRAM overclock (14.5 GHz)
was left as it was.

## Decision

The user chose design C: the encoder runs inside strata.exe, in borrowed expert slots, and the slots are refilled right after
the picture. Prototypes A (upstream's elastic reader process) and B (a persistent reader process) were not built. A's
per-picture process start was not measured either: it would first have needed a ggml-cuda reader build for Windows.

## What was built

**Engine (strata.exe, CUDA only).**
- New flags: `--vision-gpu <mmproj>`, `--vision-max-tokens N` (default 1024), `--vision-min-tokens N`, `--vision-gpu-preload`.
- New serve command `VENC <image> <out>`, sent between requests. It answers `VOK n nx ny ms` or `ERR <reason>`.
- The reader is the own tower's GPU path (`tools/vision/own/vit_gpu.cu`: cuBLAS, F32 attention, no ggml), linked as
  `strata_vit_engine`.
- The new `VitGpu` modes:
  - `stage` lays the weights out once in pinned RAM (cudaHostAlloc, 862 MiB) in exactly the form the device holds them.
  - `arena_need` gives the room a picture needs.
  - `bind` makes one H2D copy into memory the caller lends, runs the BF16->F16 conversion kernel, sets cuBLAS's workspace in
    that room and rebases the weight pointers. `unbind` undoes this; the cuBLAS handle is destroyed with it.
  - Without these modes the reader program (DIRECT mode) is unchanged.
- `VENC` (generate.cpp, beside `VRAM`), step by step:
  1. Read and prepare the picture, using the same image_prep as the CPU reader.
  2. Quiesce: `ver.wait_commit`, `adapt_quiesce`, `bgr_poll(true)`, then a sync.
  3. Take the LAST k slots of the cache, the end the prompt path lends. They must be at least `g_lend_keep` slots and never reach
     into a grown K/V. Mark their experts non-resident (`res_upload`).
  4. Encode and write the SVE1 file.
  5. Refill those slots from RAM at once. This uses the PF_REFILL_BG stream and code (hottest first, cudaMemcpyBatchAsync,
     batched events, landed batches handed over at the window boundaries). Where that path is off, the refill is queued and
     waited for. A failed refill exits the engine, so the server restarts it.
- It refuses, and the server uses the CPU reader instead, when: the engine runs a layer split, `--peer-device` or `--batch`
  slots; it is in the resident low-RAM mode; a loan is still out; there is no room. `STRATA_VISION_GPU_TEST_FAIL` refuses on
  purpose, for the tests.
- `--vision-gpu-preload` pins the weights at the start and sends one 256x256 grey picture through the whole path.
  `STRATA_VISION_GPU_LOG=0` silences its log lines.

**Server (serve/server.py).**
- `vision_in_engine(cfg)` is true when: the vision section has `"in_engine": true`; the backend is CUDA on one GPU with no
  parallel slots; it is not a Mac; and `STRATA_VISION_IN_ENGINE` is not `0` (the switch back to the CPU reader).
- `engine_args` then adds `--vision-gpu <mmproj> --vision-max-tokens N`, plus `--vision-gpu-preload` for `"preload": true`.
- `StrataEngine.vision_encode()` sends the command. `EngineVision(Vision)`:
  - reads through the engine under the request FIFO;
  - keeps the same cache of encoded pictures and the same files;
  - starts the CPU reader of the section on the first refusal, and asks the engine again at the next picture. An engine without
    the option is not asked again.

**Setup and bundle.**
- `setup.py` (the NVFP4 path): with pictures on (`--vision yes`/`gpu`, or the question answered y) and an engine that reads
  pictures, it writes `"in_engine": true, "preload": true` beside the CPU reader. `--vision cpu` keeps the CPU reader alone.
- How setup knows the engine reads pictures (`engine_reads_pictures`): BUILD.json `vision_gpu`, or `--vision-gpu MMPROJ` in a
  local build. `in_engine` and `preload` are setup's keys (`SETUP_VISION`).
- `strata_control/vision_reader.py` gets `in_engine_wanted`.
- `release/make_windows_bundle.py` writes `"vision_gpu": true`. The bundle gets no new file: the cuBLAS DLLs ship already and
  strata-vision.exe stays as the fallback.
- Size: the vit_gpu.cu object is 0.67 MB for sm_120. Inferred: about 3 MB more in strata.exe for the release's 4 architectures,
  against a 553 MB zip.

**docs/NVFP4.md**: a paragraph under "Images".

## Phase 1: baseline (release exe through serve, GPTQ tray profile, CPU reader)

The runs are out/base1, base2, mbase1 and mbase2 (600 W). TTFT is the image turn's time to the first token: encode, then the
prompt read.

| picture | image tokens | CPU reader's own time | image-turn TTFT | prompt read in it |
| --- | ---: | ---: | ---: | ---: |
| shapes 640x400 | 260 | 714-748 ms | 1,281-1,320 ms | 533-557 ms |
| big1024 1024x1024 | 1,024 | 3,933-5,472 ms | 4,586-4,922 ms | 782-1,012 ms |
| code4217 800x450 | 350 | 1,030-1,564 ms | 1,580-1,692 ms | 569-597 ms |

- The reader's own times come from its `OK` lines (16 threads). The high ends of the ranges were measured beside other
  agents' CPU work.
- Decode of a picture-free 400-token turn: 228-281 tok/s. The first turn after start is lower; the conversation cache then
  serves the text turns.
- Decode after an image turn: 247-267 tok/s, the same.
- The engine holds 8,189 expert slots (21.09 GiB). nvidia-smi shows 31.4 GB used at the steady state, with about 0.4-0.5 GB
  free.

## Phase 3: the result (same hour, interleaved, 600 W)

The runs are mbase1, meng1, mbase2 and meng2; the in-engine arms use `"preload": true`. Each arm reads three pictures between
text turns (t,t,s,t,b,t,c,t). "Text" is the 400-token turns.

| | CPU reader (release) | in the engine |
| --- | ---: | ---: |
| TTFT, 260 tokens | 1,286 / 1,281 ms | **622 / 641 ms** |
| TTFT, 1,024 tokens | 4,592 / 4,719 ms | **951 / 938 ms** |
| TTFT, 350 tokens | 1,588 / 1,580 ms | **669 / 653 ms** |
| text tok/s, before any picture (2 turns) | 236.5, 261.3 / 228.4, 255.8 | 243.3, 276.1 / 227.5, 272.9 |
| text tok/s, right after each picture (3 turns) | 262.4, 260.6, 259.8 / 259.3, 266.6, 252.1 | 271.4, 249.7, 256.3 / 264.1, 260.2, 261.4 |
| expert slots | 8,189 | 8,189 |
| nvidia-smi steady state | 31,428-31,456 MiB | 31,580 MiB (see below) |

Engine-side time per picture, warm (the `picture read on the GPU` lines):

| picture | total | lend wait | weight upload | encode | slots borrowed | refill done after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 260 tokens | 42-55 ms | 0-17 | 16-17 | 14-22 | 357 (940 MiB) | 20-35 ms |
| 1,024 tokens | 108-128 ms | 0-15 | 16-17 | 82-83 | 375 (987 MiB) | 21-34 ms |
| 350 tokens | 46-66 ms | 0-15 | 16-17 | 18-23 | 359 (946 MiB) | 20-33 ms |

- "Refill done after" counts from the end of the encode. In serve, the image turn's prompt read waited for it ("the last waited
  for"), so the prompt starts with every slot back.
- The lend wait is a previous refill still landing, which happens only when pictures come back to back (the direct VENC client).

**First picture.**
- Without preload it costs +0.7 s for the weights from the page cache and +0.5 s for the GPU's first use (the kernels' module
  load and cuBLAS's first batched GEMM). The image-turn TTFT was then 1,923 ms (eng1).
- With preload, the start takes +0.72 s (pin) and +0.47 s (warm-up) instead. The first picture is then as fast as the rest:
  622-641 ms.
- In the very first direct run (v1) the first picture's encode took 14.4 s. It never came back in 5 later engine starts.
  Inferred cause: cold cuBLAS DLL pages.

**VRAM outside the slots.**
- After the first picture, 74 MiB stay allocated: kernel code that stays loaded. It comes out of the engine's reserve, not the
  slots (measured with cudaMemGetInfo around VENC).
- With the cuBLAS handle kept, it was 138 MiB; now the handle is destroyed after each picture, which measured no slower.
- Later pictures add +3 MiB or nothing. The pinned weights take 862 MiB of RAM, not VRAM.

**Accuracy against the release CPU reader (same pictures, the own tower on both sides).**

| picture | row cosine min | mean | max abs | rel L2 |
| --- | ---: | ---: | ---: | ---: |
| 260 tokens | 0.999234 | 0.999993 | 0.0056 | 3.5e-3 |
| 1,024 tokens | 0.999515 | 0.999998 | 0.0033 | 1.8e-3 |
| 350 tokens | 0.999530 | 0.999994 | 0.0026 | 3.5e-3 |

- Two GPU reads of the same picture are bit-identical.
- Greedy answers through serve (temperature 0):
  - code4217: "CODE 4217" on both readers.
  - shapes: the same text hash as the CPU reader in the same hour (702b58 and 42bd0f).
  - big1024: the same content, different wording ("line items" against "item lines"; each reader is stable across its own runs).
- Serve's text is not bit-stable run to run even on the release, so the strict check is the CLI identity below.

**CPU fallback** (out/fail1, `STRATA_VISION_GPU_TEST_FAIL=1`): the engine refused, the server printed one line, started the
CPU reader (own tower, 16 threads) and answered the picture.

## Identity (picture-free) and speed with no picture

`bench-vision/ident_v.sh` uses rel04/ident.sh's method: exact.sh, CLI fixed placement, STRATA_SPEC_FIXED=1.

| case | release reference | this build (vg1, vg2) |
| --- | --- | --- |
| 16K defaults, text / first-token logits | 93c4688100 / e9d907e164 | 93c4688100 / e9d907e164 |
| decode conv D (new prefill switches =0) | 26c5e96e8e | 26c5e96e8e |
| decode conv P8 (MTP rt-tuned, spec 8) | 26c5e96e8e | 26c5e96e8e |

Conv D speed, the same hour, alternating, 600 W: release 124.2 / 127.2 tok/s, this build 129.7 / 128.5 tok/s (the same within
noise).

## Tests

| suite | result |
| --- | --- |
| new: serve/test_vision_in_engine.py | 10 tests OK |
| tools/test_vision_reader.py (+3 InEngine) | 31 OK |
| tools/test_setup_fork_engine.py (+3 EngineReadsPictures) | 14 OK |
| serve/test_vision_* (busy, ctx_retry, elastic, elastic_http, no_room, ready_pipe, start_error, token_limit, tool_image, workdir), serve/test_images | OK |
| tools/test_setup_choices, test_fork_update_check, test_setup_hotfix_tag | OK |

- Failures that are the same on the release tree:
  - serve/test_vision_cpu_fallback: 3 errors (WinError 32 on a temp log);
  - tools/test_setup_nvfp4: 4 failures (disk-size checks on this PC).
- Not run: the full serve suite (~1,700 tests), a 4-arch release build, the bundle and the tray.

## Measured against inferred

- **Measured:** every number in the tables, the identity hashes, the cosine figures, the fallback, the 74/138 MiB, the pinning
  and warm-up times.
- **Inferred, not measured:**
  - the exe-size growth of a 4-arch build;
  - the cause of the one 14.4 s first encode;
  - that the 74 MiB of module code cannot hurt a card that is full to its reserve;
  - other GPUs (the release's sm_75/86/89);
  - pictures above 1,024 tokens (not tried; the borrowed room is sized per picture by arena_need).

## Files

- bench-vision\vb.py, run.sh: a serve run with the tray profile (out\TAG.json / .log / .smi).
- venc_client.py, vc.sh: the engine directly, VENC x2 per picture.
- cmp_sve.py: embedding comparison.
- ident_v.sh: identity.
- sve\: the CPU and GPU embeddings.
