# Strata NVFP4

**Qwen3.8-Flash-Next (125B hybrid MoE) in NVFP4 on one RTX 20, 30, 40 or 50 card (12 GB of VRAM or more; built
and measured on an RTX 5090) + 64 GB of RAM or more, text and pictures.** A fork of
[Niko1221/Strata](https://github.com/Niko1221/Strata) that runs a ModelOpt **NVFP4** checkpoint — here
[OrcaRouter's abliterated Flash-Next](https://huggingface.co/jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4) —
instead of the Q2/Q3 quants Strata ships for. NVFP4 keeps the experts at 4.5 bits with calibrated scales, which is
why this fork exists: the 2-3 bit quants were noticeably less accurate on the same model.

Strata itself keeps the routed experts in RAM (63 GiB here; with less than 92 GiB of RAM installed, a 64 GB PC, this
fork keeps only the ones outside VRAM and reads the rest from the SSD), caches the most-used ones in VRAM, computes the
misses on the CPU and over PCIe in parallel with the GPU, and decodes with an MTP draft head. Everything about that
design is upstream's; the original README is kept as [README.upstream.md](README.upstream.md).

## Quick start

```bat
git clone https://github.com/sergqwer/strata-nvfp4
cd strata-nvfp4
START-HERE.bat
```

(Linux: `./setup.sh`.) Setup checks the PC, asks which model, how much context and whether the model should read
images, installs this fork's engine and starts the model on http://127.0.0.1:8080 (OpenAI and Anthropic APIs, a
chat page). It offers:

| `--family` | the model | license |
| --- | --- | --- |
| `huihui-nvfp4` (the default) | [huihui-ai's abliterated Qwen3.8-Flash-Next](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated), every expert NVFP4 by GPTQ: [Maximilian228/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-GPTQ-Strata](https://huggingface.co/Maximilian228/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-GPTQ-Strata) | Qwen Community License 1.0 |
| `orca-nvfp4` | [OrcaRouter's uncensored Flash-Next](https://huggingface.co/OrcaRouter/Qwen3.8-Flash-Next-Uncensored), the same quantization: [Maximilian228/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-NVFP4-GPTQ-Strata](https://huggingface.co/Maximilian228/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-NVFP4-GPTQ-Strata) | Qwen Community License 1.0 (the LICENSE OrcaRouter ships; its card says Apache 2.0) |
| `qwen`, `swift`, `coder`, `unsloth` | upstream Strata's GGUF models: ISTA-DASLab's GSQ-RCO quants of Qwen3.8-Flash-Next, Swift 1.5, the Coder, Unsloth's ~4-bit files | as upstream lists them |

- **The NVFP4 models come ready-made:** the experts pack, the dense GGUF, the FP8 n-gram table, the BF16 embedding
  and the fine-tune's MTP draft head are downloaded from Hugging Face at a pinned revision and checked against their
  SHA-256; nothing is converted on your PC. They need an RTX 20 card or newer with 12 GB of VRAM, 64 GB of RAM and
  ~130 GB of disk (see [Requirements](#requirements)). Images go through ISTA-DASLab's image encoder, on the CPU.
- **`huihui-nvfp4` is the default** where the PC meets their requirements; a PC below them gets a GGUF model as the
  default, and `--family huihui-nvfp4 --yes` installs one anyway.
- **The engine is always this fork's:** the ready-made one from this repository's releases (Windows), else compiled
  from this source. Upstream's ready-made engine has no NVFP4 path, so setup never installs it, and replaces one an
  older setup installed (one that cannot be replaced - no release engine for this card or OS yet - is kept for the
  GGUF models, and setup says so once).
- **AMD cards need an engine you build:** this fork publishes no AMD engine and setup does not fall back to
  upstream's. On Linux setup compiles it, as upstream's does; on Windows build it first - `tools\hip\build_windows.bat`
  makes `strata-windows-x64-hip.zip`, then `START-HERE.bat --backend hip --prebuilt <its dist folder>`
  ([docs/AMD_HIP.md](docs/AMD_HIP.md)). The NVFP4 models run on NVIDIA cards only.
- `--family orca-nvfp4` picks a model without the menu, `--dry-run` shows what setup would download, install and
  write and changes nothing, `--check` says what fits this PC. `UPDATE.bat` (`./update.sh`) updates a clone: `git
  pull`, then the engine of the new release. The release zip has its own `update.cmd`.

Setting it all up by hand - the engine, and converting the ModelOpt checkpoint yourself - is under
[Build it yourself](#build-it-yourself-advanced).

## How it differs from upstream Strata

- **Runs an NVFP4 checkpoint** (ModelOpt, 4.5-bit experts with calibrated scales) instead of Strata's Q2/Q3 quants:
  lossless repack, per-expert FP32 scales kept and applied to each projection's output.
- **Nothing is rounded coarser than the checkpoint where it can be avoided:** the n-gram (PLE) table in its shipped
  FP8 (upstream reads it as IQ4_NL, 8% off per row), the token embedding in its shipped BF16, RoPE angles from a
  float64 table in every kernel (upstream's fast-math angles were ~0.02 rad off at 262K), FP32-exact inputs to the
  prompt path's BF16 projections (router, indexer, gates), int8 KV behind a Hadamard rotation.
- **An int8 (W4A8) prompt path for Blackwell:** llama.cpp builds NVFP4 MMQ as FP4 x FP4 on sm_120; this keeps
  8-bit activations - 16x smaller error per product - at 86% of its speed.
- **NVFP4 CPU kernels** for the CPU share of the experts: AVX-512 (1.8-3.7x ggml-cpu) and, on a CPU without it,
  AVX2 (2.0-2.4x at 4-8 tokens, every row bit-equal to ggml-cpu's own AVX2 dot).
- **Faster start:** experts read unbuffered into the pinned arena on their own thread, registered with CUDA one layer
  ahead of the readers, beside everything else - the arena is in at ~7 s from a PCIe 5 drive.
- **The K/V grows with the context:** at a 262K window upstream allocates the whole context's K/V at start (3.35 GiB
  here with the draft layer's) - room for ~1,200 more experts in VRAM. Here the K/V takes VRAM only as the requests
  reach further, from the expert cache, and gives it back after them (CUDA virtual memory: no address moves, no graph
  is captured again). The window is never smaller; `--no-kv-grow` allocates it whole.
- **Faster prompt reading:** an MMQ group's experts gathered in one launch after one wait (under WDDM every streamed
  expert's wait and event had left ~10 us of GPU idle, 226 ms of a 32K prompt), the n-gram rows read 256 at a time
  and beside layer 0, upstream's split hyper-connection kernels on.
- **Small prompt chunks shared with the CPU:** an agent's turn (a tool result, a test's output) was a fixed ~0.6 s
  of PCIe copies of the experts it routes to. The idle decode pool now computes the experts few of its tokens reach,
  from RAM, while the rest stream: at 95K context 229 new tokens 605 -> 477 ms, 1,193 1,279 -> 788 ms. The share
  is measured each layer (where the CPU and the GPU end together), so a smaller CPU takes less and never makes a
  prompt slower; the prompt reads as close to the FP16 path as before.
- **The PCIe share of decode's misses is measured too:** each layer's count from fitted costs instead of a fixed
  share, so a slow CPU sends more over the link (one pool worker, emulated: 34.7 -> 20.8 ms a round) and a
  DRAM-bound one, as here, sends few.
- **An adaptive VRAM tier with a longer memory:** it re-ranks the experts every 2 rounds, up to 192 swaps, and the
  routing counts fade x0.92 per pass. Upstream re-ranks every 4 rounds, up to 96, x0.7. In fixed 1,000-token runs on
  the 5090 it missed 30-40% fewer experts and the round did not change measurably. About half of those runs' tokens
  came after the answer had ended, though, where the model loops over a few experts, so the gain on an answer alone
  is not measured yet. It applies to a cache holding 20-60% of the experts with every expert in RAM. Elsewhere
  upstream's settings stay; `--adapt-every`, `--adapt-swaps` and `--adapt-decay` override.
- **Tuned for NVFP4's larger experts** (PCIe share, prompt chunks up to 32K, fused scale passes, a verify commit
  that overlaps the draft) and the fine-tune's own abliterated MTP draft head.
- **A draft vocabulary with Cyrillic:** the MTP draft head proposes only tokens of its subset, and upstream's held
  142 of the vocabulary's 18,580 Cyrillic tokens - an answer in Cyrillic decoded at 83 tokens/s with 1.4 tokens a round;
  with the whole Cyrillic script (`tools/draft_vocab.py --add cyrillic`), 109 and 2.1. English is unchanged.
  Upstream's CJK subset is one `--add cjk` away (`data/draft_vocab_en.bin` is the English/code one).
- **64 GB of RAM is enough:** with less than 92 GiB installed the engine runs upstream's file tier
  (`--mmap-experts --resident-budget-gib`) with a budget of the free RAM less 6 GiB: the experts outside VRAM,
  hottest first, pinned; the rest is read from `experts.bin` when needed - unbuffered, in merged requests (this
  fork's addition, upstream since 0.1.38 as #362). The whole 63 GiB arena needs a 96 GB PC. On an RTX 5090 + 64 GB a 32K
  prompt waits 9.3 s instead of 5.7, and the answer after it runs at 97 tokens/s.
- **Every RTX 20, 30, 40 and 50 card with 12 GB or more:** the NVFP4 path needed Blackwell only for the optional
  FP4 x FP4 prompt path, which falls back; the release carries code for all four generations, each one's own path
  tested on the 5090 (built as its PTX, the host answering as that card: `STRATA_EMULATE_CC`).
- **Images on the CPU, from the checkpoint's own vision tower:** the encoder (`strata-vision`, FP32 weights) runs in
  RAM, so the expert cache keeps all its VRAM and text decodes as fast as without images; a picture takes 2-6 s.
  Upstream's GPU encoder took ~1.6 GB of VRAM (~600 cached experts, 10-20% of decode speed in served runs here) and, measured
  against an FP32 reference, was up to 11% off (ggml-cuda's FP16 flash attention); this one is 0.1% off.
  A picture an agent opens itself (Claude Code's `Read`) reaches the encoder too: it arrives inside a tool result,
  which the server used to turn into text only (fixed here, sent upstream as #529).
- **Fixes:** a scale fold that left NVFP4 hidden activations in FP16's subnormals (2-12% expert error), and the
  batched verify path skipping the query rotation of rotated KV caches.
- **A more accurate experts pack, re-quantized from the BF16 checkpoint** (optional, `tools/requant.py`):
  - The 4-bit part is NVFP4 by GPTQ on the model's own activations instead of ModelOpt's rounding to the nearest value. It has the same format and the same speed.
  - In 27 of the 48 layers the down projection is Q8_0. These are the layers where it removes the most error per byte.
  - The result is 73.8 GiB of experts instead of 63.3 GiB. Against an all-Q8_0 reference, the answers' KL is half of the ModelOpt pack's, about 2.6x less once run-to-run noise is taken out, for 12-15% of the decode speed.
  - GPTQ alone, without the 8-bit layers, is 1.7x closer at no speed cost.
  - docs/NVFP4.md, "Re-quantized from BF16", has the method and every measurement.
- **0.1.41-nvfp4.2: a 96 GB PC keeps every expert in RAM.** Windows lists a 96 GB PC as 93-95.6 GiB, so the
  low-RAM mode, on below 96 GB installed until now, started there although the 63 GiB of experts fit. It now starts
  only below 92 GiB installed (a 64 GB PC), and setup follows the same rule. Emulated (95.6 GiB installed, 88 GiB
  left to the engine by a RAM ballast, 3 pairs): a chat 13.7 ms a round instead of the low-RAM mode's 20.3, a 32K
  prompt 3.9 s instead of 8.4, 18-20 GiB of RAM still free, nothing paged out.
  - **The low-RAM mode starts faster:** it reads its VRAM cache and its RAM copy from `experts.bin` unbuffered, with
    the file's mapped view closed meanwhile: ready in 17.5 s instead of 53 at 88 GiB, a peak working set of 47 GiB
    instead of 85 (Windows kept 40 GiB free instead of 1), the same bits.
  - **The page file is checked:** below 60000 MB in all, the engine prints a `WARNING` at start and setup a framed
    warning (`--check` and the install): short of commit the expert cache opens smaller and the model can run
    significantly slower, or not start. 64000 MB is the advice ([Requirements](#requirements)).
  - **Two texts fixed:** the low-RAM start's warning about lent cache slots says what they cost (the RAM budget keeps
    no copy of them; a 32K prompt read 14.3 GiB from the SSD), and setup no longer suggests `--prefill auto:32768`,
    which this fork's `auto` already is.

- **On upstream Strata 0.1.41 (0.1.41-nvfp4.1).** Since 0.1.40.3 (and its hotfix 0.1.40.4, a Pascal decode fix)
  upstream changed, of what runs on an NVIDIA card:
  - the CPU share of small prompt chunks is on by default upstream too (`auto`, one GPU without batch slots), with
    the contention gate this fork sent as #1379, and the chunked DeltaNet recurrence (#1372) is upstream's opt-in;
    this fork keeps its additions on top: the recurrence in chunks from 128 tokens, and NVFP4 packs sharing chunks
    up to 4,096 tokens;
  - with the share on, the prompt path borrows expert slots for the staged chunk only (a 2K prompt 364 slots instead
    of 795), so more experts stay in VRAM and a 2K prompt's logits move a little (KL 0.016 on the GPTQ + Q8_0-down
    pack, 0.0017 on ModelOpt, the same top token), closer to the share-off result than before (KL 0.0067 against
    0.0645). With `STRATA_PREFILL_CPU_SHARE=0` the bits are 0.1.40.3-nvfp4.1's; a 2K prompt with the default share
    takes as long as before (918 against 923 ms, 5 pairs);
  - setup warns when the card is not in the Default compute mode (#1445), the server takes several `--api-key`s
    and refuses oversized bodies, and an engine that hangs with an idle GPU is restarted.

  The rest is multi-GPU batch groups, Windows prompts from a model bigger than RAM (#1323), AMD and Intel. 71 of the
  fork's 76 commits are carried; the gate and the chunked recurrence are upstream's now, the default `auto` and the
  4,096-token NVFP4 limit shrink to the fork's additions on upstream's CPU share, and `qsa_select_bench`'s accuracy
  floor gives way to upstream's own fix (#1399, closed). The 32K logits and tokens are identical to
  0.1.40.3-nvfp4.1's; with the fork's own defaults off they equal upstream 0.1.41's byte for byte on IQ2_XS. Decode
  is as fast as 0.1.40.3-nvfp4.1's (5 interleaved chats each: 175.9 against 174.8 tokens/s).

- **Setup installs this fork (0.1.41-nvfp4.1):** `START-HERE.bat` / `setup.sh` install this fork's NVFP4 GPTQ quants
  of huihui-ai's abliterated Flash-Next (the default) and of OrcaRouter's, ready-made from Hugging Face, and always
  this fork's engine, never upstream's (see [Quick start](#quick-start)). The release carries that engine as
  `strata-windows-x64.zip`, and the bundle updates itself with `update.cmd` (in `config\` and `data\` only a
  file still as a release shipped it is replaced).

- **0.1.40.3-nvfp4.1:** a second round of agents after a profile of the prompt and of decode. Decode turned out
  bound by host RAM bandwidth, not by PCIe: the CPU pool and the PCIe share read the same DDR5. So the PCIe share's
  blobs now go into the VRAM tier instead of the tier reading them from RAM a second time (decode +9% in a chat, +10% deep in a 32K document).
  The prompt reads its experts in place, fuses the MoE and hyper-connection kernels, reads the PLE rows with 8
  threads and loads cuBLAS's kernels at start (a 32K prompt 3921 -> 3465 ms, -12%; the first 2K request after a load -150 ms). A fix: with the tier not waiting for its
  copies (this fork's default) the CPU could compute an expert it had just evicted from a stale activation.
  docs/NVFP4.md, "A second round of kernels", has every measurement.

- **0.1.40.3-nvfp4.1, on upstream 0.1.40.3:** since 0.1.40.2 upstream changed, of what runs on an NVIDIA card:
  - the MTP drafter calls the native top-10 router only for the 512-expert, top-10 router it was written for, as
    the main layers already did (a model with fewer experts read past each row, #1357; this model is one, so the
    same bits here);
  - the tokenizer's BPE merge loop reads its rank table once per word instead of once per symbol pair (#1385), and
    the web page sends a turn with no answer text back in the history, so the turns keep alternating (#1392);
  - an automatic expert cache keeps 2,560 MiB free on Windows (#1376), for AMD cards only: CUDA keeps its own
    sizing, so the cache here is as large as before. The verify window's interleaved q8_1 copy was made opt-in and
    then reverted, so it stays on.

  The rest is AMD and Intel: setup, the bundled HIP runtime's DLLs and why one did not load (#461), the Arc
  A-series, Docker. Every commit of the fork is carried; the card check, which the fork runs before the arena
  thread starts, takes upstream's new HIP runtime probe. With the CPU and PCIe shares fixed, the logits and tokens
  are identical to the fork's head before the port on both packs; with the fork's own defaults off they equal
  upstream 0.1.40.3's byte for byte on IQ2_XS.

- **0.1.40.2-nvfp4.1:** on upstream 0.1.40.2, which added:
  - this fork's CPU share of small prompt chunks (`STRATA_PREFILL_CPU_SHARE=auto`, opt-in there), with the CPU's
    thread kept off the expert source (its blobs are taken on the prompt thread);
  - routed expert uploads that overlap the shared expert (#789), and an interleaved q8_1 copy for the verify
    window's 2-4 token dense projections (Eddoursul's fork), both the same bits;
  - per-layer slot sizes with `--expert-cache-per-layer`, opt-in verify-window work (PDL, graph branches), `--gpu`,
    Prometheus `/metrics`, the vision encoder loaded with a lazy model, request-body and timeout fixes in the server,
    and #615's token accounting, which this fork carried.

  The CPU share is upstream's code now, with this fork's three additions: `auto` is the default (upstream: off), the
  CPU's NVFP4 rows come with their down scale applied (`row_sd` 1), and NVFP4 packs read routed-only up to 4,096
  tokens. `auto` also takes the gate sent upstream as #1379: it times layers with and without the share and shares
  only while sharing is faster, so where the CPU's work slows the stream down the experts stay on the GPU. The DeltaNet
  recurrence now runs in chunks from 128 tokens (its scratch is kept per device; a 2K prompt 1,044 / 1,039 / 1,035
  against 1,064 / 1,044 / 1,047 ms). With the CPU share fixed, the logits and tokens are identical to
  0.1.40-nvfp4.3 plus that change on both packs; with the fork's own defaults off they equal upstream 0.1.40.2 byte
  for byte on IQ2_XS. Decode is as fast as 0.1.40-nvfp4.3's (10 interleaved chats each: 157.6 against 157.8
  tokens/s).

- **0.1.40-nvfp4.3:** a round of kernels after a roofline of where the time goes. A 32K prompt reads 8% faster:
  the experts' gate/up on the FP4 tensor cores with two FP4 terms per activation (w4a4x2), the prompt attention on
  INT8 tensor cores (closer to FP32 than before), the DeltaNet recurrence in chunks for long chunks, the
  hyper-connection up projection fused with its mix. Decode: the hyper-connection read v4 (15.8 against 16.1 ms a round, 5 interleaved chats, the GPU ring 7.06 against 7.41 ms).
  Answers against a high-precision reference are as close as before (8 prompts); docs/NVFP4.md, "A round of
  kernels", has the table and every measurement.

- **0.1.40-nvfp4.2:** upstream's hotfix 0.1.40.1, server only. Its #1058 gate replaces this fork's (#1068): the
  same three conditions, plus no call from a code fence or inline code, in the thinking or in the answer. A restart
  keeps waiting requests (#1012), and an engine that exits after an ERR line says why. The engine is 0.1.40-nvfp4.1's,
  byte for byte.

- **0.1.39-nvfp4.4:** small prompt chunks shared with the CPU (agent turns 15-38% faster), AVX2 NVFP4 rows for CPUs
  without AVX-512, the CPU share and decode's PCIe share measured instead of fixed, and upstream's Q2_0 in the CPU
  share. docs/NVFP4.md, "Small prompt chunks with the CPU", has the measurements.

- **0.1.39-nvfp4.3:** two fixes for a second card as a peer tier (`--peer-device`), from @chimpera's report on two RTX 3090s. The peer's cache was created on the first GPU. It also went through the elastic K/V's VRAM mapping, which only the primary cache needs, and is one allocation again, as in upstream. The server's handling of a request body it answers before reading follows upstream #594's current version. Bodies of any size get their answer, and `/load`, `/unload` and `/config` no longer hold the connection 5.5 s after answering. Logits and tokens are identical to 0.1.39-nvfp4.2.

- **0.1.39-nvfp4.2:** a long Claude Code conversation turned into "!" mid-reply, and every later request of it
  answered "!" until the engine was restarted by hand. The server now starts a fresh engine after such a reply (one
  bad reply instead of a broken session). The shared expert's fused SwiGLU quantizer keeps its fp16 scale finite, as
  0.1.39's other two do (sent upstream as #838; not this incident's cause). What corrupted the state is not found
  yet: docs/NVFP4.md lists what was ruled out. Logits and tokens are identical to 0.1.39-nvfp4.1 on both packs.

- **0.1.38-nvfp4.2:** the VRAM reserve is 700 MiB again on Windows (1500 no longer avoided a stall on 0.1.38, only
  cost expert slots). From other people's upstream pull requests: a prompt tokenised from the last shared prefix
  (#567: 6.8 ms instead of 128 ms at 125K tokens), token accounting across reasoning continuations (#615), and no
  lost 401/403 on Windows (#594). docs/NVFP4.md lists what else was measured and why it was not taken.

- **Fixes from a code review (0.1.37-nvfp4.2):** the prompt path borrows 1.25 GiB less VRAM at a 32K chunk; the
  elastic K/V no longer runs past its mapped cells when it cannot lend slots; pictures Claude Code reads that the
  server cannot read become a note instead of a 400; safer image sources. Each upstream bug went up as a pull
  request (#546, #547, #550, #553, #554, #555; 0.1.38 fixed #546 its own way); docs/NVFP4.md has the list.

Each change was measured - first-token KL against a reference, and interleaved speed A/B runs;
[docs/NVFP4.md](docs/NVFP4.md) has the numbers, and everything that was tried and dropped.

## Requirements

- **GPU:** an RTX 20, 30, 40 or 50 card with 12 GB of VRAM or more (the release has code for sm_75, 86, 89 and
  120, and the optional FP4 x FP4 prompt unit for 120a). Built and measured on an RTX 5090, 32 GB: dense weights and
  the 262K KV cache first, then ~19 GB of cached experts (~7,200). The other generations were tested on the 5090
  through their own code paths (docs/NVFP4.md, "Other GPUs"). A card with less VRAM caches fewer experts and decodes
  slower; lower `--max-context` with it (measured with 0.1.28-nvfp4.4):

  | card's VRAM | `--max-context` | expert slots | decode, measured* |
  | --- | ---: | ---: | ---: |
  | 32 GB (RTX 5090) | 262144 | 7,352 | 111 tok/s |
  | 24 GB (RTX 3090 / 4090) | 131072 | 5,158 | 95 tok/s |
  | 16 GB (RTX 4080 / 5080 / 4060 Ti 16 GB) | 65536 | 2,422 | 67 tok/s |
  | 12 GB (RTX 3060 12 GB / 4070) | 32768 | 1,055 | 59 tok/s |

  \* On the RTX 5090 with the smaller card's VRAM budget (`--vram-reserve-mib`); a real card's own compute and PCIe
  make it slower. 12 GB at 262144 and 8 GB cards at any context stop with *no VRAM is left for the expert cache*.

- **RAM:** 64 GB minimum. With 92 GiB installed or more (a 96 GB PC: Windows lists 93-95.6 GiB) the engine pins
  all 63 GiB of experts (~67 GiB of physical RAM for the engine, ~71 with images and the server; with 88 GiB left
  to it 16-21 GiB stayed free, and it ran as on 128 GB: chat 13.1 ms a round against 12.7). With less, the low-RAM
  mode starts by itself (`--low-ram`; `--no-low-ram` turns it off; `--ram-budget GIB` caps it): the experts outside
  VRAM, hottest first, pinned up to the free RAM minus 6 GiB; the rest is read from the file when needed. With 64 GB
  and an RTX 5090 all 44 GiB of them fit:

  | 64 GB of RAM (measured*) with an RTX 5090 | |
  | --- | ---: |
  | a 32K prompt: to the first token | 9.3 s (96+ GB: 5.7 s) |
  | decode after it | 97 tok/s (0.1.28-nvfp4.4: 72) |
  | start, to the session | ~16 s (the profile fill 4.3 s, the 44 GiB budget 9.5 s, read unbuffered) |

  \* On the RTX 5090 + 128 GB PC with 60 GiB of RAM locked away by a large-page ballast (58 GiB stay available, as
  on a 64 GB PC whose Windows uses 6 GB). The smaller cards' 64 GB figures (0.1.28-nvfp4.4: 24 GB 86-90 tok/s,
  16 GB 54-56) were not measured again on this release.

- **Pagefile: set 64000 MB** (System Properties - Win+R, `sysdm.cpl` - > Advanced > Performance Settings >
  Advanced > Virtual memory > Change: Custom size, initial and maximum 64000 MB, then restart). Windows lets all
  programs together commit at most RAM + pagefile, and the engine commits ~100 GiB with all experts pinned (63 GiB
  of experts plus ~30 GiB that WDDM charges for the VRAM it uses), ~80 GiB in the low-RAM mode; nothing of the model
  is ever paged out. Short of commit, the expert cache in VRAM opens smaller (a quarter less a try), and the model
  can run significantly slower (measured: one step made a chat's round 17% slower, two steps 38%); an arena that
  cannot be committed stops the start with an allocation error. The engine (one `WARNING` line at start)
  and setup (`--check` and the install) warn when all pagefiles together are below 60000 MB; a system-managed one
  counts as what Windows lets it grow to (3 x RAM, at most an eighth of its drive, within the free space).
- **Disk:** with setup's ready-made files ~130 GB: the expert pack ~70 GB, the n-gram table 51 GB, the dense GGUF,
  the embedding (1.3 GB) and the MTP head (0.8 GB); setup counts the exact sizes from its file table and checks the
  free space first. Converting the checkpoint yourself: ~200 GB for the model files (GGUF 74 GB, expert pack 70 GB,
  n-gram table 51 GB, image encoder 1.8 GB, embedding 1.3 GB, MTP head 0.8 GB), ~340 GB while preparing them (the
  135 GB checkpoint and the MTP intermediates can go afterwards). Use the fastest NVMe drive you have: every start
  reads 63 GiB.

## Measured

RTX 5090 (32 GB, PCIe 5 x16), Ryzen 9 9950X3D, 128 GB DDR5-5600, Samsung 9100 PRO, Windows 11, CUDA 13.3.
262,144-token context, KV cache int8, large pages on. The machine is also a desktop: runs vary by ~5%.

| | |
| --- | ---: |
| Writes a chat answer (~520 tokens, to its end) | 138 tokens/s on average, 125-146 in 6 runs; 18.2 ms a round |
| Writes a ~1,000-token answer after a 32K prompt | 132 tokens/s on average, 130-133 in 4 runs; 20.0 ms a round |
| The same with the re-quantized pack (GPTQ + Q8_0 down in 27 layers) | chat 126 tokens/s (-12%), after a 32K prompt 116 (-15%), the prompt read as fast |
| Reads a 32K prompt | 6,740-6,930 tokens/s (0.1.31-nvfp4.1: 5,400-5,900) |
| Start: the expert arena loaded | ~7 s (63 GiB of experts read at 10-11 GiB/s) |

0.1.31-nvfp4.3 measured the same in the same interleaved runs: 18.1 and 20.0 ms a round. Its notes said 161-173
tokens/s after a 32K prompt, but that prompt's answer ends after 22 tokens. Those runs went on to a fixed 256 tokens
with no end-of-turn stop, so that was the speed of the loop after the answer, not of an answer. The rates vary with
the draft acceptance of the path the tokens take (docs/NVFP4.md, "On upstream 0.1.32"). With
64 GB of RAM (the low-RAM mode): 9.3 s to the first token of a 32K prompt instead of 5.7, then 97 tokens/s.

Where precision was still being lost, first-token KL divergence from the more exact variant (8 prompts of 1K-8K
tokens plus one of 32K; two runs of the same configuration differ by a median 0.000007):

| source | KL mean | at 32K | now |
| --- | ---: | ---: | --- |
| RoPE: fast-math float angles | - | 0.0063 | float64 table |
| token embedding stored as Q8_0 | 0.0025 | 0.0041 | BF16 as shipped (`--embd-gguf`) |
| prompt path: BF16-rounded activations into BF16 projections | 0.0023 | 0.0093 | hi + lo split (0.0029 left: not the hyper-connection) |
| n-gram table as IQ4_NL | 0.0026 | - | FP8 as shipped |
| int8 KV without rotation (vs fp16 KV) | 0.0017 | 0.0068 | rotated: 0.0011 / 0.0038 |
| prompt path W4A8 (vs FP16 activations) | 0.0018 | - | default; `fp16` is 24% slower |

## Build it yourself (advanced)

Setup does everything below for you ([Quick start](#quick-start)). These are the manual steps: the engine built by
hand, and jpezzulli's ModelOpt checkpoint converted on your own PC (~340 GB of disk while preparing), for the
ModelOpt pack or a pack of your own.

### Build (Windows)

Needs Visual Studio 2022 Build Tools, CUDA 13+ (sm_120 wants 13), CMake, Ninja, Python 3.11+. From an x64 Native
Tools prompt:

```bat
git clone https://github.com/ggml-org/llama.cpp third_party\llama.cpp
git -C third_party\llama.cpp checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 ^
      -DSTRATA_GGML_DIR=%CD%\third_party\llama.cpp
cmake --build build --target strata
```

Without `-DSTRATA_GGML_DIR` CMake fetches the same llama.cpp commit itself. `120` is enough: CMake builds the one
FP4 x FP4 unit for `120a` by itself, and the rest runs on sm_121 too.

### Prepare the model

```bat
python -m venv .venv
.venv\Scripts\python -m pip install numpy torch safetensors transformers sentencepiece
set PYTHONPATH=third_party\llama.cpp\gguf-py

:: 1. the checkpoint (126 GiB)
hf download jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4 --local-dir models\orca-nvfp4

:: 2. the n-gram (PLE) table, 51.2 GB: its FP8 bytes copied as they are (read from the SSD, never loaded into RAM)
.venv\Scripts\python tools\ple_fp8_pack.py --model models\orca-nvfp4 --out models\ple-fp8.gguf

:: 3. the token embedding, 1.3 GB: BF16 as shipped (the GGUF below stores it as Q8_0)
.venv\Scripts\python tools\embd_bf16_pack.py --model models\orca-nvfp4 --out models\token-embd-bf16.gguf

:: 4. GGUF (NVFP4 experts, Q8_0/BF16 dense) and the pack (experts.bin 63 GiB, tokenizer)
.venv\Scripts\python tools\nvfp4_convert.py --model models\orca-nvfp4 --outfile models\orca-nvfp4.gguf
.venv\Scripts\python tools\iq_pack.py --gguf models\orca-nvfp4.gguf --out packs\orca-nvfp4

:: 5. the fine-tune's own MTP draft head
.venv\Scripts\python tools\mtp_extract.py --model models\orca-nvfp4 --out mtp-orca
.venv\Scripts\python tools\mtp_pack.py --src mtp-orca --experts q2_0 --out mtp-orca\mtp-q2_0.gguf
.venv\Scripts\python tools\mtp_rt.py --gguf mtp-orca\mtp-q2_0.gguf --out mtp-orca\rt
copy data\draft_vocab.bin mtp-orca\rt\
```

Put `packs\` and the GGUF on the fastest drive you have: the start is a 63 GiB read.

**Optional: the more accurate experts pack.** It needs the BF16 checkpoint (360 GB) and ~2 hours on an RTX 5090. The pack from step 4 stays the dense, index and tokenizer source.

```bat
hf download orcarouter/Qwen3.8-Flash-Next-Uncensored --local-dir models\orca-bf16
:: calibration: the engine's own MoE inputs of every layer for the four token files in data\requant_calib (uk, en, code, chat)
set STRATA_DUMP_MOE_INPUT=calib\d
set STRATA_DUMP_MOE_LAYER=all
build\strata.exe <the usual arguments> --tokens-file data\requant_calib\calib_uk.txt --max-new 1   (and en, code, chat)
:: errors per layer and method, then the plan for a size budget
.venv\Scripts\python tools\requant.py analyze --bf16 models\orca-bf16 --calib calib\d --out calib\errors.jsonl
.venv\Scripts\python tools\requant_plan.py calib\errors.jsonl calib 74
:: the pack: GPTQ NVFP4, Q8_0 down in the planned layers
.venv\Scripts\python tools\requant.py pack --bf16 models\orca-bf16 --calib calib\d --plan calib\plan_d8_74.json --base packs\orca-nvfp4 --out packs\orca-nvfp4-gptq-q8d
```

Then run with `--pack packs\orca-nvfp4-gptq-q8d`. The engine reads mixed expert formats from 0.1.32-nvfp4.2 on.

## Run

One-shot:

```bat
build\strata.exe --pack packs\orca-nvfp4 --native models\orca-nvfp4.gguf --native-dense-gguf models\orca-nvfp4.gguf ^
  --ple-gguf models\ple-fp8.gguf --embd-gguf models\token-embd-bf16.gguf ^
  --mtp mtp-orca\rt --spec 6 --spec-min-p 0.7 --prefill auto ^
  --expert-profile data\expert-profile.bin --expert-cache auto ^
  --max-context 262144 --kv int8 --tokens-file prompt.txt --max-new 256
```

OpenAI-compatible server: save the same arguments as a config (`{"exe": "build/strata.exe", "args": [...],
"tokenizer": "packs/orca-nvfp4/tokenizer", "host": "127.0.0.1", "port": 8097}`) and start
`python -m serve.server --engine strata --config that.json`.

`--native` and `--native-dense-gguf` both point at the GGUF: `--native` alone would take the PLE file for a second
shard of the same model.

**Images:** the image encoder comes from the checkpoint (`convert_hf_to_gguf.py <checkpoint> --mmproj --outtype f32`,
1.8 GB) and is built with `release\build-vision.cmd` (CPU only). The engine takes `--vision`, the config a
`"vision": {"exe": "build-vision-cpu/bin/strata-vision.exe", "mmproj": "models/mmproj-f32.gguf", "model":
"models/orca-nvfp4.gguf", "gpu": false, "max_tokens": 1024}` entry; then OpenAI `image_url` parts and Anthropic
image blocks (a screenshot pasted into Claude Code) work. The encoder uses one thread per core while it runs; the
engine is idle then.

## Claude Code

```bat
set ANTHROPIC_BASE_URL=http://127.0.0.1:8097
set ANTHROPIC_API_KEY=local
set ANTHROPIC_MODEL=strata-nvfp4
set ANTHROPIC_SMALL_FAST_MODEL=strata-nvfp4
claude
```

The model is hybrid (Gated DeltaNet layers keep a recurrent state that cannot be cut back to a position), so an
agent turn that differs from the last one a few tokens in would re-read everything without checkpoints. The server
keeps one at every turn boundary. Measured with a real `claude -p` session doing three tool turns: the first request
read its 21,964-token system prompt once (5.8 s), the next ones 209 and 106 new tokens (0.55 s and 0.4 s). An edit
in the middle of a history falls back to the checkpoint just before it. `/v1/messages/count_tokens` is served, and
a request that asks for no thinking (Claude Code's small helper calls) gets none.

## Large pages (why a reboot)

This is not more memory, it is bigger pages. Windows maps memory in 4 KB pages, so the 63 GiB expert arena is
16.5 million of them; the CPU part of every token reads experts from it at DRAM speed, and each page needs a TLB
entry. With 2 MB pages it is 32 thousand, and the CPU pool holds steady: 7.3-7.7 ms per round on this machine
against 7.4-11 ms with 4 KB pages. Decode itself is GPU-bound here, so the rate moves little; the point is the
steadiness (and a slightly faster start and exit).

Windows only gives large pages to an account that holds *Lock pages in memory* (SeLockMemoryPrivilege), and it
puts a privilege into a sign-in's token only when that sign-in starts. So:

1. `powershell -ExecutionPolicy Bypass -File tools\enable-large-pages.ps1` — asks for admin (UAC) and grants it to
   the current user through `secedit` (works on Windows Home, which has no `secpol.msc`); the policy as it was is
   saved to `%LOCALAPPDATA%\strata-large-pages\before.inf`, and `-Revoke` takes it back.
2. **Sign out and back in, or reboot.** Locking the screen is not a new sign-in.
3. Check: the same script with `-Check`, or the engine's start line `expert arena: ... large pages (2097152 B)`.

Without the privilege the engine says `large pages refused ... VirtualAlloc error 1314` and runs on 4 KB pages.
Error 1450 instead means the privilege is there but Windows found no 32 thousand free 2 MB blocks (memory
fragmented after a long uptime): it falls back the same way, and a reboot clears it. The arena is locked in RAM
either way - CUDA pins it for the GPU's copies.

## Switches

| | |
| --- | --- |
| `STRATA_PREFILL_NVFP4=w4a8\|w4a4\|w4a4x2\|fp16` | prompt path precision of the NVFP4 experts (default `w4a4x2`: gate/up on the FP4 tensor cores with two FP4 terms per activation, down w4a8; `w4a8` before 0.1.40-nvfp4.3) |
| `STRATA_QUANT_GATHER=1` | the prompt path quantizes a token's row once per expert again (default: once per token, scattered; the same bytes) |
| `STRATA_PROMPT_ATTN_IMMA=0` | the prompt attention's v2 kernel (FP16 MMA) instead of the INT8 one (default since 0.1.40-nvfp4.3) |
| `STRATA_GDN_CHUNKED=0` | the prompt path's DeltaNet recurrence token by token (default: in chunks of 32 for prompt chunks of 128+ tokens; for 16384+ before its scratch was kept) |
| `STRATA_MMVQ_V2=1` | decode: attention k+v+q and the drafter's k+v in one Q8_0 launch (the same bits; under 1%) |
| `--embd-gguf PATH` | the token embedding from this GGUF (BF16 from `tools/embd_bf16_pack.py`) |
| `STRATA_PREFILL_BF16X2=2\|1\|0` | exact inputs to the prompt path's BF16 projections: all but the hyper-connection (default), all (~9% slower prompt reading), off |
| `STRATA_KV_ROT=0` | int8 K/V without the Hadamard rotation (A/B) |
| `STRATA_ROPE_TABLE=0` | the fast-math RoPE angles instead of the float64 table (upstream's default; A/B) |
| `--prefill auto:8192\|16384` | cap `--prefill auto`'s chunk (default 32768 here, 8192 upstream) |
| config `"anthropic_thinking": "on_request"\|"model"` | an Anthropic request that does not ask for thinking renders without it (the bundle's default) / thinks as the template does (upstream's default) |
| `STRATA_COMMIT_SYNC=1` | the verify commit waits for its graph again (A/B) |
| `--pcie-frac F` | a fixed share of cache misses fetched over PCIe (default: each layer's count from measured costs, from the link probe's share; NVFP4 0.25) |
| `STRATA_PCIE_BALANCE=0` | decode keeps the link probe's fixed PCIe share |
| `STRATA_PREFILL_CPU_SHARE=x\|0` | a fixed CPU share of a small prompt chunk's streamed experts / none (default: `auto`, measured, and only while sharing is faster; `STRATA_DBG_CPU_GATE=1` prints its readings) |
| `STRATA_NO_LARGEPAGES=1` | 4 KB pages even when large pages are allowed (A/B) |
| `STRATA_NO_NVFP4_512=1` | CPU pool on ggml-cpu's NVFP4 dot instead of the AVX-512 rows |
| `STRATA_NO_NVFP4_256=1` | without AVX-512: ggml-cpu's NVFP4 dot instead of the AVX2 rows |
| `STRATA_FORCE_AVX2=1` | tests: the engine's CPU dispatch as on a CPU without AVX-512 |
| `STRATA_UNBUFFERED_LOAD=1\|0` | force the expert reads unbuffered / through the file cache (default: unbuffered only when the cache cannot keep the files) |
| `STRATA_DEFERRED_REGISTER=1`, `STRATA_ARENA_SYNC=1` | register the arena per layer on a thread ahead of its readers (upstream's opt-in, this fork's default until 0.1.39; the same decode speed here) / load the arena after the dense weights (A/B) |
| `STRATA_ADAPT_WAIT=1` | each decode window waits for the adaptive tier's copies, so greedy decode repeats exactly (upstream's default since 0.1.38; ~11% slower with this fork's tier, ~2% with `STRATA_ADAPT_LAG=2`, #764) |
| `STRATA_ADAPT_FETCH=0\|1\|2` | decode: the PCIe share's blobs admitted into the VRAM tier (2, the default since 0.1.40.3-nvfp4.1: instead of the tier's own swaps where an admission ran; 1: beside them; 0: off, the tier's swaps only) |
| `STRATA_ADAPT_EVICT_SYNC=0` | the tier's evictions reach the device's table only when its copies land (the old behaviour: with the tier not waiting, a stale activation could be computed) |
| `STRATA_ADAPT_FETCH_CHECKRES=1` | before each decode window, compare the device's residency table with the host's and count windows that computed a CPU expert without its activation |
| `STRATA_PLE_READERS=N` | the prompt's PLE rows through N reader threads (default 8 on Windows; 0 = one thread reaps every read) |
| `STRATA_PREFILL_IN_PLACE=0` | the prompt's MMQ reads experts from a gathered group buffer again (default: in place, from their ring / cache slots; the same bits) |
| `STRATA_PREFILL_TOKEN_ROWS=0` / `STRATA_PREFILL_SWIGLU_QUANT=0` / `STRATA_PREFILL_COMBINE_WRITE=0` | the prompt path's earlier kernels: activations quantized per expert row / swiglu and quantize apart / combine then the hyper-connection write (the same bits) |
| `STRATA_GEMM_WARM=0` | cuBLASLt's GEMM kernels load on the first prompt again (default: on the prewarm thread at start) |
| `STRATA_SPEC_STATS=1` | decode: round time, misses and accepted drafts per window size, and the drafts' acceptance by their probability |
| `STRATA_VERIFY_ARENA=1` | print a checksum of the loaded arena |
| `STRATA_DUMP_FIRST_LOGITS=file` | write the first generated token's logits (compare prompt paths) |
| `STRATA_DUMP_MOE_INPUT=file`, `STRATA_DUMP_MOE_LAYER=l` | dump one layer's real MoE input rows |
| `STRATA_REQUEST_LINES=1` | `serve.server` echoes one summary line per request to stdout |
| `STRATA_EMULATE_CC=75\|86\|89` | tests: answer as that generation (with an engine built as its PTX, `86-virtual`) |
| `STRATA_QSA_WARP=1\|select\|attn` | the pre-sm_80 QSA kernels on any card, as RTX 20 runs them (A/B) |
| `--vram-reserve-mib N` | VRAM left unused (default 700, as upstream; this fork used 1500 on Windows until 0.1.38, where 700 measured 0.8% faster with no stalls); a smaller card's budget on a bigger one |
| `--image-max-tokens N`, `--image-min-tokens N` (`serve.server`) | image tokens a picture becomes at most / at least: a bigger picture is scaled down, a smaller one up, keeping its aspect ratio (also `"max_tokens"` / `"min_tokens"` in the config's `"vision"`; the bundle's config has 1024 and the model's minimum 8; the model reads up to 4096, and more tokens read smaller text and encode slower) |
| `--low-ram` / `--no-low-ram` | the low-RAM mode on / off (default: on with less than 92 GiB installed) |
| `--ram-budget GIB` | the low-RAM mode with at most GIB of pinned expert copies (upstream's `--resident-budget-gib`) |
| `STRATA_RESIDENT_HEADROOM_GIB=6` | RAM the low-RAM mode leaves free |
| `--no-kv-grow`, `STRATA_KV_GROW=0` | the K/V allocated for the whole context at start (before 0.1.31-nvfp4.2) |
| `STRATA_KV_GROW_INIT` / `_STEP` | the K/V's cells at start (16384) and its growth step (8192) |
| `STRATA_PREFILL_GROUP_GATHER=0` | the prompt path gathers, waits and releases one expert at a time (A/B) |
| `STRATA_HC_UPMIX=0\|1` | the prompt path's hyper-connection up projection with the mix fused (gr_upmix): default on compute capability 12.x (sm_120, bitwise the cuBLAS + mix pair there at chunks of 33+ tokens) / opt-in elsewhere / off |
| `STRATA_GR_V3=0` | upstream's default hyper-connection read (0.1.32: #315's staged variant, `STRATA_HC_SPLIT`) instead of the split V3 kernels (A/B: 18.49 vs 18.23 ms a round) |
| `STRATA_GR_V4=0\|1` | the hyper-connection read: the split V3 kernels / V4 without PDL (default: V4 with programmatic dependent launch on sm_90+; the same bits as V3, its kernels 1.16-1.39x faster at 1-8 tokens) |
| `--ple-inflight N` | outstanding n-gram row reads (default 256; 64 before 0.1.31-nvfp4.2) |
| `--adapt-every N`, `--adapt-swaps N`, `--adapt-decay F` | the adaptive VRAM tier: re-rank every N rounds, up to N swaps, counts x F after each (default 2 / 192 / 0.92 with a cache of 20-60% of the experts and every expert in RAM, else upstream's 4 / 96 / 0.7) |

## Tests

| executable | checks |
| --- | --- |
| `nvfp4_avx512_parity [experts.bin]` | AVX-512 and AVX2 rows vs ggml-cpu, the AVX2 ones bit for bit (ctest `nvfp4_cpu_parity`; `--expert`: one whole expert vs FP64; `--bw T N [ggml\|avx2]`: DRAM rate) |
| `nvfp4_expert_gpu_parity experts.bin` | the GPU decode path's experts vs FP64, one per sampled layer |
| `mmq_nvfp4_parity experts.bin` | MMQ products vs FP64 (`--group`, `--layers`, `--real dump layer`) |

## Limits

- Built and measured on Windows with one RTX 5090 and 128 GB of RAM. RTX 20/30/40 cards, smaller VRAM and 64 GB of
  RAM were tested on that PC through their own code paths and budgets (docs/NVFP4.md), not on the real hardware; a
  CPU without AVX-512 runs the AVX2 rows, and smaller CPUs were emulated (`STRATA_FORCE_AVX2=1`, `--pool-workers`). The low-RAM mode's unbuffered reads
  are Windows-only (elsewhere it reads through the page cache).
- A decode round (~21 ms) is the GPU running back to back, ~4.6 ms of it pulling the PCIe share of the experts
  and ~3.4 ms waiting for the CPU's share, which reads DRAM at ~55 of the ~65 GB/s this platform does. The pool and
  the link read the same DRAM, so moving misses between them does not shorten it here (the measured PCIe count
  moves few). More VRAM for the expert cache or more memory bandwidth are what would move it; docs/NVFP4.md lists
  what was tried.
- The model is an abliterated fine-tune: it does not refuse. What it is used for is on whoever runs it.

## Releasing

```bat
python release\make_windows_bundle.py      :: clean tree only; builds build-release\ itself, zips, SHA-256
git push origin HEAD:main
python release\publish.py --title "Strata NVFP4 v... - what changed" --notes notes.md   :: --dry-run first
```

`publish.py` names this repository in every `gh` call (a clone's gh default can point at upstream) and refuses a
zip whose `engine\BUILD.json` is not the pushed HEAD, this version and a clean tree. It uploads two assets: the
bundle and `strata-windows-x64.zip`, the same engine folder in the layout setup.py installs (a clone's
`START-HERE.bat` / `UPDATE.bat` takes the engine from there, checked against GitHub's SHA-256).

## License

MIT, as upstream ([LICENSE](LICENSE), copyright Niko1221 and the Strata contributors). llama.cpp / ggml code compiled
into the build is MIT as well.
