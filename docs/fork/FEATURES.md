# Features

What Strata NVFP4 changes on top of upstream Strata, and where it stops. Back to the [README](../../README.md).

The design is upstream's: the routed experts live in RAM, the most-used ones are cached in VRAM, the misses are
computed on the CPU and fetched over PCIe in parallel with the GPU, and an MTP draft head speculates. Upstream's README
is kept as [README.upstream.md](../../README.upstream.md). [docs/NVFP4.md](../NVFP4.md) has every measurement behind this
fork, and the [releases](https://github.com/sergqwer/strata-nvfp4/releases) say what changed in each.

## What this fork adds

Each change was measured (first-token KL against a reference, interleaved speed runs); the links go to the numbers.

- **NVFP4 experts:** 4.5 bits a weight (E2M1 codes, an FP8 scale per 16 values, an FP32 scale per expert matrix
  applied to each projection's output) instead of upstream's 2-4-bit GGUF quants
  ([Where the global scales go](../NVFP4.md#where-the-global-scales-go)).
- **GPTQ re-quantization from BF16** (`tools/requant.py`), which made the packs above, calibrated on the engine's own
  MoE inputs; Q8_0 `down` in chosen layers is an option for a size budget
  ([Re-quantized from BF16](../NVFP4.md#re-quantized-from-bf16-2026-10-02-release-0132-nvfp42)).
- **Less rounding elsewhere:** the n-gram table in its shipped FP8, the token embedding in BF16, RoPE angles from a
  float64 table, FP32-exact inputs to the prompt path's BF16 projections, int8 K/V behind a Hadamard rotation
  ([Where precision is still lost](../NVFP4.md#where-precision-is-still-lost-audit-2026-09-29)).
- **The prompt path on tensor cores:** the experts' gate/up on RTX 50's FP4 tensor cores with two FP4 terms per
  activation, int8 elsewhere; the prompt attention on INT8 tensor cores; prompt chunks up to 32K
  ([Prompt path precision](../NVFP4.md#prompt-path-precision),
  [A round of kernels](../NVFP4.md#a-round-of-kernels-2026-10-07-release-0140-nvfp43)).
- **NVFP4 CPU kernels** for the CPU's share: AVX-512 1.8-3.7x ggml-cpu, AVX2 2.0-2.4x at 4-8 tokens
  ([CPU experts](../NVFP4.md#cpu-experts)).
- **A faster start:** the expert arena loads unbuffered on its own thread beside the dense weights, ~7 s for 63 GiB
  from a PCIe 5 drive ([Loading](../NVFP4.md#loading)).
- **The K/V grows with the context** (upstream's `--kv-grow`, on here): VRAM the window does not use yet caches
  experts, 7,060 -> 8,290 slots at 262K; `--no-kv-grow` allocates the whole window
  ([Faster without changing an answer](../NVFP4.md#faster-without-changing-an-answer-2026-10-01-release-0131-nvfp42)).
- **Small prompt chunks shared with the CPU** (this fork's change, upstream's default since 0.1.41); on top of
  upstream's, NVFP4 packs share chunks up to 4,096 tokens and the DeltaNet recurrence runs in chunks from 128 tokens
  ([Small prompt chunks with the CPU](../NVFP4.md#small-prompt-chunks-with-the-cpu-2026-10-06-release-0139-nvfp44)).
- **The adaptive VRAM tier** re-ranks every 2 rounds, up to 192 swaps, counts x0.92 (upstream: 4 / 96 / x0.7), and
  admits the PCIe share's experts into VRAM
  ([The adaptive VRAM tier](../NVFP4.md#the-adaptive-vram-tier-and-an-audit-2026-10-01-release-0131-nvfp43),
  [A second round of kernels](../NVFP4.md#a-second-round-of-kernels-2026-10-08-release-01403-nvfp41)).
- **The low-RAM mode** below 92 GiB installed, or where the pack's experts and 16 GiB more do not fit, read
  unbuffered (see [Requirements](INSTALL.md#requirements)).
- **Q8_0 experts in this fork's packs** (`qwen-q8`, all of them; `qwen-nvfp4-mix`, 38 layers beside GPTQ NVFP4) for a
  128 GB PC ([Q8_0 for a 128 GB PC](../NVFP4.md#q8_0-for-a-128-gb-pc-2026-10-11-release-0142-nvfp42)).
- **Every RTX 20-50 card with 12 GB or more:** only the optional FP4 x FP4 unit needs Blackwell; each generation's
  code path was tested on the RTX 5090 ([Other GPUs](../NVFP4.md#other-gpus-rtx-20-30-and-40-2026-09-30)).
- **Images on the CPU:** the encoder runs in RAM, so the expert cache keeps all its VRAM; upstream's GPU encoder
  (FP16 flash attention) was up to 11% off an FP32 reference. A picture a tool returns (Claude Code's `Read`) reaches
  the encoder too ([Images](../NVFP4.md#images-2026-09-30)).
- **A draft vocabulary with Cyrillic:** an answer in Ukrainian decodes 2.11 tokens a round instead of 1.40;
  `tools/draft_vocab.py --add cjk` builds upstream's CJK subset
  ([Upstream 0.1.28 merged](../NVFP4.md#upstream-0128-merged-2026-09-30)).
- **The server starts a fresh engine after a reply of "!"** ([Troubleshooting](TROUBLESHOOTING.md#if-a-reply-turns-into-)).
- **RTX 50 decode kernels and drafting after [Basalt](https://github.com/jesdga95/basalt)** (0.2.15-nvfp4.4): the int8
  tensor-core dense GEMV, the 8-bit hyper-connection read, staged INT8 attention, the persistent DeltaNet layer, the
  attention layer's placement and Basalt's MTP drafting (the two-stage draft head, the cumulative window rule, a window
  of 8), plus kernels of our own; on by default on RTX 50 cards only. Decode on the RTX 5090 23-56% faster than
  0.2.15-nvfp4.3, KL 0.015 against the base engine, whose own runs differ by 0.012
  ([measurements](../NVFP4.md#rtx-50-decode-and-prompt-kernels-after-basalt-2026-11-20-release-0215-nvfp44)).
- **Faster long prompts on RTX 50** (0.2.15-nvfp4.4): the DeltaNet scan on tensor cores, equal chunks, the grouped FP4
  expert GEMM, int8 QSA block scores, f16-accumulate dense GEMMs, CUDA graphs for the prompt's kernel chains, less
  hyper-connection traffic and the borrowed expert slots refilled beside decode: prompts of 16K and more about 30-35%
  faster than 0.2.15-nvfp4.3 (the two rounds' ratios multiplied)
  ([measurements](../NVFP4.md#rtx-50-faster-prompts-2026-11-20-release-0215-nvfp44)).
- **KV streaming on by default** (0.2.15-nvfp4.3) from a context of 131,072 tokens on one GPU: 32,768 cells of each
  attention layer's K/V in VRAM and all of it in pinned RAM, so decode holds its speed to the end of a long dialog
  ([measurements](../NVFP4.md#kv-streaming-on-by-default-2026-11-19-release-0215-nvfp43)).
- **The censorship switch** for the original Qwen (0.1.41-nvfp4.5): the engine removes the model's refusal direction at
  run time in layers 8-33, off by default ([Quick start](INSTALL.md#quick-start)).

## Limits

- Built and measured on Windows with one RTX 5090, a Ryzen 9 9950X3D and 128 GB of RAM. RTX 20/30/40 cards, smaller
  VRAM, 64 and 96 GB of RAM and smaller CPUs were tested on that PC through their own code paths and budgets
  (`STRATA_EMULATE_CC`, `--vram-reserve-mib`, a RAM ballast, `STRATA_FORCE_AVX2=1`, `--pool-workers`), not on the
  real hardware.
- The NVFP4 model runs on one GPU: the low-RAM mode and the growing K/V are one-GPU.
- Not for WSL2: the driver there page-locks only about 1 GB of RAM ([Requirements](INSTALL.md#requirements)).
- The low-RAM mode's unbuffered reads are Windows-only; elsewhere it reads through the page cache.
- Decode is bound by host RAM bandwidth: the CPU pool and the PCIe share read the same DDR5 (~84 of ~90 GB/s here).
  More VRAM for the expert cache or faster RAM is what moves it
  ([A second round of kernels](../NVFP4.md#a-second-round-of-kernels-2026-10-08-release-01403-nvfp41)).
- The RTX 50 kernels of 0.2.15-nvfp4.4 run on RTX 50 cards only, and Auto's MTP with Basalt's drafting runs with one
  request at a time on one card's layers; a layer split or several dialogs at once keep the older drafting.
- The models are the original Qwen, censored as Qwen ships it. With the censorship switch on they do not refuse, and
  what they are used for is on whoever runs them. The switch is never offered for NVIDIA's model.
