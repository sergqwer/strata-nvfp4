# Strata NVFP4

**Qwen3.8-Flash-Next (125B hybrid MoE) with NVFP4 experts on one RTX 20, 30, 40 or 50 card (12 GB of VRAM or more)
and 64 GB of RAM or more, text and pictures.** A fork of [Niko1221/Strata](https://github.com/Niko1221/Strata), on
upstream 0.1.42. Setup installs, ready-made from Hugging Face, the original Qwen in NVFP4 (our GPTQ, or NVIDIA's
checkpoint converted), censored as Qwen ships it; on ours an opt-in switch turns the censorship off. Upstream's GGUF
models are still offered.

**The models on Hugging Face** (setup downloads them; each card has the measurements and how to run the files by hand):
- [Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata):
  the original Qwen, our GPTQ (`--family qwen-nvfp4-gptq`, the default; with a switch that turns its censorship off)
- [Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata):
  the original Qwen, censored, NVIDIA's NVFP4 converted (`--family qwen-nvidia-nvfp4`)

## What's new

- **On upstream 0.1.42 (0.1.42-nvfp4.1).** Upstream's own changes now run here too:
  - **The route tail skip is on by default** (`STRATA_ROUTE_TAIL_SKIP=7`): a missed expert that every token of a
    verify window routes at rank 7-9 is neither fetched nor computed. With `qwen-nvfp4-gptq` decode is 11.5% faster
    (193.7 against 174.0 tokens/s), but the answers change more than upstream's models show: teacher-forced KL
    0.035 against the skip off (top token the same 93.7% of the time), more than the censorship switch's 0.026.
    No loops in 3 long Claude Code replays. `STRATA_ROUTE_TAIL_SKIP=0` (in the config's `"env"`) turns it off
    ([numbers](docs/NVFP4.md#upstream-0142-2026-10-10-release-0142-nvfp41)).
  - **Decode's PCIe share is upstream's measured one** (from the CPU pool's speed: 0.35 here); this fork's per-layer
    cost model is gone. The same speed as the old fixed share.
  - Four fixes this fork sent upstream are upstream's code now: the unbuffered start of the resident RAM mode, the
    page-file advice (a fixed 65536 MB), the K/V trim's residency upload, the prompt buffers' size.
- **Censorship can be turned off (0.1.41-nvfp4.5).** The original Qwen (`qwen-nvfp4-gptq`, and the original Qwen's
  GGUFs `qwen` and `unsloth`) now runs with or without censorship. The switch is off by default. To turn it on:
  - for one chat: the web app's **"Disable censorship"** switch;
  - for one request: the API field `"uncensored": true`;
  - as the default: install with `START-HERE.bat --uncensored on`, or set **Disable censorship by default** in the
    web page's Model settings (0.1.41-nvfp4.7; used from the model's next start).

  The engine removes the model's refusal direction at run time in layers 8-33 only; the weights are untouched.
  - With it on and thinking on, 0 of 104 English and 0 of 30 Ukrainian test requests were refused.
  - The model thinks as long as the original: 1,947 against 1,699 tokens on a long Claude Code request.
  - Never for NVIDIA's model: its license forbids it.
  - [Details](#quick-start).
- **orca-nvfp4 is removed from the fork (0.1.41-nvfp4.5).** OrcaRouter's uncensored model, the former default, is no
  longer offered.
  - Its weight edit made it a much weaker agent. On the same Claude Code task it took 10-70 steps against the
    original's 175 and gave a clearly worse result. It thinks ~40% less.
  - Instead, use `--family qwen-nvfp4-gptq --uncensored on`: the original's weights, without the censorship.
  - An existing orca install keeps working.
- **Recommended sampling (0.1.41-nvfp4.6).** Setup writes the model's own `temperature 1.0, top_p 0.95, top_k 20`
  into the config, as the default for requests that send none. Before, Claude Code decoded greedy.
- **Model settings has a Disable censorship switch too (0.1.41-nvfp4.7):** the default for requests that do not
  say, saved in the run config (`"uncensored"`) and used from the next start; greyed out for a model without the
  refusal projection (NVIDIA's, Swift, the Coder).

The design is upstream's: the routed experts live in RAM, the most-used ones are cached in VRAM, the misses are
computed on the CPU and fetched over PCIe in parallel with the GPU, and an MTP draft head speculates. Upstream's README
is kept as [README.upstream.md](README.upstream.md). [docs/NVFP4.md](docs/NVFP4.md) has every measurement behind this
fork, and the [releases](https://github.com/sergqwer/strata-nvfp4/releases) say what changed in each.

## Quick start

```bat
git clone https://github.com/sergqwer/strata-nvfp4
cd strata-nvfp4
START-HERE.bat
```

(Linux: `./setup.sh`.) Setup checks the PC, asks which model, how much context and whether the model should read
images, installs the engine and the model, and starts it on http://127.0.0.1:8080 (OpenAI and Anthropic APIs, a chat
page).

| `--family` | the model | license |
| --- | --- | --- |
| `qwen-nvfp4-gptq` (the default) | the original [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), every expert NVFP4 by our GPTQ; censored unless `--uncensored on`: [Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata) | Qwen Community License 1.0 |
| `qwen-nvidia-nvfp4` | **NVIDIA's** NVFP4 of the original, censored Qwen3.8-Flash-Next ([nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4), ModelOpt), converted for Strata with its experts bit for bit: [Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata) | NVIDIA Open Model License and the Qwen Community License 1.0 |
| `qwen`, `swift`, `coder`, `unsloth` | upstream Strata's GGUF models: ISTA-DASLab's GSQ-RCO quants of Qwen3.8-Flash-Next, Swift 1.5 and the Coder, Unsloth's ~4-bit files | as upstream lists them |

- **The NVFP4 models come ready-made:** the expert pack, the dense GGUF, the FP8 n-gram table, the BF16 embedding
  and the MTP draft head are downloaded at a pinned revision and checked against their SHA-256; nothing is converted
  on your PC. `qwen-nvfp4-gptq`'s routed experts are NVFP4 by our GPTQ from Qwen's BF16: 33.3% of plain rounding's
  error and 44.8% of NVIDIA's (NVIDIA's own is 74.4% of plain rounding's), lower in all 48 layers, at the same size
  and speed (179 against 175 tokens/s). `qwen-nvidia-nvfp4` is NVIDIA's own checkpoint, quantized by NVIDIA and only
  converted here. The two share the n-gram table, the embedding and the draft head: the second one links them
  ([The original Qwen in NVFP4](docs/NVFP4.md#the-original-qwen-in-nvfp4-2026-10-09-release-0141-nvfp44)).
- **`qwen-nvfp4-gptq` is the default** where the PC meets the [requirements](#requirements); a PC below them gets a
  GGUF model as the default, and `--family qwen-nvfp4-gptq --yes` installs it anyway.
- **Disable censorship (off by default):** for `qwen-nvfp4-gptq` and the original Qwen's GGUFs (`qwen`, ISTA's;
  `unsloth`) setup always loads this fork's refusal-direction projection ([data/uncensor](data/uncensor/README.md))
  and leaves it off: the web app's "Disable censorship"
  switch (Sampling) and the API field `"uncensored": true` turn it on for a request, and setup's question "Disable
  censorship?" (default no; `--uncensored on|off`) sets the default, `"uncensored"` in `strata-<model>.json`. Off,
  a request runs the original model. On, the engine removes the model's refusal direction from its residual stream
  after layers 8-33: with thinking on, 0 of 104
  English and 0 of 30 Ukrainian held-out harmful requests refused (the original refuses almost all), KL 0.026 to the
  original, and its thinking as long as the original's (1,947 against 1,699 tokens on a long Claude Code request,
  log p(`</think>`) +0.019 nats). On the GGUFs, against each quant's own stock model (measured on Q2_0, IQ2_XS and
  UD-Q4_K_XL): thinking-on refusals 1, 0 and 0 of 104, KL 0.016, 0.017 and 0.028, log p(`</think>`) +0.027 to
  +0.040 on Q2_0 and +0.050 on UD-Q4_K_XL's thinking (upstream's vector +0.087 to +0.149 and +0.163); the same
  vector, unchanged, on all three. It replaces upstream's speed projection there. IQ2_XS loops
  in long greedy agentic thinking by itself (3 of 3 replays of the long request, with the switch on or off): choose
  another size for agents. Upstream's `experimental_speed_projection` is the same switch. Never for
  `qwen-nvidia-nvfp4`, where nothing is loaded: the NVIDIA Open Model License does not allow bypassing its safety
  guardrails. The vector is
  under the Qwen Community License 1.0; removing refusals removes a safety behaviour, and what the model writes with
  it on is your responsibility.
- **`orca-nvfp4` was withdrawn in 0.1.41-nvfp4.5: a much weaker agent.** OrcaRouter's abliteration edits all 149
  residual writers of the model. On the same Claude Code task it ran 10-70 steps against the original Qwen's 175,
  with a clearly worse result, and it ends its thinking sooner (log p(`</think>`) +0.119 nats; ~1,000-1,150 thinking
  tokens on the long request against the original's ~1,700). `qwen-nvfp4-gptq` with its censorship switch on is
  the uncensored model with the original's weights instead. An install of `orca-nvfp4` keeps working and nothing of it is deleted; setup says so
  on each run.
- **`huihui-nvfp4` was withdrawn in 0.1.41-nvfp4.3: it often loops in long thinking** (9 of 10 greedy replays of a
  real 72K-token Claude Code request, on this engine and on 0.1.40.3's; the other models none of 5). An install of it
  keeps working too.
- **The engine is always this fork's:** on Windows the ready-made one from this repository's releases
  (`strata-windows-x64.zip`, checked against GitHub's SHA-256), on Linux compiled from this source. Upstream's engine
  has no NVFP4 path, so setup never installs it and replaces one an older setup installed.
- **AMD cards:** the NVFP4 models run on NVIDIA cards only, and this fork publishes no AMD engine. For the GGUF models
  setup compiles one on Linux; on Windows build it first with `tools\hip\build_windows.bat`, then run
  `START-HERE.bat --backend hip --prebuilt <its dist folder>` ([docs/AMD_HIP.md](docs/AMD_HIP.md)).
- **Sampling:** setup writes the model's recommended sampling into `strata-<model>.json` (`"sampling":
  {"temperature": 1.0, "top_p": 0.95, "top_k": 20}`, Qwen's generation_config.json) as the default for requests;
  a request's own values win, and temperature 0 means greedy. Before, a client that sent no temperature (Claude
  Code) decoded greedy, which Qwen advises against with thinking. A setup run again keeps the values you changed;
  `UPDATE.bat` adds the block to an older config that has none.
- **Options:** `--family qwen-nvidia-nvfp4` picks a model without the menu, `--uncensored on|off` answers the
  censorship question (the switch's default), `--check` says what fits this PC, `--dry-run`
  shows what setup would download, install and write for an NVFP4 model (the engine's arguments too) and changes
  nothing.
- **Updates:** `UPDATE.bat` (`./update.sh`) runs `git pull`, then installs the engine the new version needs; the
  model files stay. The release also has a stand-alone Windows bundle (`strata-nvfp4-v<version>-windows-x64.zip`) that
  updates itself with `update.cmd`; its `prepare-model.cmd` downloads `qwen-nvfp4-gptq` ready-made, the files and
  revision setup pins, with the same censorship switch, off by default.

## Requirements

- **GPU:** an NVIDIA RTX 20, 30, 40 or 50 card with 12 GB of VRAM or more, driver 580 or newer (CUDA 13). The
  release engine has code for sm_75, 86, 89 and 120, and the optional FP4 x FP4 prompt unit for 120a (RTX 50); the
  other cards read prompts on int8 tensor cores instead. The dense weights, the draft head and the K/V come first and
  the rest of the VRAM caches experts, so a smaller card decodes slower. Setup picks the context:

  | card's VRAM | `--max-context` |
  | --- | ---: |
  | 32 GB (RTX 5090) | 262144 |
  | 24 GB (RTX 3090 / 4090) | 131072 |
  | 16 GB (RTX 4080 / 5080 / 4060 Ti 16 GB) | 65536 |
  | 12 GB (RTX 3060 12 GB / 4070) | 32768 |

  12 GB at 262144 and 8 GB cards at any context stop with *no VRAM is left for the expert cache*. Slots and decode
  rates per card size: [Other GPUs](docs/NVFP4.md#other-gpus-rtx-20-30-and-40-2026-09-30).
- **CPU:** x86-64 with AVX2, FMA and F16C (Intel Haswell, AMD Zen or newer); the release engine names any other CPU
  and stops. The CPU computes its share of the experts with kernels chosen at run time
  ([CPU experts](docs/NVFP4.md#cpu-experts)):
  - **AVX-512** with F, BW, VL, VNNI and VBMI (e.g. Zen 4 and 5): NVFP4 rows in 512-bit lanes;
  - **otherwise AVX2 + FMA + F16C** (e.g. Zen 2 and 3, Intel Core 12th-14th gen): NVFP4 rows in 256-bit lanes, bit
    for bit ggml-cpu's own AVX2 dot. Where the CPU also has AVX-VNNI (Intel Core 12th-14th gen and Core Ultra), the
    GGUF models' i-quant and Q2_0 rows take AVX-VNNI copies of their kernels (the same sums);
  - **AVX or SSE4.2 only:** upstream's experimental older-CPU build, which setup compiles on that PC; slow
    ([Older CPUs](docs/INSTALL.md#older-cpus-experimental)).
- **RAM:** 64 GB or more.
  - From 92 GiB installed every expert stays in RAM: 63 GiB of experts, ~67 GiB of physical RAM for the engine, ~71
    with the image encoder and the server. A 96 GB PC qualifies: Windows lists it as 93-95.6 GiB.
  - Below 92 GiB (a 64 GB PC) the low-RAM mode starts by itself: the experts outside VRAM are pinned, hottest first,
    up to the free RAM less 6 GiB, and the rest are read from `experts.bin` when needed. `--low-ram` forces it,
    `--no-low-ram` turns it off, `--ram-budget GIB` caps it
    ([64 GB of RAM](docs/NVFP4.md#64-gb-of-ram-2026-09-30), [A 96 GB PC](docs/NVFP4.md#a-96-gb-pc-2026-10-09-release-0141-nvfp42)).
- **Page file: set a fixed 65536 MB** (initial = maximum). Windows lets all programs together commit at most RAM +
  page file, and the engine commits ~100 GiB with every expert in RAM (~80 GiB in the low-RAM mode): the experts
  plus ~30 GiB that WDDM charges for the VRAM it uses. Nothing of the model is paged out, but short of commit the
  expert cache in VRAM is made smaller, and the model can run significantly slower (a chat's round 17% and 38%
  slower, measured) or not start. Below 60000 MB of page file in all for sure, the engine prints a `WARNING` at
  start and setup a framed warning (`--check` and the install). Only what is there for sure counts: a
  system-managed or growing file (initial below maximum) counts at its size now, since it may not grow in time
  while WDDM charges the VRAM (upstream #60), and the warning then says to make it fixed. To set it: Win+R
  `sysdm.cpl` > Advanced > Performance Settings > Advanced > Virtual memory > Change: Custom size, initial and
  maximum 65536 MB, then restart Windows.
- **Disk:** ~130 GB for an NVFP4 model (128.8 GB of files; the expert pack and the 51 GB n-gram table are most of
  it). Setup checks the free space first. Use the fastest NVMe drive you have: every start reads 63 GiB of experts.

## Speed

RTX 5090 (32 GB, PCIe 5 x16), Ryzen 9 9950X3D, DDR5-5600, Samsung 9100 PRO, Windows 11. Measured with `huihui-nvfp4`
(withdrawn since; the original Qwen's packs have the same format and size) and the arguments setup writes (262K context, int8 K/V,
images on). The card also drives the desktop, so runs vary by a few
percent.

| | 128 GB | 96 GB (emulated) | the low-RAM mode, 88 GiB free |
| --- | ---: | ---: | ---: |
| a chat answer, ms a round (~2.3 tokens a round) | 12.7 | 13.7 | 20.3 |
| a 32K prompt | 3.7 s | 3.9 s | 8.4 s |

The 96 GB PC is the 128 GB one with 95.6 GiB reported as installed and 88 GiB of RAM left free; the last column is the
low-RAM mode forced on it. A 64 GB PC runs the low-RAM mode with a smaller RAM budget, and a smaller card caches
fewer experts: [A 96 GB PC](docs/NVFP4.md#a-96-gb-pc-2026-10-09-release-0141-nvfp42),
[64 GB of RAM](docs/NVFP4.md#64-gb-of-ram-2026-09-30) and
[Other GPUs](docs/NVFP4.md#other-gpus-rtx-20-30-and-40-2026-09-30) have those runs.

## What this fork adds

Each change was measured (first-token KL against a reference, interleaved speed runs); the links go to the numbers.

- **NVFP4 experts:** 4.5 bits a weight (E2M1 codes, an FP8 scale per 16 values, an FP32 scale per expert matrix
  applied to each projection's output) instead of upstream's 2-4-bit GGUF quants
  ([Where the global scales go](docs/NVFP4.md#where-the-global-scales-go)).
- **GPTQ re-quantization from BF16** (`tools/requant.py`), which made the packs above, calibrated on the engine's own
  MoE inputs; Q8_0 `down` in chosen layers is an option for a size budget
  ([Re-quantized from BF16](docs/NVFP4.md#re-quantized-from-bf16-2026-10-02-release-0132-nvfp42)).
- **Less rounding elsewhere:** the n-gram table in its shipped FP8, the token embedding in BF16, RoPE angles from a
  float64 table, FP32-exact inputs to the prompt path's BF16 projections, int8 K/V behind a Hadamard rotation
  ([Where precision is still lost](docs/NVFP4.md#where-precision-is-still-lost-audit-2026-09-29)).
- **The prompt path on tensor cores:** the experts' gate/up on RTX 50's FP4 tensor cores with two FP4 terms per
  activation, int8 elsewhere; the prompt attention on INT8 tensor cores; prompt chunks up to 32K
  ([Prompt path precision](docs/NVFP4.md#prompt-path-precision),
  [A round of kernels](docs/NVFP4.md#a-round-of-kernels-2026-10-07-release-0140-nvfp43)).
- **NVFP4 CPU kernels** for the CPU's share: AVX-512 1.8-3.7x ggml-cpu, AVX2 2.0-2.4x at 4-8 tokens
  ([CPU experts](docs/NVFP4.md#cpu-experts)).
- **A faster start:** the expert arena loads unbuffered on its own thread beside the dense weights, ~7 s for 63 GiB
  from a PCIe 5 drive ([Loading](docs/NVFP4.md#loading)).
- **The K/V grows with the context** (upstream's `--kv-grow`, on here): VRAM the window does not use yet caches
  experts, 7,060 -> 8,290 slots at 262K; `--no-kv-grow` allocates the whole window
  ([Faster without changing an answer](docs/NVFP4.md#faster-without-changing-an-answer-2026-10-01-release-0131-nvfp42)).
- **Small prompt chunks shared with the CPU** (this fork's change, upstream's default since 0.1.41); on top of
  upstream's, NVFP4 packs share chunks up to 4,096 tokens and the DeltaNet recurrence runs in chunks from 128 tokens
  ([Small prompt chunks with the CPU](docs/NVFP4.md#small-prompt-chunks-with-the-cpu-2026-10-06-release-0139-nvfp44)).
- **The adaptive VRAM tier** re-ranks every 2 rounds, up to 192 swaps, counts x0.92 (upstream: 4 / 96 / x0.7), and
  admits the PCIe share's experts into VRAM
  ([The adaptive VRAM tier](docs/NVFP4.md#the-adaptive-vram-tier-and-an-audit-2026-10-01-release-0131-nvfp43),
  [A second round of kernels](docs/NVFP4.md#a-second-round-of-kernels-2026-10-08-release-01403-nvfp41)).
- **The low-RAM mode** below 92 GiB installed, read unbuffered (see [Requirements](#requirements)).
- **Every RTX 20-50 card with 12 GB or more:** only the optional FP4 x FP4 unit needs Blackwell; each generation's
  code path was tested on the RTX 5090 ([Other GPUs](docs/NVFP4.md#other-gpus-rtx-20-30-and-40-2026-09-30)).
- **Images on the CPU:** the encoder runs in RAM, so the expert cache keeps all its VRAM; upstream's GPU encoder
  (FP16 flash attention) was up to 11% off an FP32 reference. A picture a tool returns (Claude Code's `Read`) reaches
  the encoder too ([Images](docs/NVFP4.md#images-2026-09-30)).
- **A draft vocabulary with Cyrillic:** an answer in Ukrainian decodes 2.11 tokens a round instead of 1.40;
  `tools/draft_vocab.py --add cjk` builds upstream's CJK subset
  ([Upstream 0.1.28 merged](docs/NVFP4.md#upstream-0128-merged-2026-09-30)).
- **The server starts a fresh engine after a reply of "!"** ([below](#if-a-reply-turns-into-)).

## Run

Setup writes `strata-<family>.json` (the server's config: the engine and its arguments) and `run-<family>.bat`
(`.sh`) next to `setup.py`. `START-HERE.bat` starts the installed model again without questions; `run-qwen-gptq-nvfp4.bat`
starts the server with that config without setup's checks (`serve/server.py --engine strata --config
strata-qwen-gptq-nvfp4.json --port 8080`). A key or an `"env"` entry you add to the config stays when setup runs again;
engine flags added to its `"args"` by hand do not.

**Images:** answer yes to setup's question (or `--vision yes`). The image encoder runs on the CPU while the engine
waits; OpenAI `image_url` parts and Anthropic image blocks (a screenshot pasted into Claude Code) work.

Running the engine and the server without setup, and converting the model yourself:
[Build it yourself](#build-it-yourself-advanced).

## Claude Code

```bat
set ANTHROPIC_BASE_URL=http://127.0.0.1:8080
set ANTHROPIC_API_KEY=local
set ANTHROPIC_MODEL=strata-nvfp4
set ANTHROPIC_SMALL_FAST_MODEL=strata-nvfp4
claude
```

The server answers under its own model name whatever the request names. The model is hybrid: its Gated DeltaNet
layers keep a recurrent state that cannot be cut back to a position, so the server keeps a checkpoint at every turn
boundary, and an agent's next turn reads only its new tokens. An edit in the middle of a history falls back to the
checkpoint just before it. `/v1/messages/count_tokens` is served. In `strata-<family>.json`:

- `"anthropic_thinking": "on_request"`: a request that does not ask for thinking (Claude Code's small helper calls)
  gets none;
- `"fit_max_tokens": true`: a request whose `max_tokens` would run past the context is shortened to the room left
  instead of refused ([docs/DETAILS.md](docs/DETAILS.md#using-it)).

## Large pages (why a reboot)

This is not more memory, it is bigger pages. Windows maps memory in 4 KB pages; the CPU part of every token reads
experts from the 63 GiB arena at DRAM speed, and each page needs a TLB entry. With 2 MB pages the CPU pool holds
steady at ~7.5 ms a round on this machine, where 4 KB pages wandered between 7.4 and 11 ms. The decode rate moves
little (4 KB pages cost a chat 0.4 ms a round here, a 32K prompt nothing); the point is the steadiness.

Windows gives large pages only to an account that holds *Lock pages in memory* (SeLockMemoryPrivilege), and puts a
privilege into a sign-in's token only when that sign-in starts. So:

1. `powershell -ExecutionPolicy Bypass -File tools\enable-large-pages.ps1` asks for admin (UAC) and grants it to the
   current user through `secedit` (works on Windows Home, which has no `secpol.msc`). The policy as it was is saved
   to `%LOCALAPPDATA%\strata-large-pages\before.inf`, and `-Revoke` takes it back.
2. **Sign out and back in, or reboot.** Locking the screen is not a new sign-in.
3. Check: the same script with `-Check`, or the engine's start line `expert arena: ... large pages (2097152 B)`.

Without the privilege the engine says `large pages refused ... VirtualAlloc error 1314` and runs on 4 KB pages.
Error 1450 means the privilege is there but Windows found too few free 2 MB blocks (memory fragmented after a long
uptime): it falls back the same way, and a reboot clears it. The arena is locked in RAM either way.

## Switches

Engine flags go into the config's `"args"`, environment variables into its `"env"` (`{"STRATA_KV_ROT": "0"}`).
The switches kept only for A/B runs (each gives the same bits or a measured alternative) are in
[docs/NVFP4.md](docs/NVFP4.md#switches-kept-for-ab-runs).

| | |
| --- | --- |
| `--prefill auto:8192\|16384` | cap `--prefill auto`'s chunk (default 32768 here, 8192 upstream) |
| `--low-ram` / `--no-low-ram` | the low-RAM mode on / off (default: on below 92 GiB installed) |
| `--ram-budget GIB` | the low-RAM mode with at most GIB of pinned expert copies |
| `STRATA_RESIDENT_HEADROOM_GIB=6` | RAM the low-RAM mode leaves free |
| `--no-kv-grow`, `STRATA_KV_GROW=0` | the K/V allocated for the whole context at start |
| `STRATA_KV_GROW_INIT` / `_STEP` | the K/V's cells at start (16384) and its growth step (8192) |
| `--vram-reserve-mib N` | VRAM left unused (default 700); also a smaller card's budget on a bigger one |
| `--pcie-frac F` | a fixed share of decode's cache misses fetched over PCIe (default: upstream's, set from the CPU pool's speed in the first decode windows) |
| `STRATA_PCIE_FRAC_DEFAULT=old` | decode keeps the link probe's fixed PCIe share (upstream 0.1.41's rule) |
| `STRATA_ROUTE_TAIL_SKIP=0` | decode computes every missed expert again (upstream's default since 0.1.42 skips those routed only at rank 7-9; this fork measured KL 0.035 with it on) |
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
| `STRATA_BANG_AUDIT=1` | the "!" audit ([below](#if-a-reply-turns-into-)) |
| `STRATA_KVG_CHECK=1\|2\|3`, `STRATA_DBG_NAN_VERIFY=1` | the audit's checks at every K/V growth and trim and at each prompt's and reply's end (2: the bytes too; 3: the table after every window) / name the first decode window with non-finite logits |
| `STRATA_VERIFY_ARENA=1` | print a checksum of the loaded arena |
| `STRATA_DUMP_FIRST_LOGITS=file` | write the first generated token's logits (compare prompt paths) |
| `STRATA_DUMP_MOE_INPUT=prefix`, `STRATA_DUMP_MOE_LAYER=l\|all` | dump the real MoE input rows of one layer / every layer (the GPTQ calibration) |
| `STRATA_EMULATE_CC=75\|86\|89` | tests: answer as that generation (with an engine built as its PTX, e.g. `86-virtual`) |
| `STRATA_QSA_WARP=1\|select\|attn` | the pre-sm_80 QSA kernels on any card, as RTX 20 runs them |
| `STRATA_EMULATE_RAM_GIB=N` | tests: the engine reads N GiB as installed, which decides the low-RAM mode |

## If a reply turns into "!"

A long Claude Code conversation once turned into "!" mid-reply (token 0, what the sampler gives when no logit is
finite), and the cause is not found yet
([docs](docs/NVFP4.md#-after-a-long-conversation-2026-10-05-release-0139-nvfp42)). The server starts a fresh engine
at the next request after such a reply. To find out what was damaged, give the engine `STRATA_BANG_AUDIT=1`
(`"env": {"STRATA_BANG_AUDIT": "1"}` in `strata-<family>.json`). It costs nothing until a reply turns into token 0;
then, once, it compares every expert in the VRAM cache with its RAM copy and with `experts.bin` and logs what
differs (`strata kvg audit` lines). Please send us the engine log (`strata-<family>.log`) from the engine's start to
the end of those lines ([Upstream 0.1.41](docs/NVFP4.md#upstream-0141-2026-10-08-release-0141-nvfp41)).

## Tests

| executable | checks |
| --- | --- |
| `nvfp4_avx512_parity [experts.bin]` | AVX-512 and AVX2 rows vs ggml-cpu, the AVX2 ones bit for bit (ctest `nvfp4_cpu_parity`; `--expert`: one whole expert vs FP64; `--bw T N [ggml\|avx2]`: DRAM rate) |
| `nvfp4_expert_gpu_parity experts.bin` | the GPU decode path's experts vs FP64, one per sampled layer |
| `mmq_nvfp4_parity experts.bin` | MMQ products vs FP64 (`--group`, `--layers`, `--real dump layer`) |

## Limits

- Built and measured on Windows with one RTX 5090, a Ryzen 9 9950X3D and 128 GB of RAM. RTX 20/30/40 cards, smaller
  VRAM, 64 and 96 GB of RAM and smaller CPUs were tested on that PC through their own code paths and budgets
  (`STRATA_EMULATE_CC`, `--vram-reserve-mib`, a RAM ballast, `STRATA_FORCE_AVX2=1`, `--pool-workers`), not on the
  real hardware.
- The NVFP4 model runs on one GPU: the low-RAM mode and the growing K/V are one-GPU.
- The low-RAM mode's unbuffered reads are Windows-only; elsewhere it reads through the page cache.
- Decode is bound by host RAM bandwidth: the CPU pool and the PCIe share read the same DDR5 (~84 of ~90 GB/s here).
  More VRAM for the expert cache or faster RAM is what moves it
  ([A second round of kernels](docs/NVFP4.md#a-second-round-of-kernels-2026-10-08-release-01403-nvfp41)).
- The models are abliterated fine-tunes: they do not refuse. What they are used for is on whoever runs them.

## Build it yourself (advanced)

Setup does all of this; `START-HERE.bat --build` compiles the engine from this source instead of taking the
release's (it installs the build tools first, asking). By hand, clone llama.cpp at the pinned commit, then run
`release\build-release.cmd` (the release engine: AVX2 ggml-cpu, sm_75, 86, 89 and 120, into `build-release\`):

```bat
git clone https://github.com/ggml-org/llama.cpp third_party\llama.cpp
git -C third_party\llama.cpp checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d
release\build-release.cmd
```

It needs Visual Studio 2022 Build Tools, CUDA 13, CMake, Ninja and Python 3.11+.
[By hand](docs/NVFP4.md#by-hand-build-convert-re-quantize-run) in docs/NVFP4.md has the rest: a one-card CMake
build, converting jpezzulli's ModelOpt checkpoint ("Prepare the model"), re-quantizing the experts from BF16 by GPTQ,
and running the engine and the server without setup. Each Hugging Face model card also shows how to run its files by
hand.

**Releasing** (maintainers): `release/make_windows_bundle.py`, then `release/publish.py`; their headers say what each
checks.

## License

MIT, as upstream ([LICENSE](LICENSE), copyright Niko1221 and the Strata contributors). llama.cpp / ggml code compiled
into the build is MIT as well.

## Star History

<a href="https://www.star-history.com/#sergqwer/strata-nvfp4&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=sergqwer/strata-nvfp4&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=sergqwer/strata-nvfp4&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=sergqwer/strata-nvfp4&type=Date" />
 </picture>
</a>
