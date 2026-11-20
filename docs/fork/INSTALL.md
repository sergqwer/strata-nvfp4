# Install and run Strata NVFP4

How to install this fork, what the PC needs, how to run it and how to build it yourself. Back to the
[README](../../README.md). Upstream's own install guide (its desktop app and Terminal) is
[docs/INSTALL.md](../INSTALL.md).

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
| `qwen-nvfp4-mix` (128 GB) | the original Qwen3.8-Flash-Next, our GPTQ NVFP4 with Q8_0 in 38 of 48 layers, more exact, half the speed; censored unless `--uncensored on`: [Maximilian228/Qwen3.8-Flash-Next-NVFP4-Q8-Mix-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-Q8-Mix-Strata) | Qwen Community License 1.0 |
| `qwen-q8` (128 GB) | the original Qwen3.8-Flash-Next, every expert Q8_0, the most exact, runs in the low-RAM mode; censored unless `--uncensored on`: [Maximilian228/Qwen3.8-Flash-Next-Q8_0-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-Q8_0-Strata) | Qwen Community License 1.0 |
| `qwen`, `swift`, `coder`, `unsloth` | upstream Strata's GGUF models: ISTA-DASLab's GSQ-RCO quants of Qwen3.8-Flash-Next, Swift 1.5 and the Coder, Unsloth's ~3.5-4-bit files (UD-IQ4_XS, UD-Q3_K_XL, UD-Q4_K_XL) | as upstream lists them |

- **The NVFP4 models come ready-made:** the expert pack, the dense GGUF, the FP8 n-gram table, the BF16 embedding
  and the MTP draft head are downloaded at a pinned revision and checked against their SHA-256; nothing is converted
  on your PC. `qwen-nvfp4-gptq`'s routed experts are NVFP4 by our GPTQ from Qwen's BF16: 33.3% of plain rounding's
  error and 44.8% of NVIDIA's (NVIDIA's own is 74.4% of plain rounding's), lower in all 48 layers, at the same size
  and speed (179 against 175 tokens/s). `qwen-nvidia-nvfp4` is NVIDIA's own checkpoint, quantized by NVIDIA and only
  converted here. The two share the n-gram table, the embedding and the draft head: the second one links them
  ([The original Qwen in NVFP4](../NVFP4.md#the-original-qwen-in-nvfp4-2026-10-09-release-0141-nvfp44)).
  `qwen-nvfp4-mix` and `qwen-q8` share all of `qwen-nvfp4-gptq`'s files but the experts
  ([Q8_0 for a 128 GB PC](../NVFP4.md#q8_0-for-a-128-gb-pc-2026-10-11-release-0142-nvfp42)).
- **Draft layers (0.2.15-nvfp4.4):** for the GPTQ set (`qwen-nvfp4-gptq`) setup, `UPDATE.bat` (`./update.sh`) and
  the release folder's `prepare-model.cmd` also fetch [Basalt](https://github.com/jesdga95/basalt)'s self-distilled MTP draft layer (0.12 GB,
  checked by SHA-256) into the model's `mtp\rt-tuned`; the set's own experts and draft vocabulary are linked beside it.
  It is under the Qwen Community License 1.0, like the model. Auto drafts with the MTP and Basalt's drafting on CUDA,
  one request at a time; without the tuned layer it uses the set's own MTP layer, which accepts fewer drafts. The
  NVIDIA, mix and Q8_0 sets keep their own layer. On a card that holds as many experts as an RTX 5090, setup also
  fetches the DFlash2 drafter (1.2 GB once), the Auto of 0.2.15-nvfp4.2 and .3, which `STRATA_AUTO_NEW_MTP=0` brings
  back. After `update.cmd` from an older release, run `prepare-model.cmd` once to get the tuned layer.
- **`qwen-nvfp4-gptq` is the default** where the PC meets the [requirements](#requirements); a PC below them gets a
  GGUF model as the default, and `--family qwen-nvfp4-gptq --yes` installs it anyway.
- **Disable censorship (off by default):** for `qwen-nvfp4-gptq`, `qwen-nvfp4-mix`, `qwen-q8` and the original Qwen's
  GGUFs (`qwen`, ISTA's; `unsloth`) setup always loads this fork's refusal-direction projection ([data/uncensor](../../data/uncensor/README.md))
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
  `START-HERE.bat --backend hip --prebuilt <its dist folder>` ([docs/AMD_HIP.md](../AMD_HIP.md)).
- **Sampling:** a request that sends none gets the model card's sampling from the server (upstream's, since
  0.2.0-beta-nvfp4.1): with thinking temperature 1.0, top_p 0.95, top_k 20 (Qwen's generation_config.json), without
  thinking Qwen's instruct set. A request's own values win, and temperature 0 means greedy. A `"sampling"` block in
  `strata-<model>.json` (setup wrote one from 0.1.41-nvfp4.6 to 0.1.44.1-nvfp4.2, or your own) still wins over the
  card; setup writes none now and keeps the one an install has.
- **Options:** `--family qwen-nvidia-nvfp4` picks a model without the menu, `--uncensored on|off` answers the
  censorship question (the switch's default), `--check` says what fits this PC, `--dry-run`
  shows what setup would download, install and write for an NVFP4 model (the engine's arguments too) and changes
  nothing.
- **Updates:** `UPDATE.bat` (`./update.sh`) runs `git pull`, then installs the engine the new version needs; the
  model files stay. The release also has a stand-alone Windows bundle (`strata-nvfp4-v<version>-windows-x64.zip`) that
  updates itself with `update.cmd`; its `prepare-model.cmd` downloads `qwen-nvfp4-gptq` ready-made, the files and
  revision setup pins, with the same censorship switch, off by default.
  `strata update` (`python strata.py update`) runs the same updater for the folder it is in: `update.cmd`'s in a
  release folder, `UPDATE.bat` / `./update.sh` in a clone; `strata update --check` only says whether a newer release
  (or, in a clone, newer commits) is there. Upstream's signed engine feed is used only when you name it (`--feed`).
- **Update notice:** at start the server asks GitHub, at most once a day, which strata-nvfp4 release is the newest
  (one small request, no token, nothing about the PC; the answer is kept in `.update/update-check.json`). When it is
  newer than the installed one, the server's window says so with the command that updates this folder, and the web
  page shows a small link in its header (hidden with its x until the next release; About lists it too). Nothing is
  downloaded or installed by itself. Off: `STRATA_UPDATE_CHECK=0`, or `"update_check": false` in the run config
  (About > Settings); `--offline` never asks.

## Requirements

- **System: Windows 10/11 or Linux, run natively. Not WSL2.** Under WSL2 the NVIDIA driver page-locks only about 1 GB
  of RAM ([NVIDIA's known limitation](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications):
  "pinned system memory ... availability for applications is limited"), and this fork needs page-locked RAM for the
  experts' RAM copy, for the GPU's reads of missed experts over PCIe, for KV streaming and for a layer split's
  hand-offs. Under WSL2 none of that can be pinned: the KV cache must stay in VRAM, every miss goes to the CPU, and a
  two-GPU layer split does not start ([#7](https://github.com/sergqwer/strata-nvfp4/issues/7)). On a Windows PC run
  `START-HERE.bat` in Windows itself.
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
  rates per card size: [Other GPUs](../NVFP4.md#other-gpus-rtx-20-30-and-40-2026-09-30).
  The decode and prompt kernels of 0.2.15-nvfp4.4 ([Switches](ENVIRONMENT.md#rtx-50-0215-nvfp44)) are on by default on
  RTX 50 cards only; the older cards take their older paths.
- **CPU:** x86-64 with AVX2, FMA and F16C (Intel Haswell, AMD Zen or newer); the release engine names any other CPU
  and stops. The CPU computes its share of the experts with kernels chosen at run time
  ([CPU experts](../NVFP4.md#cpu-experts)):
  - **AVX-512** with F, BW, VL, VNNI and VBMI (e.g. Zen 4 and 5): NVFP4 rows in 512-bit lanes;
  - **otherwise AVX2 + FMA + F16C** (e.g. Zen 2 and 3, Intel Core 12th-14th gen): NVFP4 rows in 256-bit lanes, bit
    for bit ggml-cpu's own AVX2 dot. Where the CPU also has AVX-VNNI (Intel Core 12th-14th gen and Core Ultra), the
    GGUF models' i-quant and Q2_0 rows take AVX-VNNI copies of their kernels (the same sums);
  - **AVX or SSE4.2 only:** upstream's experimental older-CPU build, which setup compiles on that PC; slow
    ([Older CPUs](../SOURCE_INSTALL.md#older-cpus-experimental)).
- **RAM:** 64 GB or more.
  - From 92 GiB installed every expert stays in RAM: 63 GiB of experts, ~67 GiB of physical RAM for the engine, ~71
    with the image encoder and the server. A 96 GB PC qualifies: Windows lists it as 93-95.6 GiB.
  - Below 92 GiB (a 64 GB PC) the low-RAM mode starts by itself: the experts outside VRAM are pinned, hottest first,
    up to the free RAM less 6 GiB, and the rest are read from `experts.bin` when needed. `--low-ram` forces it,
    `--no-low-ram` turns it off, `--ram-budget GIB` caps it
    ([64 GB of RAM](../NVFP4.md#64-gb-of-ram-2026-09-30), [A 96 GB PC](../NVFP4.md#a-96-gb-pc-2026-10-09-release-0141-nvfp42)).
  - From 92 GiB it also starts where the pack's experts and 16 GiB more do not fit, and then leaves 10 GiB free: on
    a 128 GB PC (127.2 GiB listed) the all-Q8_0 pack (119.5 GiB) runs in it, the NVFP4 + Q8_0 mix (103.1 GiB) keeps
    every expert in RAM. Both are for a 128 GB PC
    ([Q8_0 for a 128 GB PC](../NVFP4.md#q8_0-for-a-128-gb-pc-2026-10-11-release-0142-nvfp42)).
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
[By hand](../NVFP4.md#by-hand-build-convert-re-quantize-run) in docs/NVFP4.md has the rest: a one-card CMake
build, converting jpezzulli's ModelOpt checkpoint ("Prepare the model"), re-quantizing the experts from BF16 by GPTQ,
and running the engine and the server without setup. Each Hugging Face model card also shows how to run its files by
hand.

**Releasing** (maintainers): `release/make_windows_bundle.py`, then `release/publish.py`; their headers say what each
checks.

### Tests

| executable | checks |
| --- | --- |
| `nvfp4_avx512_parity [experts.bin]` | AVX-512 and AVX2 rows vs ggml-cpu, the AVX2 ones bit for bit (ctest `nvfp4_cpu_parity`; `--expert`: one whole expert vs FP64; `--bw T N [ggml\|avx2]`: DRAM rate) |
| `nvfp4_expert_gpu_parity experts.bin` | the GPU decode path's experts vs FP64, one per sampled layer |
| `mmq_nvfp4_parity experts.bin` | MMQ products vs FP64 (`--group`, `--layers`, `--real dump layer`) |
