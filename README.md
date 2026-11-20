# Strata NVFP4

**Qwen3.8-Flash-Next (125B hybrid MoE) with NVFP4 experts on one RTX 20, 30, 40 or 50 card (12 GB of VRAM or more)
and 64 GB of RAM or more, text and pictures.** A fork of [Niko1221/Strata](https://github.com/Niko1221/Strata), on
upstream 0.2.15 (its engine, server and setup; not its app or installers). Setup installs, ready-made from Hugging Face, the original Qwen in NVFP4 (our GPTQ, or NVIDIA's
checkpoint converted), censored as Qwen ships it; on ours an opt-in switch turns the censorship off. Upstream's GGUF
models are still offered.

## Get started

Windows: download `strata-nvfp4-v<version>-windows-x64.zip` from the [latest release](https://github.com/sergqwer/strata-nvfp4/releases/latest),
run `prepare-model.cmd`, then `start-server.cmd` ([its README](release/windows/README.md)). Or clone this repository and
run `START-HERE.bat` (Linux: `./setup.sh`). The full steps and what the PC needs: [Install and run](docs/fork/INSTALL.md).

## Models

Accuracy against Qwen3.8-Flash-Next with every routed expert in Q8_0, speed on an RTX 5090 with 128 GB of RAM.
Setup downloads the models it names; each Hugging Face card has its measurements and how to run the
files by hand.

| Model | Same top token as Q8_0 | Decode, tokens/s | 32K prompt | RAM in use | PC for this speed | Get it |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Q8_0: every expert Q8_0 (ours, `qwen-q8`) | 100% (the reference) | 47 | 10.3 s | 106 GiB | 128 GB | [Maximilian228/Qwen3.8-Flash-Next-Q8_0-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-Q8_0-Strata) |
| NVFP4 + Q8_0 mix (ours, `qwen-nvfp4-mix`) | 95.0% | 91 | 4.2 s | 105 GiB | 128 GB | [Maximilian228/Qwen3.8-Flash-Next-NVFP4-Q8-Mix-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-Q8-Mix-Strata) |
| Unsloth UD-Q4_K_XL (`unsloth`, `--model UD-Q4_K_XL`) | 94.1% | 169 | 4.9 s | 77 GiB | 96 GB | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |
| NVFP4 GPTQ (ours, `qwen-nvfp4-gptq`, setup's default) | 93.6% | 177 | 3.9 s | 69 GiB | 96 GB | [Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata) |
| NVIDIA NVFP4, converted by us (`qwen-nvidia-nvfp4`) | 92.9% | 167 | 3.9 s | 69 GiB | 96 GB | [Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata) |
| NVIDIA FP4-Local, retrained by NVIDIA for FP4 (QAD), converted on the PC | 92.2% | 191¹ | 3.7 s¹ | 69 GiB | 96 GB | [nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) |
| Unsloth UD-Q3_K_XL (`unsloth`, `--model UD-Q3_K_XL`) | 90.7% | 208 | 3.8 s | 58 GiB | 96 GB | [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) |
| ISTA-DASLab IQ3_S (`qwen`, `--model IQ3_S`) | 90.0% | 228 | 3.9 s | 52 GiB | 64 GB | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) |
| ISTA-DASLab IQ3_XXS (`qwen`, `--model IQ3_XXS`, upstream's pick) | 87.3% | 239 | 3.9 s | 45 GiB | 64 GB | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) |
| ISTA-DASLab IQ2_XS (`qwen`, `--model IQ2_XS`) | 85.0% | 265 | 4.2 s | 38 GiB | 48 GB | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) |
| ISTA-DASLab Q2_0 (`qwen`, `--model Q2_0`) | 84.1% | 256 | 3.9 s | 37 GiB | 48 GB | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) |

- **Accuracy:** how often the model's most likely next token is the Q8_0 model's. Measured on 5,153 tokens of code,
  docs, English and Ukrainian chats and agent transcripts, every model fed the same text. Every model ran with its
  experts on the CPU and the route tail skip off, so only the quantization differs. The Q8_0 model also has Q8_0
  attention, Qwen's BF16 router and Qwen's own FP8 PLE, so it is the original's precision everywhere but the 8-bit
  weights.
- **Speed:** RTX 5090 (32 GB), Ryzen 9 9950X3D, 128 GB of DDR5-5600, Windows 11, Strata NVFP4 0.1.42-nvfp4.1's engine with
  the arguments 0.1.42-nvfp4.2's setup writes for each model. Median of 3 runs: 4 greedy chats (up to 1,500 tokens each) and one
  32,007-token prompt.
- **RAM:** "RAM in use" is the engine and the server at the speed shown; "PC for this speed" is the installed RAM that
  holds it with Windows beside it. Smaller PCs still run the models, slower: the NVFP4 GPTQ and NVIDIA's on a 64 GB PC
  in the low-RAM mode decode ~115 tokens/s and read a 32K prompt in ~9 s (docs/NVFP4.md, "64 GB of RAM"); Unsloth's
  files go into a RAM-budget mode there and read experts from the SSD (UD-Q4_K_XL: 7-8.5 tokens/s at 64 GB,
  upstream's measurement); setup's minimum is 48 GB for them and the ISTA 2-bit files, 60-64 GB for the ISTA 3-bit ones.
- **Unsloth's files keep every expert in RAM where they fit with 16 GiB more** (setup from 0.1.42-nvfp4.2): from ~88 GB
  of RAM for UD-Q4_K_XL and ~68 GB for UD-Q3_K_XL. Below that, setup's RAM-budget mode reads the rest from the GGUF on
  the SSD. Setup before 0.1.42-nvfp4.2 used the budget
  mode on every PC; on 128 GB UD-Q4_K_XL then decoded 118 tokens/s and read the 32K prompt in 30 s (run setup again to
  switch an existing install).
- UD-Q3_K_XL's experts are i-quants despite the name: IQ3_XXS gate/up with IQ4_NL or Q8_0 down.
- The GGUF models run upstream's route tail skip by default: ~10% faster decode, and answers a little further from
  Q8_0 than the table shows.
- The two 128 GB models need a 128 GB PC (setup asks below that). Q8_0 runs there in the low-RAM mode by itself (100 GiB
  of experts locked in RAM); the mix keeps every expert in RAM.
- ¹ **NVIDIA FP4-Local** is NVIDIA's retrained 4-bit model (quantization-aware distillation): its experts and n-gram
  table were trained again for FP4, so it is a slightly different model rather than a rounding of the original. Its
  KL to the Q8_0 model is 0.051 (the NVFP4 GPTQ and NVIDIA's NVFP4: 0.038), while its perplexity on these texts is a
  little below Q8_0's. Measured on 2026-10-24 on the same texts against the same reference, with the engine of this
  fork's 0.2.0-beta-nvfp4.3. The NVFP4 GPTQ model gives the same accuracy on that engine (93.65%), and in the same runs
  it decoded 180-191 tokens/s, so the two run at the same speed within the noise. NVIDIA ships ModelOpt safetensors
  (95 GiB) that are converted on the PC (upstream Strata's `tools/nvfp4_local_pack.py prepare`, about 10 minutes,
  99 GiB of files); this fork's setup does not install it. Its license, the NVIDIA Open Model Agreement, rules out the
  censorship switch.

## Documentation

- <a id="whats-new"></a>[Changelog](CHANGELOG.md): every release, the newest first, each linked to its GitHub release.
- <a id="quick-start"></a><a id="requirements"></a><a id="run"></a><a id="large-pages-why-a-reboot"></a><a id="build-it-yourself-advanced"></a><a id="tests"></a>[Install and run](docs/fork/INSTALL.md):
  quick start, setup's options, requirements, running, large pages, building and tests.
- <a id="what-this-fork-adds"></a><a id="limits"></a>[What this fork adds](docs/fork/FEATURES.md): each change on top of
  upstream with a link to its numbers, and the limits.
- <a id="speed"></a>[Speed](docs/fork/SPEED.md): the current RTX 50 numbers and the older RAM-size runs.
- <a id="claude-code"></a>[Claude Code](docs/fork/CLAUDE_CODE.md): the settings that point Claude Code at the server.
- <a id="switches"></a>[Switches and environment variables](docs/fork/ENVIRONMENT.md): the most useful switches, the
  RTX 50 defaults and their off switches, and every variable the code reads, by area.
- <a id="if-a-reply-turns-into-"></a>[Troubleshooting](docs/fork/TROUBLESHOOTING.md): a reply that turns into "!", and where other problems are covered.
- [NVFP4 measurements](docs/NVFP4.md): every measurement behind this fork, release by release, and the steps by hand.
- [The Windows release folder](release/windows/README.md): what the stand-alone zip needs and its two steps.
- [The censorship switch's vector](data/uncensor/README.md): how it was made and its license.
- [Upstream's README](README.upstream.md) and [upstream's reference](docs/DETAILS.md): the engine and server this fork
  is built on (config keys, APIs, memory modes).

## Credits

- [Strata](https://github.com/Niko1221/Strata) by Niko1221 and the Strata contributors (MIT): the engine, the server, setup
  and the app page this fork is built on. Upstream's own credits are in
  [docs/DETAILS.md](docs/DETAILS.md#credits-and-licenses).
- [Basalt](https://github.com/jesdga95/basalt) by Jesús Gulfo (MIT), a Strata fork for RTX 50 cards: this fork's RTX 50
  decode kernels (the int8 tensor-core dense GEMV, the 8-bit hyper-connection read, staged INT8 attention, the
  persistent DeltaNet layer, the attention layer's placement) and its MTP drafting are ported from it. Its
  self-distilled MTP draft layer (Qwen Community License 1.0, like the model) ships for the GPTQ set as
  `mtp/rt-tuned` in [our model repository](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata).
- [Geode](https://play.google.com/store/apps/details?id=xyz.tifu.geode) ([source](https://github.com/sergqwer/geode)),
  this fork's author's engine that runs big models on Android phones; several of this fork's ideas came from it.
- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) by the ggml authors (MIT), compiled into the engine at the
  pinned commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`; its converter and `gguf-py` ship with the Windows zip.
- The DFlash2 drafter: [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
  by Inco AI (Apache-2.0).
- The models: [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team (Qwen Community
  License 1.0); NVIDIA's [NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) and
  [FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) checkpoints (the NVIDIA Open Model License; FP4-Local: the NVIDIA Open Model Agreement);
  [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)'s GSQ-RCO quants,
  [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)'s GGUFs and UkisAI's
  [Swift 1.5](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), which setup offers as upstream
  does; jpezzulli's ModelOpt checkpoint, which the steps by hand convert
  ([jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4](https://huggingface.co/jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4)).
- NVIDIA's CUDA runtime and cuBLAS, redistributed in the Windows zip under the
  [CUDA EULA](https://docs.nvidia.com/cuda/eula/), and the official [Node.js](https://nodejs.org) LTS build that runs the
  new interface from the zip's `node` folder.
- The people whose upstream pull requests this fork carries, named in the [Changelog](CHANGELOG.md): xrip, gputier,
  jarydkrish, LeGeRyChEeSe, Mert Özkan, ANBAL534, shoftee, sebastianrcnt, s.anastasiou and BlueKingMuch.

## License

MIT, as upstream ([LICENSE](LICENSE), copyright Niko1221 and the Strata contributors). llama.cpp / ggml code compiled
into the build is MIT as well. Code ported from Basalt is MIT, copyright Jesús Gulfo ([LICENSE.basalt](LICENSE.basalt)).

## Star History

<a href="https://www.star-history.com/#sergqwer/strata-nvfp4&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=sergqwer/strata-nvfp4&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=sergqwer/strata-nvfp4&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=sergqwer/strata-nvfp4&type=Date" />
 </picture>
</a>
