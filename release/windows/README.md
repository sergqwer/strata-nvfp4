# Strata NVFP4 — ready-made engine for Windows

Qwen3.8-Flash-Next (125B hybrid MoE), the original model with every expert NVFP4 by this fork's GPTQ, on one GeForce
RTX 20, 30, 40 or 50 card with 12 GB of VRAM or more and 64 GB of RAM or more - text and pictures. Censored as Qwen
ships it; a switch turns that off (see Censorship).
Source, measurements and how it works: https://github.com/sergqwer/strata-nvfp4

## What you need

- **GPU:** GeForce RTX 20, 30, 40 or 50 with 12 GB of VRAM or more. Built and measured on an RTX 5090 (32 GB); the
  other generations' code paths were tested on it too. With less VRAM, lower `--max-context` in
  `config\strata-qwen-nvfp4-gptq.json` (the engine says so when it does not fit):

  | card's VRAM | `--max-context` | expert slots | decode, measured* |
  | --- | ---: | ---: | ---: |
  | 32 GB (RTX 5090) | 262144 | 7,352 | 111 tok/s |
  | 24 GB (RTX 3090 / 4090) | 131072 | 5,158 | 95 tok/s |
  | 16 GB (RTX 4080 / 5080 / 4060 Ti 16 GB) | 65536 | 2,422 | 67 tok/s |
  | 12 GB (RTX 3060 12 GB / 4070) | 32768 | 1,055 | 59 tok/s |

  \* On the RTX 5090 with the smaller card's VRAM budget (`--vram-reserve-mib`); a real card's own compute and PCIe
  make it slower. 12 GB at 262144 and 8 GB cards at any context stop with *no VRAM is left for the expert cache*.

- **Driver:** NVIDIA 580 or newer (CUDA 13). The CUDA runtime is inside `strata.exe` and cuBLAS is in `engine\`,
  so no CUDA toolkit is needed.
- **RAM:** 64 GB or more. With 92 GiB installed or more (a 96 GB PC) all 63 GiB of experts stay in RAM (~67 GiB in
  all); with less, the
  engine keeps only the ones the graphics card does not hold, hottest first, and reads the rest from the file when
  needed - by itself, 6 GB stay free for Windows. `--no-low-ram` in the config turns that off, `--low-ram` forces it,
  `--ram-budget GIB` caps it. With 64 GB and an RTX 5090 all of them fit (44 GiB): a 32K prompt waits 9.3 s
  (96+ GB: 5.7 s), the answer after it runs at 97 tokens/s, and the start takes ~16 s (measured on the RTX 5090 + 128
  GB PC with 60 GiB of RAM locked away, as on a 64 GB PC whose Windows uses 6 GB).

- **Pagefile: a fixed 64000 MB** (Win+R `sysdm.cpl` > Advanced > Performance Settings > Advanced > Virtual memory >
  Change: Custom size, initial and maximum 64000 MB, then restart). Windows allows all programs together to reserve
  only RAM + pagefile, and the engine reserves ~100 GiB (~80 in the low-RAM mode; Windows counts the GPU's memory
  too); nothing of the model is actually written to the pagefile. Short of it the expert cache is made smaller (the
  model can run significantly slower) or the start fails; the engine prints a `WARNING` below 60000 MB.
- **CPU:** any x86-64 with AVX2; AVX-512 (Zen 4/5) is used automatically for the CPU share of the experts.
- **Disk:** ~130 GB for the model files (129.7 GB with the image encoder). The start reads 63 GiB, so the fastest
  NVMe drive you have is the right place for this folder (the experts are in after ~7 s from a PCIe 5 drive; a
  PCIe 4 drive reads about half as fast).
- **Python 3.11 or newer** on PATH (the scripts make their own virtual environments here).

## Two steps

1. **`prepare-model.cmd`** — downloads the model ready-made:
   [Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata)
   at the revision `model-files.json` pins (the same files setup.py installs: the expert pack, the dense GGUF, the
   FP8 n-gram table, the BF16 embedding, the MTP draft head) and the image encoder, 129.7 GB into `models\`. Every
   file is checked against its SHA-256; nothing is converted. If it stops, run it again and it resumes;
   `prepare-model.cmd --check` only says what is there.
2. **`start-server.cmd`** — loads the model and serves it on **http://127.0.0.1:8080**:
   - a chat page at `http://127.0.0.1:8080/`
   - OpenAI API at `/v1/chat/completions`, Anthropic API at `/v1/messages` (Claude Code can point at it)
   - pictures too: attach them in the chat page, send `image_url` parts or Anthropic image blocks. The image encoder
     runs on the CPU (2-6 s a picture), so the graphics card keeps all its memory for the model

First start after a reboot is slower while Windows reads the files; later starts take ~8-15 s.

## Updating

**`update.cmd`** downloads the newest release of this fork, checks it against the SHA-256 GitHub publishes for it,
and replaces the engine, the server and the tools. `config\`, `data\`, `models\` and the `.venv-*` folders stay;
the replaced program is kept in `.previous\`. Close the server first. `update.cmd --check` only says whether a newer
release is out. When a release changes a file of `config\` or `data\`, one you did not change is replaced; one you
changed stays, and the release's goes beside it as `<name>.new`. The program is swapped by renames, each one noted
first: if a file in use stops it half way, the old program is put back - by the next `update.cmd` if it cannot be
right away.

## Measured (RTX 5090, Ryzen 9 9950X3D, 128 GB DDR5-5600, 262K context)

| | |
| --- | ---: |
| Writes a chat answer (English, Ukrainian, code, an agent turn; mean of 3 runs) | ~180 tokens/s |
| Reads a 32K prompt (an NVFP4 pack of the same format and size) | ~3.7 s |

## Settings

`config\strata-qwen-nvfp4-gptq.json` holds the engine flags: context (`--max-context`, up to 262144), KV cache
(`--kv int8`), paths. Environment switches:

- `STRATA_PREFILL_NVFP4=w4a8` (default) / `w4a4` (faster prompt reading, less accurate) / `fp16` (exact, slower)
- `--pcie-frac F` in the config: a fixed share of cache misses fetched over PCIe (by default the engine measures it)

## Censorship

The model declines what Qwen declines. The config loads this fork's refusal-direction projection
(`data\uncensor`, layers 8-33) and leaves it **off** (`"uncensored": false`): with it off, the answers are the
original model's. The chat page's **Disable censorship** switch (Sampling) and the API field `"uncensored": true`
turn it on for a request; `"uncensored": true` in the config makes on the default. Measured on this model: with
thinking on, 0 of 104 English and 0 of 30 Ukrainian held-out harmful requests refused, KL 0.026 to the original,
its thinking as long as the original's (`data\uncensor\README.md`). The vector is under the Qwen Community
License 1.0; removing refusals removes a safety behaviour, and what the model writes with it on is your
responsibility.

## Large pages (optional, needs one sign-out)

Not more memory — bigger pages. The 63 GiB of experts are 16.5 million 4 KB pages; with 2 MB pages they are 32
thousand, and the CPU share of every token runs steadier (7.3-7.7 ms per step instead of 7.4-11 on the reference
PC). Windows allows it only for an account with *Lock pages in memory*, and a new right reaches you only at the
next sign-in:

1. run **`enable-large-pages.cmd`** (it asks for admin; works on Windows Home too),
2. **sign out and back in, or reboot** — locking the screen is not enough,
3. `enable-large-pages.cmd -Check` should now say ACTIVE, and `strata.log` shows `large pages (2097152 B)`.

`large pages refused ... error 1314` in the log = the right is not active yet (step 2); `error 1450` = memory is
too fragmented after a long uptime (reboot). Either way the model still runs, on 4 KB pages.
`enable-large-pages.cmd -Revoke` undoes it.

## Claude Code

```bat
set ANTHROPIC_BASE_URL=http://127.0.0.1:8080
set ANTHROPIC_API_KEY=local
set ANTHROPIC_MODEL=strata-nvfp4
set ANTHROPIC_SMALL_FAST_MODEL=strata-nvfp4
claude
```

Each turn re-reads only what is new: measured, a 21,964-token first request (5.8 s), then 209 and 106 new tokens
per tool turn (~0.5 s).

## Folder

| | |
| --- | --- |
| `engine\` | `strata.exe`, `strata-vision.exe` (the image encoder), NVIDIA cuBLAS (`cublas64_13.dll`, `cublasLt64_13.dll`), `BUILD.json` |
| `serve\` | the server and its chat page |
| `tools\` | `bundle_model.py` (what `prepare-model.cmd` runs), the updater, and the converters for making a pack yourself (with `third_party\llama.cpp\`, llama.cpp's at its pinned commit) |
| `data\` | the expert-cache profile, the draft vocabulary, `uncensor\` (the censorship switch's vector) |
| `model-files.json` | the files `prepare-model.cmd` downloads: repository, revision, size and SHA-256 of each |
| `docs\NVFP4.md` | accuracy and speed measurements |

**Coming from a bundle before 0.1.41-nvfp4.5** (orca-nvfp4, converted from jpezzulli's ModelOpt checkpoint): that
model was withdrawn - as an agent it did much worse than the original Qwen. `start-server.cmd` still starts it
(with `config\strata-nvfp4.json`) until `prepare-model.cmd` has downloaded the original Qwen; then it starts that.
`models\checkpoint`, `models\pack`, `models\orca-nvfp4.gguf` and `models\mtp` can go afterwards.

License: MIT (see `LICENSE`); third-party parts in `THIRD-PARTY-NOTICES.txt`.
