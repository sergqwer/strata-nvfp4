# Strata NVFP4 — ready-made engine for Windows

Qwen3.8-Flash-Next (125B hybrid MoE), OrcaRouter's abliteration in ModelOpt NVFP4, on one GeForce RTX 20, 30, 40 or 50
card with 12 GB of VRAM or more and 64 GB of RAM or more - text and pictures.
Source, measurements and how it works: https://github.com/sergqwer/strata-nvfp4

## What you need

- **GPU:** GeForce RTX 20, 30, 40 or 50 with 12 GB of VRAM or more. Built and measured on an RTX 5090 (32 GB); the
  other generations' code paths were tested on it too. With less VRAM, lower `--max-context` in
  `config\strata-nvfp4.json` (the engine says so when it does not fit):

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
- **RAM:** 64 GB or more. With 96 GB or more all 63 GiB of experts stay in RAM (~69 GiB in all); with less, the
  engine keeps only the ones the graphics card does not hold, hottest first, and reads the rest from the file when
  needed - by itself, 6 GB stay free for Windows. `--no-low-ram` in the config turns that off, `--low-ram` forces it,
  `--ram-budget GIB` caps it. With 64 GB and an RTX 5090 all of them fit (44 GiB): a 32K prompt waits 9.3 s
  (96+ GB: 5.7 s), the answer after it runs at 97 tokens/s, and the start takes ~16 s (measured on the RTX 5090 + 128
  GB PC with 60 GiB of RAM locked away, as on a 64 GB PC whose Windows uses 6 GB).

- **Pagefile:** at least **32 GB with 128 GB of RAM, 64 GB with 96 GB, 48 GB with 64 GB** (System > About >
  Advanced system settings > Performance > Advanced > Virtual memory; set a fixed initial size). Windows allows all
  programs together to reserve only RAM + pagefile, and the engine reserves ~70-98 GiB (Windows counts the GPU's
  memory too); nothing of the model is actually written to the pagefile. Too small, and the start fails with an
  allocation error.
- **CPU:** any x86-64 with AVX2; AVX-512 (Zen 4/5) is used automatically for the CPU share of the experts.
- **Disk:** ~340 GB free while the model is prepared, ~200 GB afterwards (delete `models\checkpoint`, 135 GB, and
  `models\mtp\tensors` once `prepare-model.cmd` is done). The start reads 63 GiB, so the fastest NVMe drive you
  have is the right place for this folder (the experts are in after ~7 s from a PCIe 5 drive; a PCIe 4 drive reads
  about half as fast).
- **Python 3.11 or newer** on PATH (the scripts make their own virtual environments here).

## Two steps

1. **`prepare-model.cmd`** — downloads
   [jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4](https://huggingface.co/jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4)
   (126 GB) and converts it into `models\`, the n-gram (PLE) table kept in FP8 and the token embedding in BF16
   exactly as Qwen ships them, and the image encoder from the checkpoint's own vision tower. It takes a while; if
   it stops, run it again and it resumes. Coming from an older release: run it again too - it adds only what is
   missing (`models\mmproj-f32.gguf`, 1.8 GB, and `models\token-embd-bf16.gguf`, 1.3 GB; the checkpoint must still
   be in `models\checkpoint`).
2. **`start-server.cmd`** — loads the model and serves it on **http://127.0.0.1:8080**:
   - a chat page at `http://127.0.0.1:8080/`
   - OpenAI API at `/v1/chat/completions`, Anthropic API at `/v1/messages` (Claude Code can point at it)
   - pictures too: attach them in the chat page, send `image_url` parts or Anthropic image blocks. The image encoder
     runs on the CPU (2-6 s a picture), so the graphics card keeps all its memory for the model

First start after a reboot is slower while Windows reads the files; later starts take ~8-15 s.

## Measured (RTX 5090, Ryzen 9 9950X3D, 128 GB DDR5-5600, 262K context)

| | |
| --- | ---: |
| Writes answers, short chat | ~115 tokens/s (in Cyrillic too: ~110) |
| Writes answers, 32K context | ~120 tokens/s |
| Reads a 32K prompt | ~5,500 tokens/s |

## Settings

`config\strata-nvfp4.json` holds the engine flags: context (`--max-context`, up to 262144), KV cache (`--kv int8`),
paths. Environment switches:

- `STRATA_PREFILL_NVFP4=w4a8` (default) / `w4a4` (faster prompt reading, less accurate) / `fp16` (exact, slower)
- `--pcie-frac F` in the config: the share of cache misses fetched over PCIe (default 0.25)

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
| `tools\`, `third_party\llama.cpp\` | the converters `prepare-model.cmd` runs (llama.cpp's at its pinned commit) |
| `data\` | the expert-cache profile and the draft vocabulary |
| `docs\NVFP4.md` | accuracy and speed measurements |

The model is an abliterated fine-tune: it does not refuse. What it is used for is on whoever runs it.

License: MIT (see `LICENSE`); third-party parts in `THIRD-PARTY-NOTICES.txt`.
