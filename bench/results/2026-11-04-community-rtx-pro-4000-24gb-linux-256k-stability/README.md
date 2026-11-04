# RTX PRO 4000 Blackwell 24 GB, Linux source build: 10-run strict-256K stability, NVFP4 GPTQ

Community report, 2026-11-04. Measured on one machine; prepared with AI assistance (Claude Code), with every number below taken from runs on this box. Raw harness, prompts and logs: [Adam90a/local-256k-reasoning](https://github.com/Adam90a/local-256k-reasoning).

## Machine

- GPU: NVIDIA RTX PRO 4000 Blackwell, 24 GB (sm_120), 145 W limit, PCIe Gen5 x16. The only GPU used (a second card in the box was masked out by UUID).
- CPU: AMD Threadripper 7970X (32C/64T, AVX-512), 1 NUMA node.
- RAM: 128 GB DDR5-4800, 4 channels (measured ~103 GB/s read with numpy).
- OS: Ubuntu 24.04, kernel 7.0, NVIDIA driver 595.91.07.

## Build

- Fork commit `586dc5e`, llama.cpp pinned at `3cf0325` (unpacked into `third_party/llama.cpp`), built by hand with CMake + Ninja instead of `setup.py`:

```bash
cmake -G Ninja -S . -B build-cuda13 -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-13.0 \
  -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp \
  -DCUDA_cublas_LIBRARY=<cu13>/lib/libcublas.so.13 -DCUDA_cublasLt_LIBRARY=<cu13>/lib/libcublasLt.so.13 \
  -DCMAKE_CXX_FLAGS="-isystem <cu13>/include" -DCMAKE_CUDA_FLAGS="-isystem <cu13>/include" \
  -DCMAKE_BUILD_RPATH="<cu13>/lib;/usr/local/cuda-13.0/lib64"
cmake --build build-cuda13 --target strata
```

  `<cu13>` is the `nvidia/cu13` folder of a pip CUDA 13 wheel set: this machine's CUDA 13.0 toolkit ships no cuBLAS, so without the explicit paths CMake picks Ubuntu's CUDA 12 cuBLAS/cudart (the mix `toolkit_root_defs` warns about). `ldd` confirms `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`. The build was clean; no source changes.
- Server: `python -m serve.server --engine strata --config <json> --port 8091`, venv from `requirements.txt`.

## Model and configuration

[`Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata`](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata) at `9b70ad3`, all files checked against `SHA256SUMS`.

```text
--pack <m>/pack --native <m>/qwen-nvfp4-gptq-dense.gguf --native-dense-gguf <m>/qwen-nvfp4-gptq-dense.gguf
--ple-gguf <m>/ple-fp8.gguf --embd-gguf <m>/token-embd-bf16.gguf --mtp <m>/mtp --spec 4 --spec-min-p 0.6
--prefill auto --expert-profile data/expert-profile.bin --expert-cache auto
--max-context 294912 --kv int8 --kv-resident 32768 --rope-scaling dynamic --rope-scale 2
```

294,912 = 262,144 input + 32,768 output: with a 262K prompt a smaller window can run out mid-reasoning (it did once in an earlier llama.cpp series on this box).

## Method

Each run is one fresh request through `/v1/chat/completions` (streaming, thinking on, temperature 0.6, top-p 0.95, top-k 20, `max_tokens` 32,768). The prompt is real source code (llama.cpp + vLLM files) with planted facts, sized to at least 262,144 tokens; the number reported is the server's own `usage.prompt_tokens`. Thirteen auto-scored checks per run: needles at start/middle/end, a distractor set, a 3-hop chain, a contradiction pair, a whole-context count, a 6-hop variable chain with decoys, a cross-file bug fix that needs computing a value, a scheduling puzzle with scattered rules, multi-value arithmetic, a cross-file "who calls this function" list (F1 ≥ 0.8), and a standing instruction from the system prompt. Ten consecutive runs, seeds 101–110 (different code, facts and positions each run). Resource sampling at 1 Hz.

## Results (strict 256K, 10 consecutive runs)

| seed | prompt tokens | score | prefill | gen tok/s | output tokens | wall |
|---|---:|---:|---:|---:|---:|---:|
| 101 | 262,490 | 13/13 | 91.9 s | 101.3 | 9,516 | 186 s |
| 102 | 262,410 | 13/13 | 92.0 s | 104.5 | 7,079 | 160 s |
| 103 | 262,488 | 13/13 | 91.9 s | 98.5 | 10,653 | 200 s |
| 104 | 262,389 | 13/13 | 92.0 s | 103.0 | 9,534 | 185 s |
| 105 | 262,321 | 13/13 | 91.9 s | 103.5 | 5,890 | 149 s |
| 106 | 262,357 | 13/13 | 91.8 s | 100.6 | 9,751 | 189 s |
| 107 | 262,327 | 13/13 | 91.8 s | 105.0 | 4,555 | 135 s |
| 108 | 262,390 | 13/13 | 91.6 s | 102.2 | 8,263 | 172 s |
| 109 | 262,350 | 13/13 | 92.0 s | 102.8 | 8,028 | 170 s |
| 110 | 262,392 | 13/13 | 91.8 s | 104.4 | 11,061 | 198 s |

All ten stopped on EOS (no truncation, no errors). Prefill 2,857 tok/s mean (91.6–92.0 s); generation 102.6 tok/s mean at ~262K. Peaks across the series: 22.9 GiB VRAM, 75.7 GB engine RSS, ≤172 MB swap-in per run, GPU 90 °C at the 145 W cap. Per-run JSON in `runs/`.

Also measured: a 3.8K-token prompt (thinking off, 1,024 tokens out): 110.7 tok/s decode, 2,131 tok/s cold prefill. 64K run: 13/13, 3,047 tok/s prefill, 105.7 tok/s.

## Same machine, llama.cpp for reference

Same prompts' seeds and scorer, llama.cpp `29c720e` with Unsloth `UD-Q4_K_XL`, experts in RAM, YaRN ×2, 294,912 window: also 13/13 on 10/10, but 343 s prefill (~765 tok/s) and 29.7 tok/s decode at 262K. Strata: 3.7× faster prompt reading, 3.5× faster generation on this box.

## Limitations

- One machine, one configuration; no A/B of individual flags.
- The test set is synthetic (planted facts in real code); it is hard enough to separate weaker models but not to rank top ones.
- AIME/coding suites were not re-run on Strata (they were on llama.cpp).
- Did not see the "!" reply issue in these runs, but conversations were single-turn.
