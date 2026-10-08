# NVFP4 routed experts

Measured on Windows 11 with an RTX 5090 (32 GB, sm_120), Ryzen 9 9950X3D and 128 GB DDR5-5600, 2026-09-29.

The target is a ModelOpt NVFP4 checkpoint of Qwen3.8-Flash-Next (OrcaRouter's abliteration,
`jpezzulli`'s quantization). `tools/nvfp4_convert.py` repacks it without loss into a GGUF whose routed
experts are NVFP4 (64-value blocks: four UE4M3 sub-block scales and 32 bytes of E2M1 codes) and whose
per-expert `weight_scale_2` lands in `blk.N.ffn_{gate,up,down}_exps.scale`. `tools/iq_pack.py` writes
each expert blob as `[gate | up | down]` plus a 16-byte tail `{s_gate, s_up, s_down, 0}`.

## Where the global scales go

Each scale multiplies its own projection's FP32 output, as llama.cpp applies `.scale`:
`h = silu(s_gate * (G x)) * (s_up * (U x))`, `y = s_down * (D h)`.

The scales are ~1e-4 each. An earlier revision folded `s_down` into up, which is algebraically the same
but leaves the hidden near 1e-5. The hidden is then rounded to Q8_0 (CPU) or q8_1 (GPU) blocks whose
scale is FP16, and `amax / 127` lands deep in FP16's subnormals: one expert's output was off by 2-12%
instead of 1.1% (`nvfp4_avx512_parity --expert`). The FP16 prompt path had the same problem in its
dequantized up weights (~1e-8). Any future fold must be checked against FP16's range.

## Prompt path precision

`STRATA_PREFILL_NVFP4` picks how the batched prompt path multiplies the experts:

| mode | kernel | activations | one product vs FP64 (`mmq_nvfp4_parity`) |
|---|---|---|---|
| `w4a8` (default) | llama.cpp MMQ, int8 tensor cores | q8_1, float block scales | 0.53% |
| `w4a4` | llama.cpp MMQ, Blackwell FP4 MMA (sm_120a only) | NVFP4 | 8.6% |
| `fp16` | dequantize + cuBLAS FP16 GEMM | FP16 | reference |

`src/prefill/mmq_nvfp4_w4a8.cu` compiles mmq.cuh's int8 NVFP4 path with Blackwell's FP4 MMA hidden from both the
device code and the host's tile config, so it is the same on every architecture; decode already multiplies at this
precision. The FP4 MMA exists only in the arch-specific target, so `src/prefill/mmq_nvfp4_w4a4.cu` (the FP4 x FP4
kernel and its FP4 activation quantizer) is the one unit CMake builds with `12x` -> `12xa` (`strata_mmq_w4a4`);
a card without that unit's image falls back to `w4a8` with a note.

First-token KL divergence against `fp16`, 8 prompts (code and prose, 1K-8K tokens); the noise floor is the
same `fp16` path cut into 4096-token chunks instead of 8192 (another summation order, same arithmetic):

| mode | KL mean | KL median | KL max | top-1 agreement | prefill, 4-8K prompts |
|---|---|---|---|---|---|
| noise floor | 0.00023 | 0.00003 | 0.0016 | 8/8 | - |
| `w4a8` | 0.0018 | 0.0010 | 0.0096 | 8/8 | 2,503 tok/s |
| `w4a4` | 0.0080 | 0.0027 | 0.040 | 7/8 | 2,904 tok/s |
| `fp16` | reference | | | | 1,905 tok/s |

`w4a8` keeps the default: 4.5x closer to the reference than `w4a4` at 86% of its speed, and the precision
every generated token already runs at. `fp16` is the choice when the prompt must be read exactly.

On a real prompt's rows (`STRATA_DUMP_MOE_INPUT` + `mmq_nvfp4_parity --real`, layer 20, its 12 busiest
experts, max|x|/rms 4 median and 9 at most) one product is off by: `w4a8` 0.46% (gate/up) and 0.90% (down, whose
SwiGLU input has the heavier tails), `w4a4` 7.2% and 8.4%, `fp16` 0.017% and 0.021%.

## The n-gram (PLE) table

Layer 1 adds 16 rows of a 320,001,536 x 160 n-gram table per token. Qwen ships it in FP8 E4M3 (128 shards of
[2500012, 160] and one scale, 51.2 GB); no source holds more precision (OrcaRouter's BF16 copy is this FP8
widened). Strata read only IQ4_NL (ISTA-DASLab's shard 2, 28.8 GB), which is 8.1% off the FP8 values per row -
correlation 0.996-0.997 on rows from every shard, so the same table in the same order, and the abliteration left it
alone. `tools/ple_fp8_pack.py` copies the FP8 bytes into a GGUF (I8, `strata.ple.format` = f8_e4m3,
`strata.ple.scale`); `ple_fp8_parity` checks the engine's rows against torch's decode of the checkpoint, bit for bit.

| first-token KL, 8 prompts (1K-8K) | mean | median | max | top-1 |
| --- | ---: | ---: | ---: | ---: |
| noise floor (summation order) | 0.00023 | 0.00003 | 0.0016 | 8/8 |
| w4a8 prompt path vs fp16 | 0.0018 | 0.0010 | 0.0096 | 8/8 |
| **IQ4_NL PLE vs FP8 PLE** | **0.0026** | **0.0012** | **0.013** | 8/8 |

It costs 22 GB more disk and nothing else: the table stays on the SSD (16 page reads a token either way; a row is
160 B instead of 90) and the row cache grows from ~95 to ~160 MB. Prompt reading speed was unchanged.

## Where precision is still lost (audit, 2026-09-29)

First-token KL on the 8 prompts above plus one of 32K tokens; the first token runs through the decode path
(a verify window), so it sees the decode arithmetic and the KV cache. Noise floor: the same configuration twice.

| source | vs | KL mean | median | 32K | cost of the exact version |
| --- | --- | ---: | ---: | ---: | --- |
| noise (same config twice) | - | 0.00012 | 0.000007 | 0 | - |
| KV int8 (before) | fp16 KV | 0.0017 | 0.00034 | 0.0068 | 1,189 expert slots at 262K, decode -12% (116.5 -> 102.7 tok/s) |
| **KV int8 + Hadamard (now)** | fp16 KV | 0.0011 | 0.00038 | 0.0038 | - |
| decode experts, q8_1 activations | FP32 activations | 0.00076 | 0.000085 | 0.00065 | new GPU and AVX-512 BF16 kernels |
| prompt path w4a8 | fp16 | 0.0018 | 0.0010 | - | prompt reading -24% |
| token embedding Q8_0 (before) | BF16 (`--embd-gguf`, now) | 0.0025 | 0.00042 | 0.0041 | 0.6 GB of host RAM |
| RoPE: fast-math float angles (before) | the float64 table (now) | - | - | 0.0063 (also at 125K) | - |
| prompt path: BF16 activations into BF16 projections (before) | hi + lo split, all | 0.0023 | 0.00031 | 0.0093 | - |
| split without the hyper-connection (now) | hi + lo split, all | 0.0011 | 0.0012 | 0.0029 | the full split: -9% prompt reading |

- **RoPE.** The native rope kernels computed `pos * powf(...)` with fast-math `cosf/sinf` (0.0014 rad off at 32K,
  ~0.02 at 262K), the prompt path the same in precise float, and the session's float64 angle table was read only by
  the non-native path. All of them read the table now (`rope_table_set`, `mrope.hpp`). `STRATA_ROPE_LEGACY=1`
  restores the old angles for A/B.
- **Prompt path BF16 activations.** Decode feeds FP32 x to the BF16-weight projections (router, indexer, SSM
  alpha/beta, shared gate, PLE key/value, hyper-connection); the prompt path fed BF16. `STRATA_PREFILL_BF16X2`
  (default 2) adds each activation's BF16 remainder as a second GEMM for all but the hyper-connection, whose
  10240-wide activations make the split cost ~9% of prompt reading (`=1` turns it on anyway, `=0` off).
- Found by a read-only audit with a fresh context (all checked here): the above, a FP16-saturating SwiGLU in the
  prompt path's shared expert, `log1pf` in the non-fused GDN softplus, and a stale NVFP4 tail comment. Checked and
  left: int8 KV group scales (the smallest V group amax on real K/V is 0.25, FP16's subnormals start at 7.8e-3).

- **Token embedding.** The converter stored `embed_tokens` as Q8_0 (0.55% off per row). `tools/embd_bf16_pack.py`
  copies the checkpoint's BF16 table into its own GGUF and `--embd-gguf` reads it: mapped host memory like before
  (1.2 GiB instead of 0.6), no VRAM, decode and prompt speed unchanged, +0.2 s at start. It moved the first token
  more than any other source here - the embedding's error rides the residual stream through every layer, where a
  projection's error enters once.

- **KV rotation.** int8 K/V now go through the 256-point Walsh-Hadamard rotation that `--kv q4_0` already used
  (queries rotated to match, the output rotated back). On real K/V/Q (`STRATA_DUMP_QKV`, 12 QSA layers, 4K tokens,
  dense attention of the last 512 queries) the attention output error drops from 0.306% to 0.265%; same memory,
  decode speed unchanged. `STRATA_KV_ROT=0` stores them unrotated.
- **Bug found on the way:** the batched verify path (`qb`, windows of 2+ tokens) never rotated its queries, which
  is why upstream kept `--kv q4_0` off that path. With rotation it scored `<q, Hk>`: coherent text, but MTP
  acceptance fell from 2.5 to 2.2 tokens a round. Fixed in `verify.cpp`.
- **k8v4 (int8 K, q4_0 V) is not worth it here.** Only 12 of 48 layers keep K/V, so int8 at 262K is ~3.1 GiB;
  V in 4 bits would free ~0.7 GiB = ~270 expert slots, and 270 fewer slots measured no change in rounds per
  second (47.7 vs 47.5). Its attention output error is 3.15% (10x int8's); `--kv q4_0` is 3.89%.
- **Decode activations.** `STRATA_DECODE_A16=1` computes the GPU's NVFP4 experts with FP32 activations
  (`native_expert_grouped_f32`, an oracle: its unoptimized kernel adds ~11% GPU time). Run with `--pcie-frac 1`
  so no expert falls to the CPU pool. The q8_1 rounding costs 12x the noise at the median; the mean is one
  prompt's 0.0055.
- Dense projections stay Q8_0 (the source is BF16; 0.56-0.80% off per weight): BF16 would take ~3.3 GB of VRAM
  (~1,270 slots) and read ~7.1 GB of dense weights per round instead of ~3.8 - an estimated -15-20% decode. Router, SSM gates, indexer, PLE and shared-expert gates are already BF16 with
  FP32 activations; the GDN state, indexer keys and norms are FP32.

## CPU experts

`src/kernels/cpu/nvfp4_avx512.cpp` computes the CPU pool's NVFP4 rows in 512-bit lanes: each 64-value
block is decoded once for every token of a verify window (vpdpbusd on |E2M1| codes with the sign moved
onto the activation). Same arithmetic as ggml-cpu's `ggml_vec_dot_nvfp4_q8_0`, only the float additions
are ordered differently (worst 4.5e-7 relative, `nvfp4_avx512_parity`). `STRATA_NO_NVFP4_512=1` falls
back to ggml-cpu.

| tokens | ggml-cpu | AVX-512 | speedup |
|---|---|---|---|
| 1 | 0.102 ms | 0.058 ms | 1.76x |
| 4 | 0.410 ms | 0.125 ms | 3.29x |
| 7 | 0.719 ms | 0.196 ms | 3.66x |

(one expert's gate+up, 1280 rows of 2560, cache-resident; the pool itself is bound by DRAM)

Without AVX-512 (Zen 2/3, Intel 12th-14th gen), `src/kernels/cpu/nvfp4_avx2.cpp` does the same in 256-bit lanes,
up to 8 tokens a pass. Each token's arithmetic is `ggml_vec_dot_nvfp4_q8_0`'s AVX2 path step for step (vpsignb,
vpmaddubsw + vpmaddwd, the same FMA order, `hsum_float_8`), so every row is bit-equal to it: an AVX2 CPU computes
what it did before, faster. `STRATA_NO_NVFP4_256=1` falls back to ggml-cpu.

| tokens | ggml-cpu | AVX2 | speedup |
|---|---|---|---|
| 1 | 0.100 ms | 0.089 ms | 1.12x |
| 4 | 0.399 ms | 0.200 ms | 2.00x |
| 8 | 0.807 ms | 0.344 ms | 2.35x |

Under DRAM load (`--bw`, 6 threads): 47.3 against 24.0 GB/s at 4 tokens, 26.6 against 12.0 at 8.

## Loading

`experts.bin` is read unbuffered (`FILE_FLAG_NO_BUFFERING`) straight into the arena by 16 readers, while a
second thread registers the arena with CUDA one layer ahead of them (`PinnedArena::Deferred` +
`register_slices`): registering 63 GiB of 4 KiB pages alone takes 6.6 s, and the buffered reader it replaces
(`STRATA_BUFFERED_LOAD=1`, the A/B arm) copied through the file cache at ~3.3 GiB/s wall. From a PCIe 5 drive
(13.4 GiB/s unbuffered) the arena now loads in 5.6-6.0 s instead of 19 s, and the session is up in ~10 s
instead of 25. `STRATA_VERIFY_ARENA=1` prints a checksum of the loaded arena; both loaders give the same one.

Decode A/B, 300 tokens, 262K context (median of 2): pipelined per-layer registration 103.5 tok/s, whole-arena
registration 103.2; ggml-cpu's NVFP4 rows 105.3 (the pool is DRAM-bound, so the kernel above moves its time by
2-3% and the decode rate not at all).

## Tuning measured on this machine (after the fixes above)

Kept:
- **PCIe share 0.25** (was 0.55): the copy kernel fetching the PCIe share was 40% of GPU kernel time and sits on
  the GPU's critical path. Decode 103 -> 117 tok/s short, 105 -> 124 at 32K.
- **Prefill auto chunks up to 32768**: a 32K prompt reads at 5201 tok/s instead of 3535 (TTFT 10.1 -> 7.5 s).
- **Large pages** once the account holds SeLockMemoryPrivilege (needs a fresh logon): the CPU pool holds
  ~7.5 ms/round where 4 KB pages wandered 7.4-11; decode itself is GPU-bound, so the rate barely moves.

Tried and dropped (no gain, or worse):
- DMA copies for the PCIe share (`--pcie-mode dma`) at 0.25 and 0.40: ~32 rounds/s either way at 32K.
- `--spec 5/6`, `--spec-min-p 0.3/0.7`: longer windows accept more but cost more; 0.3 is 15% slower.
- More pool workers (23, 31) or other prefetch distances: the pool sits at ~55 GB/s from DDR5-5600.
- An expert profile ranked by this model's own routing (6 prompts x 1000 tokens): it covered 69% of the
  traced routing against 40% for the shipped profile, but missed MORE on held-out prompts (CPU experts per layer
  5.3 vs 4.0 short, 10.8 vs 6.6 at 32K) - it fits the traces; the shipped profile generalises.
- A q4_0 / q8_0 MTP head: tools/mtp_pack.py writes them, but the drafter only runs Q2_0 experts.

### 64 GB of RAM (2026-09-30)

The resident arena holds all 24,576 experts (63.3 GiB pinned, ~69 GiB of physical RAM for the run), including the
~7,400 the VRAM cache holds a second time, so a 64 GB PC could not run the pack. `TieredExpertSource` keeps pinned
host copies only where the engine reads them - the experts outside VRAM (the CPU and the PCIe share compute them on
every token), ranked by the profile, then VRAM's own experts from its last slot back (the prompt path borrows the
cache from its end and refills it afterwards; a 32K chunk borrows ~5,300 slots) - up to the free RAM minus
`STRATA_RAM_RESERVE_GIB` (6), and maps `experts.bin` for the rest:

- a blob outside the tier is read **unbuffered** into pinned memory wherever the engine streams many (the startup's
  VRAM fill, the prompt path's stager, the lent slots' refill): copying them through the mapped file pulled its
  pages into the working set, Windows began trimming, and decode after a 32K prompt fell from 72 to 33 tok/s;
- decode prefetches the few a layer computes on the CPU at `begin_layer`, buffered (the OS cache keeps them in RAM
  nobody else uses), and trims any mapped page it handed out;
- the adaptive tier's swaps keep it in balance: an expert leaving VRAM without a host copy is copied back from its
  VRAM slot (D2H, before the slot is refilled) into a spare slot, or into the least-ranked member's; the expert that
  moved into VRAM gives its slot back once its copy has landed.

On by itself below 96 GB installed (`--low-ram` / `--no-low-ram`; `--ram-budget` caps it). Measured against the
arena on the 128 GB PC, 300 greedy tokens:

| | decode tok/s | RAM | commit |
| --- | ---: | ---: | ---: |
| the arena | 119-122 | 67-69 GiB | 97 GiB |
| upstream's `--mmap-experts` (all experts through the OS cache) | 45 | | |
| upstream's `--mmap-experts --resident-cpu-experts` (static cache only) | 87 | | 78 GiB |
| the tier, whole budget, the same cache slots | 108 vs the arena's 109 | | |
| the tier, 48 GiB | 114 | 52 GiB | 82 GiB |
| the tier, 40 GiB (1,713 experts from the file) | 100 | 45 GiB | 74 GiB |

A 64 GB PC, emulated: a ballast process locks 59-62 GiB in large pages so that 57.6 GiB stay available (a 64 GB
PC whose Windows uses 6), `STRATA_EMULATE_RAM_GIB=64` for the automatic rule:

| | RTX 5090 | 24 GB card (131K) | 16 GB card (64K) |
| --- | ---: | ---: | ---: |
| experts in RAM / outside VRAM | 17,085 / 17,085 + 1,966 of VRAM's | 18,987 / 19,292 | 19,083 / 22,028 |
| decode, 600 tokens after a 60-token prompt | 112-118 tok/s (96+ GB: 112-122) | 86-90 (96+ GB: 95) | 54-56 (96+ GB: 67) |
| a 32K prompt: read / refill / first token | 5,635 tok/s / 1.8 s / 9.3 s (arena: 5,725 / - / 5.7 s) | | |
| decode after 32K | 70-74 tok/s (arena 77) | | |
| peak RAM (the engine) | 53-54 GiB | 54 GiB | 53-55 GiB |

With the server and the CPU image encoder beside it (the tray's setup), 57.6 GiB available before the start: ready
in 20 s, a picture read correctly, a 31.5K-token document, the lowest free RAM 3.3 GiB. Large pages were refused
there (the ballast had fragmented RAM), so these ran on 4 KB pages; the first run right after locking the ballast
was slower while Windows reorganized memory, the later ones steady.

Correct to the bit: with the PCIe share off and a static cache, the arena and a 38 GiB tier (2,677 experts from the
file) generate the same 302 tokens; `STRATA_TIER_VERIFY` found all 14,602 pinned copies - including those copied
back from VRAM - equal to the file; first-token KL to the arena with a 44 GiB tier mean 0.0007 (noise). With the
PCIe share on the outputs part after ~30 tokens: file-backed experts cannot be DMA'd, so the CPU computes them
where the GPU would have, rounding differently. The arena path is unchanged: KL 0 to the previous release.

### Other GPUs: RTX 20, 30 and 40 (2026-09-30)

Nothing on the NVFP4 path needs Blackwell except the optional `w4a4` (FP4 x FP4 MMA, sm_120a), and that already
falls back: `w4a8` is ggml's int8 MMQ with Blackwell hidden from it (`mmq_nvfp4_w4a8.cu`), which runs from sm_75;
decode's NVFP4 experts use `__dp4a` and a software FP8 scale decode. The release was built for `120a` only. It is now
built for `75-real;86-real;89-real;120a-real` (114 MB instead of 44), and the engine names a card it has no code for
instead of failing at its first kernel. CMake turns `120` into `120a` as ggml's own CMake does.
(Since 0.1.31-nvfp4.1 the build is `120-real` and only the W4A4 unit gets `120a`; see below.)

Tested on the RTX 5090 alone: an engine built as PTX for an older architecture (`86-virtual`) runs through the
driver's JIT with `__CUDA_ARCH__` = 860 in every kernel, and `STRATA_EMULATE_CC=86` makes the host side answer as
that card does (compute capability for ggml's MMQ configs and the QSA kernels' choice, shared memory per block:
64 KB on Turing). cuBLAS is the one part not covered - it runs the 5090's kernels; upstream makes the same calls on
RTX 20.

| test | sm_75 | sm_86 | sm_89 |
| --- | --- | --- | --- |
| `mmq_nvfp4_parity` (w4a8 vs a double reference, T = 1-300) | identical to sm_120a | identical | identical |
| `nvfp4_expert_gpu_parity` (decode experts) | identical, worst 1.160% | identical | identical |
| `w4a4` requested | falls back to w4a8 | falls back | falls back |
| first-token KL vs the sm_120a release, 9 prompts | mean 0.0034, max 0.027 | mean 0.0003, max 0.0012 | mean 0.0005, median 0 |
| top-1 | 9/9 | 9/9 | 9/9 |

sm_86 and sm_89 are at the noise floor (several prompts bit-identical). Turing's 1-8K prompts are too; its 32K
prompt is not, and the cause is the same on the 5090 itself: `STRATA_QSA_WARP=1` (the pre-sm_80 QSA selection and
prompt attention, FP32 FMAs instead of 3xTF32 and FP16 MMA) gives KL 0.029 at 32K there as well. Either kernel alone
stays at the floor (selection 0.004, attention 0.006):

| prompt | Turing's QSA kernels | control: another summation order (`STRATA_PROMPT_ATTN_V1`) |
| --- | ---: | ---: |
| 16K | 0.0004 | 0.0003 |
| 32K | 0.029 | 0.005 |
| 125K | 0.023 | 0.008 |

So on RTX 20 long prompts land 3-6x further from the RTX 30+ result than an FP32-level change does, with the same
top token; which of the two is nearer the exact function is open (both are FP32-level by design). The four-arch
release on the 5090 against the 120a-only build: two prompts bit-identical, KL max 0.0007 (the run-to-run noise:
the slot count follows the free VRAM at start).

VRAM: the fixed part is ~7.6 GiB at 32K context and ~11 GiB at 262K (dense weights, KV, draft head, prompt
buffers); the rest caches experts. Measured on the 5090 with each card's budget (`--vram-reserve-mib` = 700 MiB +
the difference; a ballast process instead does not work under WDDM, which moves an idle process's VRAM to RAM):

| budget | 262144 | 131072 | 65536 | 32768 |
| --- | --- | --- | --- | --- |
| 24 GB | 4,415 slots, 84 tok/s | 5,158, 95 | | |
| 16 GB | 1,309, 61 | 2,051, 65 | 2,422, 67 | |
| 12 GB | does not fit | | 869, 57 | 1,055, 59 |
| 8 GB | | | | does not fit |

Decode falls toward what the CPU pool alone carries (~57 tok/s on this CPU and DDR5-5600); a real card's own speed
and PCIe generation come on top.

### Images (2026-09-30)

Upstream's vision path (`--vision`, the `strata-vision` helper on llama.cpp's mtmd, M-RoPE positions for image
cells) was in the code already; it only needs an image encoder. The checkpoint ships its vision tower in BF16
(`model.visual.*`, 333 tensors, excluded from quantization), and llama.cpp's converter at the pinned commit turns
it into an mmproj (`Qwen4ExpVisionModel`). Two pictures (an 800x450 invoice with text and two coloured
rectangles: 350 image tokens; a 3840x2400 photo: 1,000), each encoder variant against a reference (CPU, FP32 weights,
attention without flash attention):

| encoder | invoice | photo (worst row's cosine) | time |
| --- | ---: | ---: | ---: |
| **CPU, FP32 weights, FA** (now) | **0.09%** | **0.11%** (0.9999) | 1.8 s / 6.2 s |
| CPU, BF16 weights | 1.7-2.3% | 2.4-2.5% (0.989) | 1.8-10 s |
| GPU, BF16, FA (upstream's default) | 1.9% | **11.2%** (0.67) | 0.05 s / 0.18 s |
| GPU, BF16, no FA | 2.1% | 3.1% (0.984) | 0.06 s / 0.21 s |
| GPU, FP32, no FA | 0.45% | 2.1% (0.971) | 0.06 s / 0.22 s |
| CPU, FP32, a native (AVX-512) ggml build under MSVC | 4.4% | 20.8% | 3.4 s / 21.6 s |

ggml-cuda's flash attention (K and V in FP16) is what loses the photo's rows; the portable (AVX2) CPU build is exact
to 0.1% and as fast as any other CPU variant. The native MSVC build of ggml-cpu is both wrong and slower - not used.

What each costs the text, decode A/B (300 greedy tokens, 262K context, 3-4 interleaved runs each, medians):

| | expert slots | decode tok/s |
| --- | ---: | ---: |
| text only | 7,352 | 114.9 |
| `--vision` (the M-RoPE table), no encoder | 7,377 | 113.4 |
| `--vision` + the CPU encoder resident in RAM | 7,363 | 115.2 |
| `--vision` + upstream's GPU encoder (served, 3 answers each) | 6,684-6,805 | ~95-107 (text: ~120) |

The CPU encoder costs nothing: the engine is idle while a picture is encoded. Two changes to `strata-vision` make it
so: on the CPU it hides the GPU from itself (a CUDA build opened a context there, 0.4-0.7 GB, 150-260 slots), and
it skips the warm-up at 1,024 tokens (that only reserves GPU buffers; on the CPU it delayed the engine's start by
~6 s). The release builds it CPU-only (`release/build-vision.cmd`, 4.9 MB). Answers checked: the invoice's text
exactly, the left rectangle blue and the right red (the image cells' 2-D positions are right), the photo described
correctly, in English and Ukrainian, through the OpenAI and the Anthropic API.

### Upstream 0.1.28 merged (2026-09-30)

Upstream's 0.1.25-0.1.28 came in with a merge: the draft layer's prompt pass in batches (E-9), the hyper-connection
read and write fused in the prompt path (F-1, F-2 - extended here with the BF16 remainders of the split), the
expert grouping tables through mapped memory, K8V4 KV (`--kv k8v4`, not used here), the AMD/HIP and Turing ports,
the WDDM cache-sizing steps, tool-call parsing and cancellation fixes. Measured against the previous release, both
portable builds: first-token KL mean 0.0002 / median 0.00002 (noise: the cache holds 17 more experts), prompt
reading 1K +12%, 4K +14%, 8K +8-13%, 32K +16% (4791 -> 5539 tok/s), decode unchanged (107.9 vs 106.6 tok/s).

The draft head's token subset (`rt/draft_vocab.bin`, the rows the MTP head may propose) held 142 of the
vocabulary's 18,580 Cyrillic tokens. With the whole script added (`tools/draft_vocab.py --add cyrillic`, 58,963 ids,
+50 MB of VRAM):

| prompt | subset | tok/s | tokens per round |
| --- | --- | ---: | ---: |
| Ukrainian | English/code (upstream's) | 83.2 | 1.40 |
| Ukrainian | + Cyrillic (now) | 108.8 | 2.11 |
| English | English/code | 111.9 | 2.40 |
| English | + Cyrillic | 120.5 | 2.47 |

Upstream's CJK subset (106,299 ids) is not the default here: ~180 MB more of the draft head for scripts this fork's
users do not write; `--add cjk` builds it.

### Second pass (2026-09-30, two read-only audits with a fresh context, then measured)

Where a decode round goes (nsys, 262K context, ~21 ms a round): the GPU runs back to back; 4.6 ms is the copy
kernel pulling the PCIe share and 3.4 ms is spinning on the CPU pool's flags - neither overlaps other GPU work. The
CPU pool reads DRAM at 55 GB/s against a measured ceiling of ~65 GB/s (4 x 32 GB DDR5-5600): it is memory-bound.

Kept:
- **Start: the expert arena loads on its own thread** while the dense weights, PLE, MTP and head load, and the
  cuBLAS handle (0.9 s) is created on another: first prompt token at ~8.0 s instead of ~10.1 s.
- **Prompt path: NVFP4 scales applied where they are read** (swiglu, combine) instead of passes over the MMQ
  outputs, and one launch per expert gather: +5.6% prompt reading at 32K, bit-identical.
- **The verify commit does not wait** (single GPU): it overlaps the MTP draft; +2.1% rounds/s.

Tried and dropped:
- The hyper-connection kernels (3.4 ms/round, ~5x off the weights' bandwidth): 2 or 4 warps per block instead of
  8 (more SMs), all weight chunks loaded up front, activations read without the staged tiles - none faster, some
  slower. Nsight Compute crashes here (0xC0000409), so the stall reasons were not measured.
- Prefetching the next token's inputs in the prompt path's GDN recurrence: the phase -4%, the chunk unchanged.
- Readers not waiting for the arena's per-layer registration on large pages: 0.9 s faster and a CORRUPTED arena
  (STRATA_VERIFY_ARENA different every run). The wait stays.
- A drafter window of 8K or 4K instead of 32K: 0.54 s less prompt reading at 32K, but decode -14% at that length.
- The adaptive tier's swaps spread over every round (24 a round instead of 96 every 4): 1.2% fewer rounds/s.
  Without the tier decode drops 23%. High process priority: no change.

### A second round of kernels (2026-10-08) - TODO

### Upstream 0.1.40.3 (2026-10-08, release 0.1.40.3-nvfp4.1)

- **What upstream brought that matters here** (most of 0.1.40.3 is AMD and Intel: setup, Docker, the Arc
  A-series, the bundled HIP runtime's DLL closure):
  - **The drafter's router guard** (6c0fca38, #1357): the MTP drafter's per-token native top-10 router is called only
    for `n_expert == 512` and `k == 10`, as the window path and the main layers already were; another model read past
    each row. This model is 512 / 10, so nothing changes here.
  - **#1376, the Windows auto-cache floor:** an auto cache on a card of 16 GiB or more keeps 2,560 MiB free after it
    (044c59fe), then limited to HIP builds (4895f63a). The block sits after the fork's `gemm_prewarm_wait` and free
    VRAM read and compiles to nothing on CUDA, so the cache sizing here (the elastic K/V's included) is unchanged.
  - **The verify window's interleaved q8_1 projections:** made opt-in (f7610d2c) and reverted (668072aa), so
    `STRATA_MMVQ_IL` stays on as in 0.1.40.2.
  - **The tokenizer** (4dc544ed, #1385): `_bpe` reads `self.ranks.get` once per word; a threads-vs-serial encode
    test in `serve/test_detok.py`. #567's prompt encoder, which the fork carries, does not touch `_bpe`.
  - **Web** (c13ddd62, #1392): a turn with no answer text (reasoning only, a stop, an error) goes back in the
    history as an empty assistant turn. The fork's server changes no web file.
  - **HIP diagnostics:** why the bundled `amdhip64_7.dll` lost to another copy (#461, a `LoadLibraryEx` probe) and a
    missing render group on Linux; Windows AMD telemetry says why it is empty (#1380).
- **The port.** v0.1.40.2 is an ancestor of v0.1.40.3, and rel/0.1.40.2's first-parent chain since it holds the
  0.1.40.2 port's 65 commits, `qsa_select_bench`'s floor, the round's six merges (one commit each; every merge was
  clean, `git merge-tree` gives its tree) and the `STRATA_ADAPT_FETCH=2` default: 73 commits, all carried, none
  empty. One conflict, resolved by hand: the fork's arena thread (f75be384) moves the card check before the arena
  starts loading, and upstream added the #461 probe inside the old place. The old place takes the fork's side (the
  block gone), the moved block is upstream 0.1.40.3's block line for line. `port_diffcheck.py` shows only that move;
  the other 72 have the same patch-id as before. Of the fork's extras, #567 has no newer head (closed with the
  history rewrite, 76b916e2); #1379's PR head and the fork's gate are the same change but for a comment that names
  the fork's default.
- **Checks** (release build, sha 888af0e8):
  - **Fresh references first** (`port-refs\0.1.40.2r2pre`, a release build of 99007088). They now fix decode's PCIe
    share too (`--pcie-frac 0.25`; with `STRATA_PCIE_BALANCE=0` the link probe's share was a timing, and with
    `STRATA_ADAPT_FETCH=2` that share also decides what the tier admits). Against 0.1.40.2-nvfp4.1 the logits differ (GPTQ +
    Q8_0-down 2K KL 0.00022, 32K 0.0012, ModelOpt 2K 0.000007, the same top token) by `STRATA_ADAPT_FETCH=2` alone:
    the dumped logits are the first verify window's, a decode window. With `STRATA_ADAPT_FETCH=0` the head gives
    0.1.40.2-nvfp4.1's bits exactly, so the round's other changes keep them.
  - **Against those references,** the logits and 32 tokens are identical on both packs (GPTQ + Q8_0-down after 2K and
    32K, ModelOpt after 2K; CPU share 0.5, PCIe share 0.25, 6,000 slots).
  - **With every fork default off** (the list in `port_check.sh`, `STRATA_ADAPT_FETCH=0` and
    `STRATA_ADAPT_EVICT_SYNC=0` included), the logits equal upstream 0.1.40.3's byte for byte on IQ2_XS (32K,
    12,000 slots).
  - **Tests:** 102 of 105 pass; the three that fail need model files this machine does not have (`ple_parity`,
    `expert_parity`, `pool_test`). `qsa_select_bench` passes with its floor from each sample's sum |q*k|. The
    server's tests: 571 OK (upstream's new threads-vs-serial tokenizer test included).

- **What upstream brought that matters here** (its release notes have the full list):
  - **This fork's CPU share** (#1282), opt-in upstream: `STRATA_PREFILL_CPU_SHARE=auto|x` (978d3558). Two follow-ups
    of the maintainer's: the CPU-share thread makes no ExpertSource call, its blobs are taken on the prompt thread
    through `blob_stable` (667f2eca; `blob()` counts reads without a lock and raced the prompt thread's own calls), and
    `STRATA_DBG_NAN` leaves out the CPU's rows of GU and H, which the GPU never writes (4322e241).
  - **#789:** the routing ids are read first and the shared expert is queued after the host grouping, so it overlaps
    the routed-only path's first uploads; the staging ring counts real transfers. On by default
    (`STRATA_PREFILL_STREAM_AHEAD=0`: the old schedule), the same bits here (the checks below).
  - **Eddoursul's F4:** the verify window's 2-4 token dense projections read an interleaved copy of the q8_1
    activations, bitwise the multi-column kernel's (`STRATA_MMVQ_IL=0`: off).
  - **Opt-ins:** PDL in the verify window (`STRATA_DF_PDL=1`), mixer work on graph side streams (`STRATA_DF_BRANCH`),
    per-layer slot sizes with `--expert-cache-per-layer`, the pipeline windows beside the asynchronous adaptive tier,
    sampled-draft acceptance (`STRATA_SPEC_PROB`, `STRATA_SPEC_GUMBEL`). The Stager's sleeping waits are Linux-only;
    Windows keeps the yield spin.
  - **Server:** #615's token accounting (which this fork carried; upstream's version replaces it), Prometheus
    `/metrics` (#793), chunked request bodies (#893), read timeouts for the engine's READY line and the image encoder
    (#1317), the vision encoder loaded with a lazy model, and the setup's SHA-256 check of the downloaded engine.
- **The port.** rel/0.1.40 has 70 commits and five round merges; `port_rebase.sh` had taken first parents without
  merges, which dropped the five commits the merges bring (fixed: a merge that changed the tree stands for its
  branch's commits). Of the 75, 63 are carried:
  - dropped: 0.1.40.1's seven commits (the same patch-ids are upstream), the fork's #1058 gate and `/metrics` counter
    with their two reverts (net zero, checked), and #615 (upstream's);
  - **the CPU share:** upstream's code is the base. The fork's three commits shrink to what the fork adds: `auto` is the
    default (upstream: off); the CPU's NVFP4 rows carry s_down, so their `row_sd` is 1 (`cpu_ones`); NVFP4 packs read
    routed-only up to 4,096 tokens (other formats 1,024, as upstream);
  - `STRATA_MMVQ_V2` (opt-in) groups k, v and q only without upstream's query branch (`STRATA_DF_BRANCH` makes q on a
    side stream); otherwise it falls back to upstream's `mm()` (F4's path);
  - the card check, which the fork runs before the arena thread starts, carries upstream's new HIP hints (#1318,
    #1261); `--image-max-tokens` sits beside upstream's lazy vision load.
- **The contention gate** (sent upstream as #1379). `auto` balanced the CPU's time per expert against the GPU's,
  both measured with the share on, so nothing compared a layer with and without it: on an RX 7900 GRE + Ryzen 7
  5700X3D the host's time per streamed expert rose 107 -> 200 us with the share and the prompt was 4.5% slower. Now
  each eligible layer is timed with CUDA events from before its routing sync to its combine, per non-resident expert;
  adjacent layers alternate without / with the share until three ratios are in, then `auto` shares while the median
  of the last five favours it, with one layer in 29 in the other arm. In the fork `auto` is the default, so the gate
  decides the default prompt path, and on NVFP4 packs also for 1K-4K chunks. A fixed share (the checks' 0.5) and 0 are
  untouched. Here (RTX 5090 + 9950X3D, the GPTQ + Q8_0-down pack) it keeps sharing: on 600-, 1,300- and
  2,099-token prompts its decision was "share" after 44 of its 45 readings each (`STRATA_DBG_CPU_GATE=1`), the share
  0.48-0.51 as before. Not measured yet: the gated `auto` against the ungated one in time.
- **The chunked recurrence from 128 tokens** (on 0.1.40-nvfp4.3, just before the port). PR #1372's fix: one
  cudaMalloc of the 25.6 MB scratch per device and the kernels' attributes set once, instead of an allocation and
  `cudaFuncSetAttribute` on every call, whose host time on Windows had made a 2K prompt ~35 ms slower. The recurrence
  phase: 600 tokens 9 -> 6 ms, 2K 32-33 -> 21-22, 8K 123-129 -> 82-88, 32K 514-515 -> 310-328. A quiet 2K A/B, 3
  rounds: 1,063.5 / 1,043.6 / 1,047.2 ms with it off, 1,044.2 / 1,038.7 / 1,034.7 with it on. The same bits as
  `STRATA_GDN_CHUNKED=1` before.
- **Checks** (release build, sha 09cd28dc):
  - **Fresh references first.** The chunked default changed 2K after 0.1.40-nvfp4.3's references were made, so the
    fork's head (a release build of 48fb1b5b) made new ones (`port-refs\0.1.40n4pre`): against 0.1.40-nvfp4.3, 32K
    identical, 2K different by that change alone (GPTQ + Q8_0-down KL 0.0010, ModelOpt 0.0050, the same top token).
  - **Against those references,** the logits and 32 tokens are identical on both packs (GPTQ + Q8_0-down after 2K and
    32K, ModelOpt after 2K; CPU share 0.5, decode's PCIe model off, 6,000 slots). Nothing in the port moved a bit.
  - **With every fork default off** (the 0.1.40-nvfp4.3 ones and the arena thread included), the logits equal upstream
    0.1.40.2's byte for byte on IQ2_XS (32K, 12,000 slots, a budget-bound cache).
  - **Tests:** 97 of 101 pass. Three need model files this machine does not have; `qsa_select_bench` fails its FP64
    accuracy line the same way in upstream's own 0.1.40.2 build (the fast scorer 2.4e-4 off at a score scale of 229).
    The server's tests: 570 OK.
- **Speed** with the GPTQ + Q8_0-down pack, interleaved chats (1,000 tokens, `--stop-eos`) against the tray's
  0.1.40-nvfp4.3, two sets of 5 pairs (an earlier attempt was stopped: another build started during it):

  | ms a round | 0.1.40-nvfp4.3 | 0.1.40.2-nvfp4.1 |
  |---|---|---|
  | first 5 pairs | 15.34 +- 0.22 (158.9 tokens/s) | 16.10 +- 0.88 (155.2) |
  | next 5 pairs | 15.63 +- 0.54 (156.8) | 15.62 +- 0.57 (160.0) |

  The same within the noise (10 pairs: 157.8 against 157.6 tokens/s).

### A round of kernels (2026-10-07, release 0.1.40-nvfp4.3)

Where the time goes, from nsys profiles of the tray engine (RTX 5090; the card's own peaks measured: VRAM read 1640
GB/s, BF16 tensor 233 TFLOPS, INT8 tensor 732 TOPS, FP32 113 TFLOPS, PCIe 57.3 GB/s). A 32K prompt (4.22 s of kernels):

| stage | ms | share | work | at peak | of peak |
|---|---|---|---|---|---|
| experts' MMQ (W4A8, INT8 MMA) | 900 | 21% | 151 TOP | 207 ms | 23% |
| dense projections (cuBLAS BF16) | 953 | 23% | ~192 TFLOP | 824 ms | 87% |
| hyper-connection GEMMs | 216 | 5% | 40 TFLOP | 173 ms | 80% |
| DeltaNet recurrence | 513 | 12% | 7.3 TFLOP FP32 | 64 ms | 12% (token by token) |
| prompt attention | 416 | 10% | ~19 TFLOP | ~85 ms | ~20% |
| hyper-connection elementwise | 474 | 11% | ~6.4 GB a call pair | - | ~95% of bandwidth |
| MoE elementwise | 386 | 9% | memory | - | ~65-80% |

A decode round (15.7 ms): dense GEMVs 3.5 ms (the big matrices at 81-100% of bandwidth, the rest launch latency), the
cache's experts 2.4 (~bandwidth), PCIe misses 3.4 (the link), waiting for the CPU pool 1.7, the hyper-connection read
1.9 (41% of bandwidth), attention + GDN steps 1.5, small kernels 1.3. The 26% "idle" an nsys profile shows between
windows is mostly CUPTI's own cost (node-level graph tracing): with host timestamps it is ~0.5 ms a round (3%), the
graph launches' latency on WDDM.

Seven parallel agents, each reviewed, tested and committed separately:

- **Experts' gate/up on FP4 tensor cores (w4a4x2, default).** What bounded the MMQ: the FP32 scale epilogue after
  every int8 MMA (NVFP4's scales every 16 values). Blackwell's block-scaled FP4 MMA applies them in hardware (649 TOPS
  at the engine's shapes against w4a8's 241), but plain W4A4 rounds activations to E2M1 (8.5% per element). w4a4x2
  gives each activation row two FP4 terms sharing the row scale (x ~ q1 + q2, q2 the NVFP4 of the remainder), both
  multiplied in one kernel: 0.71-0.73% against double (w4a8 0.53%); down stays w4a8 (K 640, two passes are slower).
  The same bytes as q8_1. 32K -4.4% alone. Also the prompt path quantizes each token once and scatters it to its 10
  expert rows (the same bytes; 66 -> 44 ms at 32K).
- **The prompt attention on INT8 tensor cores (default).** v2 used FP16 MMA with q and p split hi/lo and converted
  each int8 K / V value to fp16; the new kernel takes K / V as stored and the other operand as three exact 8-bit parts
  (24 bits, exact int32 sums), 3 blocks an SM instead of 2. 1.34-1.46x the kernel, 346 -> 246 ms at 32K; against
  FP64 2.2e-6 (v2 3.2e-6); end to end closer to the FP32 kernel than v2 at 2K (KL 0.007 against 0.10).
- **The DeltaNet recurrence in chunks (default for chunks of 16384+).** The gated delta rule's chunked WY form in
  FP32 (chunks of 32; T = (I + A)^-1 per chunk in parallel, then a scan over chunks): 1.69x the recurrence at 8K-32K,
  2.7e-7 - 1.1e-6 from the sequential kernel; a 32K prompt -3.7%. A 2K prompt was ~35 ms slower with it, hence the
  start at 16384. The sequential kernel itself got a variant with a thread per value head (the same bits, 1.05-1.13x).
- **The hyper-connection up projection fused with its mix (default on sm_120).** gr_mix_r and gr_write_norm_rs were
  already at ~95% of bandwidth; the fused kernel never writes / reads the up GEMM's FP32 output (2.6 GB a call at
  32K): 1.45-1.77x the pair, the same bits at chunks of 33+ tokens on the RTX 5090.
- **The hyper-connection read v4 in decode (default).** v3's arithmetic scheduled better (weights first, __ldg,
  no bank conflicts, two rows a warp, one butterfly reduction, PDL on sm_90+): the same bits, 1.16-1.39x the kernels
  at T = 1..8; bound by per-token FP32 work, not its 13 MB of weights.
- **Measured, not taken:** decode's dense GEMVs grouped (STRATA_MMVQ_V2, opt-in, the same bits, under 1%); upstream
  0.1.40's hyper-connection kernels (17.02 against v3's 16.63 ms a round); the recurrence's row groups through
  __shfl_sync (0.91-0.98x) or 8 row groups (0.83x); a GPU-side chain to hide the host between windows (~3% to gain).

**Quality**, 8 prompts against a high-precision reference (FP16 experts, FP32 attention, the sequential recurrence,
no CPU share; first-token KL, the 32 greedy tokens):

| prompt | old defaults | 0.1.40-nvfp4.3 |
|---|---|---|
| 250 | 0.00035 | 0.00004 |
| 600 | 0.0049 | 0.0032 |
| 900 | 0.0015 | 0.0019 |
| 1,300 | 0.0021 | 0.00009 |
| 1,700 | 0.052 | 0.192 |
| 2K | 0.026 | 0.0095 |
| 8K | 0.0004 | 0.0007 |
| 32K | 0.0028 | 0.0055 |

Median 0.0025 against 0.0026, the same top token on all 8, the greedy tokens the reference's on 7 of 8 (old: 6).
p1700 is the prompt a routing flip moves (0.008-0.76 across CPU shares in 0.1.39-nvfp4.4).
Against 0.1.40-nvfp4.2 itself (port_check1's deterministic settings, CPU share 0.5): the GPTQ + Q8_0-down pack after
2K KL 0.19 and 32K 0.008, the ModelOpt pack after 2K 0.006, the same top token - the new defaults; the table above
is the comparison that says which side is closer.

**Speed**, 3 interleaved pairs: a 32K prompt 4110-4410 -> 3868-3994 ms (-8%), 8K even, 2K even (3 pairs: 1036-1095 against 1043-1083 ms). Decode against 0.1.40-nvfp4.2: 15.8 against 16.1 ms a round, 5 interleaved chats, the GPU ring 7.06 against 7.41 ms.

### Upstream 0.1.40.1 (2026-10-07, release 0.1.40-nvfp4.2)

Upstream's hotfix changes the server and setup only. The engine is 0.1.40-nvfp4.1's, byte for byte (the same exe).

- **#1058, upstream's way.** The maintainer fixed it in 0.1.40.1 with the conditions of #1068, which upstream closed when it rewrote its history:
  - the call begins a line;
  - only whitespace or further calls follow it;
  - the turn ends by itself.

  Two things go further than #1068:
  - no call is taken from a code fence or inline code, which closes the gap #1068 left (an opener inside a fence that never closes);
  - a call quoted in a fence or inline code in the visible answer is text too.

  The fork drops its own gate and takes upstream's, so `finish()` takes the finish reason (`reason`), not a flag. The counter is upstream's `totals.tool_calls_from_reasoning`, and a log line notes a call kept as text on a max-tokens cut.

  One reading differs from #1068. In a mixed tail, a declared call followed by an undeclared `<function=tool_call>` envelope, upstream delivers the declared block. #1068 kept both as text. The issue's author agreed the per-block reading is the consistent one. Upstream's 37-entry corpus (`serve/fixtures/rcall_specimens.json`) passes here.
- **Restarts (#1012).** A restart keeps the admission state, so waiting requests no longer hang or fail. A request that meets a restart in progress gets EngineDied. A `--batch` request whose engine died during its prompt read ends at once instead of after 300 s.
- **An engine that exits after an unread ERR line says why** (#997, #890).
- **Checks:** the server's tests, 466 OK.

### On upstream 0.1.40 (2026-10-06, release 0.1.40-nvfp4.1)

- **Upstream since 0.1.39** (its release notes have the full list):
  - crash and NaN fixes, among them this fork's #838 (q8_1 scales) and #550 (the residency-table upload);
  - the image fixes this fork carried: #553 (image sources), #554 (vision markers), #555 (temp files), and a tool's unreadable picture as a note;
  - this fork's elastic K/V as `--kv-grow` (#1040) and its deferred arena registration as `STRATA_DEFERRED_REGISTER=1` (#1039), both opt-in upstream;
  - decode fusions and kernels from Eddoursul's fork and #783;
  - Q4_0 / Q4_1 experts, PLE tables in Q8_0 / Q5_1 / BF16, Strix Halo;
  - a tool call written inside the reasoning delivered when its name is declared (#804).
- **Conflicts:**
  - **The elastic K/V.** It is upstream's code now, with this fork's defaults: on, and `--no-kv-grow` allocates the whole window. Two fixes of this fork are kept on top, because upstream's version lacks them:
    - a run that cannot lend cache slots maps the whole window up front;
    - the d_res uploads are synchronized.
  - **The arena registration.** It is upstream's opt-in `STRATA_DEFERRED_REGISTER`, off by default. This fork had its own version on. Measured below.
  - **The adaptive tier.** The fork's no-wait stays. 0.1.40's `STRATA_ADAPT_LAG` (#764) only applies to upstream's wait (`STRATA_ADAPT_WAIT=1`). Measured below.
  - **The CPU share.** It leaves the expert order before 0.1.40's resident sort (`STRATA_MMQ_RESIDENT_SORT_NE`). That sort only runs when a layer's experts are all resident, and then none go to the CPU.
  - **NVFP4.** 0.1.40's float4 combine kernel takes `row_sd` (NVFP4's per-row down scale). Q2_0 is added to 0.1.40's new format tables.
  - **#567.** The author's current version, 76b916e. `PromptEncoder.encode(prompt, plain)` resumes across the plain spans of 0.1.40's literal marks, so a conversation that quotes `<think>` or a control token re-encodes only its new part. The fork's copy encoded such a prompt in full every turn.
- **A tool call quoted in the thinking (upstream #1058; this fork's gate, sent as #1068, was replaced by 0.1.40.1's in 0.1.40-nvfp4.2).** 0.1.40's #804 rescue delivered any complete call to a declared tool written inside the reasoning. That included an example the model only quoted, even on a reply cut by max_tokens (the issue shows a quoted `rm -rf build`). Such a call is now held until the turn ends. It is delivered only if:
  - its `<tool_call>` begins a line;
  - only whitespace and further calls follow it;
  - the turn ends by itself (finish `stop`).

  Otherwise its text stays reasoning. The issue's corpora (PR #525's 25 specimens and 12 live strandings) were fed whole, one character at a time and in random chunks:
  - 0.1.40 differs from the expectation on 12 of them, this fork on none;
  - every live stranding that is a call is still delivered.

  `GET /metrics` counts both outcomes (`reasoning_calls_delivered`, `reasoning_calls_kept_as_text`).
- **Checks** (release build):
  - **With the fork's own defaults off,** the logits and 32 tokens equal upstream 0.1.40 byte for byte on IQ2_XS (32K, the same expert cache). #353 was closed upstream, so this replaces the two-step check through it.
  - **The same command with an automatic cache** gets 248 fewer slots (0.33 GiB) than upstream. This fork makes its first cuBLAS handle during the arena load (`gemm_prewarm`), before the cache is sized, so the 700 MiB reserve stays free. Upstream makes it after the cache, out of the reserve (~335 MiB). Both end with the same VRAM free.
  - **Against 0.1.39-nvfp4.4,** the logits and 32 tokens are identical on both packs: GPTQ + Q8_0-down after 2K and 32K, ModelOpt after 2K. This holds with the CPU share fixed at 0.5 and decode's PCIe model off.
  - **With the measured share (the default),** a short prompt's result varies from run to run. The share comes from timings, and which experts the CPU takes follows from it, so a 2K prompt's first token moves by KL up to 0.08 (top token the same).
  - **The cache must also be sized by its budget, not by free VRAM.** On the Q8_0-down pack, 8000 slots were bounded by free VRAM, which the desktop moves by a few slots. Those experts then went to the CPU. `port_check` now pins the share, turns off the PCIe model and uses 6000 slots.
  - **Tests:** 84 of 87 pass; the other 3 need model files this machine does not have. The server's tests: 437 OK.
- **Speed** with the GPTQ + Q8_0-down pack, 5 interleaved chats each: 16.08 +- 0.50 ms a round (155.7 tokens/s) against 16.11 +- 0.28 (150.4) for 0.1.39-nvfp4.4 - the same.
- **Measured, not taken:**
  - **`STRATA_DEFERRED_REGISTER=1`**, the registration this fork ran on 0.1.39. 3 interleaved chats against the default:
    - with it, 15.21 / 15.23 / 16.82 ms a round; without, 16.08 / 17.20 / 15.89;
    - ready 9.4-9.7 s against 9.8-11.9 s.

    0.1.39-nvfp4.4 had it on and this release has it off, and the two decode at the same speed (above). So the fork
    keeps upstream's default: the gap is within the noise, and upstream measured it 10% slower on an RTX 5070.
  - **`STRATA_ADAPT_WAIT=1 STRATA_ADAPT_LAG=2`**, upstream's wait with #764's lag (greedy decode repeats exactly):
    16.68 / 16.66 / 16.56 ms a round, against the fork's no-wait above. That is ~2% slower, so the no-wait stays.

### Small prompt chunks with the CPU (2026-10-06, release 0.1.39-nvfp4.4)

An agent's turn - a tool result, a test's output - is a small prompt chunk at long context, and it cost a fixed
~0.55 s plus ~0.5 ms a token in every release: below 1,024 tokens the prompt path streams each routed expert that is
not in VRAM over PCIe (~9,900 of them, ~30 GB, for 600 tokens), and that copy was the whole floor.

- **The CPU share.** The decode pool is idle while a prompt is read, and the pinned arena holds the same experts in
  RAM. A chunk now hands the pool the non-resident experts routed by at most 8 of its tokens, fewest first; their
  rows go to Dm's tail (as a peer GPU's do) and the combine weights them as any other row. Below 4,096 tokens a chunk
  reads routed-only (was 1,024) on NVFP4 packs; on Q2_0 streaming every expert stayed better there (2,099 tokens 714
  against 763 ms), so other formats keep 1,024.
- **The share is measured.** Each layer times the CPU thread per expert and the GPU's expert work per streamed expert
  (CUDA events, read after the next layer's routing sync - the host reaches the combine long before the GPU does;
  timing the host gave 0.18 and was slower), and the next layers hand the CPU g / (c + g), where both end together.
  It settles at 0.51 here (~75 us an expert each side).
- **The share is gated** (since 0.1.40.2-nvfp4.1): `auto` also times layers with and without it and shares only
  while sharing is faster (#1379, "Upstream 0.1.40.2").

Warm serve turns at 95K (ms, two runs each):

| new tokens | before | 0.1.39-nvfp4.4 | STRATA_FORCE_AVX2=1, 6 workers |
|---|---|---|---|
| 229 | 605-697 | 477-546 | 468-559 (630-733 without the share) |
| 601 | 768-856 | 601-669 | 601-657 (813-893) |
| 1,193 | 1,279-1,281 | 788-880 | 772-851 (1,303-1,308) |
| 2,422 | 1,299-1,327 | 1,134-1,142 | 1,086-1,370 (1,327-1,350) |
| 4,631 | 1,344-1,373 | 1,359-1,404 | 1,344-1,404 |

Smaller CPUs, emulated (600 tokens, measured share against a fixed 0.5): 2 workers 0.42, 761 against 900 ms; one
worker on ggml-cpu's dot 0.26, 891 against 2,330 (925 with no share) - a fixed share would cost such a CPU more than
it saves.

Accuracy: layer 0's expert rows (same input) are 1.086% from the FP16 prompt path on the CPU against 1.087% for the
GPU's MMQ rows. The first token's KL to the FP16 path was lower on 10 of 12 prompts (250-2,100 tokens, both packs);
a routing flip moves a single prompt either way (p1700 went 0.008-0.76 across shares 0.3-0.7). Deterministic run to run at a fixed share. The measured share follows timings, so which experts the CPU takes,
and a short prompt's last bits, move from run to run ("On upstream 0.1.40").

**Upstream's formats** (ISTA's GSQ-RCO files): Q2_0 layers take the pool's Q2_0 kernels' activations (ActQ, as
decode's). KL to `STRATA_PREFILL_MMQ=0` in the noise (Q2_0 600 tokens 0.0033 without the share, 0.0068 with it,
0.0017 on the AVX2 rows; 1,300 0.0007 / 0.0001). Q2_0 gains nothing: its 1.15 MB experts leave a small chunk short of
PCIe-bound even on an emulated 12 GB card (`--vram-reserve-mib 20480`, 5 workers: 597 / 594 ms at 600 tokens, 717 /
704 at 1,300, with the CPU taking 28-43% of the streamed experts). IQ2_XS: 492 -> 469 ms at 600 tokens.

**Decode's PCIe share** was pcie_frac of each layer's misses, from a link probe alone. With a weak CPU (one worker on
ggml-cpu's dot) the round was 34.7 ms at 0.25 and 22.3 at 0.75. The spin kernels now stamp each step's GPU work (the
plan's arrival, the PCIe part, the wait for the CPU's rows - no extra launch), the host times its pool, and least
squares fit the pool's a + c n_cpu + d n_pcie and the GPU's g0 + g n_pcie; a layer takes the PCIe count that
minimizes the longer side. d is the link's DRAM reads slowing the pool: on this PC moving experts to PCIe left the
pool's time where it was (d ~ 0.75 c), so few move; one ggml worker has d ~ 0 and most do. Two simpler controllers
were worse: balancing sums (one PCIe expert costs the GPU ~120 us against the CPU's ~70) and per-expert means without
d (+4% here).

| ms a round, chat, 1,000 tokens | fixed | measured |
|---|---|---|
| NVFP4, this PC (5 pairs) | 15.98 (0.25) | 15.93 |
| NVFP4, 1 worker on ggml-cpu's dot | 34.7 (0.25) | 20.85 |
| Q2_0, this PC (2 pairs) | 10.85 (0.55) | 10.73 |

### A peer GPU and the request drain (2026-10-06, release 0.1.39-nvfp4.3)

- **The peer tier's chunks** (fork PR #1, @chimpera, 2x RTX 3090):
  - `vmm.cpp`'s `api()` captured the current device once. Every VMM chunk was then created on that first device, so a peer cache opened on the second card allocated on the first. The peer tier failed with "out of memory" while the peer card was empty.
  - `cuMemSetAccess` refused a second location on a pair without P2P.
  - Each range now takes its device in `reserve()`, and device 0 gets access only when a P2P path exists. Taken as 4dc829d, authored by @chimpera.
- **Only the primary cache in VMM** (32d7089). `ExpertCache::set_vmm` was a global switch. The elastic K/V's conditions exclude a layer split, but not `--peer-device`, so the peer's cache was mapped chunk by chunk too, although the K/V only ever borrows from the primary's. The switch is now per cache and set on the primary alone, so the peer cache is one `cudaMalloc`, as in upstream.
  - Fork issue #2 reported a peer arena of 19.31 GiB failing on a card with ~22.7 GiB free. The budget walk and `open_sized` map the same bytes, so the gap is not the padding. Waiting for the reporter's re-test.
- **The request drain** (062178c). The fork's copy of upstream #594 had a 64 MiB byte limit. It is replaced by the PR's current version:
  - the limit is time only (5 s, one socket read at a time);
  - `/load`, `/unload`, `/config` and `/settings` read their body through `_body()`.
  - On 0.1.39-nvfp4.2 the PR's new tests found an 80 MiB body and a late control body reset (WinError 10053). Each of those three POSTs also held its connection 5.5 s after the answer: 0.1.39's #630 reads the body in the handler, and the drain then waited for it again.
- **Upstream** has none of the three: `vmm.cpp` and `set_vmm` are the fork's. Upstream's segmented cache (#533) is per cache and off with multi-GPU. Upstream's drain exists only in #594, whose author fixed it there.
- **Checks:** against 0.1.39-nvfp4.2's references at a fixed cache, the logits and 32 greedy tokens are identical after 2K and 32K on the GPTQ + Q8_0-down pack and after 2K on the ModelOpt pack. The 95K-token serve repro still grows the K/V at admission and during decode (98,304 -> 106,496 cells) over 8,000 clean tokens. Only one GPU here, so the peer path itself is untested.

### "!" after a long conversation (2026-10-05, release 0.1.39-nvfp4.2)

- **The incident** (tray, GPTQ + Q8_0-down pack, Claude Code). On a freshly started engine, a 96,414-token request
  read normally and answered ~2,800 tokens. Then the reply turned into "!" (token 0, what the sampler answers when no
  logit is finite). This was ~1,000 tokens after the decode-time K/V growth to 106,496 cells. Every later request of
  the conversation still answered "!" from its first token. Those requests were 40-1,700 tokens longer and did not
  contain the broken reply, and each restored the clean checkpoint at 79,106. So state that existed before the
  failure was damaged: expert weights in VRAM or K/V cells below 79,106. A NaN at position 99K alone does not do that.
- **Server (fork):** a reply that is token 0 repeated `repeat_stop_tokens` times marks the engine untrusted, and the
  next request starts a fresh one (about a minute). A run of any other token, a model in a loop, restarts nothing.
- **The q8_1 clamp completed (upstream #838):** 0.1.39 clamped two q8_1 activation quantizers (#606). The shared
  expert's fused SwiGLU + q8_1 kernel, on by default, still stored the block's fp16 scale and sum raw. This pack's
  shared experts are Q8_0, so decode and the MTP verify run it. The same was true of the gfx906 routed-expert kernel.
  `iq_multi_parity` covers the shared expert's kernel: 0 failures, against 4 on the old kernel. Not the incident's
  cause: on the replayed conversation the largest block amax there is 70, and the scale overflows above 8.3e6.
- **Ruled out by measurement:**
  - the expert arena: its checksum is identical on large pages and on 4 KB pages, and the incident run had fallen
    back to 4 KB (VirtualAlloc error 1450);
  - a deterministic trigger: six replays of the conversation and two 8,000-token replies past a decode-time growth,
    on large and 4 KB pages, all ran clean;
  - the resident RAM mode, which is off, and #646's zero-doorbell verifier, whose log line is absent.
- **Read and found consistent** (here and by a second, independent pass):
  - the elastic K/V's moves (synchronized before the unmap), unmaps and zeroing;
  - `map_range`, which skips mapped chunks;
  - the prompt loan and its refill;
  - checkpoints, which are host copies of the running state only;
  - captured graphs, whose buffers are sized from max_context.
- **Found on the way, not this incident's cause:** with the fork's default no-wait adaptive tier, a window can still
  route to a swap's victim while its slot is being overwritten (`d_res` is uploaded once the copy lands). The result
  is finite and lasts one token.
- **Checks:** against 0.1.39-nvfp4.1's references at a fixed cache, the logits and 32 greedy tokens are identical
  after 2K on both packs. After 32K on the GPTQ + Q8_0-down pack the tokens are the same, and the logits are
  byte-identical to 0.1.39-nvfp4.1's own binary run the same day: both differ from the stored reference by the same
  KL 0.0004. Server tests: OK.

### On upstream 0.1.39 (2026-10-04, release 0.1.39-nvfp4.1)

- **Upstream since 0.1.38:**
  - the verify pass with fewer launches and host round trips (#646);
  - the prompt path's streamed ring sized in bytes, on by default (#583);
  - several conversations at once (`"parallel": N`, #465);
  - a hot VRAM resize (`--vram-elastic`, #533);
  - the OpenAI Responses API;
  - experimental engines for older NVIDIA cards (CUDA 12), Intel Arc (SYCL) and AMD gfx906;
  - a literal `<think>` in a message encoded as text (#537).
- **Conflicts:**
  - #533's segmented expert cache and this fork's elastic K/V both reserve the cache's address range in VMM, so only one runs. The elastic K/V is off beside `--vram-elastic` and `parallel`; the cache's open and close take the segments, else the K/V's range, else one cudaMalloc.
  - #554 (vision markers in a message's text stay text) now takes #537's path. The markers join `<think>` / `</think>` as literal tags (`frontend.VISION_TAGS`), marked before the template and encoded as text through the tokenizer's plain spans. This replaces the placeholder tokenizing #554 had. The tokenizer keeps both #537's plain spans and #567's resume marks.
  - #567's incremental tokenizing is `Service.encode_rendered`. 0.1.39's `encode_prompt` calls it when no literal tag was marked.
  - #547 is reduced to its last part: `bytes_needed` takes the same expert-source flag as `init` (0.1.39 took the rest with #583).
  - The 1500 MiB reserve commit and its revert were dropped.
- **#583 on this pack:** on 0.1.38 it was 0.6-5.7% slower here. 0.1.39's version (the byte ring only where it gains) measured the same speed (two rounds each):
  - 32K: 6,886 tokens/s with it, 6,867 without;
  - 125K: 7,378 with it, 7,391 without;
  - the prompt path borrows 1.2 GiB less with it. The fork keeps upstream's default.
- **A test fixed** (also sent upstream as #785): `test_json_schema_text_format` failed on every install without `jsonschema` (optional), 0.1.39's own as well.
- **Checks** (each once, release build):
  - Against 0.1.38-nvfp4.2's references at a fixed cache:
    - GPTQ + Q8_0-down pack after 2K: logits and 32 tokens identical.
    - The same pack after 32K: the first token's logits move by KL 0.0004 with the same 32 tokens. #583 changes a long prompt's bits.
    - ModelOpt pack after 2K: identical logits, and the 32 tokens differ from the 21st on. With `STRATA_ADAPT_WAIT=1`, where greedy decode repeats exactly, both engines give the reference tokens, twice each. Without that wait, the adaptive tier's copies land at other points of 0.1.39's shorter verify rounds, and the GPU/CPU rounding of a swapped expert flips a near tie.
  - With the fork's own defaults off, the logits equal upstream 0.1.39 + #353 byte for byte (32K). #353 equals upstream on IQ2_XS (32K).
  - 66 of 69 tests pass; the other 3 need model files this machine does not have. The server's tests: 291 OK.
- **Speed** with the GPTQ + Q8_0-down pack, 5 interleaved chats each: 15.72 +- 0.10 ms a round (150.6 tokens/s) against 16.88 +- 0.17 (144.8) for 0.1.38-nvfp4.2, 6.9% shorter rounds. That gain is #646's.

### Other people's pull requests, measured here (2026-10-03, release 0.1.38-nvfp4.2)

- **The VRAM reserve is 700 MiB again on Windows, as upstream.** On 0.1.38 (RTX 5090 driving the desktop, IQ2_XS, 5 interleaved pairs), 700 left 281 MiB free after load and no run stalled. 1500 left 1,079 MiB free, but cost 570 expert slots and 0.8% of the round (13.65 against 13.55 ms), and 3.7% on a 16 GB card emulated. The stalls that made 1500 the default on 0.1.28 no longer show. Upstream #279 was closed with these numbers.
- **Taken from upstream pull requests:**
  - #567: the prompt is tokenised from the last shared prefix. With this pack's tokenizer, over a conversation growing to 125K tokens, it took 6.8 ms against 128 ms (221 ms at 125K), and the ids matched a full encode at all 49 steps.
  - #615: token accounting across a reasoning-budget continuation.
  - #594: an answer sent before the request body was read (a 401, a 403) no longer loses itself to a reset on Windows. The drain limit here is 64 MiB instead of 1 MiB: with a 2 MiB body (a long agent conversation, a picture) the 401 was still lost, 3 of 3.
- **Measured and not taken:**
  - #583, the prompt path's ring as a byte budget. It gave +1.5% on Q4_K_XL at 8K chunks, but on this pack at 32K chunks it was 0.6-5.7% slower, while lending 0.8 GiB less.
  - #500, the pool's quantization on the workers. That phase is 0.026 ms of a 6.93 ms pool call here.
  - #603 and #575, QSA top-k past the register kernel. On the 5090 #603 is the same at 32K and +1.0% at 125K; #575 is 2.5-4.3% slower. Neither changes decode. This fork waits for upstream to pick one.
- **Checks:** with the old 1500 MiB reserve, the logits and 32 greedy tokens are byte-identical to 0.1.38-nvfp4.1. At the new 700 the requested 8,000-slot cache fits 7,544 slots instead of 7,284 on the GPTQ + Q8_0-down pack, so the prompt path streams other experts. The first token's logits move by KL 0.0012 after 2K and 0.0004 after 32K, with the same 32 tokens; the ModelOpt pack is identical. With the tray's settings 437 MiB of VRAM stay free after load (7,494 cached experts). Server tests: OK.

### On upstream 0.1.38 (2026-10-03, release 0.1.38-nvfp4.1)

- **Upstream since 0.1.37:**
  - prompts on its Q2_0 and IQ packs a few percent faster;
  - `--kv q4_0` prompts on tensor cores;
  - a 6 GB card that starts (#496: the reserve shrinks until the cache fits);
  - Q5_0 experts on the GPU;
  - `--peer-device` (a second GPU as an expert cache);
  - the server checks the Host header and refuses cross-site browser requests when it has no API key.
- **Upstream took these changes from this fork** (their copies here were dropped or reduced to what is still the fork's own):
  - the group gather of the prompt path (#372);
  - the first chunk's n-gram rows beside layer 0, 256 at a time (#374);
  - unbuffered loads on Windows (#357, #362);
  - `--adapt-decay` (#407's first part);
  - the stager's first DMA of a generation (#385);
  - its own fixes for two bugs this fork had fixed: the fused layout on mixed packs (#546; 0.1.38's fix measures the same, KL 0.0036 at 32K) and `decode_cluster_parity`'s uploads (#548).
- **Conflicts:**
  - 0.1.38's group gather had no NVFP4 parts. Each expert's scale tail now goes to its group slot, and the group's last flush zeroes the MMQ tail, in the same launch (the same change went into #353). The fork's logits after a 32K prompt are byte-identical to 0.1.37-nvfp4.2's, which gathered one expert at a time there.
  - `--peer-device`'s prompt path has no NVFP4 scales, so it declines NVFP4 packs and the prompt stays on the primary GPU. The peer's decode rows do apply the scales.
  - #463's wait is upstream's default since 0.1.38. Here it stays opt-in (`STRATA_ADAPT_WAIT=1`): with this fork's tier (192 swaps every 2 rounds) it cost 11% on 0.1.38 (5 interleaved chats each: 16.80 +- 0.10 ms a round without, 18.69 +- 0.12 with), though it does make greedy decode repeat exactly (the same 190 rounds and hit rate in every run).
- **Port script:** `port_rebase.sh fork` now cherry-picks rel/<old>'s first-parent commits. 0.1.37-nvfp4.2's `merge -s ours` sat mid-history, and a plain `rebase --onto` carried 203 commits (every older port's copies) instead of 43.
- **Checks** (each once, release build):
  - Against 0.1.37-nvfp4.2's references at a fixed cache, the first token's logits and 32 greedy tokens are identical: the GPTQ + Q8_0-down pack after 2K and 32K, and the ModelOpt pack after 2K.
  - With the fork's own defaults off, the logits equal upstream 0.1.38 + #353 byte for byte (32K).
  - #353 equals upstream 0.1.38 on IQ2_XS (32K).
  - 59 of 62 tests pass; the other 3 need model files this machine does not have. The server's tests: 205 OK, 7 skipped, after one fix to #553's test (below).
- **#553's test against 0.1.38's cross-site gate:** 0.1.38 refuses a page of another site without an API key (403) before the image rule runs, so the test that expected the rule's 400 failed. The rule still matters for a page `cors_origins` lets in (`"*"`: every page), which could otherwise name any file and read the model's description of it. The test checks that case now; #553 carries the same change.
- **Speed** with the GPTQ + Q8_0-down pack, 5 interleaved chats each: 16.82 +- 0.02 ms a round (145.1 tokens/s) against 16.92 +- 0.08 (143.6) for 0.1.37-nvfp4.2.

### Fixes from a code review (2026-10-03, release 0.1.37-nvfp4.2)

Three read-only reviews (the prompt path; VRAM/RAM tiers; decode, converters and the server) found these; each was
checked against the code before it was fixed, and each upstream bug went upstream as its own pull request.

- **The prompt path borrowed 1.25 GiB too much at a 32K chunk** (upstream, #547). `bytes_needed` counted a buffer
  `carve` allocates only under `STRATA_GR_UNFUSED=1` and missed another: 40,944 bytes a token. On IQ2_XS at 8K chunks
  the loan went from 3,340 to 3,108 slots with byte-identical logits; at this fork's 32K chunk it is 1.25 GiB.
- **`STRATA_PF_FUSED=1` wrote past its buffers on mixed native packs** (upstream, #546): the fused layout's smaller
  MoE buffers were chosen when any layer was fused, while uncovered layers ran MMQ or FP16 over every row. On the
  shipped IQ2_XS pack (three IQ1_M layers) a 32K prompt's first token was KL 3.79 off with a different top token;
  fixed, 0.0036. NVFP4 packs never took that layout.
- **Residency-table uploads were unordered** (upstream, #550): plain pageable copies, read by non-blocking streams.
  They now wait for their own copy.
- **The elastic K/V could stop at 16,384 cells** (fork and #378): with no cache slots to lend (`--no-pool`,
  `--no-token-graph`, the dump modes, a cache of 0 slots) a longer prompt wrote K/V into unmapped memory - reproduced
  as an illegal address on a 32K prompt. Such a run now maps the whole window up front.
- **`--adapt-decay` of 1 or more** made the swap gains NaN; it is refused now (fork and #407).
- **The early cuBLAS handle** (#285's part 3, fork only) went to the first Gemm on any device; under a layer split a
  stage on another GPU got device 0's handle. It now stays on its device.
- **A picture a tool returned that the server cannot read** gave a 400 on every later turn of a Claude Code
  conversation (this fork's own change, #529); it becomes a note in its place.
- **The server's image path** (upstream #553, #554, #555): network (UNC) image paths are refused before Windows
  connects to them, URLs are capped at 32 MiB and fetched outside the request FIFO, a page of another origin cannot
  have a local file read; marker text inside a message no longer takes a picture's place; a refused request's image
  file is deleted.
- **`STRATA_ADAPT_WAIT=1`** (upstream #463, opt-in here): each decode window waits for the adaptive tier's copies,
  so greedy decode repeats exactly (3 of 3 runs identical, against 2 different outputs without); off by default
  because with this fork's tier it cost ~5% of decode.
- **`tools/requant.py`** checks the checkpoint's tensor shapes: a transposed one has the same byte count.
- **Checks** (against 0.1.37-nvfp4.1's references, fixed cache): logits and 32 greedy tokens identical after 2K on
  both packs; after 32K the GPTQ + Q8_0-down pack's first-token logits moved by KL 0.0003 (same top token, the same
  32 tokens) - the prompt path now borrows 1.25 GiB fewer cache slots. With the fork's defaults off the logits equal
  upstream 0.1.37 + #353 byte for byte. 54 of 57 tests (3 need absent model files), serve 180 OK.
- **Speed**, 5 interleaved chats: 18.97 +- 0.33 ms a round against 19.22 +- 0.87 for 0.1.37-nvfp4.1.

### On upstream 0.1.37 (2026-10-02, release 0.1.37-nvfp4.1)

The fork was moved to upstream 0.1.37 the same evening, with the port scripts (D:\Projects\Strata-data\port-fork.md on the build machine): the fork's commits rebased with `rerere`.

- **Upstream since 0.1.36:**
  - the server restarts an engine that went silent (#481);
  - AMD on Windows counts the desktop's VRAM;
  - a steadier PCIe probe, which changes nothing above 20 GB/s;
  - setup's `--vram-reserve-mib` (#493) and other setup fixes.
- **Conflicts:**
  - #353's NVFP4 PCIe base (0.25) beside the probe's new burst report. `rerere` replayed the resolution made in the PR for the fork's copy of the commit.
  - #279's Windows minimum (1500 MiB) in setup beside #493: the vision reserve keeps the minimum, and a `--vram-reserve-mib` given to setup still overrides it.
- **The fork's own change is the same on both bases:** 75 of 75 files, whitespace collapsed (`port_diffcheck.py`).
- **Checks** (each once; the previous release's logits and tokens are kept as references instead of running it again):
  - Against 0.1.36-nvfp4.1 at a fixed cache, the first token's logits and 32 greedy tokens are identical: the GPTQ + Q8_0-down pack after 2K and 32K, and the ModelOpt pack after 2K.
  - With the fork's own defaults off, the logits equal upstream 0.1.37 + #353 byte for byte (32K).
  - 54 of 57 tests pass; the other 3 need model files this machine does not have. The server's tests: 173 OK.
  - These checks ran on the release build, whose hash the bundle's engine is compared with.
- **A flaky upstream test fixed:** `decode_cluster_parity` reported a differing token in 0-2 of its 6 graph replays, from run to run. The graph ran on a non-blocking stream right after pageable uploads, which can return before their DMA lands, as 0.1.36 found in the fused test. With a sync after the uploads: 0 failures in 20 runs.
- **Speed** with the GPTQ + Q8_0-down pack, 5 interleaved chats each: 18.79 +- 0.73 ms a round, against 18.95 +- 0.81 for 0.1.36-nvfp4.1.

### On upstream 0.1.36 (2026-10-02, release 0.1.36-nvfp4.1)

The fork was rebuilt on upstream 0.1.36 the same way: the open pull requests and the fork's commits on top of v0.1.36.

- **Upstream since 0.1.35:**
  - fused int8 prompt kernels for Q2_0, on by default, and opt-in ones for the IQ packs (#136);
  - decode on thread-block clusters on RTX 50, bit-identical;
  - `UPDATE.bat`, `--expert-profile-save` (#477), clearer cancel and draft-head messages.
- **NVFP4 and the fused kernels:** the fused kernels cover Q2_0 and the IQ formats only, so an NVFP4 pack's layers keep MMQ, and its buffers stay MMQ-sized (`fused_ring()` is false for it). The combine applies `s_down` only on a layer that ran MMQ.
- **Conflicts:**
  - **`prefill.cpp`:** 0.1.36 moved the host grouping into the non-fused branch. The fork's lines are the same there; only their indentation changed (compared with `git diff -w`).
  - **#372 with the fused path:** a ring slot gathered in a group is released by the group's last event (`used_of`). The fused path records one event per slot, so it now resets `used_of` too. Without that, a fused layer after a grouped MMQ layer could let the issuer wait on an older event and refill a slot the fused kernels still read. This only arises with `STRATA_PF_FUSED=1` on a pack that mixes covered and uncovered layers.
  - **#477's routing heat beside `--adapt-decay`:** the heat is added before the decay, so it stays proportional to the routing at any decay.
  - **The no-MMQ stubs** (a build without MMQ, e.g. HIP by default) now match the NVFP4 signatures. Before, such a build would not link.
- **Images through Claude Code:** a picture opened with Read arrives inside a `tool_result`. The server dropped it there, and the model described an image it had not seen; it now reaches the encoder (sent upstream as #529).
- **Checks:**
  - Against 0.1.35-nvfp4.1 at a fixed cache, the first token's logits and 32 greedy tokens are identical after a 2K and a 32K prompt, with the ModelOpt pack and with the GPTQ + Q8_0-down pack.
  - With the fork's own defaults off, the logits equal upstream 0.1.36 + #353 byte for byte.
  - 54 of 57 tests pass; the other 3 need model files this machine does not have. The server's tests: 159 OK.
- **Speed** with the GPTQ + Q8_0-down pack, 7 interleaved chats each:
  - 18.96 +- 0.95 ms a round, against 18.78 +- 1.03 for 0.1.35-nvfp4.1.
  - With 0.1.36's cluster kernels off (`STRATA_QSA_CLUSTER=0 STRATA_ARGMAX_MULTI=0`, 4 runs): 18.80 +- 0.79.
  - The rounds differ run to run (the adaptive tier picks other experts), so only many runs compare.

### On upstream 0.1.35 (2026-10-02, release 0.1.35-nvfp4.1)

The fork was rebuilt on upstream 0.1.35 the same way: the open pull requests and the fork's commits on top of v0.1.35.

- **Upstream since 0.1.32:** #379 is in 0.1.33.
- **Conflicts:**
  - **#420's MMQ tile check beside NVFP4.** It finds llama.cpp's NVFP4 MMQ config, so NVFP4 keeps its prompt path (no #420 message on the 5090).
  - **#369's per-layer cache admission beside #362's read-ahead.** The read-ahead only runs without it: that walk visits the whole profile.
  - **0.1.35's card check, which also names the loaded HIP runtime (#468).** It moved ahead of the arena thread.
  - **#375 and #379 in `gr_parity` / `fused_gr`.** GR_V3 stays on by default here, and Turing keeps upstream's split rule.
- **The low-RAM mode** goes through `pin_cache_complement`, so it gets #467's working-set trim: 110 GiB available on the 128 GB box before the budget is pinned.
- **Checks:**
  - Against 0.1.32-nvfp4.2 at a fixed cache, the first token's logits and 32 greedy tokens are identical after a 2K and a 32K prompt, with the ModelOpt pack and with the GPTQ + Q8_0-down pack.
  - With the fork's own defaults off, the logits equal upstream 0.1.35 + #353 byte for byte.
  - 50 of 53 tests pass; the other 3 need model files this machine does not have.
- **Speed** with the GPTQ + Q8_0-down pack, 4 interleaved chats each: 17.84 +- 0.27 ms a round, against 18.35 +- 1.10 for 0.1.32-nvfp4.2.

### Re-quantized from BF16 (2026-10-02, release 0.1.32-nvfp4.2)

The shipped experts are ModelOpt's NVFP4: each 16-value block is scaled to its largest value and every weight rounded
to the nearest FP4 value. Here they were re-quantized from the BF16 checkpoint (`orcarouter/Qwen3.8-Flash-Next-Uncensored`)
and measured end to end.

- **Checks first.**
  - ModelOpt's NVFP4 is that checkpoint's: relative weight error 9.4%, gate rows first.
  - `tools/nvfp4_codec.py` lays ModelOpt's codes out byte for byte as `experts.bin` holds them.
  - Its round-to-nearest reproduces 100% of the block scales and 99.8-100% of the codes.
  - ModelOpt's global scale for gate/up is shared and 2-7x amax/(6x448), which does not change the error.
- **Calibration.**
  - 57.5K tokens: Ukrainian and English Wikipedia, llama.cpp and Rust sources, and 16 of the model's own chat answers.
  - The engine dumped every layer's MoE input with the routing (`STRATA_DUMP_MOE_LAYER=all`).
  - An expert's Hessian is sum w^2 x x^T over the tokens routed to it, mixed with the layer's average as 256 pseudo-tokens.
  - Down's Hessian comes from the hidden the quantized gate/up produce.
- **Per layer, held-out tokens**, the routed-weighted error of the experts' whole output, summed over 48 layers:
  - GPTQ (16-column scales searched on the updated weights, "four over six" candidates): 33.5% of ModelOpt's;
  - an activation-weighted per-block scale search alone: about half;
  - after GPTQ most of the error is in down. Q8_0 down removes up to ~85% in a layer, Q8_0 gate/up only ~12%.
- **Packs:**
  - GPTQ only, 63.3 GiB;
  - Q8_0 down in every layer, 82.0 GiB;
  - budgets from `tools/requant_plan.py`: 17 layers (69.9 GiB) and 27 layers (73.8 GiB: layers 4, 6, 15, 16, 19-23, 30-47). They were assembled from the first two: the GPTQ gate/up rows are byte-identical.
- **Measured against an all-Q8_0 reference pack** built from the same BF16 (119.5 GiB, run with `--low-ram`):
  - 12 prompts (Ukrainian, English, code; half with thinking), none in the calibration;
  - the reference's answers teacher-forced (`STRATA_LOGPOS`, top 64, serve mode so the windows read them);
  - the KL is taken over the answers' positions, per prompt.
  - **The metric's own noise:** the same pack with another cache size is 0.0017 (median; GPU and CPU round differently, and the difference accumulates in the K/V and the GDN state). Over the whole 12K-token conversation it reaches 0.01, which is why the comparison is per answer.

| pack | experts | median answer KL | of ModelOpt's | chat decode | after a 32K prompt |
| --- | ---: | ---: | ---: | ---: | ---: |
| ModelOpt NVFP4 (0.1.32-nvfp4.1) | 63.3 GiB | 0.0107 | | 17.3 ms, 142 tok/s | 19.3 ms, 136 tok/s |
| GPTQ NVFP4 | 63.3 GiB | 0.0070 | 65% (better in 12/12 prompts) | 17.3 ms, 143 tok/s | |
| + Q8_0 down in 17 layers | 69.9 GiB | 0.0059 | 55% | 18.3 ms, 134 tok/s (-5.5%) | 20.9 ms, 123 tok/s (-9.6%) |
| **+ Q8_0 down in 27 layers** | 73.8 GiB | **0.0051** | **48%** | 19.3 ms, 126 tok/s (-11.6%) | 22.3 ms, 116 tok/s (-14.7%) |
| + Q8_0 down in all 48 | 82.0 GiB | 0.0045 | 42% | 22.3 ms, 108 tok/s (-24%) | |

- **Net of the noise** (0.0017): 0.0090 -> 0.0052 (GPTQ) -> 0.0042 -> 0.0034 (27 layers) -> 0.0028. Each step is better in 10 or 12 of the 12 prompts.
- **The first token after a 32K prompt** (full vocabulary): 0.0277 -> 0.0137 (GPTQ) -> 0.0090 -> 0.0069 -> 0.0010.
- **Speed.** Interleaved runs, cache auto, to each answer's end. The prompt path reads 32K at the same 4.37-4.41 s with every pack.
  - The 8-bit layers cost decode in two ways: more bytes per miss (the CPU pool runs Q8_0 down through ggml's generic dot), and fewer experts in VRAM.
  - With the 27-layer pack, 39-40 GB of the 128 GB stay free.
- **The tray runs the 27-layer pack** (`packs\orca-nvfp4-gptq-q8d`): the most accurate one inside a 10-15% decode budget. The ModelOpt pack stays beside it.

### On upstream 0.1.32 (2026-10-01, release 0.1.32-nvfp4.1)

Upstream released 0.1.32 with this fork's first wave of pull requests in it. The maintainer applied them by hand, under our name, and made some of them opt-in. The fork was rebuilt on 0.1.32 the same way as on 0.1.31:

- **Taken upstream and dropped from the fork's commits:** #276-#278, #280-#284, #287-#291, #293, and the fork's fixes from the review of those, which upstream made in its own form.
  - **#284:** the commit wait is an event at every point that touches the session.
  - **#280:** the table is forgotten when a session is freed.
  - **#289:** a placeholder name when the device query fails, and HIP builds.
- **Opt-in upstream, on in the fork,** as every 0.1.31-nvfp4.x ran:

| | upstream 0.1.32 | this fork | the switch |
| --- | --- | --- | --- |
| INT8 K/V through the Hadamard rotation (#293) | off | on | `STRATA_KV_ROT=0` / `=1` |
| the float64 RoPE angle table (#280) | off | on | `STRATA_ROPE_TABLE=0` / `=1` |
| the prompt path's BF16 remainder (#283) | 0 | 2 | `STRATA_PREFILL_BF16X2` |
| `--prefill auto`'s largest chunk (#282) | 8192 | 32768 | `--prefill auto:N`, `STRATA_PREFILL_AUTO_MAX` |
| an Anthropic request that does not ask for thinking (#278) | thinks | does not | the config's `"anthropic_thinking"`: `"model"` / `"on_request"` (the bundle's config sets `on_request`) |

- **Carried as before:** the pull requests still open, each rebased on 0.1.32. They are NVFP4 (#353), the start (#357, #358), the file tier (#362), the VRAM reserve (#279), the group gather (#372), the PLE reads (#374), the elastic K/V (#378), the stager's DMA wait (#385) and the adaptive tier (#407). The fork-only changes stay too: the low-RAM mode, the arena thread, the split hyper-connection kernels and the adaptive tier's settings.
- **Conflicts:**
  - **NVFP4 with 0.1.32's UD-Q4_K_XL support.** The K-quant MMQ instances and cases sit beside NVFP4's. `iq_pack`'s `layer_blobs` (#277's version) appends the NVFP4 scale tail, and so does its reuse check. Without the tail an NVFP4 pack never matched its GGUF. Checked on the real pack: an expert blob without the tail differs from experts.bin, and with it equals it.
  - **The elastic K/V with #340's split buffers and #284's wait.** Both are kept.
  - **The arena thread with 0.1.32's card check.** The check moves ahead of the thread in its new form (HIP, a placeholder name).
- **Checks:**
  - **Same output as upstream.** The fork with its own defaults switched off (`STRATA_KV_ROT=0 STRATA_ROPE_TABLE=0 STRATA_PREFILL_BF16X2=0 STRATA_GR_V3=0 --no-kv-grow --prefill auto:8192`) gives the first token's logits of upstream 0.1.32 + #353 byte for byte, after a 2K and a 32K prompt (fixed cache).
  - **Every pull request on IQ2_XS.** Each one, rebased, gives main's logits byte for byte at 2K and 32K.
  - **Tests.** 49 of 52 pass. The 3 others need model files this machine does not have (the Q2_0 PLE, `pack/full`).
- **The split hyper-connection read.**
  - 0.1.32 has #315's split and staged variants of the default read. They are bit for bit the plain read and faster than it.
  - Against this fork's `STRATA_GR_V3` kernels, 6 interleaved chat runs each:
    - GR_V3: 18.23 +- 0.47 ms a round, the GPU's part 8.4 ms;
    - staged: 18.49 +- 0.29 ms, 9.3 ms.
  - GR_V3 stays on.
- **Against 0.1.31-nvfp4.3**, RTX 5090, 262K, cache auto, interleaved:

| | 0.1.31-nvfp4.3 | 0.1.32-nvfp4.1 |
| --- | ---: | ---: |
| a chat answer (~520 tokens, to its end), 6 runs each: ms a round | 18.11 +- 0.65 | 18.23 +- 0.47 |
| misses a layer / hit rate | 2.44 / 0.920 | 2.43 / 0.920 |
| a ~1,000-token answer after a 32K prompt, 4 runs each: ms a round / tokens/s | 20.00 +- 0.12 / 131.9 | 19.99 +- 0.23 / 131.8 |
| reading the 32K prompt | 4,640-4,690 ms | 4,630-4,750 ms |
| first token's logits and 128 greedy tokens, fixed cache, 2K and 32K | | identical |

- **The earlier "after a 32K prompt" numbers measured the wrong thing.** That prompt asks for three file names, and its answer ends after 22 tokens.
  - The runs went on to a fixed 256 tokens with no end-of-turn stop, so 0.1.31-nvfp4.3's 161-173 tokens/s were mostly the loop after the answer.
  - The same holds for half of the 1,000-token runs behind the adaptive tier's numbers below.
  - `bench-tools/eot_share.py` reports how much of a run came after the first end-of-turn. Runs now pass `--stop-eos`, and the 32K test asks for a long explanation (`long_32k_explain.txt`).

### The adaptive VRAM tier and an audit (2026-10-01, release 0.1.31-nvfp4.3)

> **Caveat (found with 0.1.32):** the engine runs below decoded a fixed number of tokens with no end-of-turn stop,
> and about half of each 1,000-token run came after the answer had ended. There the model loops over a few experts,
> which lifts the hit rate and the tokens per second in both arms. The misses' reduction on an answer alone is not
> re-measured.

**The tier's settings.** The adaptive tier swaps the most-routed missing experts into VRAM in place of the
least-routed resident ones, from routing counts that fade after each pass.

- **Replay first.** Two recorded decode traces were replayed (`--dump-routing`); the replay matched the engine's misses within 3%.
  - A window of the last 8-16 tokens instead of fading counts was worse, with 24-96% more misses. It forgets experts that a conversation uses rarely but steadily, and swaps them back and forth.
  - Longer memory with more frequent steps was better: every 2 rounds, 192 swaps, x0.92 gave 31-38% fewer misses.
  - An LRU tail cost 8-9x the copies.
- **In the engine**, the settings alternating run by run:

| cache (share of the experts) | upstream 4 / 96 / x0.7 | 2 / 192 / x0.92 |
| --- | --- | --- |
| NVFP4, 5090 (34%), a 1,000-token chat, 6 runs each | 4.26 misses a layer, 19.9 ms a round, 144 tok/s | **2.68, 17.8 ms, 157 tok/s** |
| NVFP4, 5090 (34%), an 8K document + 600, 4 each | 3.46, 19.6 ms | **1.91, 17.4 ms** |
| IQ2_XS, 5090 (70%), 6 each | 1.01, 15.38 ms | 0.84, 15.53 ms |
| IQ2_XS, an emulated 16 GB card (14%), 4 each | 10.61, 25.1 ms | 10.36, 25.7 ms (swaps doubled) |

- **Fit.** A fit over 26 runs gives ms per round = 12.4 + 1.49 x misses per layer + 0.043 x swaps per round.
  - The runs differ in their token paths, so that is a correlation, not a cost.
- **What moves and what does not.** Measured per round, the faster settings cut:
  - the CPU expert pool's time: NVFP4 8.7 -> 5.6 ms (chat), 7.2 -> 4.2 (document);
  - the PCIe traffic, as the share of misses the GPU fetches plus the swaps: 156 -> 138 MB and 123 -> 89 MB.
- **Why the round barely moves.** In most layers the GPU's work, not the CPU's, sets a layer's time, so only the smaller PCIe share shortens the round: ~1 ms. The rounds' spread between runs is larger than that, so the 19.9 -> 17.8 ms above is about two standard deviations.
- **Unsloth UD-Q4_K_XL** (upstream #407; cache 30%, every expert in RAM), 6 pairs:
  - misses 4.06 -> 2.79;
  - CPU pool 9.9 -> 8.0 ms;
  - PCIe 329 -> 256 MB;
  - the round 32.0 +- 3.0 -> 32.3 +- 2.2 ms;
  - with 16 busy CPU workers beside it, the round was 53.9 -> 52.2 +- 5.9 ms.
- **Task Manager shows half the CPU busy.** The pool runs one worker per physical core on both CCDs (the even logical CPUs), and the SMT siblings stay free: the expert kernels are bound by DRAM bandwidth (~55 GB/s here), and 23 or 31 workers were no faster. During decoding the workers wait for the next layer by spinning (`_mm_pause`; a layer comes every ~0.35 ms), so their cores show ~100% whatever the work, and a lighter pool does not show there.
- **Why the faster settings pay off only in the middle:** with a cache that holds most of the experts there is little left to win, and with a small one the swaps cost more than they bring.
- **Where they apply:** only with a cache holding 20-60% of the experts and every expert in RAM. With a RAM tier a swap can read the drive: a 64 GB run had the tier's host time double with no shorter round. Elsewhere the tier keeps upstream's settings.

**An audit** (a codex read-only pass; every finding checked against the code before anything changed):
- **The prompt path's Stager:** a generation's first jobs did not wait for the previous generation's DMA from their buffer.
  - The window is real but narrow: no MTP, or a ring entry the routing skipped, with unpinned blobs. The audit's per-layer case is covered by the routing sync.
  - Fixed, and sent upstream as #385.
- **VMM:** a failed access setup left a chunk counted as mapped. It is now rolled back.
- **The elastic K/V:** a failed growth or trim now stops the engine instead of serving on with a half-changed tier. The trim maps one run and refills the slots with one sync.
- **Rejected:**
  - a "redundant" sync of the verifier's copy stream (it keeps a window's host function from raising flag B in the next window);
  - the verifier's unchecked DMA calls (a mode NVFP4 does not use; the error surfaces at the next sync).

### Faster without changing an answer (2026-10-01, release 0.1.31-nvfp4.2)

A list of speedups that keep every result was worked through, one commit each. What went in:

| change | measured | results |
| --- | --- | --- |
| **The K/V grows with the context** (CUDA VMM): the pools and the expert-cache arena are virtual ranges; the K/V maps 2 MiB chunks as requests reach further, taken from the slots below the prompt path's loan (a hotter expert there moves into the coldest loan slot first), and hands them back after a short request (the slots refill from the profile) | cache auto at 262K: 7,060 -> 8,290 slots; short chat 20.5-21.1 -> 18.9-19.7 ms a round; served 32K document 99 -> 123-133 tok/s | bit-identical when no expert moves (grown from 4,096 cells, and mapped whole); first-token KL 0.0006 when 139 experts go to the CPU |
| **An MMQ group gathered in one launch**, after one wait on its last copy, released by one event | 32K prompt -5.2%, 16K -7.9%, 8K -11.1% | bit-identical, the 96-slot ring too |
| **n-gram rows read 256 at a time** (was 64) | PLE 303 -> 189-217 ms of a 32K prompt | bit-identical |
| **the first chunk's n-gram rows read beside layer 0** | 32K prompt -7 to -9% | bit-identical |
| **upstream's split hyper-connection kernels** (`STRATA_GR_V3`) on | rounds 3-5% shorter | within 2e-6 relative (gr_parity) |

Measured and left out:
- **The GDN recurrence with its state in registers** (GLM's KDA scan): two warp layouts, bit-identical, no faster (687-723 vs 690-697 ms per 32K prompt). Qwen's column kernel already keeps the state on chip.
- **The PCIe share's copy kernel on its own stream:** round time unchanged, as was `--pcie-frac` 0.25-0.45. The round follows how many experts miss, not where a miss is computed.
- **Next-layer expert prediction:** layer l's MoE input through layer l+1's router finds 53-70% of the true top-10 (GLM had 95% of its top-1). The hyper-connections change the input between layers.
- **The VRAM cache's policy:**
  - Replaying two decode routing traces (the replay matched the engine's misses within 3%) showed two findings.
    - An LRU tail costs 8-9x the copies.
    - Adapting every 2 rounds with 192 swaps misses 26-34% less. In the engine, the hit rate rose (0.871 -> 0.905), but the round time did not move measurably.
  - A fit over 26 runs explains why: ms/round = 12.4 + 1.49 x misses per layer + 0.043 x swaps per round. A swap costs about its own PCIe time, because it shares the bus with the PCIe share.
  - Left at upstream's settings until a replay benchmark can tell a few percent apart.

### On upstream 0.1.31 (2026-10-01, release 0.1.31-nvfp4.1)

The fork was rebuilt on upstream 0.1.31 rather than merged, from the pull requests it had sent upstream:
- NVFP4 rebased as #353;
- the first wave #276-#293 (most taken into upstream's 0.1.32 batch);
- the start split into #357 (unbuffered reads) and #358 (per-layer registration);
- the low-RAM tier ported into 0.1.31's own tier as #362.

What changed on the way:

- **Architectures.** The build is `75-real;86-real;89-real;120-real`. Only the FP4 x FP4 unit (`mmq_nvfp4_w4a4.cu`) is built for `120a`: CMake turns `12x` into `12xa` for it alone. At run time a probe kernel decides whether `w4a4` is available, and any card without that image takes `w4a8`.
- **The low-RAM mode is 0.1.31's tier.** It is `--mmap-experts` with `--resident-budget-gib`: the hottest experts outside VRAM in a page-locked budget, the rest read from the files. The fork's own `TieredExpertSource` and its `STRATA_RAM_RESERVE_GIB` / `STRATA_EMULATE_RAM_GIB` / `STRATA_TIER_*` switches are gone. What the fork adds:
  - it starts by itself below 96 GB installed, with a budget of the available RAM less 6 GiB; `--low-ram`, `--no-low-ram` and `--ram-budget` keep their meaning;
  - `experts.bin` is read unbuffered too, as #362 reads the GGUF in place. NVFP4 needs experts.bin for the scale tails.
- **Two findings decided how those reads are done.** Both are written up in #362:
  - **NTFS serializes small unbuffered reads of a mapped file.** It runs a file's unbuffered reads one at a time while the file is mapped anywhere. 512 KiB reads at queue depth 48 make 10.4 GB/s on an unmapped file and 3.4 GB/s on a mapped one. So nearby windows are merged into requests of up to 32 MiB.
  - **The cache probe cannot decide for the tier.** With 14 of 16 probe reads cached, a mapped profile fill pulled experts.bin into the working set, the budget shrank to 31 GiB and RAM ran out. The tier goes unbuffered whenever the files cannot be kept beside the budget.
- **RoPE:** the float64 angle table (#280) is used when the session is unscaled; under 0.1.30's linear/YaRN scaling the kernels take upstream's `rope_scaled_angle`.
- **Kept in the fork, not upstream:**
  - INT8 K/V rotated by default (upstream keeps it opt-in);
  - the Cyrillic draft vocabulary as `data/draft_vocab.bin`;
  - prompt chunks up to 32K by default;
  - the arena on its own thread with an early cuBLAS handle (#285's part 3).

Against the previous release (0.1.28-nvfp4.4), RTX 5090, 128 GB, the same expert-cache size:

| | 0.1.28-nvfp4.4 | 0.1.31-nvfp4.1 |
| --- | ---: | ---: |
| greedy tokens, 64-token smoke and 256 after a 32K prompt | | identical |
| first-token KL | | 0.0002 (run-to-run noise 0.00013) |
| decode, short chat (256 tokens) | 107-116 tok/s | 106-119 tok/s |
| decode, 256 after a 32K prompt (`--pcie-frac 0.25`) | 104-112 | 106-115 |
| a 32K prompt | 5,360-5,720 tok/s | 5,440-5,970 tok/s |
| expert arena loaded | 6.7 s | 6.6-7.4 s |

The decode rate follows the draft acceptance of the path the tokens take. With the adaptive tier's timing the tokens vary from run to run on either build, which is why the rates are ranges.

The low-RAM mode on an emulated 64 GB PC (a large-page ballast leaves 58 GiB available), RTX 5090:
- **Budget:** all 44.4 GiB of experts outside VRAM fit it.
- **Start:** the profile fill takes 4.3 s and the complement 9.5 s (through the mapping: 17 s and 31 s).
- **A 32K prompt:** read in 9.3 s, 3.8 s of it the lent slots' refill from the file. Then 97 tok/s (0.1.28-nvfp4.4: 9.3 s and 72 tok/s).
- **Tokens:** the same greedy tokens as with all experts resident.

The smaller cards were not measured again on this release.

One measuring pitfall: right after another engine process with a 63 GiB pinned arena exits, the next start's PCIe probe can read ~17 GB/s instead of 57. The probe then cuts `--pcie-frac` to 0.17, and decode drops 3-5%. Speed A/Bs here pin `--pcie-frac 0.25` and leave 15 s between runs.
