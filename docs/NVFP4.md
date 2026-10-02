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
