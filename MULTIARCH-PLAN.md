# The RTX 50 set on every NVIDIA generation: inventory, matrix, plan, checks

Branch `dec/multiarch` (worktree `D:\Projects\strata-multiarch`) on main 61c2451dd = release 0.2.15-nvfp4.4. Patch:
`bench-ma\multiarch.patch` (not committed). Scripts, raw results and logs: `bench-ma\` (results.txt; engine logs in
`D:\Projects\Strata-data\dflash-check\raw\ma_*.log`).

Everything below was run on the one RTX 5090 (Windows 11, WDDM, CUDA 13.3, driver 621.14). Older cards were emulated:
an engine built as PTX for the old architecture (`75-virtual`, `86-virtual`, `89-virtual`; the release's own
configuration otherwise: `STRATA_PORTABLE=ON`, MMQ on) run with `STRATA_EMULATE_CC=75|86|89`, so every kernel is JIT-built
with that `__CUDA_ARCH__` and the host's capability / shared-memory queries answer as that card. cuBLAS is not emulated
(it runs its sm_120 kernels). Speeds measured this way are not the real card's and are not reported as such.
"Measured" below = run here; "inferred" = from the code, ptxas resource usage or the PTX ISA / CUDA documentation.

## 0. Summary

- **Release 0.2.15-nvfp4.4 on RTX 20/30/40 paths: works.** Emulated sm_86 and sm_89: no CUDA error, sane text, the 16K
  read gives the *same text* as the 5090 with the RTX 50 set off (md5 62205238cc), first-token KL 0.0016 against it.
  Emulated sm_75 crashed (`verify: unspecified launch failure`) - root cause: the two-stage head's probe, a test-mode
  artifact that a real Turing card would not hit (inferred from the code; section 2). Fixed anyway; Turing then gives the
  same 16K text as the other archs.
- **A real release bug was found by the inventory, for other RTX 50 cards**: `STRATA_EARLY_READ` (sm_120 default) parks
  220 spinning blocks that fill every SM of a card with <= 110 SMs (RTX 5080 / 5070 Ti / 5070 / 5060, RTX PRO 5000), so
  the combine they wait for cannot start. Reproduced on the 5090 with 84 SMs left free: decode **1.2 tok/s instead of
  118.9**; on a real 84-SM card the 2-s gate timeout fires and every later read uses stale inputs (wrong text). Fixed by
  a co-residency guard (section 2).
- **Patch** (`bench-ma\multiarch.patch`, 10 files, +215/-16): the guard; a portable defaults set for cc 7.5-11.x (11 / 14 / 15 / 21
  switches by arch); the two-stage head on Turing; a faithful emulation probe; a test-only SM limiter. sm_120 stays
  bit-identical (16K `93c4688100 / e9d907e164`, MTP decode `3d13f77b08` = the release).

## 1. Release check (measured, 5090, emulated archs, unpatched main 61c2451dd)

Fixed placement (`exact.sh`), 16K prompt read with 48 greedy tokens (`STRATA_SPEC_FIXED=1`) + first-token logits md5,
and 200-token decodes of `conv.txt` with Auto's new MTP (`--mtp rt-tuned --mtp-head proxy --mtp-cum auto --spec 8`) and
with DFlash2. Each run ~1-3 min (the first run of a PTX build JITs ~4 min, cached after).

| engine / env | 16K text | 16K logits | KL vs 120-off-fp4 | MTP decode | DFlash2 decode | errors |
| --- | --- | --- | --- | --- | --- | --- |
| release, sm_120 defaults | 93c4688100 | e9d907e164 (= ref) | - | 3d13f77b08 | - | none |
| release, `NVFP4_SM120=0` | 93c4688100 | ef51ae71a4 | - | 82ebb3effa | 82ebb3effa | none |
| release, `NVFP4_SM120=0 PF_FP4G=0` (no RTX 50 kernel left in the prompt) | **62205238cc** | 7a10591be6 | 0 | - | - | none |
| 86-virtual, `EMULATE_CC=86` | **62205238cc** | ddbb216eeb | 0.0016 | 9e05d7249c | 9e05d7249c | none |
| 89-virtual, `EMULATE_CC=89` | **62205238cc** | ddbb216eeb (= 86) | 0.0016 | 9e05d7249c | 9e05d7249c | none |
| 75-virtual, `EMULATE_CC=75` | - | - | - | - | - | `verify: unspecified launch failure` (all 3) |
| 75-virtual, `EMULATE_CC=75 HEAD_PLANES=0` | **62205238cc** | d0bcbcb21d | 0.0015 | 0a6a56bc1e | 0a6a56bc1e | none |

- The text of every arch is fluent and on topic (`bench-ma\detok.py`). The conv decodes differ from the 5090's from token 0
  (a near tie: "Database indexing..." vs "The fundamental difference..."; the 5090 run still had FP4G, HC_UPMIX and the
  sm_120 series on); 75 and 86 agree for 151 tokens. MTP and DFlash2 give the same text on each arch, as they must.
- 86 and 89 are bit-identical to each other (upstream's sm_86 table is on for 86 only, so it is exact here too).
- Paths that ran, from the logs: no FP4G (MMQ W4A8), no hc8 / dense_tc / gdn_mid, the staged HC read with cp.async on
  86/89, the two-stage head on 86/89 (bit planes), Auto's proxy draft head on every arch.

## 2. Bugs

### 2a. Turing (sm_75): "verify: unspecified launch failure" - root cause and fix (measured)

- `HEAD_PLANES` (default on everywhere) repacks the Q8_0 head as bit planes when `q80p_supported()` says yes.
  `arch_ok()` (q80_vhead.cu) checked the raw major (>= 8) and `binaryVersion >= 80`. The q80p kernels are
  `__CUDA_ARCH__ >= 800`, else `__trap()`.
- Under emulation the 75-virtual PTX is JIT-compiled for the 5090, so `binaryVersion` is 120 and the raw major 12. The
  head went to planes, the first verify window launched the Turing trap stub, and the context died.
- On a real Turing card with the release, the cubin is `75-real` (binaryVersion 75), so the head stays plain - inferred,
  not a release bug (no Turing card here).
- With `STRATA_HEAD_PLANES=0` the same build runs clean (table above).
- Fix 1: `arch_ok` also checks `ptxVersion`, as the PDL and cluster probes already do.
- Fix 2 (a port): the only sm_80 instruction in q80p is `__reduce_add_sync` (`redux.sync`). On Turing it becomes an
  integer shuffle sum (exact), and the guards drop to 750.
- Confirmed by runs:
  - the patched 75 build runs the two-stage head: 79 screens, no error;
  - with the portable set off but the exact switches on, its 16K text and logits and its decode equal the
    `HEAD_PLANES=0` baseline bit for bit (62205238cc / d0bcbcb21d / 0a6a56bc1e);
  - with defaults on: 16K text 93c4688100, decode e2a9ea0814, KL 0.0006 against the baseline.

### 2b. EARLY_READ on RTX 50 cards below ~147 SMs - root cause and fix (measured on the 5090 with an SM limiter)

- `hc8_down_kernel<T, 2>` is the gated read. It has 128 registers, 256 threads and `__launch_bounds__(256, 2)`, so 2
  blocks fill an SM's 64 K registers (ptxas `-res-usage` on the release exe).
- Its 220 blocks are launched on a branch before the CPU-row wait, and each spins until the combine has counted in.
  If 2 x SMs <= 220, every SM is full of spinners and the combine cannot be placed.
- Then the gate's 2-s timeout fires, prints "results invalid", and every later gate falls through, reading stale
  activations. The 5090 has 2 x 170 = 340 places for 220 blocks, so it never showed.
- To check this on the 5090: `STRATA_EMULATE_SMS=84` (new, test-only) holds 86 of 170 SMs with an idle kernel. It
  uses 99 KB of shared memory and 1024 threads, relaunched every 50 ms.
  - EARLY_READ forced on (`STRATA_EARLY_READ_FIT=0`): decode **1.20 tok/s**. Every gated read waited for a 50 ms gap of
    the limiter; a real card has no such gaps.
  - With the fix: 118.85 tok/s.
  - Both runs give the same text (3d13f77b08 = the release's).
- Fix (`fused_gr.cu`, `gated_read_fits()`): take the gated read only where occupancy x SMs >= 220 + SMs / 2.
  - Kept on: the 5090 (170 SMs) and the RTX PRO 6000 (188).
  - Turned off (one start line says so): the 5080 (84), 5070 Ti (70), 5070 (48), 5060 Ti / 5060 (36 / 30) and the RTX PRO 5000 (110).
- **Recommend shipping this as 0.2.15-nvfp4.5.** RTX 5080-and-below users get Auto's MTP decode with EARLY_READ on today.

## 3. Which NVIDIA architectures Strata supports

From `cmake/backends/cuda.cmake`, `README.md` and `docs/`:

| cc | parts | status in this fork |
| --- | --- | --- |
| 6.0 / 7.0 (and 5.2) | P100, V100 (Maxwell M6000) | `STRATA_EXPERIMENTAL_SM60=ON` community build, CUDA 12.x only (CUDA 13 dropped them); not in the release; out of scope here |
| 7.5 | RTX 20, T4, CMP 50HX | release (`75-real`) |
| 8.0 | A100, A30, CMP 170HX | builds (no refusal); upstream has measured sm_80 defaults from one CMP 170HX (`docs/DETAILS.md`, `MMVQ_IL_TABLE.md`); **not in the release** (no 80 cubin, no PTX: the release names it as a card it has no code for) |
| 8.6 | RTX 30, RTX A2000-A6000 | release (`86-real`); upstream sm_86 table (measured on a 3060 / A4000) |
| 8.7 | Jetson Orin | builds; never mentioned; out of scope |
| 8.9 | RTX 40, L4, L40, RTX 6000 Ada | release (`89-real`); no table so far |
| 9.0 | H100 / H200 | builds; nowhere tested or claimed |
| 10.0 / 10.3 | B200 / B300 | builds; nowhere tested or claimed |
| 11.0 | Thor (was 10.1) | builds; nowhere tested or claimed |
| 12.0 | RTX 50, RTX PRO Blackwell | release (`120-real` + the FP4 unit as `120a-real`) |
| 12.1 | GB10 (DGX Spark) | upstream, Linux (`docs/DGX_SPARK_LINUX.md`, the sm_121 table); not in this fork's Windows release |

So the targets that matter are 7.5, 8.6, 8.9 and 12.0 (the release), then 8.0 and 12.1 (upstream-supported), then 9.0 / 10.x
by the same features (not claimed by anyone).

## 4. Inventory: what turns on only on RTX 50

### 4a. The sm_120 table (`src/core/nv_defaults.cpp`, 29 entries)

Min sm "doc" = what the PTX ISA requires of the instructions; "code" = what the source guards / host gates require today.
"bits" = exact (same text and logits) or rounds.

| switch(es) | what it does | instructions / resources it needs | min sm doc | min sm code (today) | bits |
| --- | --- | --- | --- | --- | --- |
| `HC_Q8=1` | the verify window's hyper-connection read on int8 rows + fp16 scales (`hc8_read.cuh`) | plain int8 -> fp32 FMAs (no MMA), `ld.acquire.gpu`, `__nanosleep` (7.0); `cp.async.bulk.prefetch.L2` hint (9.0, guarded); PDL when cc >= 9; 15.6-17.7 KB static smem, 128 regs (2 blocks/SM) | 7.0 | none (any CUDA card; only the defaults key) | rounds (KL 0.0152) |
| `PREFILL_EQUAL=2` | equal prompt chunks | host | any | none | rounds (chunk order) |
| `GDN_TC=1`, `GDN_CONVL2=2`, `GDN_TC_NP=1` | the prompt's DeltaNet scan on tensor cores (`gdn_tc.cuh`) | `mma.m16n8k16.f32.f16`, `ldmatrix(.trans)`, `cp.async`; 122 KB smem (8.0 / 9.0 / 10.x plan) or the split 99 KB plan (8.6 / 8.9 / 12.x) | 8.0 | cc >= 8 and the smem plan (kernels.cu `gdn_rec_tc`); already the Linux CUDA default on every card | rounds (first-token KL ~0.001) |
| `PF_GROUP_SCRATCH=1`, `PF_HC_REGION=2` | smaller prompt buffers (fewer borrowed expert slots) | host | any | none | exact |
| `ADAPT_FETCH=1`, `ADAPT_ACROSS=1`, `ADAPT_CAND_MIN=1.5`, `ADAPT_DECAY=0.96`, `PCIE_FIXED=0.45` | the serve tier's expert placement policy | host; values tuned for this PC's PCIe 5.0 x16, 9950X3D and 32 GB | any | none | moves experts GPU <-> CPU (KL 0.0156) |
| `DENSE_TC=1` | Basalt's int8 tensor-core Q8_0 GEMV for 1-8 token windows (`dense_tc.cu`) | `mma.m16n8k32.s8` / `m16n8k16.s8` (8.0), `cp.async` (8.0, ring layout), `cp.async.bulk` + `mbarrier.expect_tx/try_wait` (9.0, the one-tile / tile / pair layouts), `griddepcontrol` (9.0), 99 KB dynamic smem | 8.0 (ring) / 9.0 (bulk) | **9.0**: every kernel is `__CUDA_ARCH__ >= 900` else `__trap()`; host `dense_tc_on()` needs major >= 9 | rounds (same at every window width) |
| `ATTN_STAGED=1` | INT8 attention chunks staged in smem, two blocks/SM (`qsa_decode_attn.cu`) | `cp.async` 16/8 B (8.0; plain copies below, guarded since 61c2451dd), 40 KB static smem | 7.5 | none | exact |
| `GDN_MID=1`, `GDN_MID_HEAD=2` | a DeltaNet layer's middle in one persistent launch (`gdn_mid.cu`) | DENSE_TC's tiles, `cp.async.bulk` + `mbarrier` (9.0), `red.release` / `ld.acquire` grid-wide arrivals, 101,376 B smem (exactly the 99 KB opt-in), **one CTA per output tile: >= 160 SMs co-resident** | 9.0 + 160 SMs | `mid_grid()`: major >= 9, optin >= 101,376, SMs >= 160 | exact |
| `QPROJ_HALF=1`, `QPROJ_GATE_AT=3`, `QPROJ_GATE_GRID=48`, `GRAPH_PRIO=1` | attention layer: q rows first, gate rows later on 48 blocks; stream priorities kept in graphs | QPROJ_* need `dense_tc_rows` (DENSE_TC, 9.0); GATE_GRID=48 is 5090-sized (170 SMs); GRAPH_PRIO is host | 9.0 (QPROJ), any (PRIO) | QPROJ_* inert below 9.0 | exact |
| `DF_PDL=1` | PDL edges in the doorbell graphs | `griddepcontrol.wait / launch_dependents` | 9.0 | cc >= 9 + ptxVersion/binaryVersion >= 90 probe | exact |
| `EARLY_READ=1` | next layer's first HC read launched before the combine, its blocks spin on the combine's counter | needs HC_Q8's hc8; `ld.acquire`, `__nanosleep`, `%globaltimer` (7.0); **co-residency: 220 spinning blocks x 128 regs x 256 threads = 2 blocks fill an SM** | 7.0 + room for the combine | none (no SM check - see section 2) | exact |
| `ROUTE_DB=2` | router top-10 + doorbell in one launch | plain CUDA | any | none | exact |
| `QSA_I8=1` | prompt QSA block scores on int8 tensor cores (6 int8 products a score) | `mma.m16n8k32.s8/u8` (8.0) | 8.0 (Turing: m8n8k16) | cc_major_of >= 8 | rounds (same blocks selected) |
| `DENSE_F16C=1` | prompt dense FP16 GEMMs with f16-accumulate chains of 64 | `mma.m16n8k16.f16.f16` (8.0), `ldmatrix`, `cp.async`, 2 x 48 KB smem stages; gain only where f16-acc runs 2x f32-acc (GeForce / RTX pro: 7.5, 8.6, 8.9, 12.x; not A100 / H100 / B200) | 8.0 (Turing: m16n8k8) | major >= 8 (raw) | rounds (rel. err ~3e-4) |
| `KV_PREFETCH=1` | KV streaming's prefix staging + download on their own stream | host / streams | any | none | exact |
| `PF_HC_INJ=1` | next half's HC inject projection folded into the HC write kernels | CUDA-core FP32 | any | none | rounds |
| `PF_DM16H=1` | FP4G's down rows stored FP16 | only acts with `PF_FP4G=2` | 12.0a / 12.1a | inert elsewhere | rounds |
| `PF_GRAPH=7` | prompt QSA indexer chain + each expert group's chain as CUDA graphs | CUDA graphs (host) | any | none | exact |

### 4b. On by default on every card already (outside the table) - what the release check exercised

| switch | needs | on older cards today |
| --- | --- | --- |
| `PF_FP4G=2` (grouped FP4 expert GEMM) | `mma.sync.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` (12.0a / 12.1a only), `cp.async.bulk.shared::cluster` + `mbarrier` (9.0), 96.8 KB smem | `available()` probes the image; `emulated_cc()` keeps it off; real 7.5-8.9 have no 12xa image: MMQ W4A8 |
| `HEAD_PLANES` (two-stage Q8_0 head, `q80_vhead.cu`) | `__reduce_add_sync` (8.0), dp4a | 8.0+ (Turing: plain head; the patch ports it, below) |
| Auto's new MTP: `--mtp-head proxy --mtp-cum auto --spec 8` (`draft_proxy.cu`, `drafting.py`) | L2 `createpolicy` + `ld.global.nc.L2::cache_hint` on 8.0+ (`__ldcs` below) | on for **every CUDA card** with one request (serve's Auto) - checked here on 75 / 86 / 89 |
| `KV_POST`, `DENSE_SLAB`, `PF_SHARED_LATE`, `PF_KV_UNSTAGE_SIDE`, `PF_UPMIX_SKIP32`, `PF_REFILL_BG`, `GEMM_WARM_FILE` | host / plain kernels | on everywhere |

### 4c. Other RTX 50-only gates (older rounds; listed for completeness)

`HC_UPMIX` (the prompt's fused HC up + mix, cc 12.x only in `prefill.cpp`; sm_80+ kernel, bitwise vs cuBLAS only on the
5090), upstream's sm_120 bitwise series (`sm120_series.hpp`: `SIDE_MIXER`, `ATTN_DAILY`, `GR_UP75`, `GR_DOWN_TG`, `GU_SWIGLU`,
`GR_UP_WIDE`; five of them are in upstream's sm_86 table, Ada has none), `GR_V3_PF` / `SMALL_UNR` (`bw_dense_defaults.hpp`),
the QSA decode attention variant (`qsa_decode_attn.cu:1365`), DFlash2 spec 8 on big Blackwell cards (`draft_cap.hpp`),
thread-block-cluster QSA top-k / sampler (9.0+, probed by ptx/binary version), Turing's own bitwise series
(`STRATA_SM75_PERF=1`, opt-in upstream because nobody here has a Turing card).

## 5. Capability matrix (decode- and prompt-relevant)

From the PTX ISA (8.8 / 9.x) and the CUDA C++ Programming Guide's compute-capability tables; the 12.0 column is also
measured (`D:\Projects\gpu-probe\CAPABILITIES.md`). Not measured here for the others: inferred from the documentation.

| feature | 7.5 Turing | 8.0 A100 | 8.6 RTX 30 | 8.9 RTX 40 | 9.0 H100 | 10.0/10.3 B200/B300 | 11.0 Thor | 12.0 RTX 50 | 12.1 GB10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dp4a` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| `ldmatrix` x1/x2/x4 b16 | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| mma f16 m16n8k8 / k16 | k8 only | both | both | both | both | both | both | both | both |
| mma bf16, tf32 | no | yes | yes | yes | yes | yes | yes | yes | yes |
| mma int8 m8n8k16 / m16n8k16 / m16n8k32 | m8n8k16 only | all | all | all | all | all | all | all | all |
| mma fp8 e4m3/e5m2 (mma.sync m16n8k32) | no | no | no | yes | yes | yes | yes | yes | yes |
| block-scaled mma.sync (`kind::mxf8f6f4`, `mxf4`, `mxf4nvf4`) | no | no | no | no | no | no (tcgen05 only) | no (tcgen05) | **120a / 120f** | **121a / 120f** |
| `cvt ... e2m1x2` (FP4 pack) | no | no | no | no | no | 100a | 110a | 120a / f | 121a / f |
| `wgmma` (90a) / `tcgen05` (100a, 103a, 110a) | no | no | no | no | wgmma | tcgen05 | tcgen05 | no | no |
| `cp.async` (LDGSTS) | no | yes | yes | yes | yes | yes | yes | yes | yes |
| `cp.async.bulk` / TMA, `mbarrier.expect_tx` / `try_wait` | no | no | no | no | yes | yes | yes | yes | yes |
| thread-block clusters, DSMEM | no | no | no | no | yes | yes | yes | yes | yes |
| `griddepcontrol` (PDL) | no | no | no | no | yes | yes | yes | yes | yes |
| `createpolicy`, `L2::cache_hint` | no | yes | yes | yes | yes | yes | yes | yes | yes |
| `cp.async.bulk.prefetch.L2` | no | no | no | no | yes | yes | yes | yes | yes |
| `redux.sync` (`__reduce_add_sync`) | no | yes | yes | yes | yes | yes | yes | yes | yes |
| `ld.acquire` / `red.release`, `__nanosleep` | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| max smem / block (opt-in) | 64 KB | 163 KB | 99 KB | 99 KB | 227 KB | 227 KB | 227 KB | 99 KB | 99 KB |
| smem + L1 / SM | 96 KB | 192 KB | 128 KB | 128 KB | 256 KB | 256 KB | 256 KB | 128 KB | 128 KB |
| registers / SM, / thread | 64 K, 255 | 64 K, 255 | 64 K, 255 | 64 K, 255 | 64 K, 255 | 64 K, 255 | 64 K, 255 | 64 K, 255 | 64 K, 255 |
| max threads / SM | 1024 | 2048 | 1536 | 1536 | 2048 | 2048 | - | 1536 | 1536 |
| f16 x f16 with f32 acc vs f16 acc | 1/2 (GeForce) | 1/1 | 1/2 (GeForce), 1/1 some pro | 1/2 (GeForce) | 1/1 | 1/1 | - | 1/2 (measured) | - |
| int8 MMA vs f16 (f16 acc) | 2x | 2x | 2x | 2x | 2x (dense) | 2x | - | 2x (measured: 1,065 vs 533) | - |
| FP4 MMA vs int8 | - | - | - | - | - | 2x (tcgen05) | 2x (tcgen05) | 2x (measured: 2,130 vs 1,065) | 2x (inferred) |
| L2 (typical part) | 4-6 MB | 40 MB | 4-6 MB | 32-72 MB | 50 MB | >= 126 MB (inferred) | - | 96 MB (5090) | - |
| SMs (typical parts) | 22-72 | 108 | 16-84 | 20-128 | 114-132 | 148 | 20 | 30-170 | 48 |
| what needs an `a` / `f` target | - | - | - | - | wgmma, setmaxnreg (90a) | tcgen05 (100a/103a; 100f family) | tcgen05 (110a) | block-scaled mma, e2m1 cvt (120a; dense forms also 120f) | same (121a; 120f covers 120 + 121) |

Two consequences for this model on this engine:
- **FP4 tensor math exists on 12.x only through `mma.sync`**; on 10.x / 11.0 it is `tcgen05` (a different programming model:
  TMEM, async UMMA). Nothing in Strata uses tcgen05. So "FP4 everywhere" means 12.0 / 12.1 only; elsewhere NVFP4 weights go
  through int8 (W4A8 MMQ, the decode `dp4a`) - which on 7.5-8.9 is already the fastest tensor path those cards have
  (Ada's FP8 MMA with f32 accumulate runs at the int8 rate at best, so "FP8 instead of FP4 on Ada" gains no rate over the
  int8 MMQ that already runs there).
- **Everything Basalt's decode kernels need beyond sm_80 is TMA bulk copies, mbarrier waits and PDL** (9.0+). On 8.x
  the same kernels can run with `cp.async` 16-byte copies and without PDL; on 7.5 without `cp.async` and with m8n8k16 int8.

## 6. Plan: feature x architecture

`as is` = works today, only the defaults key blocks it (in the patch where marked P). `port` = needs a different code
path (sketch given). `no` = not possible on that hardware. `n/a` = nothing to do. `?` = possible but needs a real card first.

| feature | 7.5 | 8.0 | 8.6 | 8.9 | 9.0 | 10.x | 12.0 (non-5090) | 12.1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HC_Q8 | as is P | as is P | as is P | as is P | as is P | as is P | on | port of the table (Linux) |
| PREFILL_EQUAL | as is P | as is P | as is P | as is P | as is P | as is P | on | as is |
| GDN_TC (+CONVL2, TC_NP) | no (64 KB, no m16n8k16; port: m16n8k8 + 64 KB plan, L) | as is P (122 KB plan) | as is P (99 KB plan) | as is P | as is P | as is P | on | Linux default already |
| PF_GROUP_SCRATCH, PF_HC_REGION | as is P | as is P | as is P | as is P | as is P | as is P | on | as is |
| ADAPT_* , PCIE_FIXED | ? (policy: measure on a real card + PCIe link) | ? | ? | ? | ? | ? | on (tuned for PCIe 5 x16) | ? |
| DENSE_TC | port: m8n8k16 x4 + register staging (L) | port: ring layout only (`launch_s`, cp.async), guard 900 -> 800, tile/one/pair layouts with cp.async loops (M) | same as 8.0 (M) | same (M) | as is (enable) | as is | on | as is |
| ATTN_STAGED | as is P (1 block/SM: 40 KB of 64) | as is P | as is P | as is P | as is P | as is P | on | as is |
| GDN_MID (+MID_HEAD) | no (needs DENSE_TC + 99 KB) | port: cp.async + tiles-per-CTA loop (L) | port (L) | port (L) | port: tiles per CTA for < 160 SMs (M) | same (M) | **off below 160 SMs** (every RTX 50 but the 5090 / PRO 6000): tiles-per-CTA port (M) | same |
| QPROJ_HALF, GATE_AT, GATE_GRID | follows DENSE_TC | follows DENSE_TC | follows | follows | as is; GATE_GRID scaled by SMs | same | GATE_GRID=48 is 5090-sized: scale as 48 x SMs / 170 (S) | same |
| GRAPH_PRIO | as is P | as is P | as is P | as is P | as is P | as is P | on | as is |
| DF_PDL | no (no griddepcontrol) | no | no | no | as is (enable) | as is | on | as is |
| EARLY_READ | ? (co-residency; with HC_Q8) | ? | ? | ? | ? | ? | **guarded by the patch** (on only where 2 x SMs >= 220 + SMs / 2) | same guard |
| ROUTE_DB | as is P | as is P | as is P | as is P | as is P | as is P | on | as is |
| QSA_I8 | port: m8n8k16 int8 (4 per m16n8k32; same exact int32 sums) (M) | as is P | as is P | as is P | as is P | as is P | on | as is |
| DENSE_F16C | port: m16n8k8 + register-staged loads (M) | no gain (f16-acc = f32-acc rate) | as is P | as is P | no gain | no gain | on | as is |
| KV_PREFETCH | as is P | as is P | as is P | as is P | as is P | as is P | on | as is |
| PF_HC_INJ | as is P | as is P | as is P | as is P | as is P | as is P | on | as is |
| PF_GRAPH | as is P | as is P | as is P | as is P | as is P | as is P | on | as is |
| PF_FP4G / W4A4 / PF_DM16H | no (no FP4 MMA) | no | no | no | no (FP8 wgmma possible, but no gain over int8 MMQ for W4) | no (tcgen05: a new kernel, XL) | on (120a) | port: build the unit as 120f, guard on `__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200` (S) |
| HEAD_PLANES (q80p) | port done in the patch (redux -> shuffles) P | on | on | on | on | on | on | on |
| upstream sm_86 kernel set (SIDE_MIXER, ATTN_DAILY, GR_UP75, GU_SWIGLU, QSA_ROWS, B6_MMVQ_ROWS) | SM75_PERF series P | ? (upstream sm_80 tables) | on (upstream) | as is P | ? | ? | sm120_series | sm_121 table |

## 7. Per-arch defaults: how nv_defaults.cpp should key them

Today `nv_default_env(major, minor)` is one `if` per exact compute capability, and each table is "measured on that card".
Most of the RTX 50 set is not about the card but about features. Proposed keying (the patch implements the first two):

1. **The card's own measured table** stays exactly as it is (sm_86, sm_60, sm_121, sm_120): `arch_table()`.
2. **The fork's portable set** (`nv_portable_env`, new): for cc 7.5 - 11.x, keyed by feature thresholds, not by card:
   every card gets the exact host-side switches and the plain-CUDA kernels; `>= 8.0` adds the sm_80 tensor-core ones
   (GDN_TC, QSA_I8); `{8.6, 8.9}` (cards whose f16-accumulate MMA runs at twice the f32 rate) adds DENSE_F16C; 7.5 adds
   upstream's bitwise Turing series (`SM75_PERF`), 8.9 the kernel switches of upstream's sm_86 table. One master switch
   `STRATA_NVFP4_PORTABLE=0`, `STRATA_DEFAULTS=026` empties it, a switch the user set is kept, the start prints its own line.
3. **Next step (not in the patch)**: when DENSE_TC / GDN_MID get their sm_80 ports, add `>= 9.0` (DF_PDL, DENSE_TC,
   QPROJ_HALF, GDN_MID with its own SM check) and keep values that depend on the SM count out of the table: compute them
   (`QPROJ_GATE_GRID = round(48 * SMs / 170)`, GDN_MID's tiles per CTA) where the kernel is launched.
4. **Policies tuned for one PC** (ADAPT_*, PCIE_FIXED) belong in a "system" table keyed by what they depend on (PCIe
   generation and width from NVML, RAM, CPU), not by compute capability; until measured elsewhere they stay 12.0-only.

## 8. The portable release build

Today: `75-real;86-real;89-real;120-real`, the FP4 unit `120a-real`.

- **Keep 75 / 86 / 89 / 120 real.** Add `80-real` only if A100 / CMP 170HX users are to be served by the release (+~12 MB;
  upstream already has measured sm_80 defaults).
- **Add a PTX image** (`120-virtual` - or `89-virtual` for size) so a card newer than the list (and the emulated tests)
  JITs instead of "no code for this card". The FP4 unit cannot be PTX-forwarded (arch-specific features never are).
- **FP4 unit: `120f-real` instead of `120a-real`** (CUDA 12.9+ family target; this machine has 13.3). One cubin then runs
  on 12.0 and 12.1 (DGX Spark gets FP4G / W4A4 from the same file); the guards in `mmq_nvfp4_fp4g.cu` / `_w4a4.cu` /
  `_a44.cu` must also accept `__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200` (today they test `__CUDA_ARCH_FEAT_SM120_ALL` /
  `SM121_ALL`, which 120f does not define). Dense `kind::mxf4nvf4` compiles for `sm_120f` (gpu-probe CAPABILITIES.md, item
  11; the sparse forms would not - FP4G does not use them). Checked here: `mmq_nvfp4_fp4g.cu` with that guard compiles for `sm_120f` to SASS **identical** to `sm_120a` (11,872 instructions, 576 `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`); `nvcc` defines `__CUDA_ARCH_FAMILY_SPECIFIC__` = 1200 for 120f, 1210 for 121f (measured, compile only).
- **9.0 / 10.x**: `90a` / `100a` are needed only for wgmma / tcgen05, which Strata does not use; their plain `90` / `100`
  cubins would run the sm_90 paths (DENSE_TC, GDN_MID, PDL, clusters). Not worth adding to a Windows consumer release
  unless someone asks; PTX covers them.

## 9. Port work items, ranked by value per effort

Effort: S = hours, M = 1-3 days with parity tests, L = a week, XL = a new kernel family. Value as the 5090 measured it
for that switch (decode = tok/s of a window, prompt = 64K read) - the real card's gain is unknown in every row.

| # | item | archs | effort | value (5090's number) | status |
| --- | --- | --- | --- | --- | --- |
| 1 | EARLY_READ co-residency guard | 12.0 cards < 147 SMs (RTX 5080 / 5070 / 5060, PRO 5000) | S | correctness | **in the patch** |
| 2 | defaults key: the portable set | 7.5-11.x | S | sum of the exact / rounding switches above | **in the patch** |
| 3 | two-stage head (q80p) on Turing | 7.5 | S | head screen (part of the +16-55% Basalt round) | **in the patch** |
| 4 | FP4 unit as `120f`, family guard | 12.1 (Spark) | S | prompt W4A4 1.11-1.23x on the Spark (upstream's number), FP4G 16K -5% / 64K -4.6% | plan |
| 5 | QPROJ_GATE_GRID scaled by SM count | 12.0 non-5090 | S | attention front -5 us/layer on the 5090 | plan |
| 6 | DENSE_TC on 8.0-8.9: ring layout + cp.async copies of the tile / one / pair layouts, PDL off | 8.0, 8.6, 8.9 | M | +2.0% struct; unlocks GDN_MID and QPROJ_HALF | plan |
| 7 | GDN_MID for < 160 SMs (tiles per CTA) | 12.0 non-5090, 9.0, 10.x; 8.x after #6 | M | GDN front 81.8 -> 78.4 us/layer (-4%) | plan |
| 8 | QSA_I8 on Turing (m8n8k16 int8) | 7.5 | M | prompt QSA scores ~2x the FP32 kernel | plan |
| 9 | DENSE_F16C on Turing (m16n8k8, register staging) | 7.5 | M | 1.2x cuBLAS on the dense GEMMs | plan |
| 10 | ADAPT_* / PCIE_FIXED by link and RAM | all | M (needs real PCs) | struct +10.4%, prose +3.9% on this PC | needs real cards |
| 11 | GDN_TC on Turing (m16n8k8, 64 KB plan) | 7.5 | L | 64K read -2.5% | plan |
| 12 | DENSE_TC on Turing (m8n8k16) | 7.5 | L | as #6 | plan |
| 13 | grouped W4A8 expert kernel (FP4G's grouping on int8 MMA) | 7.5-8.9 | L | unknown: FP4G's gain is part rate (2x FP4), part grouping | idea |
| 14 | FP4 on B200 / Thor (tcgen05) | 10.x, 11.0 | XL | - | not planned |
| - | DF_PDL below 9.0 | 7.5-8.9 | - | - | **not possible** (no griddepcontrol) |
| - | FP4G / W4A4 below 12.0 | 7.5-9.0 | - | - | **not possible** (no FP4 MMA in mma.sync) |

## 10. Test plan

On the 5090 (what this round did, reusable as `bench-ma\run_check.sh`, `run_cw.sh`, `run_hog.sh`):
- **Build** each arch as PTX (`bench-ma\build_arch.cmd 86-virtual build-ptx86`, ~2 min from scratch, ~2 min per patch).
- **Release identity per arch**: `ma_test.sh all eNN <exe> STRATA_EMULATE_CC=NN`: a 16K read (fixed placement, 48 tokens,
  first-token logits md5) and 200-token decodes with Auto's MTP and with DFlash2; text sane, no CUDA error, the same text
  as the sm_120 path with the RTX 50 set off where the arithmetic is the same.
- **Exact switches**: on each emulated arch, the patched engine with the portable set off plus only the exact switches
  by hand must give the release's md5s (text and logits) - this round's `pNNx` runs.
- **Rounding switches**: KL of first-token logits on vs off per arch next to the 5090's own on/off KL (the same switches on
  sm_120): `kl.py`.
- **Smaller cards of the same generation**: `STRATA_EMULATE_SMS=N` holds all but N SMs with an idle kernel (50 ms launches),
  so co-residency waits (EARLY_READ, any persistent grid) can be checked on the 5090. Speeds mean nothing there.
- **sm_120 identity**: `exact.sh` fixed placement, 16K text / logits md5 with defaults = `93c4688100 / e9d907e164`, and the
  MTP decode md5 = the release's, after every change.
- **Unit**: `nv_defaults_test` (tables; no GPU).

Not verifiable without real cards: every speed number (JIT'd code on a 5090 has the 5090's SMs, clocks, caches and
PCIe 5), cuBLAS's own per-arch kernel choice (it runs sm_120 kernels here: any GEMM rounding difference on a real 30/40
card is invisible), Turing's 64 KB / 1024-thread occupancy effects on speed (the limits are emulated for the host's
choices only), WDDM behaviour on other drivers, real PCIe 3 / 4 links for the tier policies, the Spark (12.1) and the
datacenter parts (8.0 / 9.0 / 10.x) - for those the emulation also misreports shared memory (`smem_optin_of` knows 75 /
80 / 86 / 89 only).

## 11. What the patch enables, and its checks (measured)

`nv_portable_env()` (nv_defaults.cpp), applied before any switch is read, printed as its own start line:

| cc | switches |
| --- | --- |
| all of 7.5-11.x | exact: ATTN_STAGED, PF_GROUP_SCRATCH, PF_HC_REGION=2, KV_PREFETCH, PF_GRAPH=7, GRAPH_PRIO, ROUTE_DB=2; rounding: HC_Q8, PREFILL_EQUAL=2, PF_HC_INJ |
| 7.5 | + SM75_PERF=1 (upstream's bitwise Turing series) - 11 in all |
| >= 8.0 | + GDN_TC, GDN_CONVL2=2, GDN_TC_NP, QSA_I8 (8.0: 14) |
| 8.6, 8.9 | + DENSE_F16C (8.6: 15 next to upstream's 8; 8.9: 15) |
| 8.9 | + SIDE_MIXER=2, ATTN_DAILY, GR_UP75, GU_SWIGLU, QSA_ROWS, B6_MMVQ_ROWS (upstream's sm_86 kernel set) - 21 |

Off switch `STRATA_NVFP4_PORTABLE=0`; `STRATA_DEFAULTS=026` empties it; a user setting wins. Not in it, with the
reasons in section 6: DENSE_TC, GDN_MID(_HEAD), QPROJ_*, DF_PDL (sm_90+), EARLY_READ, ADAPT_* / PCIE_FIXED, PF_DM16H.
9.0 / 10.x get the same set by features (DENSE_F16C excluded) - not run.

| run (fixed placement) | 16K text | 16K logits | MTP decode | DFlash2 | result |
| --- | --- | --- | --- | --- | --- |
| sm_120 patched, defaults | 93c4688100 | e9d907e164 | 3d13f77b08 | - | = release (bit-identical) |
| 86: set off + exact switches by hand | 62205238cc | ddbb216eeb | 9e05d7249c | - | = baseline bit for bit |
| 89: set off + exact switches by hand | 62205238cc | ddbb216eeb | 9e05d7249c | - | = baseline bit for bit |
| 75: set off + exact switches + SM75_PERF (+ q80p port) | 62205238cc | d0bcbcb21d | 0a6a56bc1e | - | = baseline bit for bit |
| 86 defaults | 62205238cc | 7beaf7c98c (KL 0.0003) | 8c2031d38d | 8c2031d38d | sane, same top token |
| 89 defaults | 62205238cc | 7beaf7c98c (= 86) | 8c2031d38d | 8c2031d38d | sane |
| 75 defaults | 93c4688100 | d58dcbe598 (KL 0.0006) | e2a9ea0814 | e2a9ea0814 | sane |
| 5090 with the same set by hand (SM120=0, FP4G=0) | 0291291206 | KL 0.0039 vs its off | - | - | the 5090's own on/off distance |

The rounding switches move the older archs' logits less than they move the 5090's (0.0003-0.0006 vs 0.0039), and the
paths that ran are named in the logs (`i8x6`, `f16-accumulate chains of 64`, `hc8 read`, bit-planes head).
`nv_defaults_test` passes. Emulated speeds are in results.txt and are not the real cards'.

## 12. Measured vs inferred

Measured on the 5090:
- everything in sections 1, 2 and 11;
- the 120f compile (section 8);
- the hc8 register count (ptxas resource usage of the release exe).

Inferred:
- that a real Turing card never took the planes path (from the cubin's binaryVersion);
- the real-card failure mode of 2b (the timeout and stale reads; the 5090's limiter only showed the stall);
- every number of the capability matrix except the 12.0 column (PTX ISA / CUDA guide);
- the value column of section 9 (the 5090's numbers);
- that the rounding switches keep the 5090's accuracy figures on other archs (they run the same kernels; KL measured
  only for the first token of one 16K prompt here);
- the effect of the portable set on speed on any real card.

Not run (short runs only, per the standing rule):
- no accuracy corpus, no 64K / 131K reads;
- no 8.0 / 9.0 / 10.x / 12.1 emulation (`smem_optin_of` would misreport them).

Also noticed:
- `STRATA_EMULATE_CC` with switches the table does not set can still trap: `dense_tc_on()` and `gdn_mid` check the
  raw major, so `STRATA_DENSE_TC=1` on an emulated card would launch sm_90 code. Test-mode only; set no sm_90 switch
  when emulating.
- A `strata.exe` from `C:\LLM_models` (PID 22984) was running since 00:33 with ~1.8 GB of VRAM. Its parent is gone,
  and it was started when the lock owner was the other agent's `probe-dec` run. It is not the loaded tray engine.
  common.sh's `gpu_free` waits on it, so this round used a VRAM-only check (`ma_test2.sh`). It was not touched.
