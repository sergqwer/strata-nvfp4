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
| max threads / SM | 1024 | 2048 | 1536 | 1536 | 2048 | 2048 | 1536 | 1536 | 1536 |
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
