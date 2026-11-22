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
