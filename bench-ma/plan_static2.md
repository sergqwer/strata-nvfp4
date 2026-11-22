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
  11; the sparse forms would not - FP4G does not use them). Section 9 has the compile check.
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
