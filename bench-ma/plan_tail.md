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
