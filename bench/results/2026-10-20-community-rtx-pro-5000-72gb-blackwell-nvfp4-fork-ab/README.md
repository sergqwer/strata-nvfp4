**Community benchmark — Linux source build, fork `v0.1.44.1-nvfp4.2` (engine 0.1.44.1) → `v0.2.0-beta-nvfp4.3` (engine 0.1.45.3) with `--batch-mtp`, same machine / same pack / same config.**

- Machine: RTX PRO 5000 72GB Blackwell (sm_120) · Ryzen 9 9950X3D · 249 GiB DDR5 · Ubuntu 24.04 · CUDA 13.0 build · GPTQ NVFP4 pack (OrcaRouter uncensored), 512K ctx, int8 KV, parallel 2.
- Method: official `tools/strata_bench.py --runs 3 --long` run against **both** engines (true same-machine A/B), plus a 12-tier cold bench (two tails), a 3×-repeat controlled median set with unique payloads, code-task benchmarks with MTP acceptance recorded, a 512K full-window recall check and a vision check. Timing = engine per-request log lines.
- Full poster (details, tables, protocol, limitations) attached below. **Poster text is Chinese with English terms kept — the numbers, tables and reproduce commands are language-neutral.**

Headline numbers (median [range], tok/s):

| Suite item | 0.1.44.1 | 0.1.45.3 + `--batch-mtp` | Δ |
| --- | --- | --- | --- |
| short chat, 9 reqs | 172.7 [155.1–247.8] | **189.1 [168.9–284.0]** | **+9.5%** |
| ├ coding question | 245.0 | **283.0** | **+15.5%** |
| agent session, follow-up turns (18) | 168.2 [103.1–200.1] | **177.2 [140.0–230.3]** | **+5.4%** |
| agent stale-turn re-read per session | med 72 (4–143) | **med 2 (0–4)** | conversation-cache copies improved |
| 4K / 16K / 32K fresh prefill | 4288 / 5286 / 5429 | 4309 / 5348 / 5445 | +0.5% / +1.2% / +0.3% (flat) |
| code tasks, complete script (3×, all compile-checked) | — | **260.1** at 81.6% draft acceptance | — |
| 512K full window | — | **PASS** — 524,004 tokens in, early fact recalled exactly, prefill 4,723 tok/s | — |
| vision (CPU `strata-vision`) | — | **PASS** — shape counts + string exact, 1.8 s | — |

Across the 3×-repeat set, decode +7.7%…+19.2% (median +9.4%) with prefill flat within each cell's own noise floor (the old engine's 1K rep prefill swings ±37% single-shot — one run is not a benchmark).

On the "≈197 tok/s for code tasks" figure sometimes quoted: on this machine it turns out to be an **extrapolation at ~65–68% MTP draft acceptance, not a measurement**. Fitting all 30 measured requests: `decode ≈ 18.7 + 2.78 × acceptance%`. Code/SQL/JSON run at 80–88% acceptance (256–283 tok/s); Chinese prose runs at ~40% (110–140). Decode here is essentially linear in acceptance — worth reporting both together.

One config warning found the hard way: `--prefill` must stay `auto` on this build — pinning 1024/2048/4096 cost **−16%…−55% prefill** in controlled arms.

Limitations: single shared card at its 300 W default cap; suite serial (no `--clients 4`); "same bits" claim only partly verified here (long-context recall + code probes identical, short answers reworded — inherent to batched slots); the new engine has <24 h uptime so long-run stability is not yet a claim. Raw per-request JSONs, engine log lines and the measurement scripts are available on request.

Thanks for `--batch-mtp` and the faster decode kernels — the upgrade went live on this box after passing a pre-declared gate (no decode tier worse than −3%, prefill within noise, fact probes exact).
