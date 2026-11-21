# RTX PRO 4000 Blackwell 24 GB, 64 GB of RAM (simulated): the NVFP4 GPTQ pack in the low-RAM mode

Community report, 2026-11-10. Measured on one machine and prepared with AI assistance (Claude Code). Every number
below comes from runs on this machine. It follows the [2026-11-04 strict-256K report](../2026-11-04-community-rtx-pro-4000-24gb-linux-256k-stability/README.md)
from the same box, which ran with 128 GB. Harness, scripts and the full-RAM baseline:
[Adam90a/local-256k-reasoning](https://github.com/Adam90a/local-256k-reasoning) (`research/low-ram-64gb.md`).

**Question:** how does the NVFP4 GPTQ pack run on a 24 GB card with only 64 GB of system RAM? The README's 64 GB
figure (~115 tok/s, [NVFP4.md](../../../docs/NVFP4.md#64-gb-of-ram-2026-09-30)) was measured with a 32 GB RTX 5090.

## Machine

- **GPU:** NVIDIA RTX PRO 4000 Blackwell 24 GB (sm_120), 145 W limit, PCIe Gen5 x16. It is the only GPU used.
- **CPU:** AMD Threadripper 7970X (32C/64T, AVX-512).
- **RAM:** 128 GB DDR5-4800 installed, **capped to simulate a 64 GB PC** (see Method).
- **SSD:** Corsair MP600 PRO NH NVMe. The engine measured it at 2.50 GB/s.
- **OS:** Ubuntu 24.04, kernel 7.0, NVIDIA driver 595.91.07.

## Build

- **Engine:** the same Linux CUDA 13.0 source build of fork commit `586dc5e` as the 2026-11-04 report (build steps
  there). The engine reports 0.1.45.3.
- **Server:** checked out at a local branch: `586dc5e` plus one commit that changes `serve/server.py` only (an opt-in
  `answer_reserve_max_tokens` config key). The engine binary is unchanged, and that key has no effect on these runs:
  thinking is off and `max_tokens` is 512.

## Method

- **Simulating 64 GB:** the server runs as a transient systemd unit with `MemoryMax=56G` and `MemorySwapMax=0`.
  56G is roughly what a 64 GB desktop leaves the model. The cgroup cap counts the page cache too, so `experts.bin`
  reads behave as on a real 64 GB box. The engine still sees 125 GiB installed, so the automatic switch below
  92 GiB does not fire; `--ram-budget 48` forces the low-RAM mode instead.
- **Configuration:** `config.json`; `<m>` is the model snapshot folder.
  - Changes from the 2026-11-04 run:
    - `--max-context 131072`, setup's value for a 24 GB card;
    - no `--kv-resident`: K/V in VRAM, because pinning the 288K K/V in RAM does not fit 64 GB;
    - `--ram-budget 48`;
    - `--conversation-cache-mib 0`.
  - Unchanged: NVFP4 GPTQ pack `9b70ad3`, `--kv int8`, `--rope-scaling dynamic`, MTP `--spec 4`,
    `--prefill auto`, `--expert-cache auto`.
- **Benchmark:**
  - One fresh request at a time through `/v1/chat/completions`, streaming, thinking off, 512 output tokens.
  - Prompts are varied real text: Python stdlib source, Bible prose and Markdown docs.
  - 2 reps each at 4K, 32K and 128K. The prompt token count is the server's `usage.prompt_tokens`.
  - Decode, prefill, expert hit rate and PCIe share are the server's own per-request numbers (`/metrics`).
  - 1 Hz GPU telemetry is in `telemetry/`. Cgroup memory and the engine's disk reads, every 2 s, are in `cgroup.csv`.

## What the low-RAM mode did (`engine.log`)

- **GPU expert cache:** 5,562 slots, 14.32 GiB of VRAM, pre-filled from the profile in 3.7 s.
- **RAM:** the 48.00 GiB budget held 18,641 of the experts the GPU cache does not hold, by profile rank. All 48.00
  GiB were page-locked in one allocation, built in 47 s.
- **SSD:** the remaining ~15.3 GiB of experts are read from `experts.bin` on demand.
- **Prefill plan:** the prompt chunk dropped from 32,768 to **5,632** tokens. The engine's own estimate for a 32K
  chunk was ~28.8 GB read from the drive (~41 s) per 32K prompt, against 4.5 GiB of page cache left. The log also
  warns that all 4,705 lendable slots read from the pack during long prompts.
- **Start:** ready **59 s** after launch (52 s on an earlier start).

## Results

| prompt tokens | TTFT | prefill tok/s | decode tok/s | expert hit | PCIe share | output |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,095 | 2.53 s | 1,640 | 77.7 | 77.7% | 4.1% | 512 |
| 4,095 | 2.94 s | 1,414 | 82.2 | 78.4% | 3.8% | 512 |
| 32,763 | 16.49 s | 1,994 | 83.8 | 80.6% | 3.5% | 512 |
| 32,766 | 15.93 s | 2,063 | 80.3 | 83.1% | 3.0% | 512 |
| 131,044 | 60.70 s | 2,165 | 65.9* | 72.0% | 4.9% | 20 |
| 131,052 | 61.07 s | 2,152 | 62.2* | 79.7% | 3.2% | 12 |

\* The 128K prompts nearly filled the 131,072-token window, so only 12–20 tokens were generated. Those decode
figures are too short to compare; TTFT and prefill at 128K are valid.

Compared with the same box and harness with 128 GB and every expert in RAM
(`--kv-resident 20480`, 294,912-token window, 2026-11-09, 3 reps):

| prompt | TTFT, 64 GB / 128 GB | decode tok/s, 64 GB / 128 GB |
| --- | ---: | ---: |
| 4K | 2.7 s / 2.0 s | 80.0 / 109.8 |
| 32K | 16.2 s / 11.4 s | 82.0 / 107.5 |
| 128K | 60.9 s / 43.5 s | n/a / 100.2 |

So on a 24 GB card, 64 GB of RAM costs about 25% of decode speed and about 40% of prompt speed. It is still a
practical setup.

**Peaks:**
- Cgroup memory hit the 56.0 GiB cap. Engine RSS was 55.8 GiB, with no OOM and no swap.
- VRAM was 23.2 GiB (23,767 MiB).
- The GPU reached 89 °C at the 145 W cap during the 128K runs.
- The engine read 161 GiB from disk over the whole run (`/proc/<pid>/io` `read_bytes`). That includes the start
  (48 GiB RAM copy plus the 14 GiB VRAM fill).

## Limitations

- **Simulated, not a real 64 GB machine.** It ran under a cgroup cap. The engine saw 125 GiB installed, so its
  automatic sizing differs: `--ram-budget` was set by hand, while a real 64 GB PC would get "free RAM less 6 GiB".
- **Small sample:** 2 reps per size and short outputs (512 tokens), thinking off, single-turn.
- **No 128K decode figure** (see above).
- **No flag A/B.** No `--kv k8v4` or `--prefill auto:8192`, and no other `--ram-budget` values.
- The expert profile was learned on this box's 128 GB workload. It was copied for this run and saved separately.
