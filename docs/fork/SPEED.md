# Speed

How fast this fork runs, and where the detailed tables are. Back to the [README](../../README.md). The README's
[models table](../../README.md#models) has decode and 32K-prompt numbers for every model.

## Long dialogs in 0.2.15-nvfp4.5

RTX 5090 at 600 W, the NVFP4 GPTQ set, serve, greedy. One engine and one dialog that grows from 2K to 370K tokens;
at each length the same JSON continuation (identical text at every length) and a prose one. 0.2.15-nvfp4.5 against
the build before its round 5, both release builds, 3 runs each, tokens/s:

| Context | JSON | prose |
| ---: | ---: | ---: |
| 32K | 363 -> 378 (+4%) | 195 -> 201 (+3%) |
| 64K | 370 -> 388 (+5%) | 175 -> 182 (+4%) |
| 128K | 366 -> 389 (+6%) | 168 -> 175 (+4%) |
| 256K | 353 -> 400 (+13%) | 162 -> 172 (+6%) |
| 370K | 324 -> 367 (+13%) | 164 -> 166 (+2%) |

Short answers against 0.2.15-nvfp4.4 (two steps on two days, multiplied): JSON about +7%, chat +4%, English prose +6%,
Ukrainian +1%. Pictures read in the engine: the first token of a 329-token picture after 631 ms instead of 1,352.
The text is bit-identical to 0.2.15-nvfp4.4's in the fixed-placement check. The switches and the runs:
[RTX 50: decode rounds 3-5](../NVFP4.md#rtx-50-decode-rounds-3-5-long-contexts-smaller-cards-2026-11-21-release-0215-nvfp45).

## RTX 50 in 0.2.15-nvfp4.4

RTX 5090, Ryzen 9 9950X3D, 128 GB, the NVFP4 GPTQ set, serve, greedy, the tray's arguments, one run of each in
the same hour (power limit 510 W). Decode, tokens/s:

| Answer | 0.2.15-nvfp4.3 (Auto: DFlash2) | 0.2.15-nvfp4.4 (Auto) | Basalt |
| --- | ---: | ---: | ---: |
| JSON | 330 | 487 | 475 |
| chat | 219 | 269 | 248 |
| English prose | 205 | 261-271 | 276 |
| Ukrainian | 154 | 241 | 235 |

These runs used Basalt's self-distilled draft layer, which setup fetches for the GPTQ set. Prompt, one request,
context 140,000, ms:

| Prompt tokens | 0.2.15-nvfp4.3 | after round 1 | | before round 2 | 0.2.15-nvfp4.4 | Basalt |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 3,072 | 924 | 853 | | 922 | 857 | 1,258 |
| 16,676 | 1,938 | 1,657 | | 1,686 | 1,350 | 1,972 |
| 66,617 | 7,845 | 6,163 | | 6,208 | 5,110 | 6,300 |
| 131,000 | 14,460 | 12,471 | | 12,496 | 10,345 | - |

The first two columns were measured at a 510 W power limit, the last three at 600 W, so compare within each pair.
Basalt's context ends at 66K. The time to the first token at 64K went from 6.19 to 5.25 s in round 2. Accuracy is
unchanged: the top token matches the base engine's 95.9-96.1% of the time with a KL of 0.015, while two runs of the
base engine itself differ by 96.5% / 0.012.

The switches and the full runs:
[RTX 50: decode and prompt kernels after Basalt](../NVFP4.md#rtx-50-decode-and-prompt-kernels-after-basalt-2026-11-20-release-0215-nvfp44)
and [RTX 50: faster prompts](../NVFP4.md#rtx-50-faster-prompts-2026-11-20-release-0215-nvfp44). Long dialogs with
KV streaming: [KV streaming on by default](../NVFP4.md#kv-streaming-on-by-default-2026-11-19-release-0215-nvfp43).

## RAM and card size

An older run that compares RAM sizes on one PC.

RTX 5090 (32 GB, PCIe 5 x16), Ryzen 9 9950X3D, DDR5-5600, Samsung 9100 PRO, Windows 11. Measured with `huihui-nvfp4`
(withdrawn since; the original Qwen's packs have the same format and size) and the arguments setup writes (262K context, int8 K/V,
images on). The card also drives the desktop, so runs vary by a few
percent.

| | 128 GB | 96 GB (emulated) | the low-RAM mode, 88 GiB free |
| --- | ---: | ---: | ---: |
| a chat answer, ms a round (~2.3 tokens a round) | 12.7 | 13.7 | 20.3 |
| a 32K prompt | 3.7 s | 3.9 s | 8.4 s |

The 96 GB PC is the 128 GB one with 95.6 GiB reported as installed and 88 GiB of RAM left free; the last column is the
low-RAM mode forced on it. A 64 GB PC runs the low-RAM mode with a smaller RAM budget, and a smaller card caches
fewer experts: [A 96 GB PC](../NVFP4.md#a-96-gb-pc-2026-10-09-release-0141-nvfp42),
[64 GB of RAM](../NVFP4.md#64-gb-of-ram-2026-09-30) and
[Other GPUs](../NVFP4.md#other-gpus-rtx-20-30-and-40-2026-09-30) have those runs.
