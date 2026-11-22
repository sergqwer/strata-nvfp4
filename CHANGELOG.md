# Changelog

Every release of Strata NVFP4, the newest first. Each version links to its GitHub release, whose notes say what
changed in it; [docs/NVFP4.md](docs/NVFP4.md) has the measurements behind each one. Back to the
[README](README.md); this page replaces its What's new list.

## 0.2.15-nvfp4.8

Released 2026-11-22: [release page](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.2.15-nvfp4.8).

- **Two or more GPUs under Windows pin 8 GiB of experts again, as upstream does (0.2.15-nvfp4.8).** On the two RTX
  5060 Ti, 0.2.15-nvfp4.7 still did not start: with 61.6 GiB pinned the cards could no longer read their free memory.
  Windows charged every pinned byte to both cards, and their sum crossed the budget each card reported. The pinning per
  card (0.2.15-nvfp4.6-.7) is now off by default; the portable pin with its 8 GiB limit started on that PC and decoded no
  slower. `STRATA_ARENA_PIN_PER_STAGE=1` turns the pinning per card back on
  ([details](docs/NVFP4.md#two-gpus-under-windows-8-gib-pinned-again-2026-11-22-release-0215-nvfp48)).
- **The same text.** One GPU and Linux are unchanged.
- **Update:** close the server. In a clone run `UPDATE.bat`; in the release folder run `update.cmd`. No new model
  files. A `"STRATA_ARENA_PIN_PER_STAGE": "0"` added to a config can stay or go: it is now the default.

## 0.2.15-nvfp4.7

Released 2026-11-21: [release page](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.2.15-nvfp4.7).

- **Two GPUs under Windows: the pin stops below the driver's limit (0.2.15-nvfp4.7).** On the two RTX 5060 Ti,
  0.2.15-nvfp4.6 still failed. The driver refused at the same 63.74 GiB in all however the cards shared it, and after the
  refusal it took nothing back, so the start hung. Now a layer split under Windows pins at most 62 GiB of experts on all
  its GPUs together. If the driver refuses earlier, the engine writes where it happened
  (`%LOCALAPPDATA%\strata\wddm-pin-limit.txt`) and ends; strata serve starts it again by itself, and it pins 2 GiB less
  than before. No setting is needed
  ([details](docs/NVFP4.md#two-gpus-under-windows-one-limit-of-about-64-gib-for-all-the-gpus-2026-11-21-release-0215-nvfp47)).
- **The same text.** One GPU is unchanged. With two, the change only decides which layers stay pinned.
- **Update:** close the server. In a clone run `UPDATE.bat`; in the release folder run `update.cmd`. No new model
  files.

## 0.2.15-nvfp4.6

Released 2026-11-21: [release page](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.2.15-nvfp4.6).

- **Two GPUs under Windows start where the driver limits pinned memory (0.2.15-nvfp4.6).** A layer split pinned its
  experts stage by stage up to the driver's refusal. Two RTX 5060 Ti with 256 GB of RAM got 64-70 GiB pinned while Windows
  reported a 115 GiB budget, and then neither card could read its free memory, so the start ended. Now the stages take
  turns (each card the same share of its layers, whatever its VRAM), the first refusal ends the pin on every card, 4 GiB
  go back (`STRATA_PIN_MARGIN_GIB`) and every card must answer before the engine goes on. No setting is needed
  ([details](docs/NVFP4.md#two-gpus-under-windows-the-pinned-experts-find-the-drivers-limit-2026-11-21-release-0215-nvfp46)).
- **The same text.** One GPU is unchanged. With two, the change only decides which layers stay pinned.
- **Update:** close the server. In a clone run `UPDATE.bat`; in the release folder run `update.cmd`. No new model
  files.

## 0.2.15-nvfp4.5

Released 2026-11-21: [release page](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.2.15-nvfp4.5).

- **Long dialogs decode faster on RTX 50 (0.2.15-nvfp4.5).** Past 24,576 cells the indexer scores its blocks with a
  kernel that reads each pooled key once for all of the window's queries (the same scores bit for bit), the drafter's
  attention runs pipelined, and the K/V resolve ends at once while every block is resident. RTX 5090, one dialog
  growing to 370K tokens, against the build before this round: JSON 353 -> 400 tokens/s at 256K and 324 -> 367 at
  370K (+13%), +4-6% at 32-128K; prose +2-6%
  ([measurements](docs/NVFP4.md#rtx-50-decode-rounds-3-5-long-contexts-smaller-cards-2026-11-21-release-0215-nvfp45)).
- **Short answers a little faster (0.2.15-nvfp4.5).** Two more decode rounds of exact kernel changes: against
  0.2.15-nvfp4.4 a JSON answer about +7%, a chat +4%, English prose +6%, Ukrainian +1%.
- **The engine reads pictures itself, on the GPU (0.2.15-nvfp4.5).** It borrows ~360 expert slots only while it
  encodes a picture and refills them right after. The first token of a 329-token picture comes after 631 ms instead of
  1,352, of a 1,085-token one after 924 ms instead of 4,777. Setup and the release's config turn it on
  (`"in_engine": true`; `update.cmd` keeps a config you edited, so add it to that one's `vision` section yourself);
  `STRATA_VISION_IN_ENGINE=0` keeps the CPU reader ([Images](docs/NVFP4.md#images-2026-09-30)).
- **Fixes for smaller RTX 50 cards and two GPUs (0.2.15-nvfp4.5).** 0.2.15-nvfp4.4's early hyper-connection read
  (`STRATA_EARLY_READ`) could wait 2 s a window on cards with fewer multiprocessors than the 5090 (RTX 5080, 5070,
  5060): it is off by default now and skipped where it cannot fit. With two cards the pinned experts are cut to the
  Windows shared segment (upstream #3725), and a layer split sizes its DeltaNet grid and prompt buffer per card.
- **The session tables grow with the context (0.2.15-nvfp4.5).** Under KV streaming the indexer's pooled keys and the
  RoPE table are no longer sized for the whole context at the start: +320 expert slots at short contexts
  (`STRATA_SESSION_GROW=0` keeps the old tables). The text past 262,144 positions (dynamic RoPE) is unchanged.
- **The same text.** Every decode and prompt change is exact: the decode texts and the 16K / 64K prompt texts and
  first-token logits are bit-identical to 0.2.15-nvfp4.4's in the fixed-placement check. The new switches and their
  `=0`: [Switches](docs/fork/ENVIRONMENT.md#rtx-50-in-0215-nvfp45).
- **Update:** close the server. In a clone run `UPDATE.bat`; in the release folder run `update.cmd`. No new model
  files.

## 0.2.15-nvfp4.4

Released 2026-11-20: [release page](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.2.15-nvfp4.4).

- **Long prompts 17-20% faster again on RTX 50 (0.2.15-nvfp4.4).** int8 QSA scores, f16-accumulate dense GEMMs,
  CUDA graphs for the prompt's kernel chains, less hyper-connection traffic and the borrowed expert slots refilled
  beside decode. RTX 5090: 16K reads in 1.35 s instead of 1.69 s, 64K in 5.1 s instead of 6.2 s, 128K in 10.3 s
  instead of 12.5 s (Basalt in the same hour: 2.0 s at 16K, 6.3 s at 64K). Accuracy and decode unchanged
  ([measurements](docs/NVFP4.md#rtx-50-faster-prompts-2026-11-20-release-0215-nvfp44)).

- **Faster decode and prompts on RTX 50 (0.2.15-nvfp4.4).** Basalt's decode kernels and drafting
  ([Credits](README.md#credits)) plus kernels of our own, on by default on RTX 50 cards only. RTX 5090, tokens/s against
  0.2.15-nvfp4.3: a JSON answer 487 vs 330 (+48%), a chat 269 vs 219 (+23%), English prose 261 vs 205 (+27%),
  Ukrainian 241 vs 154 (+56%); Basalt in the same hour 475 / 248 / 276 / 235. These decode runs used Basalt's
  self-distilled draft layer, which now ships for the GPTQ set: setup, `UPDATE.bat` and `prepare-model.cmd` fetch it
  (0.12 GB, checked) into the model's `mtp\rt-tuned`; without it the drafts are accepted less often. A 64K prompt
  reads in 6.2 s instead of 7.8 s, 128K in 12.5 s instead of 14.5 s. Accuracy: KL 0.015 against the base engine,
  whose own runs differ by 0.012. `STRATA_NVFP4_SM120=0` turns all of it off
  ([measurements](docs/NVFP4.md#rtx-50-decode-and-prompt-kernels-after-basalt-2026-11-20-release-0215-nvfp44)).
- **Auto drafts with the MTP and Basalt's drafting (0.2.15-nvfp4.4).** Auto on CUDA, one request at a time and one card's layers, is now the MTP
  with Basalt's drafting for the Qwen3.8-Flash-Next sets: its two-stage draft head (`--mtp-head proxy`), its cumulative
  window rule (`--mtp-cum auto`) and a window of 8. `STRATA_AUTO_NEW_MTP=0` keeps the old Auto (DFlash2). Without the
  self-distilled draft layer Auto uses the set's own MTP layer, which accepts fewer drafts.
- **The release engine still runs on RTX 20, 30 and 40 cards (0.2.15-nvfp4.4).** It is built for sm_75, 86, 89 and 120;
  the new RTX 50 kernels take their older paths on the other cards. Each RTX 50 switch's own `=0` turns that one off, and
  a switch you set yourself is kept ([Switches](docs/fork/ENVIRONMENT.md#rtx-50-0215-nvfp44)).
- **Update:** close the server. In a clone `UPDATE.bat` updates and fetches the draft layer. In the release folder run
  `update.cmd`, then `prepare-model.cmd` once for the draft layer.

## 0.2.15-nvfp4.3

Released 2026-11-19: [release page](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.2.15-nvfp4.3).

- **KV streaming on by default (0.2.15-nvfp4.3).** On one GPU with a context of 131,072 tokens or more (the tray's
  512K), the engine keeps 32,768 cells of each attention layer's K/V in VRAM and all of it in pinned RAM (6.2 GiB at
  512K), so the VRAM the elastic K/V took from the experts as a dialog grew stays with them. RTX 5090, the same answer
  after each prompt: 274K 195 tokens/s against 180 (+8%), 128K 196 against 193, 32K 202 against 204, 2K 190 against
  193; a long prompt reads about 2% slower. The expert cache now stays one VMM range with streaming: the plain arena it
  fell back to made every GPU stage 20-60% slower under WDDM, which is why `--kv-resident` looked slower before
  (upstream #4815). `--no-kv-stream` turns it off
  ([measurements](docs/NVFP4.md#kv-streaming-on-by-default-2026-11-19-release-0215-nvfp43)).

## 0.2.15-nvfp4.2

Released 2026-11-18: [release page](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.2.15-nvfp4.2).

- **DFlash2 drafts for the NVFP4 GPTQ (0.2.15-nvfp4.2).** On a card that holds as many experts as an RTX 5090, Auto
  drafts with DFlash2 instead of the MTP layer: 215 tokens/s against 200 on the 5090 (+7.5%), 213 against 194 with the
  censorship switch on (+10%). Setup and `prepare-model.cmd` fetch the drafter (1.2 GB once). Smaller cards, NVIDIA's
  NVFP4 and the mix keep the MTP. Saving a session file with DFlash2 works now (upstream #5153)
  ([measurements](docs/NVFP4.md#dflash2-drafts-for-the-nvfp4-gptq-2026-11-18-release-0215-nvfp42)).

## 0.2.15-nvfp4.1

Released 2026-11-18: [release page](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.2.15-nvfp4.1).

- **On upstream 0.2.15 (0.2.15-nvfp4.1).** Upstream's engine, server and setup of 0.2.14 and 0.2.15 (the engine reports
  0.1.61) under this fork's NVFP4 models. One GPU and one dialog give the same bits as 0.2.13-nvfp4.1 at 2K and 32K;
  prompts and decode run at the same speed on the 5090 (32K: 8,800 tokens/s both; decode 11.71 ms a round against
  11.75 for a 0.2.13 build)
  ([the port and its checks](docs/NVFP4.md#upstream-0215-2026-11-18-release-0215-nvfp41)).
  - Upstream 0.2.14 builds its engine without llama.cpp's MMQ kernels and runs NVFP4 prompts on its own int8 kernels.
    This fork's release is still built with MMQ and keeps its NVFP4 prompt path: on the 5090 upstream's kernels gave
    other answers (they are not the same bits) and were not faster (32K prompt 2.6% slower, 128K 5% slower).
    `STRATA_PF_FUSED=1` selects upstream's kernels.
  - New in this fork: the hyper-connection read v5 (opt-in, `STRATA_GR_V5=1`): the same bits as the plain read, 0.2 ms
    off a decode round on the 5090.
  - Behaviour change from upstream's #5979: a tool argument `True` / `False` (any case) is a boolean where the tool
    declares one; with no schema, or where a string is allowed, the text stays text. This fork's Python booleans
    also still apply to `anyOf` / untyped parameters.
  - Setup: DeepSeek is gone, as upstream removed it.

## 0.2.13-nvfp4.1

Released 2026-11-09: [release page](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.2.13-nvfp4.1).

- **On upstream 0.2.13 (0.2.13-nvfp4.1).** Upstream's engine, server and setup of 0.2.1 to 0.2.13 (the engine reports
  0.1.59) under this fork's NVFP4 models. One GPU and one dialog give the same bits as 0.2.0-beta-nvfp4.3 at 2K and 32K;
  decode and a 32K prompt run at the same speed as 0.2.0-beta-nvfp4.3 on the 5090 (12.2 ms a round both;
  the 32K prompt 8,589 tokens/s both)
  ([the port and its checks](docs/NVFP4.md#upstream-0213-2026-11-09-release-0213-nvfp41)). New in this fork:
  - **#20 fixed:** a model copy that is page-locked slice by slice (a 64 GiB Mix, for example) no longer stops with
    "adaptive swap copy failed ... invalid argument". Each place in the copy that crosses a slice's edge gets its own
    page-locked copy, kept up to date as the adaptive tier moves experts in and out of it.
  - **#15 / #16:** the browser interface attached to a server (a tray's, for example) shows the server's tools. Its
    Monitor shows the server's live numbers, but asks for them only while the Monitor tab is open, and first asks the
    keyless `/health`. An idle page never wakes a sleeping model.
  - Upstream's new defaults that change answers and were not measured on NVFP4 stay at 0.2.0's behaviour: the PCIe
    share that follows the CPU's speed, the pool-sized PCIe start, the server's wider verify window, and the async
    adaptive tier that switches itself on. Each has its variable
    ([which](docs/NVFP4.md#upstream-0213-2026-11-09-release-0213-nvfp41)).
  - Behaviour change from upstream: a call to a tool the request did not declare now arrives as text;
    `"undeclared_tool_calls": "call"` restores the old behaviour. This fork's opt-in `tool_name_repair` gave way to
    upstream's tool-name aliases.
  - Setup: upstream's UD-Q3_K_XL (4 files) replaces this fork's 3-file one.

## 0.2.0-beta-nvfp4.3

Released 2026-10-20: [release page](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.2.0-beta-nvfp4.3).

- **0.2.0-beta-nvfp4.3: upstream 0.2.0's new interface, in your browser.** The release folder's `start-server.cmd` now also
  starts upstream Strata's app page on **http://127.0.0.1:8791** once the model is ready, and opens it: chat with
  pictures and the thinking switch, chats kept by the browser, skills, workspaces and the Monitor with the engine's own
  numbers. It runs on the folder's own `node\node.exe` and uses the server's model - it starts no engine of its own - and
  stops with the server. The old page stays at http://127.0.0.1:8080/. `start-ui.cmd <address>` opens it for a model
  something else runs (a tray); it never wakes a sleeping model except for a chat you send. A run config with
  `"app_ui": true` starts it with its server wherever that server is started from (a tray: the page lives while the
  tray's model is loaded), and `"app_ui_open": false` keeps the browser closed. Model downloads, tuning and
  updates stay where they are (`prepare-model.cmd`, `update.cmd`); upstream's desktop app is not part of it
  ([details](docs/NVFP4.md#the-new-interface-in-the-browser-2026-10-20-release-020-beta-nvfp43)).

## 0.2.0-beta-nvfp4.2

Released 2026-10-19: [release page](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.2.0-beta-nvfp4.2).

- **0.2.0-beta-nvfp4.2: this fork updates from this fork.** At start the server asks GitHub, at most once a day,
  which strata-nvfp4 release is the newest. When it is newer than the installed one, the server's window and the web
  page's header say so, with the command that updates the folder (`update.cmd` in a release folder, `UPDATE.bat` in a
  clone). Nothing is downloaded or installed by itself; `STRATA_UPDATE_CHECK=0` or `"update_check": false` turns the
  check off. `strata update` without `--feed` now runs this fork's updater instead of refusing; upstream's signed
  feed is used only when named. The engine is unchanged
  ([details](docs/NVFP4.md#strata-update-runs-this-forks-updater-the-start-says-when-a-newer-release-is-out-2026-10-20-release-020-beta-nvfp42)).

## 0.2.0-beta-nvfp4.1

Released 2026-10-19: [release page](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.2.0-beta-nvfp4.1).

- **On upstream 0.2.0-beta (0.2.0-beta-nvfp4.1).** Upstream's engine, server and setup of 0.1.45 and 0.2.0-beta
  (the engine says 0.1.45.3) under this fork's NVFP4 models; upstream's new app, Terminal and installers are not part
  of this release, which stays the Windows zip. One GPU and one dialog give the same bits as 0.1.44.1-nvfp4.2 at 32K
  and in the kit's identity runs, and decode ~8% faster on the 5090 (203 against 180 tokens/s, 5 of 5 pairs)
  ([the port and its checks](docs/NVFP4.md#upstream-020-beta-2026-10-19-release-020-beta-nvfp41)). What you get from
  upstream:
  - the prompt tokenised from the last shared prefix and cheaper conversation-cache copies (#567/#1852, #1163, #1164),
    `--conversation-cache-park-gain`, the model card's sampling applied by the server per thinking state (thinking:
    the 1.0 / 0.95 / 20 setup used to write; not thinking: Qwen's instruct set), pinned memory kept free under Windows
    while the expert arena pins, so the buffers after it never fail;
  - in setup: the Gyro family (27.6 and 35 GB), downloads over 8 connections that resume, every file pinned by size and
    SHA-256, the RAM + VRAM memory tiers for the GGUF default, Qwen3.6-35B-A3B with pictures and several requests;
  - our earlier pull requests in upstream's form (IMMA prompt attention, the scattered MoE input, the fused up
    projection, the tier's eviction sync): this fork keeps its defaults on them.
  This fork's own: setup no longer writes a `"sampling"` block (the server's model card does the same); `--family
  nvfp4 --model NVFP4-QAD` (upstream's NVIDIA FP4-Local, prepared on the PC) is not offered here; `strata update`
  (upstream's signed engine feed) needs a feed named explicitly, so an install of this fork never switches to
  upstream's engine by itself.

## 0.1.44.1-nvfp4.2

Released 2026-10-15: [release page](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.44.1-nvfp4.2).

- **0.1.44.1-nvfp4.2: Claude Code's system prompt is rewritten for the local model.** The server recognises Claude
  Code's prompt by its billing header and drops what only Claude can use: its identity, Anthropic's security
  policy, the list of Claude model ids, the pronoun rule. The memory how-to is cut to its path. In their place comes a
  short note for a local agent: call only the listed tools, by their exact names, never as plain text. The harness,
  context management, plan mode, output styles and every other section pass as sent. The rewrite gives the same text
  on every turn, so the conversation cache keeps hitting. Claude Code 2.1.296's prompt goes from 1,361 to 676 tokens.
  `"claude_code_prompt": "as_sent"` in the config keeps it as it comes. The engine is unchanged.

## 0.1.44.1-nvfp4.1

Released 2026-10-14: [release page](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.44.1-nvfp4.1).

This release also carries the prepared 0.1.44-nvfp4.1 and 0.1.43.1-nvfp4.1, which were not published on their own.

- **On upstream 0.1.44.1 (0.1.44.1-nvfp4.1).** Upstream's hotfix: CPUs without AVX2 start again (#2324: the NVFP4 CPU
  rows' tables were built before `main` with AVX instructions, an illegal instruction on such a CPU, now none), a
  crashed engine start is named instead of retried as a GPU runtime start, the Responses API (Codex) honours the
  config's `reasoning_effort`, and setup run again over an installed model keeps its size instead of moving it to the
  new recommendation (here also for Qwen3.6, whose re-run kept neither its size nor its family). One GPU and one
  dialog give the same bits as 0.1.44 at the same speed
  ([the port and its checks](docs/NVFP4.md#upstream-01441-2026-10-14-release-01441-nvfp41)). New in this fork since
  0.1.44:
  - **Kept dialogs give RAM back when it runs short:** below `--conversation-cache-min-free-mib` the parked dialogs
    leave RAM oldest first (to the disk tier when one is set), checked before every request and every 2 s while idle
    (on the 5090: three dialogs, 3.3 GB, released in 0.1 s).
  - **The low-RAM mode pins its RAM copy in slices** (Q8_0 on 128 GB: decode 53 tokens/s instead of 49, a 32K prompt
    in 8.5 s instead of 9.9) and keeps 1500 MiB of VRAM free under Windows for the desktop.
  - **`--rope-scaling dynamic` for contexts past 256K:** no scaling while a dialog fits the trained 262,144 positions
    (the same bits as none), yarn past them with the factor the length needs; past 262K as good as static yarn, which
    changes every short dialog too. Setup and Model settings write it on one NVIDIA GPU with one dialog at a time,
    static yarn elsewhere.
  - **Every Qwen3.6-35B-A3B size can be installed on any PC:** a size bigger than the RAM gets a note, not a question.
- **On upstream 0.1.44 (prepared as 0.1.44-nvfp4.1, published within 0.1.44.1-nvfp4.1), which ships this fork's NVFP4
  engine path.** Upstream's "NVFP4 packs" are a
  port of this fork's work: its NVFP4 experts, MMQ tails, mixed formats and CPU rows are upstream's code now, and this
  fork runs upstream's copy. Upstream's `--family nvfp4 --model NVFP4 | NVFP4-NVIDIA` install this fork's
  `qwen-nvfp4-gptq` / `qwen-nvidia-nvfp4` here, one install path
  ([the port and its checks](docs/NVFP4.md#upstream-0144-2026-10-14-release-0144-nvfp41)). One GPU and one dialog
  run as before: with the same expert cache, the default settings give the same bits as 0.1.43.1-nvfp4.1's engine.
  Upstream's changes that matter here:
  - **Unsloth's UD-Q5_K_XL, UD-Q6_K_XL and Q8_0** are new sizes of `--family unsloth`, and UD-Q4_K_XL is no longer
    experimental; images are on by default for them. Upstream writes `STRATA_HC_Q8=1` (and `STRATA_MMQ_KQ=1` for Q5/Q6)
    into their configs. This fork's UD-Q3_K_XL stays a size.
  - **Qwen3.6-35B-A3B** (`--family qwen36`, Unsloth's GGUFs with the MTP layer, text only so far) is a new family
    with its own engine path. It runs here exactly as on upstream (the same tokens and logits, bit for bit): on an
    RTX 5090 UD-Q4_K_XL keeps every expert in VRAM and answers at ~410 tokens/s with its MTP drafts, a 33K prompt in
    5.3 s. UD-Q6_K_XL does not fit a 32 GB card whole: 153 of its 256 experts per layer stay in VRAM, the rest
    run on the CPU, ~130-150 tokens/s. Setup recommends the largest size the RAM allows (Q8_0 from 64 GB); on a
    32 GB card UD-Q4_K_XL is the fast one.
  - **IQ3_S is the recommended GGUF size** from 64 GB of RAM (it was IQ3_XXS).
  - **A picture in a long conversation no longer loses the cached state** (#2228); **a long prompt gives way to a
    short request** at the next chunk (#2170); a quoted `<tool_call>` / `<tool_response>` in a message is text (#2026).
  - **Two GPUs:** the auto split keeps room on its fullest card (#1440); a slow link costs more only from a 4x gap
    (#2007); the VRAM check covers every card (#2192).
  - Opt-in: `--adapt-async 1` in the arena (#1904), `STRATA_PIN_BY_NEED=1` on Windows (#1905), `--prefill-head-alias`.
- **Also new since 0.1.42-nvfp4.2** (prepared as 0.1.44-nvfp4.1):
  - **Several GPUs (layer split, #7):** a prompt is read in chunks per stage so the stages work at the same time, a
    card on a slow link holds every expert of its layers, the CPU prompt share works on a split. On rented RTX 4090s
    a 32K prompt reads at 6,936 tokens/s on 2 cards (4,056 before; one card 5,472) and 9,703-9,997 on 4 (3,880
    before); a pair with one card on a ~2 GB/s link is no longer slower than its first card alone.
  - **Pinned memory on Windows:** on a split each GPU page-locks its own layers up to its adapter's budget, not 8 GiB
    for all; where little can be page-locked the copies run faster (this RTX 5090 with nothing pinned: a 32K prompt
    at 87-88% of the pinned speed, 70-71% before).
  - **Several dialogs at once (`"parallel"`):** a window graph out of VRAM no longer ends the engine, and the slots'
    K/V grows with their context instead of taking it all up front (4 slots at 262K: 2.8 GiB instead of 14.6; one 24
    GB card with 4 dialogs: 94 tokens/s in all where the engine ended before). `parallel` stays 1 by default.
  - **Dialogs parked on the disk** (opt-in `--conversation-disk-dir`, from xrip's upstream PR #1937): a dialog the RAM
    cache has no room for goes to a folder; a 100K dialog comes back in ~1 s instead of 11-12 s, also after the tray
    restarts the engine ([numbers](docs/NVFP4.md#dialogs-parked-on-the-disk-2026-10-13-branch-extrasdialog-park)).
    For one user `--conversation-cache-mib 8192` (4 dialogs in RAM) is enough; the disk tier is off by default.
  - **Model settings** has the dialogs and the context size (to 512K, with the RoPE extension past 262K); the
    controls are back inside the card.
  - The Windows zip ships `strata_kq.dll` (upstream's K-quant prompt kernels, for UD-Q5_K_XL / UD-Q6_K_XL and
    Qwen3.6).
- **On upstream 0.1.43.1 (prepared as 0.1.43.1-nvfp4.1, published within 0.1.44.1-nvfp4.1).** Upstream's changes
  that matter here ([the port and its checks](docs/NVFP4.md#upstream-01431-2026-10-13-release-01431-nvfp41)):
  - **Non-finite logits are caught** (#879, #1892): when a decode window's logits turn NaN, the server sends none of
    its tokens, drops the damaged caches and reads the prompt again from token 0, instead of answering "!". On by
    default (`STRATA_NAN_GUARD=0` turns it off); this fork's fresh engine after a "!" reply stays as a backstop.
  - **The PLE probe only warns** (#1984, #2040): a drive it measures slow keeps the direct reads
    (`STRATA_PLE_AUTO_MMAP=1` brings back 0.1.42's switch to mapped reads). Here it still runs once the expert arena
    is in, so the arena's own reads cannot make the SSD look slow.
  - **Windows: the stall watchdog no longer ends the engine** while it parks a conversation or waits for a draft with
    little free VRAM (#2049, #2065).
  - **A request's smaller prompt chunk no longer ends the engine** when its slot loan does not fit: the chunk is
    halved instead (#2090).
  - **Server fixes:** a quoted `</tool_call>` in a message stays text (#2026); a JSON tool call in a Markdown fence
    inside `<tool_call>` is read (#2037); a client that leaves while its pictures load stops the work (#1999).
  - **Unsloth's files in the RAM-budget mode on Windows** read the experts outside RAM in parallel: the GGUF views
    are closed once the file tier reads unbuffered (NTFS ran those reads one at a time).
  - **`BENCH.bat` / `./bench.sh`** (in a clone): measures your install through the HTTP API in about 10 minutes and
    writes a report you can share.
  - The route tail skip stays off for this fork's packs. For the GGUF models upstream now leaves it off when 90% or
    more of the experts are in VRAM (#1884). Upstream's adaptive overlap (`STRATA_ADAPT_OVERLAP=1`) is opt-in.
- **Also in 0.1.43.1-nvfp4.1:**
  - **The server's prompt encoder keeps one prompt per conversation** (#1852, by gputier): a second conversation,
    such as a Claude Code subagent beside the main one, no longer pushes out the main conversation's kept prompt, so
    its next turn is not tokenized from the start again.
  - **Not for WSL2:** setup now frames a warning under WSL2, in `--check` and at install: the driver there
    page-locks only about 1 GB of RAM ([Requirements](docs/fork/INSTALL.md#requirements), [#7](https://github.com/sergqwer/strata-nvfp4/issues/7)).
  - **Two GPUs (layer split, #7):** the expert arena loads on its own thread only with one GPU, so on a split it no
    longer takes page-locked memory before the second card's context exists; a refused hand-off buffer between the
    cards makes the prompt chunk smaller instead of stopping the start; and when the token embedding cannot be
    page-locked and the draft head runs on the other card, the engine says so at start instead of failing later with
    an illegal memory access; a later card's PCIe share starts from NVFP4's 0.25 like the first card's, not 0.55.
    One GPU runs exactly as before.
  - **Tool calls:** Python-style `True` / `False` become JSON booleans where the schema allows one (jarydkrish); a
    call missing its `</tool_call>` ends at its own `</function>` (upstream PR #1881, LeGeRyChEeSe); a quoted
    `</tool_response>` stays text (upstream PR #1840, Mert Özkan); opt-in `"tool_name_repair": true` maps a call to an
    undeclared tool onto the declared one it meant (ANBAL534).
  - **Image URLs** are fetched only from public addresses, re-checked on every redirect (shoftee); `data:` URLs and
    local files work as before.
  - **A NaN stays a NaN** in the PLE and prompt path's BF16 conversions, so the NaN guard sees it (sebastianrcnt);
    every other value converts to the same bits.
  - **A stop token the model writes as text no longer cuts the reply:** asked to write `<|im_end|>`, the model
    writes the token itself; where no turn can end (a tool call's argument, after an opening backtick or quote, after
    a space) it is now that token's text and the reply goes on (`"stop_token_text": false` turns it off).
  - **Parallel dialogs** (`--batch` with `--conversation-cache-mib`): conversation-cache and batch fixes from upstream
    pull requests #1503, #1163, #1164 (s.anastasiou, BlueKingMuch) and opt-in `--prompt-cache-spread` (#1959, xrip)
    ([Several dialogs at once](docs/NVFP4.md#several-dialogs-at-once-2026-10-13-branch-extrasparallel-dialogs)).
    One dialog runs as before.
  - **A warning** when `--max-context` is past the model's trained 262,144 tokens without rope scaling.
  - **The Monitor shows every GPU:** its role (layer-split stage, peer tier, helper cache, image encoder), VRAM, load,
    temperature, power and PCIe link (current / max, with the engine's measured bandwidth); the Disk card reads the
    model's drives. The Windows bundle's server env gets `psutil` (`update.cmd` installs it).

## 0.1.42-nvfp4.2

Released 2026-10-11: [release page](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.42-nvfp4.2).

- **Two more exact models for a 128 GB PC, and Unsloth's files faster (0.1.42-nvfp4.2).**
  - **`--family qwen-nvfp4-mix`**: our GPTQ NVFP4 with Q8_0 experts in 38 of its 48 layers. It picks Q8_0's top token
    95.0% of the time (GPTQ: 93.6%), at about half GPTQ's speed (91-95 tokens/s on an RTX 5090). Every expert stays
    in RAM: 106 GiB in use with images and the server.
  - **`--family qwen-q8`**: every routed expert Q8_0, the reference the others are measured against. 46-47 tokens/s,
    a 32K prompt in 10.3 s. Its 119.5 GiB of experts do not fit 128 GB beside Windows, so it runs in the low-RAM mode
    (100 GiB of them locked in RAM).
  - Both are the original, censored Qwen, with the censorship switch: 5 of 5 held-out harmful requests refused with it
    off, 0 of 5 with it on. Both share qwen-nvfp4-gptq's dense weights, n-gram table, embedding and draft head; an
    install after it downloads only the experts.
  - Below 128 GB, setup asks before installing them: they would run in the low-RAM mode, which was not measured with
    them.
  - **The low-RAM mode also follows the pack's size:** from 92 GiB installed, it now starts when the experts and 16 GiB
    more do not fit. In that case it leaves 10 GiB free, not 6. Below 92 GiB nothing changes.
  - **The route tail skip is off by default for all of this fork's packs**: NVFP4, Q8_0 and the mix. Before, only
    packs with NVFP4 experts left it off, so the all-Q8_0 pack would have had it on. The GGUF models keep upstream's
    default.
  - **Unsloth's files keep every expert in RAM where they fit with 16 GiB more** (a 96 GB PC for UD-Q4_K_XL). On a
    128 GB PC, UD-Q4_K_XL now decodes 169 tokens/s and reads a 32K prompt in 4.9 s. Setup's old RAM-budget mode gave
    118 tokens/s and 30 s, and it used more RAM (98.5 against 77.3 GiB). A 64 GB PC keeps the budget mode. To get the
    new config on an existing install, run setup again (`START-HERE.bat --setup`).
  - **Unsloth's UD-Q3_K_XL** is a size of `--family unsloth`, as a user asked. It picks Q8_0's top token 90.7% of the
    time, at 208 tokens/s and a 3.8 s 32K prompt, with every expert in RAM (57.5 GiB in use).

## 0.1.42-nvfp4.1

Released 2026-10-10: [release page](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.42-nvfp4.1).

- **On upstream 0.1.42 (0.1.42-nvfp4.1).** Upstream's own changes now run here too:
  - **The route tail skip** (`STRATA_ROUTE_TAIL_SKIP=7`: a missed expert that every token of a verify window routes
    at rank 7-9 is neither fetched nor computed) is upstream's default for the GGUF models, and **off by default for
    the NVFP4 models**. On `qwen-nvfp4-gptq` it moves the answers by teacher-forced KL 0.035 against the skip off
    (top token the same 93.7% of the time), as much as the whole GPTQ quantization error against an all-Q8_0
    reference (0.038), and thinking ends sooner (p(`</think>`) early x1.35). It makes decode 11.5% faster (193.7
    against 174.0 tokens/s): `"env": {"STRATA_ROUTE_TAIL_SKIP": "7"}` in the model's config turns it on
    ([numbers](docs/NVFP4.md#upstream-0142-2026-10-10-release-0142-nvfp41)).
  - **Decode's PCIe share is upstream's measured one** (from the CPU pool's speed: 0.35 here); this fork's per-layer
    cost model is gone. The same speed as the old fixed share.
  - Four fixes this fork sent upstream are upstream's code now: the unbuffered start of the resident RAM mode, the
    page-file advice (a fixed 65536 MB), the K/V trim's residency upload, the prompt buffers' size.

## 0.1.41-nvfp4.7

Released 2026-10-10: [release page](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.41-nvfp4.7).

- **Model settings has a Disable censorship switch too (0.1.41-nvfp4.7):** the default for requests that do not
  say, saved in the run config (`"uncensored"`) and used from the next start; greyed out for a model without the
  refusal projection (NVIDIA's, Swift, the Coder).

## 0.1.41-nvfp4.6

Released 2026-10-10: [release page](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.41-nvfp4.6).

- **Recommended sampling (0.1.41-nvfp4.6).** Setup writes the model's own `temperature 1.0, top_p 0.95, top_k 20`
  into the config, as the default for requests that send none. Before, Claude Code decoded greedy.

## 0.1.41-nvfp4.5

Released 2026-10-10: [release page](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.41-nvfp4.5).

- **Censorship can be turned off (0.1.41-nvfp4.5).** The original Qwen (`qwen-nvfp4-gptq`, and the original Qwen's
  GGUFs `qwen` and `unsloth`) now runs with or without censorship. The switch is off by default. To turn it on:
  - for one chat: the web app's **"Disable censorship"** switch;
  - for one request: the API field `"uncensored": true`;
  - as the default: install with `START-HERE.bat --uncensored on`, or set **Disable censorship by default** in the
    web page's Model settings (0.1.41-nvfp4.7; used from the model's next start).

  The engine removes the model's refusal direction at run time in layers 8-33 only; the weights are untouched.
  - With it on and thinking on, 0 of 104 English and 0 of 30 Ukrainian test requests were refused.
  - The model thinks as long as the original: 1,947 against 1,699 tokens on a long Claude Code request.
  - Never for NVIDIA's model: its license forbids it.
  - [Details](docs/fork/INSTALL.md#quick-start).
- **orca-nvfp4 is removed from the fork (0.1.41-nvfp4.5).** OrcaRouter's uncensored model, the former default, is no
  longer offered.
  - Its weight edit made it a much weaker agent. On the same Claude Code task it took 10-70 steps against the
    original's 175 and gave a clearly worse result. It thinks ~40% less.
  - Instead, use `--family qwen-nvfp4-gptq --uncensored on`: the original's weights, without the censorship.
  - An existing orca install keeps working.

## Older releases

These releases came before this changelog's entries. Each line is the release's own title; the release page has its notes.

- [0.1.41-nvfp4.4](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.41-nvfp4.4) (2026-10-09): The original, censored Qwen in NVFP4 - NVIDIA's checkpoint converted, or our GPTQ. [Measurements](docs/NVFP4.md#the-original-qwen-in-nvfp4-2026-10-09-release-0141-nvfp44).
- [0.1.41-nvfp4.3](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.41-nvfp4.3) (2026-10-09): huihui-nvfp4 withdrawn (it loops in long thinking), orca-nvfp4 the default, the page file counted for sure. [Measurements](docs/NVFP4.md#huihui-nvfp4-withdrawn-the-page-file-counted-for-sure-2026-10-09-release-0141-nvfp43).
- [0.1.41-nvfp4.2](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.41-nvfp4.2) (2026-10-09): A 96 GB PC keeps every expert in RAM, the low-RAM mode starts 3x faster, the page file is checked. [Measurements](docs/NVFP4.md#a-96-gb-pc-2026-10-09-release-0141-nvfp42).
- [0.1.41-nvfp4.1](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.41-nvfp4.1) (2026-10-09): Upstream 0.1.41, setup installs our NVFP4 GPTQ quants of huihui and OrcaRouter. [Measurements](docs/NVFP4.md#upstream-0141-2026-10-08-release-0141-nvfp41).
- [0.1.40.3-nvfp4.1](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.40.3-nvfp4.1) (2026-10-08): Upstream 0.1.40.3 and a second round - decode +9-10%, a 32K prompt -12%, the tier's eviction fix. [Measurements](docs/NVFP4.md#a-second-round-of-kernels-2026-10-08-release-01403-nvfp41).
- [0.1.40.2-nvfp4.1](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.40.2-nvfp4.1) (2026-10-07): Upstream 0.1.40.2 - its CPU share with a gate that shares only where it pays, the chunked recurrence from 128 tokens. [Measurements](docs/NVFP4.md#upstream-01402-2026-10-07-release-01402-nvfp41).
- [0.1.40-nvfp4.3](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.40-nvfp4.3) (2026-10-07): FP4 tensor cores for the experts, INT8 prompt attention, chunked recurrence - a 32K prompt 8% faster. [Measurements](docs/NVFP4.md#a-round-of-kernels-2026-10-07-release-0140-nvfp43).
- [0.1.40-nvfp4.2](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.40-nvfp4.2) (2026-10-06): Upstream 0.1.40.1 - upstream's fix for quoted tool calls, restart fixes. [Measurements](docs/NVFP4.md#upstream-01401-2026-10-07-release-0140-nvfp42).
- [0.1.40-nvfp4.1](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.40-nvfp4.1) (2026-10-06): On upstream 0.1.40, a quoted tool call stays text, image tokens per picture. [Measurements](docs/NVFP4.md#on-upstream-0140-2026-10-06-release-0140-nvfp41).
- [0.1.39-nvfp4.4](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.39-nvfp4.4) (2026-10-06): Small prompt chunks shared with the CPU, AVX2 CPUs, measured CPU and PCIe shares. [Measurements](docs/NVFP4.md#small-prompt-chunks-with-the-cpu-2026-10-06-release-0139-nvfp44).
- [0.1.39-nvfp4.3](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.39-nvfp4.3) (2026-10-05): A peer GPU's cache on its own card, #594's current request drain. [Measurements](docs/NVFP4.md#a-peer-gpu-and-the-request-drain-2026-10-06-release-0139-nvfp43).
- [0.1.39-nvfp4.2](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.39-nvfp4.2) (2026-10-04): A fresh engine after NaN logits, the q8_1 clamp completed. [Measurements](docs/NVFP4.md#-after-a-long-conversation-2026-10-05-release-0139-nvfp42).
- [0.1.39-nvfp4.1](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.39-nvfp4.1) (2026-10-04): On upstream 0.1.39 (7% shorter decode rounds, same answers). [Measurements](docs/NVFP4.md#on-upstream-0139-2026-10-04-release-0139-nvfp41).
- [0.1.38-nvfp4.2](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.38-nvfp4.2) (2026-10-03): 700 MiB VRAM reserve again, faster tokenising of long conversations. [Measurements](docs/NVFP4.md#other-peoples-pull-requests-measured-here-2026-10-03-release-0138-nvfp42).
- [0.1.38-nvfp4.1](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.38-nvfp4.1) (2026-10-03): On upstream 0.1.38 (same answers, same speed). [Measurements](docs/NVFP4.md#on-upstream-0138-2026-10-03-release-0138-nvfp41).
- [0.1.37-nvfp4.2](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.37-nvfp4.2) (2026-10-02): Fixes from a code review (1.25 GiB less VRAM borrowed, elastic K/V, image path). [Measurements](docs/NVFP4.md#fixes-from-a-code-review-2026-10-03-release-0137-nvfp42).
- [0.1.37-nvfp4.1](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.37-nvfp4.1) (2026-10-02): On upstream Strata 0.1.37: a frozen engine no longer hangs the server. [Measurements](docs/NVFP4.md#on-upstream-0137-2026-10-02-release-0137-nvfp41).
- [0.1.36-nvfp4.1](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.36-nvfp4.1) (2026-10-02): On upstream Strata 0.1.36; pictures opened by an agent reach the model. [Measurements](docs/NVFP4.md#on-upstream-0136-2026-10-02-release-0136-nvfp41).
- [0.1.35-nvfp4.1](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.35-nvfp4.1) (2026-10-02): On upstream Strata 0.1.35, same answers and speed. [Measurements](docs/NVFP4.md#on-upstream-0135-2026-10-02-release-0135-nvfp41).
- [0.1.32-nvfp4.2](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.32-nvfp4.2) (2026-10-01): A more accurate experts pack re-quantized from BF16 (GPTQ + Q8_0 down). [Measurements](docs/NVFP4.md#re-quantized-from-bf16-2026-10-02-release-0132-nvfp42).
- [0.1.32-nvfp4.1](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.32-nvfp4.1) (2026-10-01): On upstream Strata 0.1.32, same answers and speed. [Measurements](docs/NVFP4.md#on-upstream-0132-2026-10-01-release-0132-nvfp41).
- [0.1.31-nvfp4.3](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.31-nvfp4.3) (2026-10-01): A longer-memory VRAM tier; audit fixes. [Measurements](docs/NVFP4.md#the-adaptive-vram-tier-and-an-audit-2026-10-01-release-0131-nvfp43).
- [0.1.31-nvfp4.2](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.31-nvfp4.2) (2026-10-01): The K/V grows with the context; 137 tok/s short chat, 6,500 tok/s 32K prompts. [Measurements](docs/NVFP4.md#faster-without-changing-an-answer-2026-10-01-release-0131-nvfp42).
- [0.1.31-nvfp4.1](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.31-nvfp4.1) (2026-10-01): On upstream 0.1.31; 64 GB answers 97 tok/s after a 32K prompt. [Measurements](docs/NVFP4.md#on-upstream-0131-2026-10-01-release-0131-nvfp41).
- [0.1.28-nvfp4.4](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.28-nvfp4.4) (2026-09-30): Steady decode on a desktop GPU.
- [0.1.28-nvfp4.3](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.28-nvfp4.3) (2026-09-30): Faster drafting after long prompts.
- [0.1.28-nvfp4.2](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.28-nvfp4.2) (2026-09-30): Runs on 64 GB of RAM. [Measurements](docs/NVFP4.md#64-gb-of-ram-2026-09-30).
- [0.1.28-nvfp4.1](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.28-nvfp4.1) (2026-09-30): Upstream 0.1.28, pictures, Cyrillic drafts, RTX 20-50. [Measurements](docs/NVFP4.md#upstream-0128-merged-2026-09-30).
- [0.1.24-nvfp4.3](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.24-nvfp4.3) (2026-09-29): Precision audit and a speed pass. [Measurements](docs/NVFP4.md#where-precision-is-still-lost-audit-2026-09-29).
- [0.1.24-nvfp4.2](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.24-nvfp4.2) (2026-09-29): The n-gram table in FP8, as shipped. [Measurements](docs/NVFP4.md#the-n-gram-ple-table).
- [0.1.24-nvfp4.1](https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.1.24-nvfp4.1) (2026-09-29): Windows, RTX 50-series.
