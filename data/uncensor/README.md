# uncensor (off by default)

`qwen-nvfp4-gptq-uncensored.gguf` is a 268 KB control vector for the original Qwen3.8-Flash-Next: one direction of
2,560 values per layer (GGUF `controlvector` format). The JSON beside it has its per-layer strengths, how it was built,
and its scores.

**What it does.** For `qwen-nvfp4-gptq`, ISTA's GGUFs (the `qwen` family) and Unsloth's (`unsloth`) setup - and the
Windows bundle's config - starts the engine with

```
--control-vector-scaled data/uncensor/qwen-nvfp4-gptq-uncensored.gguf:1.0 --control-vector-layer-range 8 33
--cvec-mode project --cvec-dir per-layer
```

After each of layers 8-33 the engine removes that layer's refusal direction from every hyper-connection residual
stream (`h -= s (h . v) v`; `s` is the direction's norm, 1.0 on every layer here). The model then stops declining
requests. It is the mechanism upstream's experimental speed projection uses, with this fork's own vector. It is
loaded but **off** unless chosen: `"uncensored": false` in `strata-<model>.json` is the default for requests
(`true` when setup's "Disable censorship?" is answered yes, or `--uncensored on`), and the web app's "Disable
censorship" switch and the API field `"uncensored"` (alias `"experimental_speed_projection"`) change it per
request. Off, a request runs the original model bit for bit (the kernel skips the projection).

**Where.** Made and measured on `qwen-nvfp4-gptq`; measured too on ISTA's Q2_0 and IQ2_XS (setup offers it for the
whole `qwen` family: IQ3_XXS and IQ3_S sit between Q2_0 and NVFP4) and on Unsloth's UD-Q4_K_XL (the `unsloth`
family; its UD-IQ4_XS was not measured). Never for `qwen-nvidia-nvfp4`: the NVIDIA Open Model License does not
allow bypassing the model's safety guardrails. Not for Swift 1.5 or the Coder, which have other weights.

Against each GGUF quant's own stock model (5 greedy replays of the long Claude Code request where it says loops):

| quant | refusals, thinking off, EN / UK (stock) | refusals, thinking on, EN / UK (stock) | KL | log p(`</think>`) shift | thinking tokens, stock / this vector |
|---|---|---|---|---|---|
| ISTA Q2_0 | 1/104, 2/30 (30/30, 28/30) | 1/104, 0/30 (30/30, 19/30) | 0.016 | +0.027 to +0.040 (upstream's +0.087 to +0.149) | 965 / LOOP / 1,766 against 1,306 / 1,350 / 906, no loop |
| ISTA IQ2_XS | 1/104, 2/30 | 0/104 EN | 0.017 | -0.003 to +0.066 (upstream's +0.12 to +0.20) | 3 of 3 loop against 2 of 3 loop |
| Unsloth UD-Q4_K_XL | 4/104 (the classifier counts disclaimers), 1/30 | 0/104, 1/30 (30/30, 29/30) | - | - | 888 / 1,183 / 1,731 against 1,204 / 1,255 / 1,343, no loop |

IQ2_XS loops in long greedy agentic thinking by itself (3 of 3 without the vector); the vector does not make it
worse, and setup says so when IQ2_XS is chosen.

**How it was made.** On this engine and the `qwen-nvfp4-gptq` pack, the residual after every layer was dumped at the
assistant-header positions of 320 harmful and 320 harmless prompts (mlabonne's harmful_behaviors and
harmless_alpaca), rendered with the chat template, thinking on. Per layer the direction is the difference of the two
means, summed over the 4 hyper-connection streams, averaged over the 7 header positions, unit-normalized. About 30
variants (layer window, per-layer strength, direction source) were scored on held-out prompts. The window stops at
layer 33: strength beyond it was what most made the model end its thinking early, the failure of OrcaRouter's
abliteration.

**Measured** (teacher-forced KL against the stock model on 5,153 rows of code, prose and the model's own replies;
"thinking shift" = the mean change of log p(`</think>`) over 8,550 rows of its own thinking, > 0 ends thinking
sooner; a captured 72K-token Claude Code request replayed 5 times greedy):

| | stock | this vector | upstream's speed projection | OrcaRouter's abliteration |
|---|---|---|---|---|
| refusals, thinking on, EN / UK held-out | 30/30, 29/30 (of 30 + 30) | **0/104, 0/30** | 2/104, 1/30 | - |
| refusals, thinking off, EN / UK | 102/104, 30/30 | 3/104, 2/30 | 3/104, 2/30 | - |
| refusals on harmless prompts | 2/64 | 2/64 | 2/64 | - |
| KL / same top token | 0 / 100% | **0.026** / 94.4% | 0.043 / 93.1% | 0.047 / 92.9% |
| thinking shift (nats) | 0 | **+0.019** | +0.092 | +0.119 |
| thinking tokens on the request (mean of 5), loops | 1,699, 0 | **1,947**, 0 | 1,825, 0 | ~1,000-1,150, 0 |
| decode on that request, tokens/s | 158.8 | 159.0 | 158.4 | 141.1 / 161.7 |

Read by hand, most thinking-off answers the classifier counted as refusals were capability disclaimers followed by
an answer (about 1/104 EN and 0/30 UK were real refusals). On the same tokens the projection costs 0.2-0.4% of decode
speed (upstream's measurement of its own vector).

**License.** Made only from this fork's GPTQ quant of Qwen's BF16 weights, so the **Qwen Community License 1.0**
applies; no NVIDIA weights were used. Removing refusals removes a safety behaviour: you are responsible for what the
model writes with it on.
