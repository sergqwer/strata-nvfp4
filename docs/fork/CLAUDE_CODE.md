# Claude Code

How to point Claude Code at this fork's server. Back to the [README](../../README.md).

```bat
set ANTHROPIC_BASE_URL=http://127.0.0.1:8080
set ANTHROPIC_API_KEY=local
set ANTHROPIC_MODEL=strata-nvfp4
set ANTHROPIC_SMALL_FAST_MODEL=strata-nvfp4
claude
```

The server answers under its own model name whatever the request names. The model is hybrid: its Gated DeltaNet
layers keep a recurrent state that cannot be cut back to a position, so the server keeps a checkpoint at every turn
boundary, and an agent's next turn reads only its new tokens. An edit in the middle of a history falls back to the
checkpoint just before it. `/v1/messages/count_tokens` is served. In `strata-<family>.json`:

- `"anthropic_thinking": "on_request"`: a request that does not ask for thinking (Claude Code's small helper calls)
  gets none;
- `"fit_max_tokens": true`: a request whose `max_tokens` would run past the context is shortened to the room left
  instead of refused ([docs/DETAILS.md](../DETAILS.md#using-it)).

## What the server does for Claude Code

- It rewrites Claude Code's system prompt for the local model (since 0.1.44.1-nvfp4.2):
  `"claude_code_prompt": "as_sent"` in the config keeps it as it comes
  ([Changelog](../../CHANGELOG.md#01441-nvfp42)).
- Its prompt encoder keeps one prompt per conversation, so a subagent beside the main conversation does not push
  out the main one's kept prompt (0.1.43.1-nvfp4.1).
- A picture a tool returns (Claude Code's `Read`) reaches the image encoder ([Features](FEATURES.md#what-this-fork-adds)).
- With the censorship switch on, the model thinks as long as the original on a long Claude Code request
  ([Quick start](INSTALL.md#quick-start)).
