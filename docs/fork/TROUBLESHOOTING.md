# Troubleshooting

Problems seen with this fork and what to send us. Back to the [README](../../README.md). Upstream's own page is
[docs/TROUBLESHOOTING.md](../TROUBLESHOOTING.md).

## If a reply turns into "!"

A long Claude Code conversation once turned into "!" mid-reply (token 0, what the sampler gives when no logit is
finite), and the cause is not found yet
([docs](../NVFP4.md#-after-a-long-conversation-2026-10-05-release-0139-nvfp42)). Since 0.1.44.1-nvfp4.1
upstream's guard (#879) catches non-finite logits first: the window's tokens are not sent, the caches are dropped, the
prompt is read again and the engine log has a `non-finite logits (#879)` line. The server still starts a fresh engine
at the next request after a reply of "!". To find out what was damaged, give the engine `STRATA_BANG_AUDIT=1` and
`STRATA_NAN_GUARD=0` (`"env": {"STRATA_BANG_AUDIT": "1", "STRATA_NAN_GUARD": "0"}` in `strata-<family>.json`; with
the guard on, a reply never turns into "!", so the audit never runs). It costs nothing until a reply turns into token 0;
then, once, it compares every expert in the VRAM cache with its RAM copy and with `experts.bin` and logs what
differs (`strata kvg audit` lines). Please send us the engine log (`strata-<family>.log`) from the engine's start to
the end of those lines ([Upstream 0.1.41](../NVFP4.md#upstream-0141-2026-10-08-release-0141-nvfp41)).

## Other places to look

- WSL2, the page file and the RAM the engine needs: [Requirements](INSTALL.md#requirements).
- `large pages refused ... VirtualAlloc error 1314` or error 1450 at start: [Large pages](INSTALL.md#large-pages-why-a-reboot).
- Switches that print what the engine does (`STRATA_SPEC_STATS=1`, `STRATA_REQUEST_LINES=1`, `STRATA_VERIFY_ARENA=1`):
  [Switches](ENVIRONMENT.md).
- Bugs: [github.com/sergqwer/strata-nvfp4/issues](https://github.com/sergqwer/strata-nvfp4/issues), with the engine log
  (`strata-<family>.log`).
