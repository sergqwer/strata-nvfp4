"""serve/runconfig.py - #564: the web page's Settings view of the run config (strata-<model>.json).

A short list of documented keys can be read and changed from Strata's own page (GET / POST /config).  Everything
else in the file - the keys setup writes, the network and security keys (host, api_key, cors_origins,
trusted_origins, allowed_hosts), the MCP servers and before_load (which run programs), sampling keys not listed
here - is kept as it is: a change touches only the keys it names (as setup run again keeps the user's keys, #629).
The earlier file is kept as strata-<model>.json.bak.  The server reads the config when it starts, so a change is
used from the next start on.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

# (key, kind, help).  kind: "bool", "int>=0", "num>=0", ("enum", values), "names", or ("sampling", check) for a key
# of the "sampling" block, ("arg", flag) for an engine option kept in "args" (a whole number; ("arg", flag, "text"): a
# string, ("arg", flag, "flag"): an option without a value, on or absent), ("count", lo, hi) for a whole number the
# page offers as a list.
EDITABLE = [
    ("sampling.temperature", ("sampling", "num>=0"),
     "Default temperature for requests that send none (0 = greedy, the default without a sampling block)"),
    ("sampling.top_p", ("sampling", "0<x<=1"), "Default top_p for requests that send none"),
    ("sampling.top_k", ("sampling", "1..64"), "Default top_k for requests that send none (1-64)"),
    ("sampling.min_p", ("sampling", "0<=x<=1"), "Default min_p for requests that send none"),
    # this fork: the censorship switch's default (setup writes it beside the refusal projection it loads)
    ("uncensored", "bool",
     "Disable censorship by default: the refusal projection (layers 8-33) on for requests that do not say; a chat's "
     "Disable censorship switch or the API field \"uncensored\" still decide per request. Only where setup loaded "
     "it (the original Qwen), never for NVIDIA's model"),
    ("reasoning_budget_tokens", "int>=0", "Cap the thinking of every request at this many tokens (0 or empty: no cap)"),
    ("fit_max_tokens", "bool", "Shorten a max_tokens that does not fit the context instead of answering 400"),
    ("anthropic_thinking", ("enum", ["model", "on_request"]),
     "Anthropic requests that do not ask for thinking: think as the model does (model) or not (on_request)"),
    ("effort_position", ("enum", ["start", "end"]),
     "Where a non-default reasoning effort goes: start (the default) or end (keeps the cache when it changes)"),
    ("aliases", "names", "Other model names the server lists and answers to (comma-separated)"),
    ("idle_unload_s", "num>=0", "Unload the model after this many seconds without requests (0 or empty: never)"),
    ("lazy_load", "bool", "Start without loading the model; the first request loads it (text only)"),
    ("engine_silence_s", "num>=0", "End a request when the engine says nothing for this long (default 300 s, 0 = wait)"),
    ("api_monitor", "bool", "Keep the last 100 requests' prompts and answers in memory for /api-monitor"),
    ("open_browser", "bool", "Open the chat page in the browser when the model is ready"),
    ("vram_reserve_mib", ("arg", "--vram-reserve-mib"),
     "VRAM in MiB the engine leaves free for other programs (engine default 700)"),
    # this fork: dialogs at once (server.parallel_args: "parallel": N -> --batch N; 1 or absent: one at a time)
    ("parallel", ("count", 1, 8),
     "Dialogs that run at once (the engine's batch slots; more wait their turn). Empty: 1 at a time, the fastest for "
     "a dialog that runs alone. Each one more slows it: one RTX 5090 with 2 at once 158.6 instead of 175.8 tok/s "
     "(-10%); a layer split over 2 GPUs with 4 at once 95.6 instead of ~147 tok/s (-35%)"),
    # this fork: dialogs kept between requests - the engine's conversation cache and its disk tier
    ("conversation_cache_slots", ("arg", "--conversation-cache-slots"),
     "Dialogs kept in RAM between requests: at most this many (engine default 4). Needs the RAM below; without it "
     "only the last dialog is kept, and another one is read again from its start"),
    ("conversation_cache_mib", ("arg", "--conversation-cache-mib"),
     "RAM for the kept dialogs, in MiB (empty or 0: off). Measured per dialog: 0.5 GB at 8K tokens, 1 GB at 32K, "
     "2.2 GB at 100K; 16384 kept 8 agents' dialogs (back in 0.2-0.6 s instead of 1-11 s read again)"),
    ("conversation_disk_dir", ("arg", "--conversation-disk-dir", "text"),
     "Keep dialogs on the disk too: a folder on a fast SSD (empty: off). A dialog the RAM above has no room for is "
     "written there and read back (100K tokens: 1 s instead of 11-12 s read again). Needs the RAM above"),
    ("conversation_disk_mib", ("arg", "--conversation-disk-mib"),
     "Disk space for those dialogs, in MiB (engine default 32768)"),
    ("conversation_disk_persist", ("arg", "--conversation-disk-persist", "flag"),
     "Keep the dialogs on the disk for the next start (off: removed when the model stops, and after 30 minutes). "
     "Saved after 60 s without requests, so closing Strata's window loses at most what came after"),
]
SPEC = {k: kind for k, kind, _ in EDITABLE}
# the view's headings: the first key of each group -> its title
SECTIONS = {"sampling.temperature": "Sampling defaults", "reasoning_budget_tokens": "Requests",
            "idle_unload_s": "Loading and the engine", "parallel": "Dialogs"}
# what an empty field means, where "default" says too little (the page's placeholder / first choice)
DEFAULTS = {"conversation_cache_slots": "default: 4", "conversation_cache_mib": "off", "conversation_disk_dir": "off",
            "conversation_disk_mib": "default: 32768", "conversation_disk_persist": "default: off"}
DIALOG_KEYS = ("conversation_cache_slots", "conversation_cache_mib", "conversation_disk_dir", "conversation_disk_mib",
               "conversation_disk_persist")
NO_PROJECTION = ("this model loads no refusal projection (--control-vector-scaled): setup loads it for the original "
                 "Qwen's families only, never for NVIDIA's model (its license does not allow bypassing the model's "
                 "safety guardrails), nor for Swift 1.5 or the Coder")


def has_projection(cfg: dict) -> bool:
    """The engine is started with a control vector (the censorship switch's refusal projection)."""
    a = cfg.get("args") if isinstance(cfg.get("args"), list) else []
    return any(str(x) in ("--control-vector-scaled", "--control-vector") for x in a)


def _arg(cfg: dict, flag: str, sub: str = "int>=0"):
    a = cfg.get("args") if isinstance(cfg.get("args"), list) else []
    if sub == "flag":
        return True if flag in a else None
    if flag in a[:-1]:
        v = str(a[a.index(flag) + 1])
        return int(v) if v.isdigit() and sub != "text" else v
    return None


def _sub(kind) -> str:
    """An "arg" key's value kind: "int>=0" (the default), "text" or "flag"."""
    return kind[2] if len(kind) > 2 else "int>=0"


def value_of(cfg: dict, key: str):
    kind = SPEC[key]
    if isinstance(kind, tuple) and kind[0] == "sampling":
        s = cfg.get("sampling")
        return s.get(key.split(".", 1)[1]) if isinstance(s, dict) else None
    if isinstance(kind, tuple) and kind[0] == "arg":
        return _arg(cfg, kind[1], _sub(kind))
    return cfg.get(key)


def gpu_count(cfg: dict) -> int:
    """The cards the config names (server.gpu_list: "gpu": 2, [0, 1] or "0,1"), at least 1."""
    g = cfg.get("gpu")
    items = g if isinstance(g, (list, tuple)) else [] if g is None else str(g).split(",")
    return max(1, sum(str(x).strip() != "" for x in items))


def _count_view(cfg: dict, key: str, kind) -> dict:
    """A "count" key as a list: lo..hi (and a value outside it the file already has); for "parallel" the choice that
    is one dialog per GPU says so, and an engine option that sets it already (--batch / --slots) greys it out."""
    v, gpus = value_of(cfg, key), gpu_count(cfg)
    ns = list(range(kind[1], kind[2] + 1)) + ([v] if isinstance(v, int) and not isinstance(v, bool) and
                                              not kind[1] <= v <= kind[2] else [])
    out = {"kind": "choice", "default": "default: 1 at a time",
           "choices": [[n, ("1 at a time" if n == 1 else f"{n} at once") +
                        (f" - 1 per GPU ({gpus} GPU{'s' if gpus > 1 else ''})" if n == gpus else "")] for n in ns]}
    a = cfg.get("args") if isinstance(cfg.get("args"), list) else []
    if any(f in a for f in ("--batch", "--slots")):
        out["available"] = False
        out["note"] = " - not available: the config's args set it (--batch / --slots)"
    return out


def view(cfg: dict, path: str | Path) -> dict:
    """GET /config: the editable keys with their values (None: not set, the default applies)."""
    out = []
    for key, kind, help_ in EDITABLE:
        k = kind[0] if isinstance(kind, tuple) else kind
        if k == "arg" and _sub(kind) != "int>=0":
            k = {"text": "text", "flag": "bool"}[_sub(kind)]
        off = key == "uncensored" and not has_projection(cfg)   # nothing to switch: shown, not settable
        extra = _count_view(cfg, key, kind) if k == "count" else {}
        out.append({"key": key, "value": value_of(cfg, key), "help": help_ + (f" - not available: {NO_PROJECTION}"
                                                                                 if off else extra.pop("note", "")),
                    "kind": "number" if k in ("sampling", "arg", "int>=0", "num>=0") else k,
                    **({"choices": kind[1]} if k == "enum" else {}), **({"available": False} if off else {}),
                    **({"section": SECTIONS[key]} if key in SECTIONS else {}),
                    **({"default": DEFAULTS[key]} if key in DEFAULTS else {}), **extra})
    return {"file": Path(path).name, "keys": out,
            "note": "Saved to the run config; used from the next start of the model."}


def _number(key, v, whole=False, lo=0.0, hi=None, lo_open=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (whole and v != int(v)) or \
            (v <= lo if lo_open else v < lo) or (hi is not None and v > hi):
        rng = f"{'more than' if lo_open else 'at least'} {lo:g}" + (f" and at most {hi:g}" if hi is not None else "")
        raise ValueError(f"{key}: expected a {'whole ' if whole else ''}number, {rng}, not {v!r}")
    return int(v) if whole else v


def check(key: str, v, cfg: dict):
    """The value to store for `key`, or a ValueError naming what is expected.  None removes the key (its default)."""
    if key not in SPEC:
        raise ValueError(f"{key!r} cannot be changed here (only the keys the Settings view lists)")
    if v is None:
        return None
    kind = SPEC[key]
    k = kind[0] if isinstance(kind, tuple) else kind
    if k == "bool":
        if not isinstance(v, bool):
            raise ValueError(f"{key}: expected true or false, not {v!r}")
        if key == "lazy_load" and v and cfg.get("vision"):
            raise ValueError("lazy_load: lazy loading is text-only, and this model reads images (\"vision\")")
        if key == "uncensored" and v and not has_projection(cfg):
            raise ValueError(f"uncensored: {NO_PROJECTION}")
        return v
    if k == "enum":
        if v not in kind[1]:
            raise ValueError(f"{key}: expected one of {', '.join(kind[1])}, not {v!r}")
        return v
    if k == "names":
        if isinstance(v, str):
            v = [x.strip() for x in v.split(",") if x.strip()]
        if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() and len(x) <= 128 for x in v):
            raise ValueError(f"{key}: expected a list of model names, not {v!r}")
        return v or None
    if k == "int>=0":
        return _number(key, v, whole=True)
    if k == "num>=0":
        return _number(key, v)
    if k == "arg" and _sub(kind) == "flag":
        if not isinstance(v, bool):
            raise ValueError(f"{key}: expected true or false, not {v!r}")
        return v or None                               # off: the option is left out, as by default
    if k == "arg" and _sub(kind) == "text":
        if not isinstance(v, str) or not v.strip() or len(v) > 1024 or any(ord(c) < 32 for c in v):
            raise ValueError(f"{key}: expected a folder (one line of text), not {v!r}")
        return v.strip()
    if k == "arg":
        return _number(key, v, whole=True)
    if k == "count":
        return _number(key, v, whole=True, lo=kind[1], hi=kind[2])
    rule = kind[1]                                     # a sampling key: the server's own rules
    if rule == "num>=0":
        return float(_number(key, v))
    if rule == "0<x<=1":
        return float(_number(key, v, hi=1, lo_open=True))
    if rule == "0<=x<=1":
        return float(_number(key, v, hi=1))
    return _number(key, v, whole=True, lo=1, hi=64)


def apply(cfg: dict, changes: dict) -> tuple[dict, list[str]]:
    """The config with `changes` ({key: value or None}) made, and the keys that changed.  Every value is checked
    first, so a request with one bad value changes nothing (ValueError)."""
    if not isinstance(changes, dict) or not changes:
        raise ValueError('send {"set": {"<key>": <value or null>, ...}}')
    checked = {k: check(k, v, cfg) for k, v in changes.items()}
    new = json.loads(json.dumps(cfg))                  # a deep copy: the caller's dict is left alone
    changed = []
    for key, v in checked.items():
        if value_of(new, key) == v:
            continue
        kind = SPEC[key]
        if isinstance(kind, tuple) and kind[0] == "sampling":
            s = dict(new.get("sampling") or {}) if isinstance(new.get("sampling"), dict) else {}
            sub = key.split(".", 1)[1]
            if v is None:
                s.pop(sub, None)
            else:
                s[sub] = v
            if s:
                new["sampling"] = s
            else:
                new.pop("sampling", None)
        elif isinstance(kind, tuple) and kind[0] == "arg" and _sub(kind) == "flag":
            a = [x for x in (new.get("args") or []) if x != kind[1]]
            new["args"] = a + [kind[1]] if v else a
        elif isinstance(kind, tuple) and kind[0] == "arg":
            a = list(new.get("args") or [])
            flag = kind[1]
            if flag in a[:-1]:
                i = a.index(flag)
                if v is None:
                    del a[i:i + 2]
                else:
                    a[i + 1] = str(v)
            elif v is not None:
                a += [flag, str(v)]
            new["args"] = a
        elif v is None:
            new.pop(key, None)
        else:
            new[key] = v
        changed.append(key)
    if any(k in changed for k in DIALOG_KEYS):
        _check_dialogs(new)
    return new, changed


def _check_dialogs(cfg: dict):
    """The kept dialogs' keys work together: a refusal names what is missing (an engine given one without the other
    would start and keep nothing)."""
    v = {k: value_of(cfg, k) for k in DIALOG_KEYS}
    if not v["conversation_cache_mib"]:
        for k in ("conversation_cache_slots", "conversation_disk_dir"):
            if v[k] is not None:
                raise ValueError(f"{k}: dialogs are kept only with RAM for them - set conversation_cache_mib too "
                                 "(e.g. 16384), or clear both")
    if not v["conversation_disk_dir"]:
        for k in ("conversation_disk_mib", "conversation_disk_persist"):
            if v[k] is not None:
                raise ValueError(f"{k}: set the folder (conversation_disk_dir) too, or clear both")


def save(path: str | Path, cfg: dict) -> Path:
    """The config written whole (a temporary file moved over the old one), the earlier one kept as <name>.bak."""
    path = Path(path)
    bak = path.with_name(path.name + ".bak")
    shutil.copyfile(path, bak)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return bak


def load(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{Path(path).name} is not a JSON object")
    return cfg
