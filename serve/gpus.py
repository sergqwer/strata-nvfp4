"""serve/gpus.py - every GPU of the PC on the web page's Monitor: which physical card each of the engine's devices is,
and what the engine does with it (a layer-split stage and its layers, the peer expert tier, a helper expert cache, the
image encoder, or nothing).

- The cards are NVML's (nvidia-smi's numbering, serve/telemetry.py reads them), each with its PCI bus id and UUID.
- The engine's CUDA ordinals -> those cards, from the engine's own environment (CUDA_VISIBLE_DEVICES and
  CUDA_DEVICE_ORDER, as the server set them for a config's "gpu" or as the user did): indices in PCI order
  (PCI_BUS_ID, what the server sets), UUIDs, and CUDA's default FASTEST_FIRST order when every card is the same model.
  Anything else (FASTEST_FIRST over different cards, MIG) asks the CUDA driver once, in a child process, for each
  ordinal's PCI bus id.  AMD: HIP_VISIBLE_DEVICES in the KFD order the server and setup use.
- The roles come from the engine's start-up lines in its log, which it prints once before READY ("layer split: CUDA1
  runs layers 24-47, expert cache ...", "peer GPU 1: ...", "CUDA2: N additional experts ..."); before a start (a lazy
  or unloaded engine) the args say what is planned.
Nothing here runs per request or per token: the telemetry thread refreshes the layout once a second from cached
pieces (the mapping once per environment, the log once per engine start).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time

# ------------------------------------------------------------------------------------------------ CUDA ordinal -> card


def pci_key(s) -> str | None:
    """A PCI bus id in one spelling: NVML's "00000000:01:00.0", CUDA's "0000:01:00.0" and "01:00.0" -> "0000:01:00.0"
    (lower case); None when `s` is not one."""
    m = re.fullmatch(r"(?:([0-9a-fA-F]{1,8}):)?([0-9a-fA-F]{1,2}):([0-9a-fA-F]{1,2})\.([0-7])", str(s or "").strip())
    if not m:
        return None
    return "%04x:%02x:%02x.%s" % (int(m.group(1) or "0", 16), int(m.group(2), 16), int(m.group(3), 16), m.group(4))


def visible_tokens(env: dict, var: str = "CUDA_VISIBLE_DEVICES") -> list[str] | None:
    """The entries of CUDA_VISIBLE_DEVICES (None: unset, every card)."""
    v = env.get(var)
    if v is None:
        return None
    return [t.strip() for t in str(v).split(",")]


_PROBE = (
    "import ctypes,os,sys\n"
    "lib=ctypes.CDLL('nvcuda.dll' if os.name=='nt' else 'libcuda.so.1')\n"
    "assert lib.cuInit(0)==0\n"
    "n=ctypes.c_int()\n"
    "assert lib.cuDeviceGetCount(ctypes.byref(n))==0\n"
    "for i in range(n.value):\n"
    "    d=ctypes.c_int()\n"
    "    assert lib.cuDeviceGet(ctypes.byref(d),i)==0\n"
    "    b=ctypes.create_string_buffer(32)\n"
    "    assert lib.cuDeviceGetPCIBusId(b,32,d)==0\n"
    "    print(b.value.decode())\n")


def cuda_probe(env: dict, timeout: float = 20.0) -> list[str] | None:
    """The PCI bus id of each CUDA ordinal under `env` (the engine's environment), from the CUDA driver API in a child
    process - the server process itself never loads the driver.  None when it cannot say."""
    try:
        r = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=timeout,
                           env={k: str(v) for k, v in env.items()})
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    out = [pci_key(x) for x in r.stdout.split()]
    return out if all(out) else None


def cuda_cards(env: dict, cards: list[dict], probe=cuda_probe) -> tuple[list, str]:
    """The card (its "index", nvidia-smi's numbering) of each CUDA ordinal the engine sees, and how it was matched:
    "pci_bus_id" (CUDA_DEVICE_ORDER=PCI_BUS_ID: CUDA counts in bus order), "same_model" (FASTEST_FIRST, CUDA's default,
    over identical cards: they tie and CUDA breaks the tie by bus id), "uuid", "cuda_driver" (asked the driver: `probe`)
    or "assumed" (nothing to go by: ordinal k is card k).  CUDA_VISIBLE_DEVICES is read the way CUDA reads it: in its
    order, up to the first entry that names no card."""
    known = [c for c in cards if c.get("index") is not None]
    order = str(env.get("CUDA_DEVICE_ORDER") or "FASTEST_FIRST").strip().upper()
    by_bus = sorted(known, key=lambda c: (c.get("pci") is None, c.get("pci") or "", c["index"]))
    if any(c.get("pci") is None for c in known):
        by_bus = sorted(known, key=lambda c: c["index"])      # NVML numbers the cards in bus order too
    base, how = None, None
    if order == "PCI_BUS_ID":
        base, how = [c["index"] for c in by_bus], "pci_bus_id"
    elif len({c.get("name") for c in known}) <= 1:
        base, how = [c["index"] for c in by_bus], "same_model"
    tokens = visible_tokens(env)
    out, need = [], False
    if tokens is None:
        need = base is None
        out = list(base or [])
    else:
        for t in tokens:
            if not t:                                   # CUDA_VISIBLE_DEVICES="" hides every card
                break
            if re.fullmatch(r"-?\d+", t):
                i = int(t)
                if i < 0:
                    break
                if base is None:
                    need = True
                    break
                if i >= len(base) or base[i] in out:
                    break
                out.append(base[i])
            elif t.upper().startswith("GPU-"):
                hit = [c for c in known if str(c.get("uuid") or "").lower().startswith(t.lower())]
                if len(hit) != 1 or hit[0]["index"] in out:
                    break
                out.append(hit[0]["index"])
                how = how if how and how != "same_model" else "uuid"
            else:                                       # MIG-..., or anything CUDA may know better
                need = True
                break
    if need:
        pcis = probe(env) if probe else None
        if pcis is not None:
            idx = {c.get("pci"): c["index"] for c in known}
            return [idx.get(p) for p in pcis], "cuda_driver"
        if tokens is None:
            return [c["index"] for c in sorted(known, key=lambda c: c["index"])], "assumed"
        ints = [int(t) for t in tokens if re.fullmatch(r"\d+", t)]
        have = {c["index"] for c in known}
        return [i if i in have else None for i in ints], "assumed"
    return out, how or "pci_bus_id"


def hip_cards(env: dict, count: int) -> tuple[list, str]:
    """AMD: HIP_VISIBLE_DEVICES in the KFD order (the server sets it from the config's "gpu"), else every card."""
    tokens = visible_tokens(env, "HIP_VISIBLE_DEVICES")
    if tokens is None:
        return list(range(count)), "hip_order"
    out = []
    for t in tokens:
        if not re.fullmatch(r"\d+", t) or int(t) >= count or int(t) in out:
            break
        out.append(int(t))
    return out, "hip_order"


# ------------------------------------------------------------------------------------------------ the engine's layout
_ACROSS = re.compile(r"layer split across (\d+) GPUs: CUDA0, then CUDA([\d,]+)")
_STAGE = re.compile(r"layer split: CUDA(\d+) runs layers (\d+)-(\d+)(?:, expert cache (\d+) slots \(([\d.]+) GiB\))?")
_SKIPPED = re.compile(r"layer split skipped \(--split-skip-if-fits\)")
_PEER = re.compile(r"peer GPU (\d+): (\d+) experts, ([\d.]+) GiB")
_HELPER = re.compile(r"strata generate: CUDA(\d+): (\d+) additional experts, ([\d.]+) GiB")
_CACHE = re.compile(r"strata generate: expert cache (\d+) slots, ([\d.]+) GiB of VRAM")
# the engine's own PCIe measurement: CUDA0's at start, each later stage's on a layer split (-1.0: the probe failed)
_PROBE0 = re.compile(r"strata generate: PCIe probe: (-?[\d.]+) GB/s host->device(?:.*?pcie_frac ([\d.]+))?")
_PROBE_STAGE = re.compile(r"layer split: CUDA(\d+) PCIe probe (-?[\d.]+) GB/s.*?pcie_frac ([\d.]+)")


def _add_stage(stages: dict, d: int, layers: list, experts=None, gib=None):
    """A stage on card d; a second stage on the same card (`--layer-split 24 --split-device 0` runs both stages on
    one card) widens its layer range and adds its expert cache."""
    st = stages.get(d)
    if st is None:
        st = stages[d] = {"layers": list(layers)}
    else:
        a, b = st["layers"], layers
        firsts = [x for x in (a[0], b[0]) if x is not None]
        st["layers"] = [min(firsts) if firsts else None, None if a[1] is None or b[1] is None else max(a[1], b[1])]
        st["stages_here"] = st.get("stages_here", 1) + 1
    if experts is not None:
        st["experts"] = st.get("experts", 0) + experts
        st["gib"] = round(st.get("gib", 0) + (gib or 0), 2)


def parse_layout(lines) -> dict:
    """The engine's multi-GPU layout from its start-up lines (one start's): stages {ordinal: {layers, experts, gib}},
    the peer tier and the helper caches {ordinal: {experts, gib}}, CUDA0's own expert cache, and whether a split was
    asked and then skipped (--split-skip-if-fits: the model fit on CUDA0)."""
    out = {"split": False, "skipped": False, "stages": {}, "peer": {}, "helpers": {}, "cache": None, "links": {}}
    for line in lines:
        if "strata generate" not in line:
            continue
        m = _PROBE_STAGE.search(line) or _PROBE0.search(line)
        if m:
            d, gbps, frac = (int(m.group(1)), m.group(2), m.group(3)) if m.re is _PROBE_STAGE else (0, *m.groups())
            out["links"][d] = {"link_gbps": float(gbps) if float(gbps) > 0 else None,
                               "pcie_frac": float(frac) if frac else None}
            continue
        m = _ACROSS.search(line)
        if m:
            out["split"] = True
            continue
        m = _STAGE.search(line)
        if m:
            out["split"] = True
            _add_stage(out["stages"], int(m.group(1)), [int(m.group(2)), int(m.group(3))],
                       int(m.group(4)) if m.group(4) else None, float(m.group(5)) if m.group(5) else None)
            continue
        if _SKIPPED.search(line):
            out["skipped"], out["split"], out["stages"] = True, False, {}
            continue
        m = _PEER.search(line)
        if m:
            out["peer"][int(m.group(1))] = {"experts": int(m.group(2)), "gib": float(m.group(3))}
            continue
        m = _HELPER.search(line)
        if m:
            out["helpers"][int(m.group(1))] = {"experts": int(m.group(2)), "gib": float(m.group(3))}
            continue
        m = _CACHE.search(line)
        if m:                                           # the last one: the engine may try smaller sizes first
            out["cache"] = {"experts": int(m.group(1)), "gib": float(m.group(2))}
    return out


def _arg(args: list, name: str):
    return args[args.index(name) + 1] if name in args[:-1] else None


def planned_layout(args: list, n_visible: int) -> dict:
    """What the args ask for, before the engine has said what it did (a lazy or unloaded engine): the stages of
    --layer-split (the first layer of each later stage - "24,36" - or "auto" over every visible card), on CUDA0 and
    then --split-device's cards or CUDA1, 2, ...; --peer-device; --expert-cache-device1..3 (CUDA1..3, or with a split
    the visible cards no stage runs on, in order)."""
    out = {"split": False, "skipped": False, "stages": {}, "peer": {}, "helpers": {}, "cache": None, "links": {}}
    args = [str(a) for a in args or []]
    helpers = [r for r in (1, 2, 3) if (_arg(args, f"--expert-cache-device{r}") or "0") not in ("0", "")]
    split = _arg(args, "--layer-split")
    if split:
        parts = [x.strip() for x in split.split(",")]
        points = [int(x) for x in parts] if all(x.isdigit() for x in parts) else []
        devs_arg = _arg(args, "--split-device")
        devs = [int(x) for x in devs_arg.split(",") if x.strip().isdigit()] if devs_arg else []
        n_later = len(points) if points else max(0, n_visible - 1)
        devs = (devs or list(range(1, n_later + 1)))[:n_later]
        if devs:
            out["split"] = True
            for k, d in enumerate([0] + devs):
                first = 0 if k == 0 else (points[k - 1] if k - 1 < len(points) else None)
                last = (points[k] - 1) if k < len(points) else None
                _add_stage(out["stages"], d, [first, last])
    if out["split"]:
        free = [d for d in range(1, max(n_visible, 1 + len(out["stages"]) + len(helpers))) if d not in out["stages"]]
        for r, d in zip(helpers, free):
            out["helpers"][d] = {}
    else:
        for r in helpers:
            out["helpers"][r] = {}
    peer = _arg(args, "--peer-device")
    if peer is not None and peer.isdigit() and int(peer) >= 1:
        out["peer"][int(peer)] = {}
    return out


def roles(layout: dict, n_visible: int, info: dict | None = None, planned: bool = False) -> list[dict]:
    """One role per CUDA ordinal 0..n-1: "model" (one GPU runs it all), "stage" (a layer-split stage: its layers and
    its own expert cache), "peer" (--peer-device's expert tier), "helper" (a helper expert cache), or "unused"."""
    info = info or {}
    n = max(n_visible, 1 + max([*layout["stages"], *layout["peer"], *layout["helpers"], 0]))
    out = [{"role": "unused"} for _ in range(n)]
    primary = None
    if isinstance(info.get("expert_slots_primary"), int):
        primary = {"experts": info["expert_slots_primary"],
                   "gib": round((info.get("expert_cache_primary_mib") or 0) / 1024, 2)}
    elif layout.get("cache"):
        primary = dict(layout["cache"])
    elif isinstance(info.get("expert_slots"), int) and not layout["split"] and not layout["peer"] \
            and not layout["helpers"]:
        primary = {"experts": info["expert_slots"], "gib": round((info.get("expert_cache_mib") or 0) / 1024, 2)}
    if layout["split"] and len(layout["stages"]) == 1:   # every stage on one card (--split-device 0)
        d, st = next(iter(layout["stages"].items()))
        row = {"role": "model", "layers": st["layers"],
               "note": "a layer split onto one card (--split-device 0): every stage runs here"}
        mine = primary if d == 0 and primary else {}     # CUDA0's own cache + the later stage's cache on it
        if st.get("experts") or mine.get("experts"):
            row.update(experts=(st.get("experts") or 0) + (mine.get("experts") or 0),
                       gib=round((st.get("gib") or 0) + (mine.get("gib") or 0), 2))
        out[d] = row
    elif layout["split"] and layout["stages"]:
        order = sorted(layout["stages"])
        for k, d in enumerate(order):
            st = dict(layout["stages"][d])
            if d == 0 and primary and "experts" not in st:
                st.update(primary)
            out[d] = {"role": "stage", "stage": k + 1, "stages": len(order), **st}
    else:
        out[0] = {"role": "model", **(primary or {})}
        if layout.get("skipped"):
            out[0]["note"] = "the layer split was skipped: the model fits on this card (--split-skip-if-fits)"
    for d, v in layout["peer"].items():
        out[d] = {"role": "peer", **v}
    for d, v in layout["helpers"].items():
        out[d] = {"role": "helper", **v}
    for d, v in (layout.get("links") or {}).items():   # the engine's measured link and the PCIe share it chose
        if 0 <= d < len(out):
            out[d].update({k: x for k, x in v.items() if x is not None})
    if planned:
        for r in out:
            if r["role"] != "unused":
                r["planned"] = True
    return out


def read_start_lines(path: str | None, start: int | None, cap: int = 8 << 20) -> list[str]:
    """One engine start's lines of its log: from `start` (where that start began) on, at most `cap` bytes - the layout
    lines come before READY, so the first megabytes hold them."""
    if not path:
        return []
    try:
        with open(path, "rb") as f:
            f.seek(int(start or 0))
            data = f.read(cap)
    except (OSError, ValueError):
        return []
    return [x for x in data.decode("utf-8", errors="replace").splitlines() if "strata generate" in x]


# ------------------------------------------------------------------------------------------------ the page's rows
def view(cards: list[dict], readings: dict, engines: list[dict], encoder: int | None = None) -> list[dict]:
    """The Monitor's GPU rows: every card (static facts + its last reading), with the engine's ordinal for it and what
    the engine does with it.  `engines`: [{"replica": r or None, "cards": [card index per CUDA ordinal], "roles":
    [role per ordinal], "state": "ready" | "starting" | "unloaded"}]; `encoder`: the card the image encoder alone
    runs on (the config's vision cuda_device)."""
    rows = []
    for c in cards:
        row = {k: c.get(k) for k in ("index", "name", "pci", "uuid", "pcie_gen_max", "pcie_width_max",
                                     "pcie_gen_gpu_max")}
        r = readings.get(c.get("index")) or {}
        for k in ("util", "mem_used", "mem_total", "temp", "power", "power_limit", "pcie_gen", "pcie_width"):
            row[k] = r.get(k)
        if row["mem_total"] is None:
            row["mem_total"] = c.get("mem_total")
        row.update(cuda=None, replica=None, role="unused", visible=False)
        best = None
        for e in engines:
            for k, ci in enumerate(e.get("cards") or []):
                if ci != c.get("index"):
                    continue
                rl = (e.get("roles") or [])[k] if k < len(e.get("roles") or []) else {"role": "unused"}
                cand = (rl.get("role") != "unused", e, k, rl)
                if best is None or (cand[0] and not best[0]):
                    best = cand
        if best is not None:
            _, e, k, rl = best
            row.update(cuda=k, replica=e.get("replica"), visible=True, engine_state=e.get("state"))
            row.update({kk: vv for kk, vv in rl.items()})
        if encoder is not None and c.get("index") == encoder:
            row["encoder"] = True
            if row["role"] == "unused":
                row["role"] = "encoder"
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------------ the layout keeper
class Layout:
    """The engines' GPU layout, refreshed by the telemetry thread (refresh(), once a second) and read by /metrics
    (rows()).  Each piece is cached: the ordinal -> card mapping per environment (a CUDA driver probe at most once per
    environment), the roles per engine start (its log read once, after READY)."""

    def __init__(self, svc):
        self.svc = svc
        self.lock = threading.Lock()
        self.maps: dict = {}
        self.parsed: dict = {}
        self.engines: list[dict] = []
        self.how = None

    def _sources(self):
        """(replica, args, env, lines-or-None, info, state) for each engine; a mock fixture's engine in mock mode."""
        fx = mock_fixture()
        if fx is not None:
            e = fx.get("engine") or {}
            return [(None, e.get("args") or [], e.get("env") or {}, e.get("log") or [], e.get("info") or {}, "ready")]
        eng = getattr(self.svc, "engine", None)
        group = list(vars(eng).get("engines") or []) if eng is not None and hasattr(eng, "__dict__") else None
        out = []
        for r, e in enumerate(group or ([eng] if eng is not None else [])):
            spawn = getattr(e, "spawn", None)
            if not spawn:
                continue
            args, env = list(spawn[1] or []), dict(spawn[4] if len(spawn) > 4 and spawn[4] else os.environ)
            if getattr(e, "unloaded", False):
                state = "unloaded"
            elif getattr(e, "ended", True) or not getattr(e, "max_context", 0):
                state = "starting"
            else:
                state = "ready"
            lines = None
            path, start = getattr(e, "log_path", None), getattr(e, "log_start", None)
            if state == "ready" and path:
                if self.parsed.get(path, (None,))[0] != start:      # each log's newest start, read once
                    self.parsed[path] = (start, read_start_lines(path, start))
                lines = self.parsed[path][1]
            out.append((r if group else None, args, env, lines, dict(getattr(e, "info", {}) or {}), state))
        return out

    def _map(self, env: dict, cards: list[dict], amd: bool):
        keys = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "HIP_VISIBLE_DEVICES")
        k = (amd, tuple(env.get(x) for x in keys), tuple((c.get("index"), c.get("pci")) for c in cards))
        if k not in self.maps:
            self.maps[k] = hip_cards(env, len(cards)) if amd else cuda_cards(env, cards)
        return self.maps[k]

    def refresh(self, cards: list[dict], amd: bool = False) -> list[int]:
        """Recompute the layout (the telemetry thread); returns the cards the engines use (for the aggregates)."""
        engines, used, hows = [], set(), set()
        for replica, args, env, lines, info, state in self._sources():
            ords, how = self._map(env, cards, amd)
            hows.add(how)
            if lines is not None:
                layout, planned = parse_layout(lines), False
            else:
                layout, planned = planned_layout(args, len(ords)), True
            rl = roles(layout, len(ords), info, planned)
            engines.append({"replica": replica, "cards": ords, "roles": rl, "state": state})
            used.update(ci for ci, r in zip(ords, rl) if ci is not None and r.get("role") != "unused")
        with self.lock:
            self.engines, self.how = engines, ",".join(sorted(hows)) or None
        return sorted(used)

    def rows(self, cards: list[dict], readings: dict, encoder=None) -> dict:
        with self.lock:
            engines, how = list(self.engines), self.how
        return {"cards": view(cards, readings, engines, encoder), "mapping": how,
                "mock": mock_fixture() is not None}


# ------------------------------------------------------------------------------------------------ mock mode
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
_mock_lock = threading.Lock()
_mock_cache: dict = {}


def mock_fixture() -> dict | None:
    """STRATA_MOCK_GPUS: fake cards for the page and the API without the GPUs - a number N (N identical RTX 4090
    in a layer split, the 4x 4090 test server's shape) or the path of a fixture JSON (serve/fixtures/gpus-*.json:
    "cards", and the "engine" whose env, args, log lines and INFO facts the layout is read from).  None when unset."""
    spec = os.environ.get("STRATA_MOCK_GPUS", "").strip()
    if not spec:
        return None
    with _mock_lock:
        if spec in _mock_cache:
            return _mock_cache[spec]
        if spec.isdigit():
            fx = generated_fixture(max(1, min(16, int(spec))))
        else:
            path = spec if os.path.isabs(spec) or os.path.exists(spec) else os.path.join(FIXTURES, spec)
            with open(path, encoding="utf-8") as f:
                fx = json.load(f)
        _mock_cache[spec] = fx
        return fx


def generated_fixture(n: int, layers: int = 48) -> dict:
    """N identical RTX 4090 (24 GiB) with the model's 48 layers split evenly across them."""
    cards = [{"name": "NVIDIA GeForce RTX 4090", "mem_total_mib": 24564, "power_limit": 450.0,
              "pci": "0000:%02x:00.0" % (0x01 + 0x20 * i), "uuid": "GPU-00000000-0000-4090-0000-%012d" % i,
              "mem_used_mib": 22100 - 300 * i, "util": 70 - 5 * i}
             for i in range(n)]
    points = [round(layers * k / n) for k in range(1, n)]
    log = ["strata generate: PCIe probe: 24.8 GB/s host->device (best of 24.8 24.5 24.6 24.7) -> pcie_frac 0.25 "
           "(default 0.25)"]
    if n > 1:
        log.append(f"strata generate: layer split across {n} GPUs: CUDA0, then CUDA{','.join(map(str, range(1, n)))} "
                   f"(split auto)")
        for d in range(1, n):
            last = (points[d] if d < len(points) else layers) - 1
            log.append(f"strata generate: layer split: CUDA{d} runs layers {points[d - 1]}-{last}, expert cache "
                       f"{7800 - 150 * d} slots ({(7800 - 150 * d) * 2.68 / 1024:.2f} GiB), {7800 - 150 * d} of its "
                       f"9984 profiled pairs; slot 0 verified")
            log.append(f"strata generate: layer split: CUDA{d} PCIe probe {24.6 - 0.3 * d:.1f} GB/s (best of 4) -> "
                       f"pcie_frac 0.55")
        log.append(f"strata generate: layer split: CUDA0 runs layers 0-{points[0] - 1}")
    else:
        log.append("strata generate: expert cache 8644 slots, 22.26 GiB of VRAM; policy is")
    args = ["--layer-split", "auto"] if n > 1 else []
    return {"cards": cards, "engine": {"env": {"CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                                               "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(n)))},
                                       "args": args, "log": log,
                                       "info": {"expert_slots_primary": 6900 if n > 1 else 8644,
                                                "expert_cache_primary_mib": 18060 if n > 1 else 22794}}}


class MockGpu:
    """A fixture card with telemetry's reader interface; its load and power move so the page looks alive."""

    def __init__(self, index: int, card: dict):
        self.index, self.card = index, card

    def ok(self):
        return True

    def name(self):
        return self.card.get("name") or "Mock GPU"

    def info(self):
        c = self.card
        return {"name": self.name(), "uuid": c.get("uuid"), "pci": pci_key(c.get("pci")),
                "pcie_gen_max": c.get("pcie_gen_max", 4), "pcie_width_max": c.get("pcie_width_max", 16),
                "pcie_gen_gpu_max": c.get("pcie_gen_gpu_max", c.get("pcie_gen_max", 4))}

    def read(self):
        import math
        c, t = self.card, time.time()
        base = float(c.get("util", 50))
        util = max(0.0, min(100.0, base + 25 * math.sin(t / 5 + self.index)))
        lim = float(c.get("power_limit") or 300)
        total = int(c.get("mem_total_mib") or 24564) << 20
        used = c.get("mem_used_mib")
        return {"util": round(util), "mem_used": (int(used) << 20) if used is not None else None, "mem_total": total,
                "temp": round(45 + util * 0.3 + self.index), "power": round(lim * (0.15 + 0.6 * util / 100), 1),
                # an idle card drops its link to Gen1 to save power, as real ones do
                "power_limit": lim, "pcie_gen": c.get("pcie_gen", c.get("pcie_gen_max", 4)) if util >= 15 else 1,
                "pcie_gen_max": c.get("pcie_gen_max", 4), "pcie_width": c.get("pcie_width", 16),
                "pcie_rx_mb": round(util * 3, 1), "pcie_tx_mb": round(util, 1)}
