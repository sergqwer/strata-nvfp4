"""tools/sim_tier.py - replay `--dump-routing` traces through the engine's adaptive VRAM tier, counted in host RAM bytes.

Decode on a host-RAM-bound machine pays for every expert blob read from RAM in a round: the distinct experts a verify
window misses (the CPU pool and the PCIe share both read them) plus the adaptive tier's swaps (each one copies a blob
from RAM into a slot).  This replays the routing of a real run window by window through a model of the tier in
src/program/generate.cpp (adapt(): per layer, the hottest missing experts with a count >= 2 against the coldest
resident ones, a swap while hot >= cold + 1.5, the best `swaps` by gain, every `every` windows, counts x `decay`
after each round; a round in flight blocks the next one) and prints, per policy:

    hit (routed entries computed in VRAM), miss/L (distinct misses per layer per window), swaps per window,
    RAM MB per window = (distinct misses + swaps) x the layer's blob bytes.

    python tools/sim_tier.py TRACE [--slots N] [--profile P] [--pack DIR] [--grid] [--oracle]

The trace's verify windows: one record per token per layer (drive_pool_multi), layers in order, so a window is a run
of records from layer 0 to layer 47 and its size is the records of layer 0.  The start: the profile's first N pairs
(the engine's fill: slot i = rank i); a K/V that gave slots away holds the hottest N - given (see kvg_ensure).
`--slots`: the "N of M slots hold experts" the engine's log reports at the decode's start.

`--lag`: the windows until a round's swaps are resident.  STRATA_ADAPT_WAIT=1 is 1; the default (no wait) is 2 on an
RTX 5090 (STRATA_TRACE_ADAPT=1: every round's copies were still in flight at the next window).  Checked against the
engine's own counters on the same traces (misses = the CPU's + the PCIe share's distinct experts a layer, swaps):
chat 2.67 / 6177 (engine, wait) vs 2.68 / 6182 (here, lag 1), 32K 4.16 / 10949 vs 4.16 / 10961, nowait chat
2.86 / 6340 vs 2.87 / 6356 (lag 2), 32K 4.10 / 10361 vs 4.10 / 10342, 96K 4.06 / 10622 vs 4.07 / 10618.

Policy(fetch=True) models STRATA_ADAPT_FETCH: a `fetch_frac` share of each layer's misses goes over PCIe (the engine's
balance model picks ~0.25 here), steered to the most-routed ones, and each of them routed >= `umin` and `fmargin` more
than the layer's least-routed resident expert not routed in the window takes that one's slot for free (the blob
crossed the link anyway).  `--variants` and `--oracle` print the other candidates and the bounds.
"""
from __future__ import annotations

import argparse
import itertools
import struct
import sys
from pathlib import Path

import numpy as np

L, E = 48, 512


def read_profile(path):
    blob = Path(path).read_bytes()
    if blob[:4] != b"STRP":
        raise SystemExit(f"{path}: not a Strata profile")
    _, nl, ne, _, n = struct.unpack_from("<5I", blob, 4)
    assert (nl, ne) == (L, E)
    pairs = np.frombuffer(blob, dtype=np.uint16, count=2 * n, offset=24).reshape(n, 2)
    return pairs[:, 0].astype(np.int64) * E + pairs[:, 1].astype(np.int64)   # codes in rank order


def read_blob_bytes(pack):
    bb = np.zeros(L, dtype=np.int64)
    for line in (Path(pack) / "native_experts.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        bb[int(f[0])] = int(f[4])
    return bb


class Trace:
    """Windows of a trace: per window its size T and its distinct (layer, expert) codes with their routed counts."""

    def __init__(self, path, skip_big=16):
        raw = np.fromfile(path, dtype=np.int32)
        k = int(raw[1])
        rec = raw.reshape(-1, 2 + 2 * k)
        assert (rec[:, 1] == k).all(), "variable k"
        layer = rec[:, 0].astype(np.int64)
        ids = rec[:, 2:2 + k].astype(np.int64)
        start = np.flatnonzero(np.r_[True, layer[1:] < layer[:-1]])
        end = np.r_[start[1:], len(layer)]
        self.k = k
        self.T, self.codes, self.counts, self.layers_of = [], [], [], []
        self.skipped = 0
        for s, e in zip(start, end):
            n0 = int((layer[s:e] == 0).sum())
            if n0 == 0 or e - s != n0 * L:
                raise SystemExit(f"{path}: a window of {e - s} records is not {L} layers x its tokens")
            if n0 > skip_big:   # a prompt read through the pool would show here; none expected
                self.skipped += 1
                continue
            c = (layer[s:e, None] * E + ids[s:e]).ravel()
            u, cnt = np.unique(c, return_counts=True)
            self.T.append(n0)
            self.codes.append(u)
            self.counts.append(cnt.astype(np.float32))
        self.T = np.array(self.T)
        self.n = len(self.T)


class Policy:
    """The tier's knobs.  Defaults = the engine's (fork defaults for a cache holding 20-60% of the experts)."""

    def __init__(self, every=2, swaps=192, decay=0.92, umin=2.0, margin=1.5, count="entries", scope="layer",
                 tie="id", lag=1, prior=0.0, prior_scale=7000.0, fetch=False, fetch_frac=0.25, steer=True,
                 fmargin=None, fmin=None, name=None):
        self.every, self.swaps, self.decay, self.umin, self.margin = every, swaps, decay, umin, margin
        self.count, self.scope, self.tie, self.lag = count, scope, tie, lag
        self.prior, self.prior_scale = prior, prior_scale
        # admit from the PCIe share: the misses the GPU reads over PCIe land in VRAM anyway; one that the tier would
        # swap in takes its layer's coldest slot not routed in the window at no RAM cost (`steer`: the share is the
        # layer's hottest misses, else any)
        self.fetch, self.fetch_frac, self.steer = fetch, fetch_frac, steer
        self.fmargin = margin if fmargin is None else fmargin
        self.fmin = umin if fmin is None else fmin
        self.name = name or f"e{every} s{swaps} d{decay:g} m{margin:g}" + \
            ("" if umin == 2.0 else f" u{umin:g}") + ("" if count == "entries" else f" {count}") + \
            ("" if scope == "layer" else f" {scope}") + ("" if tie == "id" else f" tie-{tie}") + \
            ("" if lag == 1 else f" lag{lag}") + ("" if prior == 0 else f" prior{prior:g}/{prior_scale:g}") +             ("" if not fetch else f" FETCH f{fetch_frac:g}{'' if steer else ' unsteered'} fm{self.fmargin:g}")


def initial_resident(profile, n_slots):
    res = np.zeros(L * E, dtype=bool)
    res[profile[:n_slots]] = True
    return res


def pick_layer(usage, res, pol, tiekey, bias=None):
    """adapt()'s choice, per layer: (gain, in, out) arrays over all layers, best gain first, capped.  `bias`: a
    per-pair term added to the counts for the ranking (the profile prior); the candidates still need count >= umin."""
    U = usage.reshape(L, E)
    S = U if bias is None else (usage + bias).reshape(L, E)
    R = res.reshape(L, E)
    g_all, i_all, o_all = [], [], []
    for l in range(L):
        raw, u, r = U[l], S[l], R[l]
        cand = np.flatnonzero(~r & (raw >= pol.umin))
        if cand.size == 0:
            continue
        vict = np.flatnonzero(r)
        if vict.size == 0:
            continue
        cand = cand[np.argsort(-u[cand], kind="stable")]
        # coldest first; ties by `tiekey` (larger = evicted first), else by id
        vict = vict[np.lexsort((-tiekey[l * E + vict], u[vict]))] if tiekey is not None else \
            vict[np.argsort(u[vict], kind="stable")]
        nc = min(cand.size, vict.size)
        d = u[cand[:nc]] - u[vict[:nc]]
        ok = u[cand[:nc]] >= u[vict[:nc]] + np.float32(pol.margin)
        m = nc if ok.all() else int(np.argmin(ok))
        if m == 0:
            continue
        g_all.append(d[:m])
        i_all.append(l * E + cand[:m])
        o_all.append(l * E + vict[:m])
    if not g_all:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    g = np.concatenate(g_all)
    i = np.concatenate(i_all)
    o = np.concatenate(o_all)
    ordr = np.argsort(-g, kind="stable")[:pol.swaps]
    return i[ordr], o[ordr]


def pick_global(usage, res, pol, bb, slot_big, tiekey):
    """Cross-layer: a hot missing expert takes the coldest resident anywhere whose slot fits its blob (a slot keeps
    the size of the expert it was filled with).  Values in bytes a window: count x the blob's bytes."""
    lay = np.arange(L * E) // E
    val = usage * bb[lay]
    cand = np.flatnonzero(~res & (usage >= pol.umin))
    if cand.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    cand = cand[np.argsort(-val[cand], kind="stable")]
    vict = np.flatnonzero(res)
    key2 = -tiekey[vict] if tiekey is not None else np.zeros(vict.size)
    vict = vict[np.lexsort((key2, val[vict]))]
    big_v = [v for v in vict if slot_big[v]]
    sml_v = [v for v in vict if not slot_big[v]]
    bi = si = 0
    ins, outs = [], []
    big_layer = bb == bb.max()
    for c in cand:
        if len(ins) >= pol.swaps:
            break
        need_big = big_layer[c // E]
        opts = []
        if bi < len(big_v):
            opts.append((val[big_v[bi]], 0))
        if not need_big and si < len(sml_v):
            opts.append((val[sml_v[si]], 1))
        if not opts:
            continue
        vv, which = min(opts)
        if val[c] < vv + pol.margin * bb[c // E]:
            if need_big:
                continue
            break
        if which == 0:
            o = big_v[bi]; bi += 1
        else:
            o = sml_v[si]; si += 1
        ins.append(c)
        outs.append(o)
    return np.array(ins, np.int64), np.array(outs, np.int64)


def simulate(tr, profile, n_slots, bb, pol, give=None):
    """`give`: {window: slots} the elastic K/V takes at that window's start (the coldest residents go)."""
    res = initial_resident(profile, n_slots)
    rank = np.full(L * E, len(profile), dtype=np.int64)
    rank[profile] = np.arange(len(profile))
    slot_big = np.zeros(L * E, dtype=bool)   # whether the slot an expert sits in holds the larger blob
    slot_big[profile[:n_slots]] = (bb[profile[:n_slots] // E] == bb.max())
    tiekey = rank.astype(np.float64) if pol.tie == "rank" else None
    # the prior: a pair's routing rate by its profile rank (~0.2 a window at the top, e-folding every
    # `prior_scale` ranks), worth `prior` windows of counts (x1.3 entries a routed window)
    bias = None
    if pol.prior > 0:
        bias = (1.3 * pol.prior * 0.2 * np.exp(-rank / pol.prior_scale)).astype(np.float32)
    usage = np.zeros(L * E, dtype=np.float32)
    lay_bb = bb[np.arange(L * E) // E]
    pending, due = None, -1
    miss_n = swap_n = fetch_n = 0
    miss_b = swap_b = 0
    hit_e = tot_e = 0
    for w in range(tr.n):
        if pending is not None and w >= due:
            res[pending[0]] = True
            slot_big[pending[0]] = pending[2]
            pending = None
        if give and w in give:
            heat = usage.astype(np.float64) * 1048576.0 - rank
            r = np.flatnonzero(res)
            drop = r[np.argsort(heat[r], kind="stable")[:give[w]]]
            res[drop] = False
        c, cnt = tr.codes[w], tr.counts[w]
        hit = res[c]
        miss = c[~hit]
        miss_n += miss.size
        miss_b += int(lay_bb[miss].sum())
        hit_e += float(cnt[hit].sum())
        tot_e += float(cnt.sum())
        usage[c] += cnt if pol.count == "entries" else np.float32(1.0)
        if pol.fetch and miss.size:
            score = usage if bias is None else usage + bias
            pend_in = pending[0] if pending is not None else np.zeros(0, np.int64)
            ml = miss // E
            routed_l = c // E
            admitted = []
            for l in np.unique(ml):
                M = miss[ml == l]
                m = int(M.size * pol.fetch_frac + 0.5)
                if m == 0:
                    continue
                M = M[np.argsort(-score[M], kind="stable")] if pol.steer else M
                P = [x for x in M[:m] if usage[x] >= pol.fmin and x not in pend_in]
                if not P:
                    continue
                r = np.flatnonzero(res[l * E:(l + 1) * E]) + l * E
                r = np.setdiff1d(r, c[routed_l == l], assume_unique=True)
                if r.size == 0:
                    continue
                r = r[np.argsort(score[r], kind="stable")]
                for j, x in enumerate(P):
                    if j >= r.size or score[x] < score[r[j]] + np.float32(pol.fmargin):
                        break
                    res[r[j]] = False
                    admitted.append(x)
            if admitted:
                res[np.array(admitted)] = True   # from the next window on
                fetch_n += len(admitted)
        if pol.every > 0 and (w + 1) % pol.every == 0 and pending is None:
            if pol.scope == "layer":
                i, o = pick_layer(usage, res, pol, tiekey, bias)
            else:
                i, o = pick_global(usage, res, pol, bb, slot_big, tiekey)
            if i.size:
                big = slot_big[o].copy()
                res[o] = False
                pending, due = (i, o, big), w + pol.lag
                swap_n += i.size
                swap_b += int(lay_bb[i].sum())
            usage *= np.float32(pol.decay)
    W = tr.n
    return dict(policy=pol.name, windows=W, hit=hit_e / tot_e, miss_l=miss_n / (W * L), swaps=swap_n / W,
                swaps_total=swap_n, miss_mb=miss_b / W / 1e6, swap_mb=swap_b / W / 1e6,
                ram_mb=(miss_b + swap_b) / W / 1e6, fetched=fetch_n / W)


# ---- bounds ----------------------------------------------------------------------------------------------------

def presence(tr):
    """windows x codes presence as a list of index arrays (already: tr.codes)."""
    return tr.codes


def oracle_static(tr, profile, n_slots, bb):
    """The best fixed set in hindsight (the most windows routed, per byte), reached from the profile's start by
    swaps counted once: misses of that set + the swaps to build it."""
    lay_bb = bb[np.arange(L * E) // E]
    win = np.zeros(L * E)
    for c in tr.codes:
        win[c] += 1
    # a slot is a slot (sizes ignored here): the top n by bytes saved
    best = np.argsort(-(win * lay_bb), kind="stable")[:n_slots]
    res = np.zeros(L * E, dtype=bool)
    res[best] = True
    start = initial_resident(profile, n_slots)
    swaps_in = np.flatnonzero(res & ~start)
    miss_b = sum(int(lay_bb[c[~res[c]]].sum()) for c in tr.codes)
    miss_n = sum(int((~res[c]).sum()) for c in tr.codes)
    W = tr.n
    return dict(policy="oracle static (hindsight set)", windows=W, hit=float("nan"), miss_l=miss_n / (W * L),
                swaps=swaps_in.size / W, swaps_total=swaps_in.size, miss_mb=miss_b / W / 1e6,
                swap_mb=int(lay_bb[swaps_in].sum()) / W / 1e6, ram_mb=(miss_b + int(lay_bb[swaps_in].sum())) / W / 1e6)


def oracle_future(tr, profile, n_slots, bb, horizon, margin=1.0, every=1, cap=10 ** 9):
    """Looks ahead: every `every` windows, per layer, the experts routed in the most of the next `horizon` windows
    replace resident ones routed in fewer, while the difference exceeds `margin` windows (a swap reads one blob,
    a window routed while missing reads one)."""
    lay_bb = bb[np.arange(L * E) // E]
    res = initial_resident(profile, n_slots)
    W = tr.n
    # future presence counts via a sliding window
    fut = np.zeros(L * E, dtype=np.float32)
    for w in range(min(horizon, W)):
        fut[tr.codes[w]] += 1
    miss_n = swap_n = miss_b = swap_b = 0
    hit_e = tot_e = 0
    pend = None
    for w in range(W):
        if pend is not None:
            res[pend] = True
            pend = None
        c, cnt = tr.codes[w], tr.counts[w]
        hit = res[c]
        miss_n += int((~hit).sum())
        miss_b += int(lay_bb[c[~hit]].sum())
        hit_e += float(cnt[hit].sum())
        tot_e += float(cnt.sum())
        # slide: fut covers windows w+1 .. w+horizon (the swap lands for window w+1)
        fut[c] -= 1
        if w + horizon < W:
            fut[tr.codes[w + horizon]] += 1
        if (w + 1) % every:
            continue
        F = fut.reshape(L, E)
        R = res.reshape(L, E)
        ins, outs, gains = [], [], []
        for l in range(L):
            f, r = F[l], R[l]
            cand = np.flatnonzero(~r & (f > margin))
            vict = np.flatnonzero(r)
            if cand.size == 0 or vict.size == 0:
                continue
            cand = cand[np.argsort(-f[cand], kind="stable")]
            vict = vict[np.argsort(f[vict], kind="stable")]
            nc = min(cand.size, vict.size)
            ok = f[cand[:nc]] - f[vict[:nc]] > margin
            m = nc if ok.all() else int(np.argmin(ok))
            ins.append(l * E + cand[:m]); outs.append(l * E + vict[:m]); gains.append(f[cand[:m]] - f[vict[:m]])
        if ins:
            i = np.concatenate(ins); o = np.concatenate(outs); g = np.concatenate(gains)
            ordr = np.argsort(-g, kind="stable")[:cap]
            i, o = i[ordr], o[ordr]
            if i.size:
                res[o] = False
                pend = i
                swap_n += i.size
                swap_b += int(lay_bb[i].sum())
    return dict(policy=f"oracle future h{horizon} m{margin:g} e{every}", windows=W, hit=hit_e / tot_e,
                miss_l=miss_n / (W * L), swaps=swap_n / W, swaps_total=swap_n, miss_mb=miss_b / W / 1e6,
                swap_mb=swap_b / W / 1e6, ram_mb=(miss_b + swap_b) / W / 1e6)


def oracle_belady(tr, profile, n_slots, bb):
    """Belady's MIN per layer at window granularity, every miss loaded (a swap) - the fewest misses any policy with
    these per-layer slot counts can have; its swaps show what that costs."""
    lay_bb = bb[np.arange(L * E) // E]
    res = initial_resident(profile, n_slots)
    W = tr.n
    nxt_w = {}
    # next use per (window, code): process backwards
    nxt = [None] * W
    last = {}
    for w in range(W - 1, -1, -1):
        c = tr.codes[w]
        nxt[w] = np.array([last.get(int(x), W) for x in c], dtype=np.int64)
        for x in c:
            last[int(x)] = w
    cur_next = np.full(L * E, W, dtype=np.int64)
    for x, w in last.items():
        cur_next[x] = w
    miss_n = swap_n = miss_b = swap_b = 0
    for w in range(W):
        c = tr.codes[w]
        hit = res[c]
        miss = c[~hit]
        miss_n += miss.size
        miss_b += int(lay_bb[miss].sum())
        cur_next[c] = nxt[w]
        # load each miss whose next use is sooner than the furthest resident's in its layer
        for x in miss:
            l = x // E
            r = np.flatnonzero(res[l * E:(l + 1) * E]) + l * E
            if r.size == 0:
                continue
            v = r[np.argmax(cur_next[r])]
            if cur_next[v] > cur_next[x]:
                res[v] = False
                res[x] = True
                swap_n += 1
                swap_b += int(lay_bb[x])
    return dict(policy="oracle Belady (bypass)", windows=W, hit=float("nan"), miss_l=miss_n / (W * L),
                swaps=swap_n / W, swaps_total=swap_n, miss_mb=miss_b / W / 1e6, swap_mb=swap_b / W / 1e6,
                ram_mb=(miss_b + swap_b) / W / 1e6)


def fmt(r):
    return (f"  {r['policy']:<44} hit {r['hit']:.4f}  miss/L {r['miss_l']:5.2f}  swaps/w {r['swaps']:6.1f} "
            f"({r['swaps_total']:6d})  miss {r['miss_mb']:6.1f} + swap {r['swap_mb']:6.1f} = RAM {r['ram_mb']:6.1f} MB/w"
            + (f"  admitted/w {r['fetched']:.1f}" if r.get('fetched') else ""))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("trace")
    p.add_argument("--slots", type=int, default=6700, help="experts in VRAM at the decode's start")
    p.add_argument("--profile", default="C:/LLM_models/strata-nvfp4/data/expert-profile.bin")
    p.add_argument("--pack", default="C:/LLM_models/Strata-data/packs/orca-nvfp4-gptq-q8d")
    p.add_argument("--lag", type=int, default=1, help="windows until a round's swaps are resident (WAIT mode: 1)")
    p.add_argument("--tie", default="id", help="victims of equal count: id (any) or rank (the profile's coldest)")
    p.add_argument("--grid", action="store_true", help="sweep every / swaps / decay / margin")
    p.add_argument("--variants", action="store_true", help="the policy variants beside the default")
    p.add_argument("--oracle", action="store_true", help="the bounds")
    p.add_argument("--csv", help="write the grid's rows here")
    a = p.parse_args(argv)
    tr = Trace(a.trace)
    prof = read_profile(a.profile)
    bb = read_blob_bytes(a.pack)
    print(f"trace {a.trace}: {tr.n} windows ({tr.skipped} skipped), {tr.T.sum()} positions, "
          f"{tr.T.mean():.2f} a window; slots {a.slots}")
    base = Policy(lag=a.lag, tie=a.tie)
    print(fmt(simulate(tr, prof, a.slots, bb, base)))
    if a.variants:
        for pol in [Policy(every=0, name="no tier (profile only)"),
                    Policy(lag=a.lag, tie="rank"),
                    Policy(lag=a.lag, count="windows", umin=2.0, margin=1.5),
                    Policy(lag=a.lag, scope="global"),
                    Policy(lag=a.lag, every=4, swaps=96, decay=0.7),
                    Policy(lag=a.lag, fmin=1.0, fetch=True, fmargin=0.5, name="STRATA_ADAPT_FETCH=1"),
                    Policy(lag=a.lag, margin=1e9, umin=1.0, fetch=True, fmargin=0.5, name="STRATA_ADAPT_FETCH=2")]:
            print(fmt(simulate(tr, prof, a.slots, bb, pol)))
    if a.oracle:
        print(fmt(oracle_static(tr, prof, a.slots, bb)))
        for h, m in [(8, 1), (16, 1), (32, 1), (64, 1), (32, 2)]:
            print(fmt(oracle_future(tr, prof, a.slots, bb, h, m)))
        print(fmt(oracle_belady(tr, prof, a.slots, bb)))
    if a.grid:
        rows = []
        for ev, sw, dc, mg in itertools.product([1, 2, 4], [32, 64, 192], [0.85, 0.92, 0.96, 0.98],
                                                [1.5, 3, 6]):
            r = simulate(tr, prof, a.slots, bb, Policy(every=ev, swaps=sw, decay=dc, margin=mg, lag=a.lag, tie=a.tie))
            rows.append(r)
        rows.sort(key=lambda r: r["ram_mb"])
        for r in rows[:15]:
            print(fmt(r))
        if a.csv:
            with open(a.csv, "w") as f:
                f.write("policy,hit,miss_l,swaps,ram_mb\n")
                for r in rows:
                    f.write(f"{r['policy']},{r['hit']:.4f},{r['miss_l']:.3f},{r['swaps']:.2f},{r['ram_mb']:.2f}\n")


if __name__ == "__main__":
    main()
