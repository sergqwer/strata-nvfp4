"""tools/route_eval.py - how well the next verify round's expert-cache misses can be predicted (qwen-speedup-plan #13, step 1).

    STRATA_ROUTE_TRACE=routes.srd strata.exe ...          the engine writes the trace (generate.cpp route_dump; not the
                                                          older text STRATA_ROUTE_DUMP of expert_source.cpp)
    python tools/route_eval.py routes.srd [--gguf DENSE.gguf] [--export pairs.npz]

A miss is an expert a layer routed that the VRAM tier did not hold when the layer started.  For every round r+1 of a
decode sequence, each predictor scores the (layer, expert) pairs from what was known after round r; the B best pairs the
VRAM tier does not hold are "copied", and the report is the share of round r+1's misses they cover (recall) at each
budget B, with the bytes those copies take over PCIe.  A prefetch that misses costs only bandwidth: the CPU computes
what did not arrive, as now.

Predictors (all from rounds <= r of the same sequence):
  last      the experts each layer routed in round r (scored by how many rows routed them)
  ema-X     the routing counts of the past rounds, each round back weighted X times less
  prior     how often each (layer, expert) was routed over the first half of the dump (static)
  ema+prior ema-0.5 with the prior as a tie-break
With --gguf (and the MoE input of a layer t in the dump): next-router, the router of layer t+1 applied to layer t's MoE
input of the SAME round (upstream's look-ahead heuristic; qwen-speedup-plan #3), scored against ema-0.5 on those layers.

--export writes, for every pair (r, r+1) of a sequence, the features DFlash2 also sees (the dumped layers' MoE input of
round r: the last row and the mean of the rows) and round r+1's routed / missed / resident bits: the training set for a
learned route drafter (step 2).
"""
from __future__ import annotations

import argparse
import mmap
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b"SRD1"
REC = struct.Struct("<BBHHHIQ")          # tag, has_x, layer, n_tok, k, round, t_us
WIN = struct.Struct("<BBHIq")            # tag 'W', 0, n_tok, 0, position of the first row; then i32 tokens[n_tok]


@dataclass
class Dump:
    n_layers: int
    n_expert: int
    n_embd: int
    x_layers: list[int]
    rounds: list["Round"] = field(default_factory=list)


@dataclass
class Round:
    index: int
    t_us: int
    n_tok: int
    ids: list[np.ndarray]                 # per layer: (n_tok, k) int16
    resident: np.ndarray                  # (n_layers, n_expert) bool
    x: dict[int, np.ndarray]              # layer -> (n_tok, n_embd) float16
    tokens: np.ndarray | None = None      # the window's token ids (a 'W' record before the round), when the engine wrote one
    pos: int = -1                         # the position of its first row


def read_dump(path: str | Path, keep_x: bool = True, max_rounds: int | None = None) -> Dump:
    with open(path, "rb") as fh:
        data = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    if data[:4] != MAGIC:
        raise SystemExit(f"{path}: not a route dump (no SRD1 header)")
    version, n_layers, n_expert, n_embd, n_x = struct.unpack_from("<5I", data, 4)
    if version != 1:
        raise SystemExit(f"{path}: route dump version {version}, this tool reads 1")
    x_layers = list(struct.unpack_from(f"<{n_x}I", data, 24))
    pos = 24 + 4 * n_x
    nb = (n_expert + 7) // 8
    dump = Dump(n_layers, n_expert, n_embd, x_layers)
    cur: dict | None = None
    win = None                              # the last 'W' record: (tokens, position), for the next round

    def close(c):
        if c is not None and len(c["ids"]) == n_layers and all(v is not None for v in c["ids"]):
            tok, p0 = c["win"] if c["win"] is not None else (None, -1)
            dump.rounds.append(Round(c["round"], c["t_us"], c["n_tok"], c["ids"], c["res"], c["x"], tok, p0))

    while pos + REC.size <= len(data):
        if data[pos] == ord("W"):
            _, _, n_w, _, p_w = WIN.unpack_from(data, pos)
            if pos + WIN.size + 4 * n_w > len(data):
                break
            win = (np.frombuffer(data, dtype="<i4", count=n_w, offset=pos + WIN.size).copy(), p_w)
            pos += WIN.size + 4 * n_w
            continue
        tag, has_x, layer, n_tok, k, rnd, t_us = REC.unpack_from(data, pos)
        if tag != ord("L"):
            raise SystemExit(f"{path}: bad record at byte {pos}")
        pos += REC.size
        n = n_tok * k
        if pos + 2 * n + nb + (2 * n_tok * n_embd if has_x else 0) > len(data):
            break                           # a run killed mid-record: its last record is cut
        ids = np.frombuffer(data, dtype="<i2", count=n, offset=pos).reshape(n_tok, k).copy()
        pos += 2 * n
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8, count=nb, offset=pos), bitorder="little")[:n_expert].astype(bool)
        pos += nb
        x = None
        if has_x:
            m = n_tok * n_embd
            if keep_x:
                x = np.frombuffer(data, dtype="<f2", count=m, offset=pos).reshape(n_tok, n_embd).copy()
            pos += 2 * m
        if layer == 0:
            close(cur)
            if max_rounds is not None and len(dump.rounds) >= max_rounds:
                cur = None
                break
            cur = {"round": rnd, "t_us": t_us, "n_tok": n_tok, "ids": [None] * n_layers,
                   "res": np.zeros((n_layers, n_expert), dtype=bool), "x": {},
                   "win": win if win is not None and len(win[0]) == n_tok else None}
            win = None
        if cur is None or rnd != cur["round"] or layer >= n_layers or n_tok != cur["n_tok"]:
            cur = None                      # a round that does not start at layer 0, or changes width: dropped
            continue
        cur["ids"][layer] = ids
        cur["res"][layer] = bits
        if x is not None:
            cur["x"][layer] = x
    close(cur)
    data.close()                            # (every array above is a copy: Windows cannot delete a mapped file)
    return dump


def sequences(rounds: list[Round], gap_ms: float, max_tok: int) -> list[list[Round]]:
    """Runs of rounds with no gap longer than gap_ms between their starts (a new request or a prompt in between), each
    round no wider than max_tok rows (wider windows are prompt reads, not decode)."""
    out, cur, last = [], [], None
    for r in rounds:
        broken = last is not None and (r.t_us - last.t_us > gap_ms * 1000 or r.index != last.index + 1)
        if r.n_tok > max_tok or broken:
            if len(cur) > 1:
                out.append(cur)
            cur = []
        if r.n_tok <= max_tok:
            cur.append(r)
        last = r
    if len(cur) > 1:
        out.append(cur)
    return out


def routed_counts(r: Round, n_layers: int, n_expert: int) -> np.ndarray:
    """(n_layers, n_expert): how many rows of the round routed each expert."""
    c = np.zeros((n_layers, n_expert), dtype=np.float32)
    for l, ids in enumerate(r.ids):
        v = ids.reshape(-1).astype(np.int64)
        v = v[(v >= 0) & (v < n_expert)]
        np.add.at(c[l], v, 1.0)
    return c


def misses(r: Round, counts: np.ndarray) -> np.ndarray:
    return (counts > 0) & ~r.resident


def recall_at(score: np.ndarray, resident_next: np.ndarray, miss_next: np.ndarray, budgets: list[int], rng) -> list[int]:
    """How many of miss_next the top-B scored pairs the tier does not hold cover, for each budget B (ties broken at random)."""
    s = np.where(resident_next, -np.inf, score.astype(np.float64) + rng.random(score.shape) * 1e-6)
    flat = s.reshape(-1)
    order = np.argsort(-flat, kind="stable")
    hit = miss_next.reshape(-1)[order]
    valid = np.isfinite(flat[order])
    hit = hit & valid
    cum = np.cumsum(hit)
    return [int(cum[min(b, len(cum)) - 1]) if b > 0 else 0 for b in budgets]


def gguf_tensors(path: str) -> tuple[dict[str, tuple[int, list[int], int]], int]:
    """name -> (ggml type, dims, absolute byte offset), from a GGUF v2/v3 header (no gguf-py: it needs yaml)."""
    f = open(path, "rb")
    rd = lambda n: f.read(n)
    u32 = lambda: struct.unpack("<I", rd(4))[0]
    u64 = lambda: struct.unpack("<Q", rd(8))[0]
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

    def skip(t):
        if t == 8:
            rd(u64())
        elif t == 9:
            et, n = u32(), u64()
            if et in sizes:
                rd(sizes[et] * n)
            else:
                for _ in range(n):
                    skip(et)
        else:
            rd(sizes[t])

    if rd(4) != b"GGUF":
        raise SystemExit(f"{path}: not a GGUF file")
    u32()
    n_t, n_kv = u64(), u64()
    align = 32
    for _ in range(n_kv):
        key = rd(u64()).decode("utf-8", "replace")
        t = u32()
        if key == "general.alignment" and t == 4:
            align = u32()
        else:
            skip(t)
    info = {}
    for _ in range(n_t):
        name = rd(u64()).decode("utf-8", "replace")
        dims = [u64() for _ in range(u32())]
        info[name] = (u32(), dims, u64())
    base = (f.tell() + align - 1) // align * align
    f.close()
    return {k: (t, dims, base + off) for k, (t, dims, off) in info.items()}, base


def load_routers(gguf: str, layers: list[int], n_expert: int, n_embd: int):
    info, _ = gguf_tensors(gguf)
    out = {}
    mm = np.memmap(gguf, dtype=np.uint8, mode="r")
    for l in layers:
        name = f"blk.{l}.ffn_gate_inp.weight"
        if name not in info:
            continue
        t, dims, off = info[name]
        n = n_expert * n_embd
        if t == 0:                           # F32
            w = np.frombuffer(mm, dtype="<f4", count=n, offset=off).copy()
        elif t == 1:                         # F16
            w = np.frombuffer(mm, dtype="<f2", count=n, offset=off).astype(np.float32)
        elif t == 30:                        # BF16: the high half of an f32
            w = (np.frombuffer(mm, dtype="<u2", count=n, offset=off).astype(np.uint32) << 16).view(np.float32)
        else:
            raise SystemExit(f"{name}: ggml type {t}, this tool reads F32/F16/BF16 routers")
        out[l] = w.reshape(n_expert, n_embd)
    del mm
    if not out:
        raise SystemExit(f"{gguf}: no blk.N.ffn_gate_inp.weight tensors")
    return out


def norm(x: np.ndarray) -> np.ndarray:
    m = float(x.max()) if x.size else 0.0
    return x / m if m > 0 else x


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def accepted_rows(r: Round, nxt: Round) -> int:
    """the rows of r its next window builds on: the next window starts where r's accepted tokens end"""
    if r.pos >= 0 and nxt.pos > r.pos:
        return max(1, min(r.n_tok, nxt.pos - r.pos))
    return r.n_tok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dumps", nargs="+", help="route traces, scored in this order (the online tables carry over)")
    ap.add_argument("--budgets", default="25,50,100,150,200,300,400", help="copies per round to score (all layers together)")
    ap.add_argument("--gap-ms", type=float, default=250.0, help="a longer pause between rounds starts a new sequence")
    ap.add_argument("--max-tok", type=int, default=8, help="wider rounds are prompt reads, not decode")
    ap.add_argument("--max-rounds", type=int, default=None, help="per trace")
    ap.add_argument("--expert-mib", type=float, default=2.64, help="one expert's bytes (NVFP4 GPTQ: 2.64 MiB a slot)")
    ap.add_argument("--cpu-ms", type=float, default=None, help="the run's CPU pool ms a round, to turn recall into ms")
    ap.add_argument("--gguf", default=None, help="the dense GGUF: its routers for the router-based predictors")
    ap.add_argument("--export", default=None, help="write the (round r features, round r+1 labels) pairs as .npz")
    ap.add_argument("--export-layers", default="5,15,25,35,45", help="the MoE inputs written as features (DFlash2's taps)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    budgets = [int(b) for b in a.budgets.split(",") if b.strip()]
    rng = np.random.default_rng(a.seed)

    preds = ["last", "ema-0.5", "freq", "tok", "tok+ema"]
    if a.gguf:
        preds += ["prev-router", "tok+ema+router"]
    hits = {p: np.zeros(len(budgets), dtype=np.int64) for p in preds}
    bl = [1, 2, 3, 4, 6, 8, 16]                          # the layer-ahead heuristic: copies per layer
    la_hit = {"next-router": np.zeros(len(bl), dtype=np.int64), "ema-0.5": np.zeros(len(bl), dtype=np.int64)}
    la_tot = 0
    agree = n_chk = 0
    total_miss = n_pairs = rounds_all = routed_tot = miss_tot = n_seqs = 0
    widths: dict[int, int] = {}
    per_file = []
    ex = {"x": [], "tokens": [], "n_tok": [], "prev": [], "routed": [], "miss": [], "resident": [], "seq": []}
    L = E = H = None
    R = None
    freq = None
    tok_tab: dict[int, np.ndarray] = {}
    tok_n: dict[int, int] = {}
    xl_export = [int(v) for v in a.export_layers.split(",") if v.strip()]

    for path in a.dumps:
        d = read_dump(path, keep_x=bool(a.gguf or a.export), max_rounds=a.max_rounds)
        if L is None:
            L, E, H = d.n_layers, d.n_expert, d.n_embd
            freq = np.zeros((L, E), dtype=np.float64)
            if a.gguf:
                rt = load_routers(a.gguf, list(range(L)), E, H)
                R = np.stack([rt[l] for l in range(L)]).transpose(0, 2, 1).copy()      # (L, H, E)
                del rt
        elif (d.n_layers, d.n_expert, d.n_embd) != (L, E, H):
            raise SystemExit(f"{path}: another model's geometry")
        have_x = set(range(L)) <= set(d.x_layers)
        if a.gguf and not have_x:
            print(f"{path}: the router predictors need every layer's MoE input (STRATA_ROUTE_TRACE_X=all); skipped there")
        seqs = sequences(d.rounds, a.gap_ms, a.max_tok)
        n_seqs += len(seqs)
        f_hits = {p: np.zeros(len(budgets), dtype=np.int64) for p in preds}
        f_miss = 0
        for s in seqs:
            ema5 = np.zeros((L, E), dtype=np.float64)
            prev = prev_r = prev_own = None
            for r in s:
                c = routed_counts(r, L, E)
                m = misses(r, c)
                routed_tot += int((c > 0).sum())
                miss_tot += int(m.sum())
                rounds_all += 1
                widths[r.n_tok] = widths.get(r.n_tok, 0) + 1
                own = None
                if R is not None and have_x:
                    X = np.stack([r.x[l] for l in range(L)]).astype(np.float32)        # (L, n_tok, H)
                    own = X @ R                                                         # (L, n_tok, E): each layer's router
                    if n_chk < 200_000:                                                 # the data check
                        kk = r.ids[0].shape[1]
                        top = np.argpartition(-own, kk - 1, axis=2)[:, :, :kk]
                        for l in range(L):
                            for row in range(r.n_tok):
                                agree += len(set(top[l, row].tolist()) & set(r.ids[l][row].tolist()))
                                n_chk += r.ids[l].shape[1]
                    ahead = X[:-1] @ R[1:]                                              # layer t+1's router on layer t's input
                    if prev is not None:
                        for t in range(L - 1):
                            mt = m[t + 1]
                            if not mt.any():
                                continue
                            la_tot += int(mt.sum())
                            k = r.ids[0].shape[1]
                            top = np.argpartition(-ahead[t], k - 1, axis=1)[:, :k]
                            v = np.zeros(E, dtype=np.float64)
                            np.add.at(v, top.reshape(-1), 1.0)
                            v += 1e-3 * norm(ahead[t].max(axis=0) - ahead[t].max(axis=0).min())
                            la_hit["next-router"] += recall_at(v[None], r.resident[t + 1][None], mt[None], bl, rng)
                            la_hit["ema-0.5"] += recall_at(ema5[t + 1][None], r.resident[t + 1][None], mt[None], bl, rng)
                if prev is not None:
                    total_miss += int(m.sum())
                    f_miss += int(m.sum())
                    n_pairs += 1
                    tokp = np.zeros((L, E), dtype=np.float64)
                    if r.tokens is not None:
                        for v in r.tokens.tolist():
                            if v in tok_tab:
                                tokp += tok_tab[v] / tok_n[v]
                    sc = {"last": prev, "ema-0.5": ema5, "freq": freq, "tok": tokp + 1e-3 * norm(ema5),
                          "tok+ema": norm(tokp) + 0.5 * norm(ema5)}
                    if "prev-router" in preds:
                        if prev_own is not None:
                            acc = accepted_rows(prev_r, r)
                            rp = softmax(prev_own[:, :acc, :], axis=2).mean(axis=1)    # (L, E)
                        else:
                            rp = np.zeros((L, E))
                        sc["prev-router"] = rp
                        sc["tok+ema+router"] = norm(tokp) + 0.5 * norm(ema5) + norm(rp)
                    for p in preds:
                        h = recall_at(sc[p], r.resident, m, budgets, rng)
                        hits[p] += h
                        f_hits[p] += h
                    if a.export and all(t in prev_r.x for t in xl_export):
                        acc = accepted_rows(prev_r, r)
                        xs = np.stack([prev_r.x[t][:acc].astype(np.float32) for t in xl_export])  # (T, acc, H)
                        ex["x"].append(np.stack([xs[:, -1], xs.mean(axis=1)]).astype(np.float16))
                        tk = np.full(8, -1, dtype=np.int32)
                        if r.tokens is not None:
                            tk[:min(8, len(r.tokens))] = r.tokens[:8]
                        ex["tokens"].append(tk)
                        ex["n_tok"].append(r.n_tok)
                        ex["prev"].append(np.packbits(prev > 0, axis=1, bitorder="little"))
                        ex["routed"].append(np.packbits(c > 0, axis=1, bitorder="little"))
                        ex["miss"].append(np.packbits(m, axis=1, bitorder="little"))
                        ex["resident"].append(np.packbits(r.resident, axis=1, bitorder="little"))
                        ex["seq"].append(n_seqs)
                if r.tokens is not None:                         # the token table learns round r after it is scored
                    for row, v in enumerate(r.tokens.tolist()):
                        t_ = tok_tab.get(v)
                        if t_ is None:
                            t_ = tok_tab[v] = np.zeros((L, E), dtype=np.float32)
                            tok_n[v] = 0
                        tok_n[v] += 1
                        for l in range(L):
                            t_[l, r.ids[l][row].astype(np.int64)] += 1.0
                ema5 = 0.5 * ema5 + c
                freq += c
                prev, prev_r, prev_own = c, r, own
        per_file.append((Path(path).stem, len(d.rounds), f_miss, {p: f_hits[p].copy() for p in preds}))
        print(f"{path}: {len(d.rounds)} rounds, {len(seqs)} sequences, {sum(len(s) - 1 for s in seqs)} pairs", flush=True)
        del d

    if n_pairs == 0:
        print("nothing to score")
        return 1
    print(f"\n{len(a.dumps)} traces, {rounds_all} rounds, {n_pairs} round pairs; rows a round: "
          + ", ".join(f"{w}: {n}" for w, n in sorted(widths.items())))
    print(f"per layer and round: {routed_tot / rounds_all / L:.2f} experts routed, {miss_tot / rounds_all / L:.2f} missed; "
          f"round r+1's misses to find: {total_miss / n_pairs:.1f} a round; {len(tok_tab)} distinct tokens in the table")
    if n_chk:
        print(f"data check: each layer's router on its traced MoE input finds {agree / n_chk * 100:.1f}% of the routed experts")
    mb = [b * a.expert_mib * 1.048576 for b in budgets]          # MB of copies a round; MB / (GB/s) = ms
    print("\nROUND AHEAD (known before round r+1 starts: rounds <= r, the next window's tokens)")
    print("copies a round (B)      " + "".join(f"{b:>8d}" for b in budgets))
    print("  PCIe 5 ms (~50 GB/s)  " + "".join(f"{x / 50:>8.1f}" for x in mb))
    print("  PCIe 4 ms (~25 GB/s)  " + "".join(f"{x / 25:>8.1f}" for x in mb))
    print("share of round r+1's misses covered:")
    for p in preds:
        print(f"{p:<24s}" + "".join(f"{h / total_miss * 100:>7.1f}%" for h in hits[p]))
    if len(per_file) > 1:
        j = budgets.index(100) if 100 in budgets else len(budgets) // 2
        show = [p for p in ("ema-0.5", "tok", "tok+ema", "tok+ema+router") if p in preds]
        print(f"\nper trace at B={budgets[j]} (the token table is online: later traces start warmer): " + ", ".join(show))
        for name, nr_, fm, fh in per_file:
            if fm:
                print(f"  {name:<20s} {nr_:>6d} rounds " + " ".join(f"{fh[p][j] / fm * 100:6.1f}%" for p in show))
    if a.cpu_ms:
        best = max(preds, key=lambda p: hits[p][-1])
        print(f"\n{best} at B={budgets[-1]}: {hits[best][-1] / total_miss * 100:.0f}% of the misses, at most "
              f"~{hits[best][-1] / total_miss * a.cpu_ms:.1f} of the {a.cpu_ms:.1f} CPU ms a round")
    if la_tot:
        print("\nLAYER AHEAD (layer t+1's router on layer t's MoE input, the same round: ~0.25 ms to copy), share of layer "
              "t+1's misses at a budget of copies per layer")
        print("  copies per layer        " + "".join(f"{b:>7d}" for b in bl))
        for p, h in la_hit.items():
            print(f"  {p:<24s}" + "".join(f"{x / la_tot * 100:>6.1f}%" for x in h))

    if a.export and ex["x"]:
        np.savez_compressed(a.export, x=np.array(ex["x"]), tokens=np.array(ex["tokens"]), n_tok=np.array(ex["n_tok"], dtype=np.int32),
                            prev=np.array(ex["prev"]), routed=np.array(ex["routed"]), miss=np.array(ex["miss"]),
                            resident=np.array(ex["resident"]), seq=np.array(ex["seq"], dtype=np.int32),
                            x_layers=np.array(xl_export, dtype=np.int32))
        print(f"\nexported {len(ex['x'])} pairs to {a.export}: x {np.array(ex['x']).shape} (last accepted row, mean of the "
              f"accepted rows) of layers {xl_export}, the next window's tokens, round r's routes, round r+1's labels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
