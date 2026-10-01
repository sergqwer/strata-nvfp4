"""tools/requant.py - Strata expert packs re-quantized from the BF16 checkpoint, measured on real activations.

    python tools/requant.py analyze --bf16 DIR --calib PREFIX --out errors.jsonl [--layers 0-47]
    python tools/requant.py pack    --bf16 DIR --calib PREFIX --plan plan.json --base PACK --out PACK2 [--method gptq]

The calibration is the engine's own MoE inputs: STRATA_DUMP_MOE_INPUT=PREFIX STRATA_DUMP_MOE_LAYER=all writes
PREFIX_lNN.bin per layer (records {T, N, K, 2} int64, T x N float32 inputs, T x K expert ids int32, T x K routing
weights float32). An expert's tokens are the rows routed to it, weighted by the square of their routing weight (its
output enters the residual times that weight).

analyze: per layer, on held-out tokens (every 4th), the routed-weighted error of each expert's whole output
  y = down(silu(gate x) * up x) against BF16, for the 4-bit methods (rtn = ModelOpt's rounding, search = per-block
  scale search weighted by the inputs' second moments, gptq) and for 8 bits (Q8_0) on gate/up, down or both.
pack: experts.bin + native_experts.txt for a plan {"layers": {"l": {"gu": "nvfp4"|"q8_0", "d": ...}}} (missing
  layers: nvfp4 both); dense.bin, index.txt and the tokenizer are copied from --base (they do not depend on the
  experts). The blob is gate | up | down [| {s_gate, s_up, s_down, 0} when any projection is NVFP4, 1.0 for the
  others].
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import nvfp4_codec as C  # noqa: E402

N_EMBD, N_FF, N_EXP = 2560, 640, 512
NVFP4_TYPE, Q8_0_TYPE = 40, 8
PRIOR_TOKENS = 256        # pseudo-tokens of the layer's average input mixed into each expert's Hessian
BATCH = 64                # experts per GPU batch


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


# ---------------------------------------------------------------------------------------------------------- inputs
class Bf16:
    def __init__(self, d):
        self.d = pathlib.Path(d)
        self.idx = json.load(open(self.d / "model.safetensors.index.json"))["weight_map"]

    def layer(self, l, dev):
        """gate_up [512, 1280, 2560] (gate rows first), down [512, 2560, 640], as bfloat16 on `dev`."""
        out = []
        for name in ("gate_up_proj", "down_proj"):
            k = "model.language_model.layers.%d.mlp.experts.%s" % (l, name)
            with safe_open(self.d / self.idx[k], "pt") as f:
                out.append(f.get_tensor(k).to(dev))
        return out


def load_calib(prefix, l):
    """All records of layer l: X [T, N] float32, ids [T, K] int64, w [T, K] float32 (CPU)."""
    raw = np.fromfile("%s_l%02d.bin" % (prefix, l), dtype=np.uint8)
    xs, ids, ws, o = [], [], [], 0
    while o < raw.size:
        T, N, K, ver = np.frombuffer(raw[o:o + 32].tobytes(), dtype=np.int64)
        assert ver == 2 and N == N_EMBD, (T, N, K, ver)
        o += 32
        xs.append(np.frombuffer(raw[o:o + 4 * T * N].tobytes(), dtype=np.float32).reshape(T, N)); o += 4 * T * N
        ids.append(np.frombuffer(raw[o:o + 4 * T * K].tobytes(), dtype=np.int32).reshape(T, K)); o += 4 * T * K
        ws.append(np.frombuffer(raw[o:o + 4 * T * K].tobytes(), dtype=np.float32).reshape(T, K)); o += 4 * T * K
    return (torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ids)).long(),
            torch.from_numpy(np.concatenate(ws)))


def routed(ids, w, mask):
    """For every expert: (token rows, routing weights) among the tokens where mask is true."""
    T, K = ids.shape
    tok = torch.arange(T).unsqueeze(1).expand(T, K)
    keep = mask.unsqueeze(1).expand(T, K)
    e, t, ww = ids[keep], tok[keep], w[keep]
    order = torch.argsort(e, stable=True)
    e, t, ww = e[order], t[order], ww[order]
    counts = torch.bincount(e, minlength=N_EXP)
    starts = torch.cumsum(counts, 0) - counts
    return [(t[s:s + c], ww[s:s + c]) for s, c in zip(starts.tolist(), counts.tolist())]


# ------------------------------------------------------------------------------------------------------ hessians
def hessian(rows_x, wts, prior, n0):
    """sum w^2 x x^T over an expert's rows, mixed with `prior` (an average second moment) as n0 pseudo-weight."""
    xw = rows_x * wts.unsqueeze(1)
    S = xw.T @ xw
    return (S + n0 * prior) / (float((wts ** 2).sum()) + n0)


# ------------------------------------------------------------------------------------------------- quantizations
def q_nvfp4(W, H, method):
    """W [E, rows, cols] float32 (CUDA), H [E, cols, cols]. Returns (dequantized W, codes, scale bytes, s2 [E])."""
    E = W.shape[0]
    if method == "gptq":
        codes, sb, s2 = C.nvfp4_gptq(W, H)
    else:
        cs, sbs, s2s = [], [], []
        for e in range(E):
            if method == "rtn":
                c, sb, s2 = C.nvfp4_rtn(W[e])
            else:   # search, weighted by the inputs' second moments (diag H)
                c, sb, s2 = C.nvfp4_search(W[e], h=torch.diagonal(H[e]))
            cs.append(c); sbs.append(sb); s2s.append(s2)
        codes, sb, s2 = torch.stack(cs), torch.stack(sbs), torch.tensor(s2s, device=W.device)
    return C.nvfp4_dequant_batched(codes, sb, s2), codes, sb, s2


def q_q8(W):
    """W [E, rows, cols] -> (dequantized, raw Q8_0 rows [E, rows, cols/32*34])."""
    raw = torch.stack([C.q8_0(W[e]) for e in range(W.shape[0])])
    deq = torch.stack([C.q8_0_dequant(raw[e], W.shape[2]) for e in range(W.shape[0])])
    return deq, raw


# --------------------------------------------------------------------------------------------------------- analyze
def expert_out(Wgu, Wd, x):
    gu = x @ Wgu.T
    h = torch.nn.functional.silu(gu[:, :N_FF]) * gu[:, N_FF:]
    return h, h @ Wd.T


def analyze_layer(bf, l, prefix, dev, methods):
    gu_all, d_all = bf.layer(l, "cpu")
    X, ids, w = load_calib(prefix, l)
    T = X.shape[0]
    test = torch.arange(T) % 4 == 3
    fit_lists, test_lists = routed(ids, w, ~test), routed(ids, w, test)
    Xd = X.to(dev)
    prior = (Xd[~test.to(dev)].T @ Xd[~test.to(dev)]) / float((~test).sum())
    n0 = PRIOR_TOKENS * float((w ** 2).mean())
    acc = {}
    for b0 in range(0, N_EXP, BATCH):
        es = list(range(b0, min(b0 + BATCH, N_EXP)))
        Wgu = gu_all[es].to(dev).float()
        Wd = d_all[es].to(dev).float()
        Hgu = torch.stack([hessian(Xd[fit_lists[e][0].to(dev)], fit_lists[e][1].to(dev), prior, n0) for e in es])
        # the reference hidden of the fit rows (for down's Hessian under the BF16 gate/up)
        q = {}
        for m in methods:
            Wgu_q = q_nvfp4(Wgu, Hgu, m)[0]
            # down's Hessian from the hidden the quantized gate/up produce (as GPTQ quantizes sequentially)
            Hd = []
            for j, e in enumerate(es):
                rows, wts = fit_lists[e]
                hq, _ = expert_out(Wgu_q[j], Wd[j], Xd[rows.to(dev)])
                pd = torch.eye(N_FF, device=dev) * float((hq ** 2).mean()) if hq.numel() else torch.eye(N_FF, device=dev)
                Hd.append(hessian(hq, wts.to(dev), pd, n0))
            Wd_q = q_nvfp4(Wd, torch.stack(Hd), m)[0]
            q[m] = (Wgu_q, Wd_q)
        Wgu8, Wd8 = q_q8(Wgu)[0], q_q8(Wd)[0]
        for j, e in enumerate(es):
            rows, wts = test_lists[e]
            if rows.numel() == 0:
                continue
            x = Xd[rows.to(dev)]
            w2 = (wts.to(dev) ** 2).unsqueeze(1)
            _, y = expert_out(Wgu[j], Wd[j], x)
            acc["energy"] = acc.get("energy", 0.0) + float((w2 * y ** 2).sum())
            cfgs = {"q8_q8": (Wgu8[j], Wd8[j])}
            for m in methods:
                cfgs[m] = q[m][0][j], q[m][1][j]
                cfgs[m + "_gu8"] = Wgu8[j], q[m][1][j]
                cfgs[m + "_d8"] = q[m][0][j], Wd8[j]
            for k, (a, b) in cfgs.items():
                _, yq = expert_out(a, b, x)
                acc[k] = acc.get(k, 0.0) + float((w2 * (yq - y) ** 2).sum())
        del Wgu, Wd, Hgu, q
        torch.cuda.empty_cache()
    rel = {k: (v / acc["energy"]) for k, v in acc.items() if k != "energy"}
    return {"layer": l, "tokens": int(T), "test_tokens": int(test.sum()), "energy": acc["energy"], "rel_mse": rel}


# ------------------------------------------------------------------------------------------------------------ pack
def layer_cfg(plan, l):
    d = plan.get("default", {"gu": "nvfp4", "d": "nvfp4"})
    return plan.get("layers", {}).get(str(l), d)


def pack_layer(bf, l, cfg, prefix, method, dev):
    """[512, blob] uint8 of layer l under cfg {"gu": .., "d": ..}, and (gu_type, d_type)."""
    gu_all, d_all = bf.layer(l, "cpu")
    need_h = method != "rtn" and ("nvfp4" in (cfg["gu"], cfg["d"]))
    if need_h:
        X, ids, w = load_calib(prefix, l)
        lists = routed(ids, w, torch.ones(X.shape[0], dtype=torch.bool))
        Xd = X.to(dev)
        prior = (Xd.T @ Xd) / float(X.shape[0])
        n0 = PRIOR_TOKENS * float((w ** 2).mean())
    blobs = []
    for b0 in range(0, N_EXP, BATCH):
        es = list(range(b0, min(b0 + BATCH, N_EXP)))
        Wgu = gu_all[es].to(dev).float()
        Wd = d_all[es].to(dev).float()
        B = len(es)
        if cfg["gu"] == "nvfp4":
            Hgu = (torch.stack([hessian(Xd[lists[e][0].to(dev)], lists[e][1].to(dev), prior, n0) for e in es])
                   if need_h else torch.eye(N_EMBD, device=dev).expand(B, -1, -1).contiguous())
            gu_deq, codes, sb, s2 = q_nvfp4(Wgu, Hgu, method)
            del Hgu
            gu_rows = [C.to_ggml(codes[j], sb[j]) for j in range(B)]          # [1280, 1440]: gate rows then up rows
            s_gu = s2.tolist()
        else:
            gu_deq, raw = q_q8(Wgu)
            gu_rows = [raw[j] for j in range(B)]
            s_gu = [1.0] * B
        if cfg["d"] == "nvfp4":
            if need_h:
                Hd = []
                for j, e in enumerate(es):
                    rows, wts = lists[e]
                    hq, _ = expert_out(gu_deq[j], Wd[j], Xd[rows.to(dev)])
                    pd = torch.eye(N_FF, device=dev) * (float((hq ** 2).mean()) if hq.numel() else 1.0)
                    Hd.append(hessian(hq, wts.to(dev), pd, n0))
                Hd = torch.stack(Hd)
            else:
                Hd = torch.eye(N_FF, device=dev).expand(B, -1, -1).contiguous()
            _, codes, sb, s2 = q_nvfp4(Wd, Hd, method)
            del Hd
            d_rows = [C.to_ggml(codes[j], sb[j]) for j in range(B)]
            s_d = s2.tolist()
        else:
            _, raw = q_q8(Wd)
            d_rows = [raw[j] for j in range(B)]
            s_d = [1.0] * B
        tail = "nvfp4" in (cfg["gu"], cfg["d"])
        for j in range(B):
            parts = [gu_rows[j].reshape(-1), d_rows[j].reshape(-1)]
            if tail:
                parts.append(torch.tensor([s_gu[j], s_gu[j], s_d[j], 0.0], dtype=torch.float32,
                                          device=dev).view(torch.uint8))
            blobs.append(torch.cat(parts).cpu())
        del Wgu, Wd, gu_deq
        torch.cuda.empty_cache()
    types = (NVFP4_TYPE if cfg["gu"] == "nvfp4" else Q8_0_TYPE, NVFP4_TYPE if cfg["d"] == "nvfp4" else Q8_0_TYPE)
    return torch.stack(blobs), types


def pack(a, bf, dev):
    plan = json.load(open(a.plan, encoding="utf-8")) if a.plan else {}
    out, base = pathlib.Path(a.out), pathlib.Path(a.base)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("dense.bin", "index.txt"):
        dst = out / name
        if not dst.exists():
            try:
                import os
                os.link(base / name, dst)        # same volume: a second name, no copy
            except OSError:
                shutil.copy2(base / name, dst)
    if not (out / "tokenizer").exists():
        shutil.copytree(base / "tokenizer", out / "tokenizer")
    lines, offset = [], 0
    part = out / "experts.bin.tmp"
    with open(part, "wb") as fo:
        for l in range(48):
            t0 = time.time()
            cfg = layer_cfg(plan, l)
            blobs, (gt, dt) = pack_layer(bf, l, cfg, a.calib, a.method, dev)
            fo.write(blobs.numpy().tobytes())
            lines.append("%d %d %d %d %d" % (l, gt, dt, offset, blobs.shape[1]))
            offset += blobs.shape[1] * N_EXP
            log("layer %2d: gu %s d %s, blob %d B, %.1f s" % (l, cfg["gu"], cfg["d"], blobs.shape[1], time.time() - t0))
    part.replace(out / "experts.bin")
    head = ("# strata native experts v3: layer gu_type d_type offset blob_bytes (n_expert %d, total %d; experts.bin "
            "re-quantized from the BF16 checkpoint by tools/requant.py, method %s)" % (N_EXP, offset, a.method))
    (out / "native_experts.txt").write_text(head + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    (out / "requant.json").write_text(json.dumps({"plan": plan, "method": a.method, "bf16": a.bf16, "calib": a.calib,
                                                   "total": offset, "time": time.strftime("%Y-%m-%d %H:%M")},
                                                  indent=1), encoding="utf-8")
    log("experts.bin: %.2f GiB" % (offset / 2 ** 30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["analyze", "pack"])
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--calib", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="0-47")
    ap.add_argument("--methods", default="rtn,search,gptq")
    ap.add_argument("--plan", default="")
    ap.add_argument("--base", default=r"C:\LLM_models\Strata-data\packs\orca-nvfp4")
    ap.add_argument("--method", default="gptq", choices=["rtn", "search", "gptq"])
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.layers.split("-"))
    bf = Bf16(a.bf16)
    dev = "cuda"
    if a.mode == "pack":
        pack(a, bf, dev)
        return
    with open(a.out, "a", encoding="utf-8") as fo:
        for l in range(lo, hi + 1):
            t0 = time.time()
            r = analyze_layer(bf, l, a.calib, dev, a.methods.split(","))
            r["seconds"] = round(time.time() - t0, 1)
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            log("layer %d: %s" % (l, " ".join("%s %.5f" % (k, v) for k, v in sorted(r["rel_mse"].items()))))


if __name__ == "__main__":
    main()
