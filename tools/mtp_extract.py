"""tools/mtp_extract.py - the MTP block from a LOCAL checkpoint, in tools/mtp_fetch.py's output format.

    python tools/mtp_extract.py --model <checkpoint dir> --out DIR

mtp_fetch.py range-reads the 31 `mtp.*` tensors of the ORIGINAL Qwen/Qwen3.8-Flash-Next. A fine-tune or an
abliteration has its own draft head, and it should be used: OrcaRouter's abliteration edits the MTP head's
residual writers together with the main model's "so speculative decoding keeps working", so the original head would
draft for a model that is no longer there. This writes the same thing from a checkpoint on disk -
DIR/tensors/<name>.bin (raw bytes, as stored) and DIR/mtp-manifest.json - so tools/mtp_pack.py packs it unchanged.
Only BF16 tensors are accepted, the form mtp_pack reads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from safetensors import safe_open

DTYPE = {"torch.bfloat16": "BF16"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model, out = pathlib.Path(a.model), pathlib.Path(a.out)
    index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    names = sorted(n for n in index if n.startswith("mtp."))
    if not names:
        print("no mtp.* tensors in", model)
        return 1
    (out / "tensors").mkdir(parents=True, exist_ok=True)
    manifest = []
    for name in names:
        shard = index[name]
        with safe_open(model / shard, framework="pt") as f:
            t = f.get_tensor(name)
        dtype = DTYPE.get(str(t.dtype))
        if dtype is None:
            print("%s is %s; mtp_pack reads BF16 only" % (name, t.dtype))
            return 1
        raw = t.contiguous().view(-1).view(dtype=__import__("torch").int16).numpy().tobytes()
        path = out / "tensors" / (name + ".bin")
        path.write_bytes(raw)
        manifest.append(dict(name=name, shard=shard, dtype=dtype, shape=list(t.shape), bytes=len(raw),
                             file=str(path.relative_to(out)).replace("\\", "/"),
                             sha256=hashlib.sha256(raw).hexdigest()))
        print("  %-62s %s %s" % (name, dtype, tuple(t.shape)))
    (out / "mtp-manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("%d tensors, %.2f GB -> %s" % (len(manifest), sum(m["bytes"] for m in manifest) / 1e9, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
