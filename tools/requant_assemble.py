"""tools/requant_assemble.py PLAN OUT [NVFP4_PACK Q8D_PACK] - an experts pack whose layers come from two packs requant.py built from the same GPTQ run:
orca-gptq4 (NVFP4 everywhere) and orca-gptq-d8 (the same GPTQ gate/up + Q8_0 down). A plan layer with "d": "q8_0"
is copied from the second, every other layer from the first - the GPTQ gate/up rows are the same in both (same
weights, same calibration, deterministic), so no re-quantization is needed. dense.bin / index.txt hardlinked,
tokenizer copied, native_experts.txt rewritten with the new offsets."""
import json
import os
import pathlib
import shutil
import sys

P = pathlib.Path(r"C:\LLM_models\Strata-data\packs")
A = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else P / "orca-gptq4"
B = pathlib.Path(sys.argv[4]) if len(sys.argv) > 4 else P / "orca-gptq-d8"
plan = json.load(open(sys.argv[1], encoding="utf-8"))
out = pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)


def layout(pack):
    rows = {}
    for line in open(pack / "native_experts.txt", encoding="utf-8"):
        if line.startswith("#") or not line.strip():
            continue
        l, gt, dt, off, blob = (int(x) for x in line.split()[:5])
        rows[l] = (gt, dt, off, blob)
    return rows


la, lb = layout(A), layout(B)
for name in ("dense.bin", "index.txt"):
    if not (out / name).exists():
        os.link(A / name, out / name)
if not (out / "tokenizer").exists():
    shutil.copytree(A / "tokenizer", out / "tokenizer")
lines, offset = [], 0
with open(A / "experts.bin", "rb") as fa, open(B / "experts.bin", "rb") as fb, open(out / "experts.bin.tmp", "wb") as fo:
    for l in range(48):
        use_b = plan.get("layers", {}).get(str(l), plan.get("default", {})).get("d") == "q8_0"
        f, (gt, dt, off, blob) = (fb, lb[l]) if use_b else (fa, la[l])
        f.seek(off)
        left = blob * 512
        while left:
            chunk = f.read(min(left, 1 << 26))
            fo.write(chunk)
            left -= len(chunk)
        lines.append("%d %d %d %d %d" % (l, gt, dt, offset, blob))
        offset += blob * 512
(out / "experts.bin.tmp").replace(out / "experts.bin")
head = ("# strata native experts v3: layer gu_type d_type offset blob_bytes (n_expert 512, total %d; GPTQ from the BF16 "
        "checkpoint, assembled by tools/requant_assemble.py from %s)" % (offset, sys.argv[1]))
(out / "native_experts.txt").write_text(head + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
(out / "requant.json").write_text(json.dumps({"plan": plan, "assembled_from": [str(A), str(B)]}, indent=1), encoding="utf-8")
print("%s: %.2f GiB, Q8_0 down in layers %s" % (out, offset / 2 ** 30,
      sorted(int(k) for k, v in plan.get("layers", {}).items() if v.get("d") == "q8_0")))
