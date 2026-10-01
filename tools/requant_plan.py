"""tools/requant_plan.py ERRORS.jsonl OUTDIR BUDGET_GIB... - plans for tools/requant.py from `requant.py analyze`.

Per layer the absolute error (relative MSE x the layer's routed output energy, held-out tokens) of GPTQ NVFP4 and of
GPTQ gate/up + Q8_0 down; the layers whose down goes to Q8_0 are taken by error removed per extra byte until
experts.bin reaches each budget. Writes OUTDIR/plan_d8_<GiB>.json ({"default": nvfp4, "layers": {l: q8_0 down}}).
"""
import json
import pathlib
import sys

errors, outdir, budgets = sys.argv[1], pathlib.Path(sys.argv[2]), [float(x) for x in sys.argv[3:]]
rows = sorted((json.loads(l) for l in open(errors, encoding="utf-8")), key=lambda r: r["layer"])
NE = 512
B4 = 1280 * 1440 + 2560 * 360 + 16        # NVFP4 gate/up/down + the scale tail
B8 = 1280 * 1440 + 2560 * 680 + 16        # NVFP4 gate/up + Q8_0 down + the tail
gib = lambda b: b / 2 ** 30
gain = []
for r in rows:
    m, E = r["rel_mse"], r["energy"]
    gain.append((m["gptq"] * E - m["gptq_d8"] * E, r["layer"]))
tot_rtn = sum(r["rel_mse"]["rtn"] * r["energy"] for r in rows)
tot4 = sum(r["rel_mse"]["gptq"] * r["energy"] for r in rows)
print("%d layers | summed error: GPTQ %.1f%% of RTN" % (len(rows), 100 * tot4 / tot_rtn))
for bud in budgets:
    size, err, pick = len(rows) * NE * B4, tot4, []
    for g, l in sorted(gain, reverse=True):
        if gib(size + NE * (B8 - B4)) > bud:
            break
        size += NE * (B8 - B4)
        err -= g
        pick.append(l)
    plan = {"default": {"gu": "nvfp4", "d": "nvfp4"}, "layers": {str(l): {"gu": "nvfp4", "d": "q8_0"} for l in pick}}
    (outdir / ("plan_d8_%d.json" % bud)).write_text(json.dumps(plan, indent=1), encoding="utf-8")
    print("budget %.0f GiB: Q8_0 down in %d layers, %.1f GiB, error %.1f%% of RTN: %s" % (
        bud, len(pick), gib(size), 100 * err / tot_rtn, sorted(pick)))
