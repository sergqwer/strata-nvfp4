"""ab_sum.py OUT NAME - summary of the last ab.sh run in OUT: per fixture, the mean (min..max) of each arm and B vs A."""
import re
import statistics as st
import sys

lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
start = max(i for i, l in enumerate(lines) if l.startswith("# ab "))
pat = re.compile(r"^([AB]) (\w+) (\d+) \| \d+ generated in \d+ ms \(([0-9.]+) tok/s\) \| (\d+) windows, avg T ([0-9.]+), "
                 r"([0-9.]+) tokens/window, ([0-9.]+) ms/window.*?verify ([0-9.]+) \(GPU-reach wait ([0-9.]+) \+ per-layer host ([0-9.]+)")
d = {}
for l in lines[start:]:
    m = pat.match(l)
    if not m:
        continue
    arm, fx = m.group(1), m.group(2)
    v = dict(tps=float(m.group(4)), T=float(m.group(6)), tpw=float(m.group(7)), ms=float(m.group(8)),
             verify=float(m.group(9)), reach=float(m.group(10)), host=float(m.group(11)))
    d.setdefault(fx, {}).setdefault(arm, []).append(v)
print("summary %s (mean of the runs; B vs A on ms/window and tok/s)" % sys.argv[2])
for fx, arms in d.items():
    for arm in "AB":
        r = arms.get(arm, [])
        if not r:
            continue
        mean = lambda k: st.mean(x[k] for x in r)
        print("  %-6s %s n=%d  ms/window %.2f (%.2f..%.2f)  tok/s %.1f  T %.2f  tok/window %.2f  verify %.2f  reach %.2f  host %.2f"
              % (fx, arm, len(r), mean("ms"), min(x["ms"] for x in r), max(x["ms"] for x in r), mean("tps"), mean("T"),
                 mean("tpw"), mean("verify"), mean("reach"), mean("host")))
    if arms.get("A") and arms.get("B"):
        a = st.mean(x["ms"] for x in arms["A"]); b = st.mean(x["ms"] for x in arms["B"])
        ta = st.mean(x["tps"] for x in arms["A"]); tb = st.mean(x["tps"] for x in arms["B"])
        print("  %-6s B vs A: ms/window %+.3f (%+.2f%%), tok/s %+.2f%%" % (fx, b - a, 100 * (b - a) / a, 100 * (tb - ta) / ta))
