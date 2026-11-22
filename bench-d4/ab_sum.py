"""ab_sum.py OUT NAME - bench-d3/ab_sum.py + ms/token, draft ms and slots (E1: windows of different widths)."""
import re
import statistics as st
import sys

lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
start = max(i for i, l in enumerate(lines) if l.startswith("# ab "))
pat = re.compile(r"^([AB]) (\w+) (\d+) \| \d+ generated in \d+ ms \(([0-9.]+) tok/s\) \| (\d+) windows, avg T ([0-9.]+), "
                 r"([0-9.]+) tokens/window, ([0-9.]+) ms/window.*?verify ([0-9.]+) \(GPU-reach wait ([0-9.]+) \+ per-layer host ([0-9.]+)"
                 r".*?CPU ([0-9.]+)\].*? draft ([0-9.]+).*?slots (\d+)")
d = {}
for l in lines[start:]:
    m = pat.match(l)
    if not m:
        continue
    arm, fx = m.group(1), m.group(2)
    v = dict(tps=float(m.group(4)), T=float(m.group(6)), tpw=float(m.group(7)), ms=float(m.group(8)),
             verify=float(m.group(9)), reach=float(m.group(10)), host=float(m.group(11)), cpu=float(m.group(12)),
             draft=float(m.group(13)), slots=int(m.group(14)))
    v["mpt"] = v["ms"] / v["tpw"]
    d.setdefault(fx, {}).setdefault(arm, []).append(v)
print("summary %s (mean of the runs; B vs A on ms/token and tok/s)" % sys.argv[2])
for fx, arms in d.items():
    for arm in "AB":
        r = arms.get(arm, [])
        if not r:
            continue
        mean = lambda k: st.mean(x[k] for x in r)
        print("  %-6s %s n=%d  ms/tok %.3f (%.3f..%.3f)  tok/s %.1f  ms/window %.2f  T %.2f  tok/window %.2f  verify %.2f  reach %.2f"
              "  host %.2f  CPU %.2f  draft %.2f  slots %d"
              % (fx, arm, len(r), mean("mpt"), min(x["mpt"] for x in r), max(x["mpt"] for x in r), mean("tps"), mean("ms"),
                 mean("T"), mean("tpw"), mean("verify"), mean("reach"), mean("host"), mean("cpu"), mean("draft"), round(mean("slots"))))
    if arms.get("A") and arms.get("B"):
        a = st.mean(x["mpt"] for x in arms["A"]); b = st.mean(x["mpt"] for x in arms["B"])
        ta = st.mean(x["tps"] for x in arms["A"]); tb = st.mean(x["tps"] for x in arms["B"])
        print("  %-6s B vs A: ms/token %+.4f (%+.2f%%), tok/s %+.2f%%" % (fx, b - a, 100 * (b - a) / a, 100 * (tb - ta) / ta))
