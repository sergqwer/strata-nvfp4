"""nsys_win.py DB MARKER_RX [--lo F --hi F] [--layer] - decode verify windows in an nsys sqlite (--cuda-graph-trace=node).
A window = the span between consecutive kernels matching MARKER_RX (one per verify window, e.g. the head).  Over the windows
in the decode's [lo, hi] fraction: wall per window, kernels per window, union GPU busy / idle, and per kernel name:
launches and summed duration per window (all streams), and on the busiest stream: summed gaps before each kernel name.
--layer: also print one window's busiest-stream sequence (name, start offset, duration, gap) for a structural comparison."""
import re
import sqlite3
import sys
from collections import defaultdict

db, mrx = sys.argv[1], re.compile(sys.argv[2])
lo = float(sys.argv[sys.argv.index("--lo") + 1]) if "--lo" in sys.argv else 0.3
hi = float(sys.argv[sys.argv.index("--hi") + 1]) if "--hi" in sys.argv else 0.9
c = sqlite3.connect(db)
S = dict(c.execute("select id, value from StringIds"))
ks = [(s, e, S.get(n, "?"), st) for s, e, n, st in
      c.execute("select start, end, shortName, streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
marks = [k for k in ks if mrx.search(k[2])]
if len(marks) < 10:
    sys.exit("only %d marker kernels" % len(marks))
# the decode: marker kernels; keep windows between the lo/hi fractions of the markers
i0, i1 = int(len(marks) * lo), int(len(marks) * hi)
ms = marks[i0:i1 + 1]
t0, t1 = ms[0][0], ms[-1][0]
nw = len(ms) - 1
win = [k for k in ks if t0 <= k[0] < t1]
# union busy
busy, cs_, ce_ = 0, None, None
for s, e, _, _ in sorted(win):
    e = min(e, t1)
    if cs_ is None or s > ce_:
        if cs_ is not None:
            busy += ce_ - cs_
        cs_, ce_ = s, e
    else:
        ce_ = max(ce_, e)
if cs_ is not None:
    busy += ce_ - cs_
wall = (t1 - t0) / nw / 1e6
print("%s: %d windows, wall %.3f ms/window, kernels %.0f/window, GPU busy (union) %.3f ms, idle %.3f ms" %
      (db, nw, wall, len(win) / nw, busy / nw / 1e6, wall - busy / nw / 1e6))
streams = defaultdict(int)
for s, e, n, st in win:
    streams[st] += e - s
print("streams (busy ms/window):", ", ".join("%d: %.3f" % (st, v / nw / 1e6) for st, v in sorted(streams.items(), key=lambda x: -x[1])[:8]))
main = max(streams, key=streams.get)
per = defaultdict(lambda: [0, 0, 0])   # count, duration, gap before (main stream)
prev_end = None
for s, e, n, st in win:
    p = per[n]
    p[0] += 1
    p[1] += e - s
    if st == main:
        if prev_end is not None and s > prev_end:
            p[2] += s - prev_end
        prev_end = max(prev_end or 0, e)
tot_gap = sum(p[2] for p in per.values())
print("main stream %d: kernels %.0f/window, gaps %.3f ms/window" % (main, sum(1 for k in win if k[3] == main) / nw, tot_gap / nw / 1e6))
print("%-60s %8s %9s %9s" % ("kernel", "n/win", "ms/win", "gap ms"))
for n, (cnt, d, g) in sorted(per.items(), key=lambda x: -x[1][1])[:int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 45]:
    print("%-60s %8.1f %9.3f %9.3f" % (n[:60], cnt / nw, d / nw / 1e6, g / nw / 1e6))
if "--layer" in sys.argv:
    mid = len(ms) // 2
    a, b = ms[mid][0], ms[mid + 1][0]
    pe = None
    for s, e, n, st in [k for k in ks if a <= k[0] < b]:
        print("%s %8.1f %7.2f %7.2f  %s" % ("*" if st == main else " ", (s - a) / 1e3, (e - s) / 1e3,
                                              (s - pe) / 1e3 if pe is not None and st == main else 0.0, n[:70]))
        if st == main:
            pe = e
