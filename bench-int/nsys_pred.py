"""nsys_pred.py DB KERNEL [lo hi] - what KERNEL waits for: for each launch of KERNEL in the middle windows, the kernels
running or ending in the 40 us before its start (name, end relative to KERNEL's start); and KERNEL's mean duration.
Also, per launch, the gap from the previous kernel on its own stream."""
import collections
import sqlite3
import sys

db, name = sys.argv[1], sys.argv[2]
lo = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
hi = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9
c = sqlite3.connect(db)
S = dict(c.execute("select id, value from StringIds"))
ks = [(s, e, S[n], st) for s, e, n, st in c.execute(
    "select start, end, shortName, streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
tg = [i for i, k in enumerate(ks) if k[2] == name]
tg = tg[int(len(tg) * lo):int(len(tg) * hi)]
last = collections.Counter()
lastgap = collections.defaultdict(float)
dur = 0.0
samegap = 0.0
for i in tg:
    s0 = ks[i][0]
    dur += (ks[i][1] - s0) / 1000.0
    # the kernel that ended last before this one started (any stream)
    best = None
    for j in range(i - 1, max(0, i - 400), -1):
        if ks[j][1] <= s0 and (best is None or ks[j][1] > ks[best][1]):
            best = j
    if best is not None:
        last[ks[best][2]] += 1
        lastgap[ks[best][2]] += (s0 - ks[best][1]) / 1000.0
    for j in range(i - 1, max(0, i - 400), -1):
        if ks[j][3] == ks[i][3]:
            samegap += (s0 - ks[j][1]) / 1000.0
            break
n = len(tg)
print(f"{name}: {n} launches, mean {dur / n:.2f} us, mean gap from its stream's previous kernel {samegap / n:.2f} us")
for k, v in last.most_common(8):
    print(f"  last to end before it: {k[:60]:60s} {v / n * 100:5.1f}%  gap {lastgap[k] / v:.2f} us")
