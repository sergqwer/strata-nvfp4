"""nsys_kern.py DB [lo hi] - per-window kernel table of the decode windows (window = q80p_screen_kernel to the next one, the
middle windows lo..hi): every kernel name's count and summed duration (us) per window, and the window's wall time."""
import collections
import sqlite3
import sys

db = sys.argv[1]
lo = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
hi = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
c = sqlite3.connect(db)
S = dict(c.execute("select id, value from StringIds"))
ks = [(s, e, S[n]) for s, e, n in c.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
heads = [k for k in ks if k[2] in ("q80p_screen_kernel", "q80p_vhead_kernel")]
if not heads:
    # Basalt: its head kernel names differ; take the most frequent once-per-window kernel the caller names
    name = sys.argv[4]
    heads = [k for k in ks if k[2] == name]
i0, i1 = int(len(heads) * lo), int(len(heads) * hi)
cnt, dur = collections.Counter(), collections.Counter()
wall = 0.0
nw = 0
j = 0
for i in range(i0, i1):
    a, b = heads[i][0], heads[i + 1][0]
    wall += (b - a) / 1000.0
    nw += 1
    while j < len(ks) and ks[j][0] < a:
        j += 1
    k = j
    while k < len(ks) and ks[k][0] < b:
        cnt[ks[k][2]] += 1
        dur[ks[k][2]] += (ks[k][1] - ks[k][0]) / 1000.0
        k += 1
print(f"{nw} windows, wall {wall / nw:.1f} us/window, kernels {sum(cnt.values()) / nw:.0f}/window, busy {sum(dur.values()) / nw:.0f} us")
for n, d in dur.most_common():
    print(f"{d / nw:9.1f} us {cnt[n] / nw:7.1f} x  {n[:90]}")
