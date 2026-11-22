"""nsys_front_list.py DB [layer] [nwin] - the kernels of one attention layer's front (combine end of layer-1 -> the doorbell
end of layer) in a few middle windows: start / end relative to the front's start (us), stream, name; plus the mean over all
middle windows of each kernel name's (start, end) in that front."""
import bisect
import collections
import sqlite3
import sys

db = sys.argv[1]
L = int(sys.argv[2]) if len(sys.argv) > 2 else 15
show = int(sys.argv[3]) if len(sys.argv) > 3 else 1
c = sqlite3.connect(db)
S = dict(c.execute("select id, value from StringIds"))
ks = [(s, e, S[n], st) for s, e, n, st in c.execute("select start, end, shortName, streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
st0 = [k[0] for k in ks]
m = [k for k in ks if k[2] == "q80p_screen_kernel"]
agg = collections.defaultdict(lambda: [0, 0.0, 0.0])
lens = []
shown = 0
for i in range(int(len(m) * .3), int(len(m) * .9)):
    a, b = m[i][1], m[i + 1][0]
    w = ks[bisect.bisect_left(st0, a):bisect.bisect_left(st0, b)]
    dk = [k for k in w if k[2] == "doorbell_publish_res_kernel"]
    if len(dk) != 48:
        continue
    lim = dk[L][0]
    cb = None
    for k in w:
        if dk[L - 1][1] <= k[0] < lim and "combine" in k[2]:
            cb = k
            break
    if cb is None:
        continue
    f0, f1 = cb[1], dk[L][1]
    lens.append((f1 - f0) / 1000)
    inner = [k for k in w if f0 <= k[0] < f1]
    seen = collections.Counter()
    for k in inner:
        seen[k[2]] += 1
        key = f"{k[2]}#{seen[k[2]]}"
        g = agg[key]
        g[0] += 1; g[1] += (k[0] - f0) / 1000; g[2] += (k[1] - f0) / 1000
    if shown < show:
        shown += 1
        print(f"window {i}: front {(f1 - f0) / 1000:.1f} us")
        for k in inner:
            print(f"  {(k[0] - f0) / 1000:7.1f} {(k[1] - f0) / 1000:7.1f}  s{k[3] % 1000:<4d} {k[2][:70]}")
n = len(lens)
print(f"{n} windows, layer {L} front mean {sum(lens) / n:.1f} us")
for key, g in sorted(agg.items(), key=lambda x: x[1][1] / x[1][0]):
    if g[0] >= n * 0.5:
        print(f"  {g[1] / g[0]:7.1f} {g[2] / g[0]:7.1f}  {g[0] / n:4.2f}  {key[:70]}")
