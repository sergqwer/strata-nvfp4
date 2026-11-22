"""nsys_layers.py DB strata|basalt [lo hi] - per-layer split of the decode windows: front = combine END (layer l-1)
-> doorbell (layer l), back = doorbell -> combine END (layer l); averages per window over the middle windows.
Strata: doorbell = end of doorbell_publish_res_kernel, combine = combine_k10_vec4 (or any *combine* on the stream).
Basalt: doorbell = end of the hc_fused_kernel right before dm_glue_gu1_kernel, combine = dm_combine4_kernel start."""
import sqlite3
import statistics
import sys

db, eng = sys.argv[1], sys.argv[2]
lo = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
hi = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9
c = sqlite3.connect(db)
S = dict(c.execute("select id, value from StringIds"))
ks = [(s, e, S[n], st) for s, e, n, st in c.execute("select start, end, shortName, streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
m = [k for k in ks if k[2] == "q80p_screen_kernel"]
i0, i1 = int(len(m) * lo), int(len(m) * hi)
fronts, backs, wins, firsts, tails = [], [], [], [], []
for i in range(i0, i1):
    a, b = m[i][1], m[i + 1][0]          # from this window's head to the next window's head
    w = [k for k in ks if a <= k[0] < b]
    if eng == "strata":
        dk = [k for k in w if k[2] == "doorbell_publish_res_kernel"]
        dbs = [k[1] for k in dk]
        cst = dk[0][3] if dk else -1
        cbs = [k[1] for k in w if k[3] == cst and "combine" in k[2]]
    else:
        dbs = []
        for j, k in enumerate(w):
            if k[2] == "dm_glue_gu1_kernel":
                prev = [x for x in w[:j] if x[2] == "hc_fused_kernel"]
                if prev:
                    dbs.append(prev[-1][1])
        cbs = [k[1] for k in w if k[2] == "dm_combine4_kernel"]
    if len(dbs) != 48 or len(cbs) < 48:
        continue
    cbs = cbs[:48]
    f = sum(dbs[l] - cbs[l - 1] for l in range(1, 48))
    bk = sum(cbs[l] - dbs[l] for l in range(48))
    fronts.append(f / 1e6)
    backs.append(bk / 1e6)
    firsts.append((dbs[0] - a) / 1e6)
    tails.append((b - cbs[47]) / 1e6)
    wins.append((b - a) / 1e6)
n = len(wins)
print("%s: %d windows; per window ms: first doorbell after the previous head %.2f | fronts (47 layers) %.2f | backs (48) %.2f"
      " | last combine -> next head %.2f | total %.2f" % (db, n, statistics.mean(firsts), statistics.mean(fronts),
                                                         statistics.mean(backs), statistics.mean(tails), statistics.mean(wins)))
print("per layer us: front %.1f, back %.1f" % (statistics.mean(fronts) / 47 * 1e3, statistics.mean(backs) / 48 * 1e3))
