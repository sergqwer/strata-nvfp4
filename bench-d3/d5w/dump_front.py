"""dump_front.py TRACE.csv [N] - print the kernels of N attention fronts (verify graph): start/end rel. to read-1 upq end, stream."""
import sys, bisect
src = open(r"D:\Projects\gpu-probe\dprof\ana.py").read().replace("\nmain()\n", "\n"); ns = {}; exec(compile(src, "ana", "exec"), ns)
K, C, S = ns["load"](sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
skip = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
at = [i for i, k in enumerate(K) if k["n"] == "attn_chunk_staged_kernel"]
shown = 0
for ai in at[skip:]:
    a = K[ai]
    # window: 150 us before attention start up to its end + 30 us
    lo = a["s"] - 160000
    ks = [k for k in K[max(0, ai - 400):ai + 60] if k["e"] > lo and k["s"] < a["e"] + 30000 and k["graph"] == a["graph"]]
    ks.sort(key=lambda k: k["s"])
    t0 = a["s"]
    print("=== attention T=%d graph %d" % (a["g"][2], a["graph"]))
    for k in ks:
        print("%8.2f %8.2f %6.2f st%-4d %-40s %s %s" % ((k["s"] - t0) / 1e3, (k["e"] - t0) / 1e3, (k["e"] - k["s"]) / 1e3, k["st"], k["n"][:40], k["g"], k["b"]))
    shown += 1
    if shown >= N: break
