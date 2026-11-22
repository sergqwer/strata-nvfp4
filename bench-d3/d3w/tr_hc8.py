"""tr_hc8.py TRACE.csv - gr_hc8_check --bench / --bench-gated under cupti_trace: per graph (arm), the hc8 read's
critical-path pieces in us (medians over every read of every replay):
  chain (ungated --bench graphs): read = upq end - previous upq end; d_tail = down end - previous upq end; u_tail = upq end - down end
  gated (--bench-gated graphs):    after = upq end - combine end; d_after = down end - combine end; u_after = upq end - down end
Arms: graphs in creation (id) order per (kind, T): chain hc8 arms 0 = grid-end, 1 = flag; gated arms = flag 0..3."""
import collections
import re
import statistics
import sys

names, K = {}, []
for ln in open(sys.argv[1], encoding="utf-8", errors="replace"):
    p = ln.rstrip("\n").split(",")
    if p[0] == "N":
        names[int(p[1])] = ",".join(p[2:])
    elif p[0] == "K":
        v = [int(x) for x in p[1:]]
        K.append(dict(s=v[0], e=v[1], st=v[2], n=names.get(v[3], "?"), graph=v[14]))
K.sort(key=lambda k: k["s"])
rx_d = re.compile(r"hc8_down_kernelILi(\d)ELi(\d)")
rx_u = re.compile(r"hc8_upq_kernelILi(\d)ELi(\d)")
by_g = collections.defaultdict(list)
for k in K:
    if k["graph"]:
        by_g[k["graph"]].append(k)
groups = collections.defaultdict(list)   # (kind, T) -> graph ids
info = {}
for g, ks in by_g.items():
    hasc = any(k["n"].find("combine_k10_vec4_sig") >= 0 for k in ks)
    d = [k for k in ks if rx_d.search(k["n"])]
    if not d:
        continue
    T = int(rx_d.search(d[0]["n"]).group(1))
    kind = "gated" if hasc else "chain"
    if kind == "chain" and len(d) < 20:
        continue   # the selftest's one-read graphs
    groups[(kind, T)].append(g)
    info[g] = (kind, T)
R = collections.defaultdict(lambda: collections.defaultdict(list))
for (kind, T), gs in sorted(groups.items()):
    for arm, g in enumerate(sorted(gs)):
        ks = sorted(by_g[g], key=lambda k: k["s"])
        if kind == "chain":
            ups = [k for k in ks if rx_u.search(k["n"])]
            dns = [k for k in ks if rx_d.search(k["n"])]
            # pair each down with the next upq (stream order), replays concatenated
            ups.sort(key=lambda k: k["s"]); dns.sort(key=lambda k: k["s"])
            for i in range(1, min(len(ups), len(dns))):
                pu, d, u = ups[i - 1], dns[i], ups[i]
                if d["s"] < pu["s"] or u["s"] - pu["e"] > 50000:
                    continue   # a replay boundary
                R[(kind, T, arm)]["read"].append(u["e"] - pu["e"])
                R[(kind, T, arm)]["d_tail"].append(d["e"] - pu["e"])
                R[(kind, T, arm)]["u_tail"].append(u["e"] - d["e"])
                R[(kind, T, arm)]["u_start"].append(u["s"] - d["s"])
        else:
            cs = [k for k in ks if "combine_k10_vec4_sig" in k["n"]]
            dns = [k for k in ks if rx_d.search(k["n"])]
            ups = [k for k in ks if rx_u.search(k["n"])]
            for c in cs:
                # the read released by this combine: the down that ends first after it, and the next upq after that down
                d = [k for k in dns if k["s"] < c["e"] and k["e"] > c["s"]]
                if not d:
                    continue
                d = min(d, key=lambda k: k["e"])
                u = [k for k in ups if k["s"] >= d["s"] and k["e"] > d["e"] - 1]
                if not u:
                    continue
                u = min(u, key=lambda k: k["s"])
                R[(kind, T, arm)]["after"].append(u["e"] - c["e"])
                R[(kind, T, arm)]["d_after"].append(d["e"] - c["e"])
                R[(kind, T, arm)]["u_after"].append(u["e"] - d["e"])
                R[(kind, T, arm)]["comb"].append(c["e"] - c["s"])
for kind, cols in (("chain", ["read", "d_tail", "u_tail", "u_start"]), ("gated", ["after", "d_after", "u_after", "comb"])):
    print("%s: us medians (n)" % kind)
    print("  T arm " + "".join("%9s" % c for c in cols) + "      n")
    for key in sorted(k for k in R if k[0] == kind):
        _, T, arm = key
        r = R[key]
        print("  %d %3d " % (T, arm) + "".join("%9.2f" % (statistics.median(r[c]) / 1e3 if r[c] else 0) for c in cols) +
              "%7d" % len(r[cols[0]]))
