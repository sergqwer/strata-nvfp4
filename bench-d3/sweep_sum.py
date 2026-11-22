import re,sys,collections
d=collections.OrderedDict()
for l in open(sys.argv[1]):
    m=re.match(r"r\d+ depth (\d+) pieces (\d+) depthz (\d+) rz (\d+)\s+(.*) (\w+)$",l.strip())
    if not m: print(l.strip()); continue
    k=tuple(int(x) for x in m.groups()[:4])
    d.setdefault(k,collections.defaultdict(list))
    for tv in m.group(5).split():
        t,v=tv.split(':'); d[k][int(t)].append(float(v))
    if m.group(6)!='bitwise': print("DIFF",l)
for k,v in d.items():
    kb = 43520/k[1]*k[0]/1024 if k[0] else 85
    print(f"depth {k[0]} pieces {k[1]} depthz {k[2]} rz {k[3]} ({kb:5.1f} KiB)", "  ".join(f"T{t} {sum(x)/len(x):5.1f} [{min(x):.1f}-{max(x):.1f}]" for t,x in sorted(v.items())))
