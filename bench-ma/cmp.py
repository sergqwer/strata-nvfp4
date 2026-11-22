"""cmp.py LOG_A LOG_B - where the greedy output token ids of two engine logs first differ"""
import sys
def out(p):
    for l in open(p, encoding="utf-8", errors="replace"):
        if "output  :" in l: return l.split("output  :")[1].split()
    return []
a, b = out(sys.argv[1]), out(sys.argv[2])
n = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
print("%s vs %s: %d / %d tokens, first difference at %d" % (sys.argv[1], sys.argv[2], len(a), len(b), n))
