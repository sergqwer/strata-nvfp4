"""kl.py REF OTHER... - KL(ref || other) of first-token logit dumps (float32 vocab), and whether the argmax agrees."""
import array, math, sys
def load(p):
    a = array.array("f"); a.frombytes(open(p, "rb").read()); return a
def logp(a):
    m = max(a); s = sum(math.exp(x - m) for x in a); ls = m + math.log(s)
    return [x - ls for x in a]
r = load(sys.argv[1]); lr = logp(r); pr = [math.exp(x) for x in lr]; ar = max(range(len(r)), key=r.__getitem__)
for p in sys.argv[2:]:
    o = load(p); lo = logp(o); ao = max(range(len(o)), key=o.__getitem__)
    kl = sum(pr[i] * (lr[i] - lo[i]) for i in range(len(r)) if pr[i] > 0)
    print("%-28s KL %.6f  top1 %s  maxabs %.4f" % (p.split("/")[-1], kl, "same" if ao == ar else "DIFF %d/%d" % (ar, ao), max(abs(a - b) for a, b in zip(r, o))))
