"""gguf_sizes.py GGUF REGEX - tensor name, type, shape and bytes of the matching tensors (minimal header reader)."""
import re
import struct
import sys

BS = {0: (1, 4), 1: (1, 2), 8: (32, 34), 30: (1, 2), 2: (32, 18), 12: (256, 144), 14: (256, 210), 13: (256, 176)}


def rd(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def rstr(f):
    (n,) = rd(f, "Q")
    return f.read(n).decode("utf-8", "replace")


def skip_val(f, t):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if t in sizes:
        f.read(sizes[t])
    elif t == 8:
        rstr(f)
    elif t == 9:
        (et,) = rd(f, "I")
        (n,) = rd(f, "Q")
        for _ in range(n):
            skip_val(f, et)


f = open(sys.argv[1], "rb")
assert f.read(4) == b"GGUF"
(ver,) = rd(f, "I")
nt, nkv = rd(f, "QQ")
for _ in range(nkv):
    rstr(f)
    (t,) = rd(f, "I")
    skip_val(f, t)
rx = re.compile(sys.argv[2])
tot = 0
for _ in range(nt):
    name = rstr(f)
    (nd,) = rd(f, "I")
    shape = rd(f, "Q" * nd)
    (typ,) = rd(f, "I")
    rd(f, "Q")
    if rx.search(name):
        n = 1
        for s in shape:
            n *= s
        b, by = BS.get(typ, (1, 0))
        nb = n // b * by
        tot += nb
        print("%-45s type %2d %-20s %10.2f MB" % (name, typ, list(shape), nb / 1e6))
print("total %.2f MB" % (tot / 1e6))
