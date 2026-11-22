"""cmp_sve.py A.sve B.sve - two SVE1 embedding files (strata-vision / VENC): grid, per-row cosine (min / mean), max abs diff, relative L2."""
import struct
import sys

import numpy as np


def read(p):
    d = open(p, "rb").read()
    magic, n, nx, ny, e = struct.unpack("<5i", d[:20])
    assert magic == 0x31455653, p
    return (n, nx, ny, e), np.frombuffer(d[20:], dtype=np.float32).reshape(n, e)


ha, a = read(sys.argv[1])
hb, b = read(sys.argv[2])
if ha != hb:
    print("header differs", ha, hb)
    sys.exit(1)
cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)
print(f"n={ha[0]} grid={ha[1]}x{ha[2]} dim={ha[3]} row-cos min {cos.min():.6f} mean {cos.mean():.6f} "
      f"maxabs {np.abs(a - b).max():.4g} relL2 {np.linalg.norm(a - b) / np.linalg.norm(b):.3e} identical={np.array_equal(a, b)}")
