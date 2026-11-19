"""route_oracle.py TRACE.srd OUT.sor - the oracle file for STRATA_FS_ORACLE (#13): each decode window's position, tokens
and every row's routed experts from a STRATA_ROUTE_TRACE trace (generate.cpp, namespace route_oracle, has the format)."""
import struct
import sys

import numpy as np

import route_eval as RE


def main(src, dst):
    d = RE.read_dump(src, keep_x=False)
    rounds = [r for r in d.rounds if r.tokens is not None and r.pos >= 0 and len(r.ids) == d.n_layers]
    k = int(rounds[0].ids[0].shape[1])
    with open(dst, "wb") as f:
        f.write(b"SOR1" + struct.pack("<4I", d.n_layers, d.n_expert, k, len(rounds)))
        for r in rounds:
            ids = np.stack(r.ids, axis=1).astype(np.int16)          # (n_tok, L, k)
            f.write(struct.pack("<qI", int(r.pos), int(r.n_tok)))
            f.write(np.asarray(r.tokens[:r.n_tok], dtype=np.int32).tobytes())
            f.write(np.ascontiguousarray(ids[:r.n_tok]).tobytes())
    print(f"{dst}: {len(rounds)} windows, {d.n_layers} layers, k {k}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
