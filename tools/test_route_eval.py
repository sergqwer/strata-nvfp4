"""tools/route_eval.py on a synthetic dump in the engine's format (generate.cpp route_dump: SRD1, version 1)."""
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import route_eval  # noqa: E402

L, E, H, K = 4, 16, 8, 2


def write_dump(path, rounds, x_layers=(1,)):
    """rounds: list of (round_index, t_us, ids[L] as (n_tok, K) arrays, resident[L] as E-bool arrays)."""
    b = bytearray(b"SRD1")
    b += struct.pack("<5I", 1, L, E, H, len(x_layers)) + struct.pack(f"<{len(x_layers)}I", *x_layers)
    for rnd, t_us, ids, res in rounds:
        for l in range(L):
            a = np.asarray(ids[l], dtype="<i2")
            n_tok = a.shape[0]
            has_x = l in x_layers
            b += route_eval.REC.pack(ord("L"), 1 if has_x else 0, l, n_tok, K, rnd, t_us)
            b += a.tobytes()
            b += np.packbits(np.asarray(res[l], dtype=bool), bitorder="little").tobytes()
            if has_x:
                b += (np.arange(n_tok * H, dtype=np.float32) / 7).astype("<f2").tobytes()
    Path(path).write_bytes(bytes(b))


def resident_none():
    return [np.zeros(E, dtype=bool) for _ in range(L)]


class RouteEval(unittest.TestCase):
    def test_parse_and_misses(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.srd"
            res = resident_none()
            res[0][3] = True                                   # layer 0 holds expert 3
            ids = [np.array([[3, 5], [5, 6]]) for _ in range(L)]
            write_dump(p, [(0, 0, ids, res), (1, 10_000, ids, res)])
            d = route_eval.read_dump(p)
            self.assertEqual((d.n_layers, d.n_expert, d.n_embd, d.x_layers), (L, E, H, [1]))
            self.assertEqual(len(d.rounds), 2)
            r = d.rounds[0]
            self.assertEqual(r.n_tok, 2)
            self.assertTrue(r.resident[0][3] and not r.resident[1][3])
            c = route_eval.routed_counts(r, L, E)
            self.assertEqual(c[0][5], 2.0)
            m = route_eval.misses(r, c)
            self.assertEqual(sorted(np.flatnonzero(m[0])), [5, 6])           # 3 is resident at layer 0
            self.assertEqual(sorted(np.flatnonzero(m[1])), [3, 5, 6])
            self.assertEqual(r.x[1].shape, (2, H))
            self.assertAlmostEqual(float(r.x[1][1][0]), 8 / 7, places=2)

    def test_sequences_split_at_gaps_and_wide_rounds(self):
        mk = lambda i, t, n=1: route_eval.Round(i, t, n, [], np.zeros((L, E), bool), {})
        rs = [mk(0, 0), mk(1, 10_000), mk(2, 20_000), mk(3, 900_000), mk(4, 910_000), mk(5, 920_000, 16), mk(6, 930_000)]
        seqs = route_eval.sequences(rs, gap_ms=250, max_tok=8)
        self.assertEqual([[r.index for r in s] for s in seqs], [[0, 1, 2], [3, 4]])

    def test_repeating_misses_are_found_by_last(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "b.srd"
            res = resident_none()
            ids = [np.array([[1, 2]]) for _ in range(L)]
            write_dump(p, [(i, i * 10_000, ids, res) for i in range(5)])
            out = subprocess.run([sys.executable, str(Path(route_eval.__file__)), str(p), "--budgets", "4,8"],
                                 capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            line = [ln for ln in out.stdout.splitlines() if ln.startswith("last ")][0]
            self.assertIn("50.0%", line)                       # 4 copies cover 4 of the 8 misses (2 a layer, 4 layers)
            self.assertIn("100.0%", line)

    def test_recall_skips_resident_pairs(self):
        rng = np.random.default_rng(0)
        score = np.array([[5.0, 4.0, 3.0, 0.0]])
        resident = np.array([[True, False, False, False]])
        miss = np.array([[False, False, True, False]])
        self.assertEqual(route_eval.recall_at(score, resident, miss, [1, 2], rng), [0, 1])

    def test_export(self):
        with tempfile.TemporaryDirectory() as td:
            p, out = Path(td) / "c.srd", Path(td) / "c.npz"
            ids = [np.array([[1, 2], [3, 4]]) for _ in range(L)]
            write_dump(p, [(i, i * 10_000, ids, resident_none()) for i in range(3)])
            self.assertEqual(route_eval.main([str(p), "--export", str(out), "--export-layers", "1"]), 0)
            with np.load(out) as z:                               # (closed: Windows cannot delete an open file)
                self.assertEqual(z["x"].shape, (2, 2, 1, H))       # 2 pairs, (last, mean), 1 dumped layer
                self.assertEqual(z["routed"].shape, (2, L, E // 8))
                self.assertEqual(np.unpackbits(z["miss"][0][0], bitorder="little").nonzero()[0].tolist(), [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
