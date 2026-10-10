"""tools/test_bundle_model.py - the Windows bundle's model: prepare-model.cmd downloads qwen-nvfp4-gptq ready-made
(tools/bundle_model.py, model-files.json) - the same pinned files setup.py installs, each checked against its
SHA-256 - and config/strata-qwen-nvfp4-gptq.json runs exactly those files, with the censorship switch loaded and off.

    python -m unittest tools.test_bundle_model
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import bundle_model as B  # noqa: E402
import setup  # noqa: E402

WIN = ROOT / "release" / "windows"
TABLE = WIN / "model-files.json"
CONFIG = WIN / "config" / "strata-qwen-nvfp4-gptq.json"


class Table(unittest.TestCase):
    def test_it_is_setup_s_table(self):
        t = B.load(TABLE)
        self.assertEqual(t["family"], "qwen-nvfp4-gptq")
        files, why = setup.nvfp4_files("qwen-nvfp4-gptq")
        self.assertIsNone(why)
        want = {(repo, rev, path, size, sha) for _c, repo, rev, path, size, sha in files}
        got = {(f["repo"], f["revision"], f["path"], f["size"], f["sha256"]) for f in t["files"]
               if f["dir"] == B.MODEL_DIR}
        self.assertEqual(got, want)                     # regenerate: python tools/bundle_model.py --write ...
        mm = [f for f in t["files"] if f["dir"] == "models"]
        self.assertEqual([f["path"] for f in mm], [setup.MMPROJ])
        self.assertEqual(mm[0]["repo"], "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF")
        self.assertEqual(mm[0]["revision"], setup.HF_REVISIONS[mm[0]["repo"]])
        self.assertRegex(mm[0]["sha256"], r"^[0-9a-f]{64}$")

    def test_the_config_runs_those_files(self):
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        t = B.load(TABLE)
        have = {f"{f['dir']}/{f['path']}" for f in t["files"]}
        a = cfg["args"]
        val = lambda k: a[a.index(k) + 1]                # noqa: E731
        for flag in ("--native", "--native-dense-gguf", "--ple-gguf", "--embd-gguf"):
            self.assertIn(val(flag), have, flag)
        self.assertIn(val("--pack") + "/experts.bin", have)
        self.assertIn(val("--pack") + "/index.txt", have)
        self.assertIn(val("--mtp") + "/experts.bin", have)
        self.assertIn(cfg["tokenizer"] + "/vocab.json", have)
        self.assertIn(cfg["vision"]["mmproj"], have)
        self.assertEqual(cfg["vision"]["model"], val("--native"))
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-gptq-nvfp4")
        # the censorship switch: setup's flags with the bundle's relative vector, loaded and off by default
        i = a.index("--control-vector-scaled")
        self.assertEqual(a[i:i + 9], ["--control-vector-scaled", f"data/uncensor/{setup.UNCENSOR_VECTOR.name}:1.0",
                                      *setup.uncensor_args()[2:]])
        self.assertTrue((ROOT / "data" / "uncensor" / setup.UNCENSOR_VECTOR.name).is_file())
        self.assertIs(cfg["uncensored"], False)

    def test_the_scripts_name_it(self):
        start = (WIN / "start-server.cmd").read_text(encoding="utf-8")
        self.assertIn(r"config\strata-qwen-nvfp4-gptq.json", start)
        self.assertIn(r"models\qwen-nvfp4-gptq\pack\experts.bin.sha256", start)   # bundle_model.py's mark
        self.assertIn(r"config\strata-nvfp4.json", start)                         # an older bundle's orca still runs
        prep = (WIN / "prepare-model.cmd").read_text(encoding="utf-8")
        self.assertIn(r"tools\bundle_model.py", prep)
        self.assertIn("requirements-model.txt", prep)
        self.assertNotIn("jpezzulli", prep)
        self.assertTrue((WIN / "requirements-model.txt").is_file())
        self.assertFalse((WIN / "config" / "strata-nvfp4.json").exists())        # orca withdrawn: not shipped


class Download(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.root = Path(t.name)
        self.data = {"pack/experts.bin": b"E" * 1000, "dense.gguf": b"D" * 300, "mmproj.gguf": b"V" * 50}
        rows = [{"repo": "o/m", "revision": "1" * 40, "path": p, "size": len(d), "sha256": hashlib.sha256(d).hexdigest(),
                 "dir": "models" if p == "mmproj.gguf" else "models/m"} for p, d in self.data.items()]
        self.table = self.root / "model-files.json"
        self.table.write_text(json.dumps({"family": "x", "title": "X", "files": rows}))
        self.calls = []

    def fake(self, bad=()):
        def download(repo_id, filename, revision, local_dir):
            self.calls.append((repo_id, filename, revision))
            dst = Path(local_dir) / filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"X" * len(self.data[filename]) if filename in bad else self.data[filename])
            return str(dst)
        return download

    def test_every_file_once_and_checked(self):
        self.assertEqual(B.prepare(self.root, self.table, download=self.fake()), 0)
        self.assertEqual(len(self.calls), 3)
        self.assertTrue(all(c[2] == "1" * 40 for c in self.calls))              # the pinned revision
        for p in self.data:
            dst = self.root / ("models" if p == "mmproj.gguf" else "models/m") / p
            self.assertEqual(B.mark_of(dst).read_text().strip(), hashlib.sha256(self.data[p]).hexdigest())
        self.calls.clear()
        self.assertEqual(B.prepare(self.root, self.table, download=self.fake()), 0)
        self.assertEqual(self.calls, [])                                        # all here and checked
        self.assertEqual(B.prepare(self.root, self.table, check_only=True), 0)

    def test_a_wrong_file_is_deleted_and_fetched_again(self):
        with self.assertRaises(SystemExit) as e:
            B.prepare(self.root, self.table, download=self.fake(bad=("dense.gguf",)))
        self.assertIn("SHA-256", str(e.exception))
        self.assertFalse((self.root / "models/m/dense.gguf").exists())
        self.assertEqual(B.prepare(self.root, self.table, check_only=True), 1)   # not done
        self.calls.clear()
        self.assertEqual(B.prepare(self.root, self.table, download=self.fake()), 0)
        # only what was missing: the bad file, and the one the stop left (experts.bin was checked before it)
        self.assertEqual([c[1] for c in self.calls], ["dense.gguf", "mmproj.gguf"])


if __name__ == "__main__":
    unittest.main()
