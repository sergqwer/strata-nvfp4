"""tools/nvfp4_table.py: a repository's files become an NVFP4_REPOS entry that setup.py accepts as it is.  Mocked
Hub API; nothing is downloaded.

    python -m unittest tools.test_nvfp4_table
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import nvfp4_table as T  # noqa: E402
import setup  # noqa: E402

REPO = setup.HUIHUI_REPO
COMMIT = "a" * 40
SMALL = {"pack/index.txt": b"index", "pack/native_experts.txt": b"native", "pack/tokenizer/vocab.json": b"{}",
         "mtp/rt/dense.txt": b"dense"}
BIG = {"pack/experts.bin": 67948118016, "pack/dense.bin": 1538625024, "dense.gguf": 5_000_000_000,
       "ple-fp8-e4m3.gguf": 51200246144, "token-embd-bf16.gguf": 1271398688, "mtp/rt/experts.bin": 707788800,
       "mtp/rt/dense.bin": 116099072}


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def hub(req, timeout=None):
    url = req.full_url
    if "/api/models/" in url:
        sib = [{"rfilename": p, "size": len(b)} for p, b in SMALL.items()]
        sib += [{"rfilename": p, "size": s, "lfs": {"sha256": hashlib.sha256(p.encode()).hexdigest(), "size": s}}
                for p, s in BIG.items()]
        sib += [{"rfilename": "README.md", "size": 900}, {"rfilename": "mmproj-x.gguf", "size": 9}]
        return Response(json.dumps({"sha": COMMIT, "siblings": sib}).encode())
    path = url.split(f"/resolve/{COMMIT}/", 1)[1]
    return Response(SMALL[path])


class Table(unittest.TestCase):
    def test_components_by_path(self):
        for path, comp in (("pack/experts.bin", "pack"), ("pack/tokenizer/vocab.json", "pack"), ("mtp/rt/dense.txt", "mtp"),
                           ("ple-fp8-e4m3.gguf", "ple"), ("token-embd-bf16.gguf", "embd"), ("x-NVFP4.gguf", "dense"),
                           ("expert-profile.bin", "profile"), ("mmproj-x.gguf", None), ("README.md", None)):
            self.assertEqual(T.component(path), comp, path)

    def test_the_entry_is_what_setup_reads(self):
        commit, files = T.hub_files(REPO, "main", "https://huggingface.co", opener=hub)
        self.assertEqual(commit, COMMIT)
        self.assertEqual(files["pack/index.txt"], (5, hashlib.sha256(b"index").hexdigest()))   # small: downloaded
        src = T.entry(REPO, commit, files)
        self.assertIn("# not a component: README.md, mmproj-x.gguf", src)
        table = eval("{" + src.split("\n    # not")[0] + "}")          # noqa: S307 - the tool's own output
        with mock.patch.object(setup, "NVFP4_REPOS", {**setup.NVFP4_REPOS, **table}):
            files, why = setup.nvfp4_files("huihui-nvfp4")
        self.assertIsNone(why)
        self.assertEqual(len(files), len(SMALL) + len(BIG))
        self.assertAlmostEqual(setup.nvfp4_experts_gib(files), 63.3, places=1)

    def test_local_folder(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "pack").mkdir()
            (Path(d) / "pack" / "index.txt").write_bytes(b"index")
            with mock.patch.object(sys, "stderr", io.StringIO()):
                files = T.local_files(Path(d))
        self.assertEqual(files, {"pack/index.txt": (5, hashlib.sha256(b"index").hexdigest())})
        self.assertIn('"revision": "",   # the upload\'s commit', T.entry(REPO, "", files))


if __name__ == "__main__":
    unittest.main()
