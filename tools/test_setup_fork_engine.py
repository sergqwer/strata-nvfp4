"""This fork's engine over an installed one: an upstream ready-made engine that runs the GGUF models is replaced by
the fork's release engine when one is published for the PC, and kept - said once, not asked again on every start -
when none is (no asset yet, offline, Linux / the CUDA 12 engine, a card the release has no code for).  Its
BUILD.json is dropped only once the asset is there.  A config set up for the GPU image encoder gets the CPU settings
under the release's CPU-only encoder.  Mocked network and a temp Strata folder: nothing is downloaded.

    python -m unittest tools.test_setup_fork_engine
"""
from __future__ import annotations

import io
import json
import re
import sys
import tempfile
import unittest
import zipfile
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

# upstream's oldest engine the GGUF models take, and the fork's current release (both follow a port)
UPSTREAM = {"version": setup.version_text(setup.UPSTREAM_MIN_ENGINE), "archs": [75, 80, 86, 89, 90, 100, 120],
            "source": "prebuilt"}
FORK = {"version": setup.version_text(setup.MIN_ENGINE), "source": "release", "archs": [75, 86, 89, 120], "ptx": False,
        "vision": "cpu", "fork": "https://github.com/sergqwer/strata-nvfp4", "cuda_libs": "bundled"}


class Head:
    def close(self):
        pass


class Case(unittest.TestCase):
    """A temp Strata folder with upstream's engine installed; GitHub, the GPU and pip mocked."""

    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.root = Path(t.name)
        self.heads, self.downloads, self.pip = [], [], []
        self.published = True                         # the release has the asset
        self.arch = 89

    def engine(self, meta=UPSTREAM, toolkit=13) -> Path:
        eng = self.root / (setup.ENGINE12_DIR if toolkit == 12 else "engine")
        eng.mkdir(exist_ok=True)
        (eng / "BUILD.json").write_text(json.dumps(meta))
        (eng / setup.EXE).write_bytes(b"upstream")
        return eng

    def patches(self):
        def urlopen(req, timeout=None):
            self.heads.append(req.full_url)
            if self.published is None:
                raise setup.urllib.error.URLError("offline")
            if not self.published:
                raise setup.urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
            return Head()

        def download(url, dst, what=None, **kw):
            self.downloads.append(url)
            with zipfile.ZipFile(dst, "w") as z:
                z.writestr("BUILD.json", json.dumps(FORK))
                z.writestr(setup.EXE, b"fork")
                z.writestr(setup.VEXE, b"fork vision")

        return [mock.patch.object(setup, "ROOT", self.root),
                mock.patch.object(setup.urllib.request, "urlopen", urlopen),
                mock.patch.object(setup, "download", download),
                mock.patch.object(setup, "engine_digest", lambda asset, base: None),
                mock.patch.object(setup, "gpu_info", lambda *a: {"arch": self.arch, "name": "X", "vram_gb": 24.0}),
                mock.patch.object(setup, "pip_cuda_libs", lambda tk=13: self.pip.append(tk))]

    def starts(self, n=3, toolkit=None) -> str:
        """`n` model starts (START-HERE): each updates the installed engine first."""
        out = io.StringIO()
        with ExitStack() as st:
            for p in self.patches():
                st.enter_context(p)
            with redirect_stdout(out):
                for _ in range(n):
                    setup.update_installed_engine(setup.PREBUILT_URL, toolkit)
        return out.getvalue()

    def kept_lines(self, out: str) -> list:
        return [x for x in out.splitlines() if "kept the installed engine" in x]

    def build(self, toolkit=13) -> dict:
        eng = self.root / (setup.ENGINE12_DIR if toolkit == 12 else "engine")
        return json.loads((eng / "BUILD.json").read_text())


class Keep(Case):
    def test_a_card_the_release_has_no_code_for(self):
        # an A100 (sm_80): nothing is downloaded, and it is said once, not on every start
        self.engine()
        self.arch = 80
        out = self.starts(3)
        self.assertEqual((self.heads, self.downloads), ([], []))
        self.assertEqual(len(self.kept_lines(out)), 1, out)
        self.assertIn("built for sm_75, sm_86, sm_89, sm_120; your GPU is sm_80", out)
        self.assertEqual(self.build(), UPSTREAM)
        self.assertEqual((self.root / "engine" / setup.EXE).read_bytes(), b"upstream")

    def test_not_published_yet_said_once_tried_again_in_a_day(self):
        self.engine()
        self.published = False
        out = self.starts(3)
        self.assertEqual(len(self.heads), 2, self.heads)              # the release's tag, then the latest: once
        self.assertEqual(self.downloads, [])
        self.assertEqual(len(self.kept_lines(out)), 1, out)
        self.assertIn("setup tries the release again in a day", out)
        self.assertNotIn("could not update the engine", out)
        self.assertEqual(self.build(), UPSTREAM)                       # its BUILD.json was never dropped
        later = setup.time.time() + 86400 + 60
        with mock.patch.object(setup.time, "time", lambda: later):
            self.starts(1)
        self.assertEqual(len(self.heads), 4)                           # a day later: asked again

    def test_the_cuda_12_engine_asks_nobody(self):
        # the fork publishes no CUDA 12 engine: known without GitHub, said once, --build named
        self.engine(FORK)
        up12 = {**UPSTREAM, "archs": [61, 70]}
        self.engine(up12, toolkit=12)
        self.arch = 61
        out = self.starts(3)
        self.assertEqual((self.heads, self.downloads), ([], []))
        self.assertEqual(len([x for x in out.splitlines() if "Windows (CUDA 13) only" in x]), 1, out)
        self.assertIn("setup --build compiles it", out)
        self.assertNotIn("Replacing upstream", out)
        self.assertEqual(self.build(12), up12)

    def test_linux_asks_nobody(self):
        self.engine()
        with mock.patch.object(setup, "PREBUILT_ASSET", "strata-linux-x64.tar.gz"):
            out = self.starts(3)
        self.assertEqual((self.heads, self.downloads), ([], []))
        self.assertEqual(len(self.kept_lines(out)), 1, out)
        self.assertIn("Windows (CUDA 13) only", out)
        self.assertNotIn("could not update", out)
        self.assertEqual(self.build(), UPSTREAM)

    def test_an_offline_rerun_keeps_the_installed_engine(self):
        # setup run again (not a start) with no internet: the GGUF models keep their engine - nothing compiled
        eng = self.engine()
        self.published = None
        out = io.StringIO()
        with ExitStack() as st:
            for p in self.patches():
                st.enter_context(p)
            with redirect_stdout(out):
                got = setup.get_prebuilt(setup.PREBUILT_URL, {"arch": 89}, "none")
        self.assertEqual(got, eng)
        self.assertEqual(self.build(), UPSTREAM)
        self.assertEqual(self.downloads, [])
        self.assertIn("kept the installed engine " + UPSTREAM["version"], out.getvalue())


class Replace(Case):
    def test_replaced_when_published(self):
        self.engine()
        (self.root / "engine" / setup.FORK_NOTE).write_text(json.dumps({"release": "old", "final": True}))
        out = self.starts(2)
        self.assertEqual(len(self.downloads), 1)
        self.assertIn("Replacing upstream Strata's ready-made engine with this fork's", out)
        self.assertEqual(self.build()["fork"], FORK["fork"])
        self.assertEqual((self.root / "engine" / setup.EXE).read_bytes(), b"fork")
        self.assertFalse((self.root / "engine" / setup.FORK_NOTE).exists())
        self.assertEqual(self.pip, [])                                 # the zip carries its cuBLAS

    def test_a_new_install_on_a_card_the_release_lacks_compiles_without_downloading(self):
        out = io.StringIO()
        with ExitStack() as st:
            for p in self.patches():
                st.enter_context(p)
            with redirect_stdout(out):
                got = setup.get_prebuilt(setup.PREBUILT_URL, {"arch": 80}, "none")
        self.assertIsNone(got)
        self.assertEqual((self.heads, self.downloads), ([], []))
        self.assertIn("compiling it from this source", out.getvalue())

    def test_the_release_s_cards_are_setup_s(self):
        src = (ROOT / "release" / "make_windows_bundle.py").read_text(encoding="utf-8")
        archs = re.search(r'"archs": \[([\d, ]+)\], "ptx": False', src)
        self.assertIsNotNone(archs)
        self.assertEqual(tuple(int(x) for x in archs.group(1).split(",")), setup.FORK_ARCHS)


class Vision(unittest.TestCase):
    def config(self, d: Path, engine_vision: str, max_tokens=1024) -> Path:
        eng = d / "engine"
        eng.mkdir()
        (eng / "BUILD.json").write_text(json.dumps({**FORK, "vision": engine_vision}))
        cfg = d / "strata-iq3_xxs.json"
        cfg.write_text(json.dumps({"exe": str(eng / setup.EXE), "args": ["--max-context", "131072", "--vision",
                                                                          "--vram-reserve-mib", "700"],
                                   "vision": {"exe": str(eng / setup.VEXE), "mmproj": "m", "model": "x", "gpu": True,
                                              "max_tokens": max_tokens}}))
        return cfg

    def upgrade(self, cfg: Path) -> tuple:
        out = io.StringIO()
        with redirect_stdout(out):
            got = setup.upgrade_config(cfg, json.loads(cfg.read_text()))
        return got, json.loads(cfg.read_text()), out.getvalue()

    def test_the_release_s_cpu_encoder_gets_the_cpu_settings(self):
        with tempfile.TemporaryDirectory() as d:
            got, saved, out = self.upgrade(self.config(Path(d), "cpu"))
        self.assertEqual(saved, got)
        v = saved["vision"]
        self.assertEqual((v["gpu"], v["max_tokens"]), (False, setup.VISION["cpu"]["max_tokens"]))
        self.assertGreaterEqual(v["threads"], 1)
        self.assertIn("--vision", saved["args"])
        self.assertEqual(saved["args"][saved["args"].index("--vram-reserve-mib") + 1],
                         str(setup.VISION["cpu"]["reserve_mib"]))
        self.assertIn("image encoder runs on the CPU", out)

    def test_a_gpu_encoder_or_the_user_s_tokens_stay(self):
        with tempfile.TemporaryDirectory() as d:
            _, saved, out = self.upgrade(self.config(Path(d), "gpu"))
        self.assertTrue(saved["vision"]["gpu"])
        self.assertEqual(out, "")
        with tempfile.TemporaryDirectory() as d:
            _, saved, _ = self.upgrade(self.config(Path(d), "cpu", max_tokens=600))
        self.assertEqual((saved["vision"]["gpu"], saved["vision"]["max_tokens"]), (False, 600))


class Texts(unittest.TestCase):
    def test_the_fork_s_version_is_written_as_its_tag(self):
        self.assertEqual(setup.version_text(setup.version_tuple("0.1.40.3-nvfp4.1")), "0.1.40.3-nvfp4.1")
        with tempfile.TemporaryDirectory() as d:
            eng = Path(d) / "engine"
            eng.mkdir()
            (eng / "BUILD.json").write_text(json.dumps(FORK))
            cfg = Path(d) / "strata-huihui-nvfp4.json"
            cfg.write_text(json.dumps({"exe": str(eng / setup.EXE), "args": ["--max-context", "131072"]}))
            out = io.StringIO()
            with mock.patch.object(setup, "pip_install", lambda *a, **k: None), \
                    mock.patch.object(setup, "update_installed_engine", lambda *a, **k: None), \
                    mock.patch.object(setup, "requirement_lines", lambda: []), redirect_stdout(out):
                setup.update_install([cfg], mock.Mock(build=False, prebuilt=setup.PREBUILT_URL))
        self.assertIn(f"Strata is updated (engine {FORK['version']})", out.getvalue())


if __name__ == "__main__":
    unittest.main()
