"""Upstream 0.1.44.1's "setup run again keeps the installed size" (tools/test_setup_rerun_size.py) with this fork's
families: Unsloth's sizes incl. this fork's UD-Q3_K_XL, an NVFP4 / Q8_0 / mix install beside a GGUF one, and the
Qwen3.6 own-path family (its size menu and the family default of a re-run).  A Qwen3.6 size bigger than the RAM
stays a note, never a question (bdd615c9).  Mocked as the tests it borrows from - nothing is fetched or started.

    python -m unittest tools.test_setup_rerun_fork
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
import test_setup_golden as g  # noqa: E402
import test_setup_nvfp4 as n  # noqa: E402
import test_setup_qwen36 as q36  # noqa: E402

PC128 = g.PROFILES["128GB-1x24GB"]
# every NVFP4 family published (test_setup_nvfp4's tables; its FILLED has the GPTQ and NVIDIA uploads only)
REPOS = {**n.FILLED, setup.QWEN_MIX_REPO: n.table("qwen-mix"), setup.QWEN_Q8_REPO: n.table("qwen-q8")}


def fork_install(ram, found, argv, configs=(), answers=None, extra=()):
    """test_setup_golden.install with this fork's release engine (test_setup_nvfp4.install_with) and every NVFP4 family."""
    patches = [mock.patch.object(setup, "NVFP4_REPOS", REPOS), mock.patch.object(setup, "get_prebuilt", n.fork_engine([])),
               mock.patch.object(setup, "verify_sha256", n.verify),
               mock.patch.object(setup, "download", lambda url, dst, what=None, unpinned_ok=True:
                                 (dst.parent.mkdir(parents=True, exist_ok=True), dst.write_bytes(b""), setup.mark(dst))),
               mock.patch.object(setup, "pip_cuda_libs", lambda tk=13: None), *extra]
    return g.install(ram, found, argv, answers=answers, extra=patches, configs=configs)


def gguf_cfg(tag: str) -> tuple:
    return (f"strata-{tag}.json", {"exe": "x", "args": ["--pack", f"p/{tag}", "--max-context", "32768"], "cwd": ".",
                                   "tokenizer": "t", "model_name": f"qwen3.8-flash-next-{tag}", "log": "l", "port": 8080})


def nvfp4_cfg(family: str) -> tuple:
    tag = (setup.NVFP4_FAMILIES[family]["tag"] + setup.nvfp4_size(family)).lower()
    return (f"strata-{tag}.json", {"exe": "x", "args": ["--native", "p", "--max-context", "131072"], "cwd": ".",
                                   "model_name": f"{setup.NVFP4_FAMILIES[family]['name']}-{setup.nvfp4_size(family).lower()}"})


def q36_cfg(size: str) -> tuple:
    tag = f"qwen36-{size.lower()}"
    return (f"strata-{tag}.json", {"exe": "x", "args": ["--native", "g.gguf", "--max-context", "131072"], "cwd": ".",
                                   "model_name": f"qwen3.6-35b-a3b-{size.lower()}", "arch": "qwen35moe"})


class GgufFamilies(unittest.TestCase):
    def test_an_installed_ud_q3_k_xl_stays_the_unsloth_default(self):
        ram, found = PC128
        code, out, cfg, asked = fork_install(ram, found, ["--family", "unsloth", "--setup", "--no-start"], answers="",
                                             configs=[gguf_cfg("unsloth-ud-q3_k_xl")])
        self.assertEqual(code, 0, out[-2000:])
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-unsloth-ud-q3_k_xl")
        line = next(x for x in out.splitlines() if "UD-Q3_K_XL" in x and "download" in x)
        self.assertIn("(installed)", line)
        line = next(x for x in out.splitlines() if "UD-Q4_K_XL" in x and "download" in x)
        self.assertIn("(recommended)", line)                  # still marked, still a choice
        self.assertTrue(any("Which size?" in a for a in asked))

    def test_an_nvfp4_install_is_no_size_of_the_gguf_family(self):
        # qwen-q8's config (strata-qwen-q8_0.json) reads back as the qwen-q8 family, so it pre-selects no GGUF size
        ram, found = PC128
        for fam in setup.NVFP4_FAMILIES:
            with self.subTest(fam=fam):
                code, out, cfg, _ = g.install(ram, found, ["--family", "qwen", "--setup", "--no-start"], answers=None,
                                              configs=[nvfp4_cfg(fam), gguf_cfg("iq3_xxs")])
                self.assertEqual(code, 0, out[-2000:])
                self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-iq3_xxs")
                code, out, cfg, _ = g.install(ram, found, ["--family", "qwen", "--setup", "--no-start"], answers=None,
                                              configs=[nvfp4_cfg(fam)])
                self.assertEqual(code, 0, out[-2000:])
                self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-iq3_s")   # the recommendation, as a first install


class Nvfp4Families(unittest.TestCase):
    def test_a_rerun_keeps_each_installed_nvfp4_family(self):
        ram, found = PC128
        for fam in setup.NVFP4_FAMILIES:
            with self.subTest(fam=fam):
                name, old = nvfp4_cfg(fam)
                code, out, cfg, _ = fork_install(ram, found, ["--setup", "--no-start"], configs=[(name, old)], answers="")
                self.assertEqual(code, 0, out[-3000:])
                self.assertEqual(cfg["model_name"], old["model_name"])

    def test_the_users_own_tail_skip_survives_a_rerun(self):
        # 0.1.44.1: STRATA_ROUTE_TAIL_SKIP / STRATA_MMQ_KQ / STRATA_HC_Q8 are setup's only in an Unsloth config;
        # this fork's packs never get them from setup, so there they are the user's own
        old = {"env": {"STRATA_ROUTE_TAIL_SKIP": "7", "STRATA_RESIDENT_PIN": "0"}}
        new = {"model_name": "qwen3.8-flash-next-gptq-nvfp4"}
        setup.carry_over(old, new)
        self.assertEqual(new["env"], {"STRATA_ROUTE_TAIL_SKIP": "7"})


class Qwen36Rerun(unittest.TestCase):
    setUp = q36.InstallOwnPath.setUp
    install = q36.InstallOwnPath.install

    def put(self, *cfgs):
        for name, c in cfgs:
            (self.root / name).write_text(json.dumps(c), encoding="utf-8")

    def native(self):
        return self.cfg["args"][self.cfg["args"].index("--native") + 1]

    def test_enter_keeps_the_installed_size(self):
        self.put(q36_cfg("UD-Q4_K_XL"))
        rc, out = self.install(ram=64.0)                    # 64 GB recommends Q8_0 (the largest that fits)
        self.assertEqual(rc, 0, out)
        self.assertTrue(self.native().endswith("Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"), self.native())
        self.assertIn("Recommended for 64 GB of RAM: Q8_0", out)
        self.assertIn("Enter keeps the installed UD-Q4_K_XL", out)
        line = next(x for x in out.splitlines() if "UD-Q4_K_XL" in x and "download" in x)
        self.assertIn("(installed)", line)

    def test_a_named_size_still_wins(self):
        self.put(q36_cfg("UD-Q4_K_XL"))
        rc, out = self.install(ram=64.0, model="UD-Q6_K_XL")
        self.assertEqual(rc, 0, out)
        self.assertTrue(self.native().endswith("UD-Q6_K_XL.gguf"))

    def test_an_installed_size_bigger_than_the_ram_is_kept_with_a_note(self):
        def no_question(*a, **k):
            raise AssertionError("a size bigger than the RAM must not ask")
        self.put(q36_cfg("Q8_0"))
        with mock.patch.object(setup, "confirm_risk", no_question):
            rc, out = self.install(ram=32.0)
        self.assertEqual(rc, 0, out)
        self.assertTrue(self.native().endswith("Qwen3.6-35B-A3B-Q8_0.gguf"))
        self.assertIn("installing Q8_0 with 32 GB of RAM, as you chose", out)

    def test_another_familys_install_is_not_a_qwen36_size(self):
        self.put(gguf_cfg("unsloth-q8_0"), nvfp4_cfg("qwen-q8"))
        rc, out = self.install(ram=64.0)
        self.assertEqual(rc, 0, out)
        self.assertTrue(self.native().endswith("Qwen3.6-35B-A3B-Q8_0.gguf"))   # the recommendation
        self.assertNotIn("(installed)", out)


class Qwen36RerunFamily(unittest.TestCase):
    def test_a_rerun_keeps_the_qwen36_family(self):
        # --setup without --family over a Qwen3.6 install: Qwen3.6 stays the default, not this fork's NVFP4 model
        ram, found = PC128
        picked = []

        def own_path(a, family, *rest):
            picked.append(family)
            return 0
        code, out, _, asked = g.install(ram, found, ["--setup", "--no-start"], answers="",
                                        configs=[q36_cfg("UD-Q4_K_XL")],
                                        extra=[mock.patch.object(setup, "install_own_path", own_path)])
        self.assertEqual(code, 0, out[-2000:])
        self.assertEqual(picked, ["qwen36"])
        self.assertTrue(any("Which model?" in a for a in asked))


if __name__ == "__main__":
    unittest.main()
