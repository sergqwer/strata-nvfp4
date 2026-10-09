"""The original, censored Qwen in this fork's NVFP4 format: qwen-nvidia-nvfp4 (NVIDIA's ModelOpt NVFP4 checkpoint
converted) and qwen-nvfp4-gptq (every expert NVFP4 by GPTQ).  NVIDIA's is uploaded and offered, the GPTQ one is a
placeholder until its upload; with filled tables the menu lists them after the uncensored model and says which are
censored, and an install names its config, served model, files and license.  tools/test_setup_nvfp4.py's harness (setup.main() on mocked PCs).

    python -m unittest tools.test_setup_qwen_nvfp4
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_setup_nvfp4 import FILLED, PROFILES, arg, run, setup, table  # noqa: E402

QNV, QG = setup.QWEN_NV_REPO, setup.QWEN_GPTQ_REPO
ALL = {**FILLED, QNV: table("qwen-nv"), QG: table("qwen-gptq")}
QWEN = {"qwen-nvidia-nvfp4": (QNV, "qwen3.8-flash-next-nvidia-nvfp4", "strata-qwen-nvidia-nvfp4",
                              "NVIDIA Open Model License"),
        "qwen-nvfp4-gptq": (QG, "qwen3.8-flash-next-gptq-nvfp4", "strata-qwen-gptq-nvfp4", "Qwen Community License 1.0")}


def menu_lines(out: str) -> list:
    menu = out[out.index("=== Step 2"):out.index("[ok] model:")]
    return [x.strip() for x in menu.splitlines() if x.strip()[:2] in {f"{i})" for i in range(1, 10)}]


class Names(unittest.TestCase):
    def test_nvidia_s_is_named_as_nvidia_s(self):
        self.assertEqual(QNV, "Maximilian228/Qwen3.8-Flash-Next-NVIDIA-NVFP4-Strata")
        self.assertEqual(QG, "Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata")
        self.assertIn("NVIDIA", setup.NVFP4_FAMILIES["qwen-nvidia-nvfp4"]["title"])
        self.assertNotIn("NVIDIA", setup.NVFP4_FAMILIES["qwen-nvfp4-gptq"]["title"])
        self.assertNotIn("qwen-nvfp4", setup.NVFP4_FAMILIES)          # the key itself says NVIDIA


class AsCommitted(unittest.TestCase):
    """NVIDIA's is uploaded (its table pinned); the GPTQ one is a placeholder until its upload."""
    def test_nvidia_s_is_offered_the_gptq_one_not_yet(self):
        files, why = setup.nvfp4_files("qwen-nvidia-nvfp4")
        self.assertIsNone(why)
        self.assertEqual({f[2] for f in files}, {"cca8f7fee5ce0af932097340a75e06df25c4e57d"})
        self.assertEqual({f[1] for f in files}, {QNV})
        self.assertEqual(round(setup.nvfp4_experts_gib(files), 2), 63.28)
        files, why = setup.nvfp4_files("qwen-nvfp4-gptq")
        self.assertEqual(files, [])
        self.assertIn("its pack files are not on Hugging Face yet", why)

    def test_the_menu_with_the_committed_tables(self):
        ram, found = PROFILES["128GB-1x24GB"]
        code, out, cfg, _ = run(ram, found, ["--no-start"], repos=setup.NVFP4_REPOS, answers="")
        self.assertEqual(code, 0, out[-3000:])
        lines = menu_lines(out)
        self.assertEqual(len(lines), 2 + len(setup.FAMILIES))
        self.assertTrue(lines[0].startswith("1) OrcaRouter"), lines)
        self.assertTrue(lines[1].startswith("2) Qwen3.8-Flash-Next (NVIDIA's NVFP4)"), lines)
        self.assertIn("(Qwen3.8-Flash-Next (NVFP4, GPTQ): not offered, not published yet", out)
        self.assertEqual(cfg["model_name"], "orcarouter-qwen3.8-flash-next-uncensored-nvfp4")   # still the default
        code, out, cfg, _ = run(ram, found, ["--family", "qwen-nvfp4-gptq", "--no-start"], repos=setup.NVFP4_REPOS)
        self.assertEqual(code, 1)
        self.assertIn("is not offered: not published yet", out)

    def test_nvidia_s_install_pins_its_upload(self):
        ram, found = PROFILES["128GB-1x24GB"]
        got = []
        code, out, cfg, _ = run(ram, found, ["--family", "qwen-nvidia-nvfp4", "--no-start"], repos=setup.NVFP4_REPOS,
                                downloads=got)
        self.assertEqual(code, 0, out[-3000:])
        rev = "cca8f7fee5ce0af932097340a75e06df25c4e57d"
        self.assertIn(f"https://huggingface.co/{QNV}/resolve/{rev}/pack/experts.bin", got)
        self.assertIn(f"https://huggingface.co/{QNV}/resolve/{rev}/ple-fp8.gguf", got)
        self.assertTrue(arg(cfg, "--native").endswith("/qwen-nvidia-nvfp4-dense.gguf"))
        self.assertTrue(arg(cfg, "--mtp").replace("\\", "/").endswith(setup.repo_dir(QNV) + "/mtp"))


class Published(unittest.TestCase):
    def test_the_menu_says_which_are_censored(self):
        ram, found = PROFILES["128GB-1x24GB"]
        code, out, cfg, _ = run(ram, found, ["--no-start"], repos=ALL, answers="")
        self.assertEqual(code, 0, out[-3000:])
        lines = menu_lines(out)
        self.assertEqual(len(lines), 3 + len(setup.FAMILIES))
        self.assertTrue(lines[0].startswith("1) OrcaRouter"), lines)
        self.assertTrue(lines[1].startswith("2) Qwen3.8-Flash-Next (NVIDIA's NVFP4)"), lines)
        self.assertTrue(lines[2].startswith("3) Qwen3.8-Flash-Next (NVFP4, GPTQ)"), lines)
        for i in (1, 2):
            self.assertIn("censored", lines[i])
            self.assertNotIn("does not refuse", lines[i])
        self.assertIn("plain rounding", lines[1])
        self.assertIn("GPTQ", lines[2])
        self.assertIn("does not refuse", lines[0])        # OrcaRouter's: uncensored
        self.assertEqual(cfg["model_name"], "orcarouter-qwen3.8-flash-next-uncensored-nvfp4")   # still the default

    def test_each_installs_its_own_files(self):
        ram, found = PROFILES["128GB-1x24GB"]
        for f, (repo, name, stem, lic) in QWEN.items():
            with self.subTest(f):
                code, out, cfg, _ = run(ram, found, ["--family", f, "--no-start"], repos=ALL)
                self.assertEqual(code, 0, out[-3000:])
                self.assertEqual(cfg["model_name"], name)
                self.assertTrue(cfg["log"].replace("\\", "/").endswith(f"/{stem}.log"), cfg["log"])
                m = setup.repo_dir(repo)
                for flag in ("--pack", "--native", "--native-dense-gguf", "--ple-gguf", "--embd-gguf", "--mtp"):
                    self.assertIn(m, arg(cfg, flag), flag)
                self.assertIn(f"Its license: {lic}", out)
                self.assertEqual((arg(cfg, "--spec"), arg(cfg, "--spec-min-p")), ("6", "0.7"))

    def test_an_install_is_read_back_as_its_own_family(self):
        with tempfile.TemporaryDirectory() as d:
            for f, (_r, _n, stem, _l) in QWEN.items():
                p = Path(d) / f"{stem}.json"
                p.write_text(json.dumps({"exe": "x", "args": ["--max-context", "131072", "--kv", "int8"]}))
                ch = setup.choices_from_config(p)
                self.assertEqual((ch["family"], ch["model"]), (f, "NVFP4"))
            p = Path(d) / "strata-iq3_xxs.json"            # the GGUF qwen family stays itself
            p.write_text(json.dumps({"exe": "x", "args": []}))
            self.assertEqual(setup.choices_from_config(p)["family"], "qwen")


if __name__ == "__main__":
    unittest.main()
