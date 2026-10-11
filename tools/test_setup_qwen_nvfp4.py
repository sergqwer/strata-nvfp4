"""The original, censored Qwen in this fork's NVFP4 format: qwen-nvfp4-gptq (every expert NVFP4 by GPTQ, the default)
and qwen-nvidia-nvfp4 (NVIDIA's ModelOpt NVFP4 checkpoint converted), both uploaded and offered: the menu lists them
first and says they are censored, an install names its config, served model, files and license, and the
files the two repositories share are downloaded once.  0.1.42-nvfp4.2 (Q8Packs): qwen-nvfp4-mix and qwen-q8 for a
128 GB PC - their tables, the files shared with qwen-nvfp4-gptq linked, the engine's RAM rule by the pack's size, the
risk below 128 GB, the names.  tools/test_setup_nvfp4.py's harness (setup.main() on mocked PCs).

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

QNV, QG, QM, QQ = setup.QWEN_NV_REPO, setup.QWEN_GPTQ_REPO, setup.QWEN_MIX_REPO, setup.QWEN_Q8_REPO
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
    """Both uploaded, their tables pinned (2026-10-09)."""
    REV = {"qwen-nvidia-nvfp4": "cca8f7fee5ce0af932097340a75e06df25c4e57d",
           "qwen-nvfp4-gptq": "640a1db97446ddd5e89d2b70ecd0297987192c93"}

    def test_both_are_offered(self):
        for f, (repo, *_r) in QWEN.items():
            files, why = setup.nvfp4_files(f)
            self.assertIsNone(why)
            self.assertEqual({x[2] for x in files}, {self.REV[f]})
            self.assertEqual({x[1] for x in files}, {repo})
            self.assertEqual(round(setup.nvfp4_experts_gib(files), 2), 63.28)

    def test_the_shared_files_are_one_file(self):
        """The PLE table (Qwen-FP8's), the embedding and the MTP head: the same SHA-256 in both; only the experts, the
        dense GGUF and the pack's dense part are each repository's own."""
        a, b = setup.NVFP4_REPOS[QNV], setup.NVFP4_REPOS[QG]
        for comp in ("ple", "embd", "mtp"):
            self.assertEqual(sorted(a[comp].values()), sorted(b[comp].values()), comp)
        self.assertNotEqual(a["pack"]["pack/experts.bin"][1], b["pack"]["pack/experts.bin"][1])
        orca = "40f95a6242e08e9aea2e525cd3b21c3239163d1040682facf5cf2484192d42ce"   # withdrawn orca's ple-fp8.gguf
        self.assertNotEqual(a["ple"]["ple-fp8.gguf"][1], orca)          # orca's file is another FP8 rounding

    def test_the_menu_with_the_committed_tables(self):
        ram, found = PROFILES["128GB-1x24GB"]
        code, out, cfg, _ = run(ram, found, ["--no-start"], repos=setup.NVFP4_REPOS, answers="")
        self.assertEqual(code, 0, out[-3000:])
        lines = menu_lines(out)
        self.assertEqual(len(lines), 4 + len(setup.FAMILIES))
        self.assertTrue(lines[0].startswith("1) Qwen3.8-Flash-Next (NVFP4, GPTQ)"), lines)
        self.assertTrue(lines[1].startswith("2) Qwen3.8-Flash-Next (NVIDIA's NVFP4)"), lines)
        self.assertTrue(lines[2].startswith("3) Qwen3.8-Flash-Next (NVFP4 + Q8_0 mix)"), lines)   # 0.1.42-nvfp4.2
        self.assertTrue(lines[3].startswith("4) Qwen3.8-Flash-Next (Q8_0)"), lines)
        self.assertNotIn("not offered", out)
        self.assertNotIn("<- needs", out)                                       # 128 GB: nothing short
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-gptq-nvfp4")   # the default since 0.1.41-nvfp4.5

    def test_each_install_pins_its_upload_and_links_what_the_other_has(self):
        ram, found = PROFILES["128GB-1x24GB"]
        with tempfile.TemporaryDirectory() as d:
            got = []

            def install(f):
                got.clear()
                return run(ram, found, ["--family", f, "--no-start", "--models-dir", d], repos=setup.NVFP4_REPOS,
                           downloads=got)

            code, out, cfg, _ = install("qwen-nvidia-nvfp4")
            self.assertEqual(code, 0, out[-3000:])
            rev = self.REV["qwen-nvidia-nvfp4"]
            self.assertIn(f"https://huggingface.co/{QNV}/resolve/{rev}/pack/experts.bin", got)
            self.assertIn(f"https://huggingface.co/{QNV}/resolve/{rev}/ple-fp8.gguf", got)
            self.assertTrue(arg(cfg, "--native").endswith("/qwen-nvidia-nvfp4-dense.gguf"))
            code, out, cfg, _ = install("qwen-nvfp4-gptq")
            self.assertEqual(code, 0, out[-3000:])
            rev = self.REV["qwen-nvfp4-gptq"]
            self.assertIn(f"https://huggingface.co/{QG}/resolve/{rev}/pack/experts.bin", got)
            self.assertIn(f"https://huggingface.co/{QG}/resolve/{rev}/qwen-nvfp4-gptq-dense.gguf", got)
            for big in ("ple-fp8.gguf", "token-embd-bf16.gguf", "mtp/experts.bin", "mtp/dense.bin"):
                self.assertNotIn(f"https://huggingface.co/{QG}/resolve/{rev}/{big}", got)   # linked, not downloaded
                self.assertIn(f"{big}: the same file as", out)
            self.assertTrue(all(QG in u for u in got), got)
            self.assertIn(setup.repo_dir(QG), arg(cfg, "--ple-gguf"))


class Published(unittest.TestCase):
    def test_the_menu_says_which_are_censored(self):
        ram, found = PROFILES["128GB-1x24GB"]
        code, out, cfg, _ = run(ram, found, ["--no-start"], repos=ALL, answers="")
        self.assertEqual(code, 0, out[-3000:])
        lines = menu_lines(out)
        self.assertEqual(len(lines), 2 + len(setup.FAMILIES))
        self.assertTrue(lines[0].startswith("1) Qwen3.8-Flash-Next (NVFP4, GPTQ)"), lines)
        self.assertTrue(lines[1].startswith("2) Qwen3.8-Flash-Next (NVIDIA's NVFP4)"), lines)
        for i in (0, 1):
            self.assertIn("censored", lines[i])
            self.assertNotIn("does not refuse", lines[i])
        self.assertIn("plain rounding", lines[1])
        self.assertIn("GPTQ", lines[0])
        self.assertIn("--uncensored on", lines[0])        # the switch, only on ours
        self.assertNotIn("uncensored", lines[1])
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-gptq-nvfp4")   # the default

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


class Q8Packs(unittest.TestCase):
    """0.1.42-nvfp4.2: qwen-nvfp4-mix (GPTQ NVFP4 + Q8_0 in 38 layers, 103.13 GiB of experts) and qwen-q8 (every expert
    Q8_0, 119.53 GiB), for a 128 GB PC.  The engine keeps the mix's arena on 128 GB and starts the Q8_0 pack in its
    low-RAM mode by itself (experts.bin + 16 GiB must fit): setup writes no RAM flag for either and says which."""
    FAM = {"qwen-nvfp4-mix": (QM, "4774b34094cec87d5e4ea7853f5551095cec574a", 103.13, "qwen-nvfp4-mix-dense.gguf",
                              "strata-qwen-mix-nvfp4", "qwen3.8-flash-next-mix-nvfp4", "NVFP4"),
           "qwen-q8": (QQ, "5202b08cff8286e075cbf1ac91695801fdc59db2", 119.53, "qwen-q8-dense.gguf",
                       "strata-qwen-q8_0", "qwen3.8-flash-next-q8_0", "Q8_0")}

    def test_offered_and_pinned(self):
        for f, (repo, rev, gib, *_r) in self.FAM.items():
            with self.subTest(f):
                files, why = setup.nvfp4_files(f)
                self.assertIsNone(why)
                self.assertEqual({x[2] for x in files}, {rev})
                self.assertEqual({x[1] for x in files}, {repo})
                self.assertEqual(round(setup.nvfp4_experts_gib(files), 2), gib)
                self.assertIn(f, setup.UNCENSOR_FAMILIES)
                self.assertIn(f, setup.FAMILY_SAMPLING)
                self.assertEqual(setup.NVFP4_FAMILIES[f]["min_ram"], (120, 128))

    def test_what_they_share_with_gptq(self):
        """Only the experts (and the mix's mix.json, the pack's native_experts.txt) are their own."""
        g = setup.NVFP4_REPOS[QG]
        for repo in (QM, QQ):
            t = setup.NVFP4_REPOS[repo]
            for comp in ("dense", "ple", "embd", "mtp"):
                self.assertEqual(sorted(t[comp].values()), sorted(g[comp].values()), (repo, comp))
            for name in ("pack/dense.bin", "pack/index.txt"):
                self.assertEqual(t["pack"][name], g["pack"][name], (repo, name))
            self.assertNotEqual(t["pack"]["pack/experts.bin"][1], g["pack"]["pack/experts.bin"][1])

    def test_installed_after_gptq_only_their_experts_are_downloaded(self):
        ram, found = PROFILES["128GB-1x24GB"]
        with tempfile.TemporaryDirectory() as d:
            got = []

            def install(f):
                got.clear()
                return run(ram, found, ["--family", f, "--no-start", "--models-dir", d], repos=setup.NVFP4_REPOS,
                           downloads=got)

            code, out, cfg, _ = install("qwen-nvfp4-gptq")
            self.assertEqual(code, 0, out[-3000:])
            for f, (repo, rev, _gib, dense, *_r) in self.FAM.items():
                with self.subTest(f):
                    code, out, cfg, _ = install(f)
                    self.assertEqual(code, 0, out[-3000:])
                    self.assertIn(f"https://huggingface.co/{repo}/resolve/{rev}/pack/experts.bin", got)
                    for big in ("pack/dense.bin", dense, "ple-fp8.gguf", "token-embd-bf16.gguf", "mtp/experts.bin",
                                "mtp/dense.bin"):
                        self.assertNotIn(f"https://huggingface.co/{repo}/resolve/{rev}/{big}", got)
                        self.assertIn(f"{big}: the same file as", out)
                    self.assertTrue(all(repo in u for u in got), got)
                    self.assertTrue(arg(cfg, "--native").endswith("/" + dense))

    def test_the_engine_rule(self):
        """setup's mirror of platform::low_ram_auto (memory_test.cpp has the same cases)."""
        self.assertEqual(setup.NVFP4_BESIDE_GIB, 16)
        for ram, gib, low, by_size in ((127.18, 119.53, True, True), (127.18, 103.13, False, False),
                                       (127.18, 63.28, False, False), (95.6, 63.28, False, False),
                                       (95.6, 103.13, True, True), (91.9, 63.28, True, False),
                                       (255.7, 119.53, False, False), (0.0, 119.53, False, False)):
            with self.subTest((ram, gib)):
                self.assertEqual(setup.engine_low_ram(ram, gib), low)
                self.assertEqual(setup.low_ram_by_size(ram, gib), by_size)

    def test_a_128_gb_pc(self):
        ram, found = PROFILES["128GB-1x24GB"]
        for f, (repo, _rev, gib, _d, stem, name, size) in self.FAM.items():
            with self.subTest(f):
                code, out, cfg, _ = run(ram, found, ["--family", f, "--no-start"], repos=setup.NVFP4_REPOS)
                self.assertEqual(code, 0, out[-3000:])
                self.assertEqual(cfg["model_name"], name)
                self.assertTrue(cfg["log"].replace("\\", "/").endswith(f"/{stem}.log"), cfg["log"])
                for flag in ("--low-ram", "--no-low-ram", "--mmap-experts", "--resident-experts", "--ram-budget"):
                    self.assertNotIn(flag, cfg["args"])                       # the engine decides by itself
                if f == "qwen-q8":
                    self.assertIn("the engine's low-RAM mode (on by itself: its 120 GiB of experts and 16 GiB beside "
                                  "them do not fit)", out)
                    self.assertIn("up to the free RAM less 10 GiB", out)
                else:
                    self.assertIn(f"all {gib:.0f} GiB of experts are loaded into RAM", out)
                self.assertIs(cfg["uncensored"], False)                      # the switch: loaded, off
                self.assertIn("--control-vector-scaled", cfg["args"])
                self.assertEqual(cfg["sampling"], setup.QWEN_SAMPLING)
                self.assertNotIn("<- needs", out)

    def test_below_128_gb_a_risk_not_a_wall(self):
        ram, found = PROFILES["96GB-1x16GB"]
        for f, (_repo, _rev, gib, *_r) in self.FAM.items():
            with self.subTest(f):
                code, out, cfg, _ = run(ram, found, ["--family", f, "--no-start"], repos=setup.NVFP4_REPOS,
                                        answers="")                          # Enter: the question's no
                self.assertEqual(code, 1, out[-3000:])
                self.assertIn("is made for a 128 GB PC", out)
                self.assertIn(f"--family {setup.NVFP4_DEFAULT} (64 GB or more)", out)
                code, out, cfg, _ = run(ram, found, ["--family", f, "--no-start"], repos=setup.NVFP4_REPOS)
                self.assertEqual(code, 0, out[-3000:])                       # --family with --yes: the consent
                self.assertIn("with 96 GB of RAM, as you chose", out)
                self.assertIn(f"its {gib:.0f} GiB of experts and 16 GiB beside them do not fit", out)
        code, out, cfg, _ = run(ram, found, ["--no-start"], repos=setup.NVFP4_REPOS, answers="")
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-gptq-nvfp4")   # the default stays GPTQ
        lines = menu_lines(out)
        self.assertIn("<- needs 128 GB of RAM or more, this PC has 96", lines[2])
        self.assertIn("<- needs 128 GB of RAM or more, this PC has 96", lines[3])
        self.assertNotIn("<- needs", lines[0])

    def test_check(self):
        for profile, want in (("128GB-1x24GB", {"qwen-nvfp4-mix": "fits (all 103 GiB of experts in RAM)",
                                                "qwen-q8": "fits in the engine's low-RAM mode (its 120 GiB of experts "
                                                           "and 16 GiB beside them do not fit"}),
                              ("96GB-1x16GB", {"qwen-nvfp4-mix": "does not fit: it needs 128 GB of RAM or more",
                                               "qwen-q8": "does not fit: it needs 128 GB of RAM or more"})):
            ram, found = PROFILES[profile]
            code, out, _, _ = run(ram, found, ["--check"], repos=setup.NVFP4_REPOS)
            self.assertEqual(code, 0, out[-3000:])
            for f, w in want.items():
                line = next(x for x in out.splitlines() if x.strip().startswith(f + " "))
                self.assertTrue(line.split(None, 1)[1].startswith(w), (profile, line))

    def test_read_back_as_its_own_family(self):
        with tempfile.TemporaryDirectory() as d:
            for f, (*_r, stem, _n, size) in self.FAM.items():
                p = Path(d) / f"{stem}.json"
                p.write_text(json.dumps({"exe": "x", "args": ["--max-context", "262144", "--kv", "int8"]}))
                ch = setup.choices_from_config(p)
                self.assertEqual((ch["family"], ch["model"]), (f, size))
                self.assertEqual(setup.nvfp4_size(f), size)


if __name__ == "__main__":
    unittest.main()
