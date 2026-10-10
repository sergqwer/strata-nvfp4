"""This fork's NVFP4 models in setup (qwen-nvfp4-gptq, the default, and qwen-nvidia-nvfp4; huihui-nvfp4 and
orca-nvfp4 withdrawn, an install of either kept; --uncensored): chosen from the menu or by --family, the
start script's engine arguments (the tray's working set), a family refused while its file table (NVFP4_REPOS) is
empty, the requirements (an NVIDIA RTX 20+ card, 12 GB of VRAM, 64 GB of RAM, the engine's low-RAM mode below 92 GiB,
the disk, the page file), the engine (this fork's, never upstream's) and --dry-run.  setup.main() on mocked PCs through
tools/test_setup_golden.py's harness: no GPU, no downloads, nothing written outside a temp folder.

    python -m unittest tools.test_setup_nvfp4
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from test_setup_golden import PROFILES, card, install  # noqa: E402

QG, QNV = setup.QWEN_GPTQ_REPO, setup.QWEN_NV_REPO
ORCA = "Maximilian228/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-NVFP4-GPTQ-Strata"   # withdrawn in 0.1.41-nvfp4.5
HUIHUI = "Maximilian228/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-GPTQ-Strata"   # withdrawn: NVFP4_WITHDRAWN
OTHER = "Maximilian228/Other-NVFP4-GPTQ-Strata"                                  # a second repository (links)
FORK = "https://github.com/sergqwer/strata-nvfp4"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def table(own: str) -> dict:
    """A filled table as the upload will give it (the tray pack's sizes); the PLE table and the embedding are the
    same files in both repositories (Qwen's own), the rest each model's."""
    pack = {"index.txt": 117515, "experts.bin": 67948118016, "dense.bin": 1538625024, "native_experts.txt": 1594,
            "tokenizer/vocab.json": 5737005, "tokenizer/merges.txt": 3600844, "tokenizer/tokenizer.json": 554,
            "tokenizer/token_type.json": 744960, "tokenizer/chat_template.jinja": 8952}
    return {"revision": sha(own)[:40],
            "pack": {f"pack/{n}": (s, sha(own + n)) for n, s in pack.items()},
            "dense": {"dense-q8_0.gguf": (5_000_000_000, sha(own + "dense"))},
            "ple": {"ple-fp8-e4m3.gguf": (51200246144, sha("ple"))},
            "embd": {"token-embd-bf16.gguf": (1271398688, sha("embd"))},
            "mtp": {"mtp/rt/experts.bin": (707788800, sha(own + "mtp")), "mtp/rt/dense.bin": (116099072, sha(own + "md")),
                    "mtp/rt/dense.txt": (1880, sha(own + "mt"))},
            "profile": {}}


FILLED = {QG: table("qwen-gptq"), QNV: table("qwen-nv")}
EMPTY = {r: {"revision": "", **{c: {} for c in setup.NVFP4_COMPONENTS}} for r in (QG, QNV)}
# the withdrawn family's PLE copy, as setup links it (the test tables' PLE SHA-256)
WITHDRAWN = {"huihui-nvfp4": {**setup.NVFP4_WITHDRAWN["huihui-nvfp4"],
                              "copies": {HUIHUI: {"ple-fp8-e4m3.gguf": sha("ple")}}}}


def fork_engine(eng_holder: list):
    """get_prebuilt that installs this fork's release engine (its BUILD.json as make_windows_bundle.py writes it)."""
    def get(url_base, gpu, vision, updating=False, toolkit=13):
        eng = setup.engine_dir(toolkit)
        eng.mkdir(exist_ok=True)
        (eng / "BUILD.json").write_text(json.dumps({
            "version": "0.1.40.3-nvfp4.1", "source": "release", "archs": [75, 86, 89, 120], "ptx": False,
            "cuda": "13.3", "vision": "cpu", "portable": True, "fork": FORK, "cuda_libs": "bundled"}))
        (eng / setup.EXE).write_bytes(b"")
        (eng / setup.VEXE).write_bytes(b"")
        eng_holder.append(url_base)
        return eng
    return get


def verify(s, size, digest):
    setup.mark(s, f"sha256 {digest}")


def run(ram, found, argv, repos=FILLED, answers=None, extra=(), downloads=None, pip=None):
    """setup.main() through the golden harness, with the NVFP4 tables and this fork's engine."""
    got = [] if downloads is None else downloads

    def fake_download(url, dst, what=None, unpinned_ok=True):
        got.append(url)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"")
        setup.mark(dst)

    patches = [mock.patch.object(setup, "NVFP4_REPOS", repos),
               mock.patch.object(setup, "get_prebuilt", fork_engine([])),
               mock.patch.object(setup, "download", fake_download),
               mock.patch.object(setup, "verify_sha256", verify),
               mock.patch.object(setup, "pip_cuda_libs", lambda tk=13: (pip if pip is not None else []).append(tk)),
               *extra]
    return install(ram, found, argv, answers=answers, extra=patches)


def install_with(ram, found, argv, configs=()):
    """run() with run configs already in the Strata folder (a re-run)."""
    patches = [mock.patch.object(setup, "NVFP4_REPOS", FILLED), mock.patch.object(setup, "get_prebuilt", fork_engine([])),
               mock.patch.object(setup, "verify_sha256", verify),
               mock.patch.object(setup, "download", lambda url, dst, what=None, unpinned_ok=True:
                                 (dst.parent.mkdir(parents=True, exist_ok=True), dst.write_bytes(b""), setup.mark(dst))),
               mock.patch.object(setup, "pip_cuda_libs", lambda tk=13: None)]
    return install(ram, found, argv, extra=patches, configs=configs)


def arg(cfg, flag):
    a = cfg["args"]
    return a[a.index(flag) + 1] if flag in a else None


class Choice(unittest.TestCase):
    def test_the_default_is_qwen_gptq(self):
        ram, found = PROFILES["128GB-1x24GB"]
        code, out, cfg, asked = run(ram, found, ["--no-start"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(asked, [])
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-gptq-nvfp4")
        self.assertIn("Maximilian228--Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata", arg(cfg, "--pack"))
        self.assertIn("Its license: Qwen Community License 1.0", out)

    def test_the_menu_offers_the_nvfp4_models_first_then_the_gguf_models(self):
        ram, found = PROFILES["128GB-1x24GB"]
        code, out, cfg, asked = run(ram, found, ["--no-start"], answers="")
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("Which model? [1]", asked[0])
        menu = out[out.index("=== Step 2"):out.index("[ok] model:")]
        lines = [x.strip() for x in menu.splitlines() if x.strip()[:2] in {f"{i})" for i in range(1, 10)}]
        self.assertTrue(lines[0].startswith("1) Qwen3.8-Flash-Next (NVFP4, GPTQ)"), lines)
        self.assertTrue(lines[1].startswith("2) Qwen3.8-Flash-Next (NVIDIA's NVFP4)"), lines)
        self.assertTrue(lines[2].startswith("3) Qwen3.8-Flash-Next "), lines)
        self.assertEqual(len(lines), 2 + len(setup.FAMILIES))
        self.assertNotIn("Huihui", menu)                                  # withdrawn
        self.assertNotIn("OrcaRouter", menu)                              # withdrawn
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-gptq-nvfp4")

    def test_qwen_gptq_by_number_and_by_flag(self):
        ram, found = PROFILES["128GB-1x24GB"]
        for argv, answers in ((["--no-start"], {"Which model": "1"}), (["--family", "qwen-nvfp4-gptq", "--no-start"], "")):
            with self.subTest(argv=argv):
                code, out, cfg, _ = run(ram, found, argv, answers=answers)
                self.assertEqual(code, 0, out[-3000:])
                self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-gptq-nvfp4")
                self.assertIn("Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata", arg(cfg, "--native"))
                self.assertIn("Its license: Qwen Community License 1.0", out)

    def test_the_gguf_families_stay(self):
        ram, found = PROFILES["128GB-1x24GB"]
        code, out, cfg, _ = run(ram, found, ["--family", "qwen", "--no-start"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-iq3_xxs")
        # a GGUF size without --family keeps the GGUF default (the NVFP4 models have no sizes)
        code, out, cfg, _ = run(ram, found, ["--model", "IQ2_XS", "--no-start"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-iq2_xs")
        code, out, _, _ = run(ram, found, ["--family", "qwen-nvfp4-gptq", "--model", "IQ2_XS", "--no-start"])
        self.assertNotEqual(code, 0)
        self.assertIn("has one size: leave out --model IQ2_XS", out)

    def test_a_rerun_keeps_the_installed_family(self):
        # --setup on a folder with a GGUF model installed: that family stays the default, not the NVFP4 one
        ram, found = PROFILES["128GB-1x24GB"]
        old = {"exe": "x", "args": ["--max-context", "65536"], "model_name": "qwen3.8-flash-next-coder-q2_0"}
        code, out, cfg, _ = install_with(ram, found, ["--setup", "--no-start"], configs=[("strata-coder-q2_0.json", old)])
        self.assertEqual(code, 0, out[-3000:])
        self.assertTrue(cfg["model_name"].startswith("qwen3.8-flash-next-coder"), cfg["model_name"])
        nv = {"exe": "x", "args": ["--max-context", "131072"], "model_name": "qwen3.8-flash-next-gptq-nvfp4"}
        code, out, cfg, _ = install_with(ram, found, ["--setup", "--no-start"], configs=[("strata-qwen-gptq-nvfp4.json", nv)])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-gptq-nvfp4")
        code, out, cfg, _ = install_with(ram, found, ["--no-start"])         # a new install: qwen-nvfp4-gptq
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-gptq-nvfp4")

    def test_cuda_12_does_not_offer_them(self):
        # --cuda 12 (or a split with a Pascal card): the NVFP4 models need the CUDA 13 engine - never the default,
        # refused by name with the reason, said so by --check
        ram, found = PROFILES["128GB-1x24GB"]
        code, out, cfg, _ = run(ram, found, ["--cuda", "12", "--no-start"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertFalse(cfg["model_name"].endswith("-nvfp4"), cfg["model_name"])
        self.assertIn("not offered, it runs on the CUDA 13 engine only", out)
        code, out, _, _ = run(ram, found, ["--cuda", "12", "--family", "qwen-nvfp4-gptq", "--no-start"])
        self.assertEqual(code, 1)
        self.assertIn("leave out --cuda 12", out)
        self.assertNotIn("--gpu N", out)
        code, out, _, _ = run(ram, found, ["--cuda", "12", "--check"])
        self.assertIn("qwen-nvfp4-gptq not offered: it runs on the CUDA 13 engine only", " ".join(out.split()))

    def test_an_earlier_install_is_read_back(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-qwen-gptq-nvfp4.json"
            p.write_text(json.dumps({"exe": "x", "args": ["--max-context", "131072", "--kv", "int8"],
                                     "vision": {"gpu": False}}))
            ch = setup.choices_from_config(p)
        self.assertEqual((ch["family"], ch["model"], ch["context"], ch["kv"], ch["vision"]),
                         ("qwen-nvfp4-gptq", "NVFP4", 131072, "int8", "cpu"))


class EngineArgs(unittest.TestCase):
    def test_the_tray_s_working_set(self):
        ram, found = PROFILES["128GB-1x24GB"]
        code, out, cfg, _ = run(ram, found, ["--family", "qwen-nvfp4-gptq", "--context", "262144", "--vision", "yes",
                                             "--no-start"])
        self.assertEqual(code, 0, out[-3000:])
        m = "<T>/models/Maximilian228--Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata"
        self.assertEqual(cfg["args"], [
            "--pack", f"{m}/pack", "--native", f"{m}/dense-q8_0.gguf", "--native-dense-gguf", f"{m}/dense-q8_0.gguf",
            "--ple-gguf", f"{m}/ple-fp8-e4m3.gguf", "--embd-gguf", f"{m}/token-embd-bf16.gguf",
            "--expert-profile", "<T>/data/expert-profile.bin", "--expert-cache", "auto", "--prefill", "auto",
            "--spec", "6", "--spec-min-p", "0.7", "--mtp", f"{m}/mtp/rt", "--max-context", "262144", "--kv", "int8",
            "--vision"])
        self.assertEqual(cfg["tokenizer"], f"{m}/pack/tokenizer")
        self.assertEqual(cfg["vision"], {"exe": "<T>/engine/strata-vision" + (".exe" if setup.WIN else ""),
                                         "mmproj": f"<T>/models/{setup.MMPROJ}", "model": f"{m}/dense-q8_0.gguf",
                                         "gpu": False, "max_tokens": 1024, "threads": cfg["vision"]["threads"]})
        self.assertEqual((cfg["gpu"], cfg["gpus_asked"]), (0, True))       # one card, not offered a split at start

    def test_the_kv_cache_only_past_8k_and_the_recommended_context(self):
        for prof, want in (("128GB-1x24GB", "131072"), ("64GB-1x32GB", "262144"), ("96GB-1x16GB", "65536")):
            ram, found = PROFILES[prof]
            with self.subTest(prof):
                code, out, cfg, _ = run(ram, found, ["--family", "qwen-nvfp4-gptq", "--no-start"])
                self.assertEqual(code, 0, out[-3000:])
                self.assertEqual(arg(cfg, "--max-context"), want)
                self.assertEqual(arg(cfg, "--kv"), "int8")
                self.assertNotIn("--vision", cfg["args"])
                self.assertNotIn("--kv-resident", cfg["args"])               # the K/V grows in VRAM instead
        ram, found = PROFILES["128GB-1x24GB"]
        code, out, cfg, _ = run(ram, found, ["--family", "qwen-nvfp4-gptq", "--context", "8192", "--no-start"])
        self.assertNotIn("--kv", cfg["args"])

    def test_a_shipped_profile_and_one_from_the_repository(self):
        repos = {**FILLED, QG: {**FILLED[QG], "profile": {"expert-profile.bin": (196632, sha("prof"))}}}
        ram, found = PROFILES["128GB-1x24GB"]
        code, out, cfg, _ = run(ram, found, ["--family", "qwen-nvfp4-gptq", "--no-start"], repos=repos)
        self.assertEqual(code, 0, out[-3000:])
        self.assertTrue(arg(cfg, "--expert-profile").endswith("-Strata/expert-profile.bin"))


class Tables(unittest.TestCase):
    def test_the_placeholders_refuse_both_families(self):
        for f in setup.NVFP4_FAMILIES:
            files, why = setup.nvfp4_files(f)
            if setup.NVFP4_REPOS[setup.NVFP4_FAMILIES[f]["sources"]["pack"]]["revision"]:
                continue                                   # filled after the upload
            self.assertEqual(files, [])
            self.assertIn("not on Hugging Face yet", why)
            self.assertNotIn("NVFP4_REPOS", why)                 # the user's words, not the maintainer's

    def test_an_empty_table_is_not_offered(self):
        ram, found = PROFILES["128GB-1x24GB"]
        code, out, cfg, _ = run(ram, found, ["--family", "qwen-nvfp4-gptq", "--no-start"], repos=EMPTY)
        self.assertEqual(code, 1)
        self.assertIn("is not offered: not published yet", out)
        self.assertIn("its pack files are not on Hugging Face yet", out)
        # without --family: the GGUF default, and a line says why the NVFP4 models are not in the menu
        code, out, cfg, _ = run(ram, found, ["--no-start"], repos=EMPTY)
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-iq3_xxs")
        self.assertIn("(Qwen3.8-Flash-Next (NVFP4, GPTQ): not offered, not published yet", out)

    def test_an_incomplete_table_is_refused(self):
        def without(repo, comp, path=None, **change):
            t = json.loads(json.dumps(FILLED))
            if path:
                del t[repo][comp][path]
            else:
                t[repo].update(change)
            return {r: {k: ({p: tuple(x) for p, x in v.items()} if isinstance(v, dict) else v) for k, v in d.items()}
                    for r, d in t.items()}

        cases = {"index.txt": without(QG, "pack", "pack/index.txt"),
                 "tokenizer/vocab.json": without(QG, "pack", "pack/tokenizer/vocab.json"),
                 "upload is not finished yet": without(QG, "pack", revision="main"),
                 "its mtp files are not on Hugging Face yet": without(QG, "mtp", mtp={})}
        for why, repos in cases.items():
            with self.subTest(why), mock.patch.object(setup, "NVFP4_REPOS", repos):
                files, got = setup.nvfp4_files("qwen-nvfp4-gptq")
                self.assertEqual(files, [])
                self.assertIn(why, got)


class Requirements(unittest.TestCase):
    def test_nvidia_rtx_20_or_newer_only(self):
        with mock.patch.object(setup, "NVFP4_REPOS", FILLED):
            self.assertIsNone(setup.nvfp4_offer("qwen-nvfp4-gptq", card(0, "RTX 2080 Ti", 11.0, "75"), False))
            self.assertIn("RTX 20 card or newer", setup.nvfp4_offer("qwen-nvfp4-gptq", card(0, "GTX 1080 Ti", 11.0, "61"),
                                                                    False))
            amd = {"index": 0, "name": "Radeon RX 7900 XTX", "vram_gb": 24.0, "arch": "gfx1100"}
            self.assertIn("NVIDIA cards only", setup.nvfp4_offer("qwen-nvfp4-gptq", amd, True))

    def test_a_small_card_or_little_ram_is_not_the_default_but_can_be_chosen(self):
        for prof in ("32GB-2x24GB", "8GB-card"):
            ram, found = (63.7, [card(0, "NVIDIA GeForce RTX 4060", 7.99, "89")]) if prof == "8GB-card" \
                else PROFILES[prof]
            with self.subTest(prof):
                code, out, cfg, _ = run(ram, found, ["--no-start"])
                self.assertEqual(code, 0, out[-3000:])
                self.assertFalse(cfg["model_name"].endswith("-nvfp4"))      # --yes alone: a GGUF model
                self.assertIn("<- needs", out)
                code, out, cfg, _ = run(ram, found, ["--family", "qwen-nvfp4-gptq", "--no-start"])   # explicit: consent
                self.assertEqual(code, 0, out[-3000:])
                self.assertIn("as you chose", out)
                code, out, cfg, _ = run(ram, found, ["--family", "qwen-nvfp4-gptq", "--no-start"], answers="")
                self.assertEqual(code, 1)                                    # Enter declines the risk

    def test_the_low_ram_mode_below_92_gib(self):
        ram, found = PROFILES["64GB-1x32GB"]
        code, out, cfg, _ = run(ram, found, ["--family", "qwen-nvfp4-gptq", "--no-start"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("the engine's low-RAM mode (on by itself below 92 GiB)", out)
        for flag in ("--low-ram", "--no-low-ram", "--mmap-experts", "--resident-experts", "--ram-budget"):
            self.assertNotIn(flag, cfg["args"])                              # the engine decides, as the tray runs
        for argv, want in ((["--low-ram", "on"], ["--low-ram"]), (["--low-ram", "off"], ["--no-low-ram"]),
                           (["--resident-budget-gib", "40"], ["--ram-budget", "40"])):
            with self.subTest(argv):
                code, out, cfg, _ = run(ram, found, ["--family", "qwen-nvfp4-gptq", "--no-start", *argv])
                self.assertEqual(code, 0, out[-3000:])
                i = cfg["args"].index(want[0])
                self.assertEqual(cfg["args"][i:i + len(want)], want)
        ram, found = PROFILES["128GB-1x24GB"]
        code, out, cfg, _ = run(ram, found, ["--family", "qwen-nvfp4-gptq", "--no-start"])
        self.assertIn("all 63 GiB of experts are loaded into RAM", out)

    def test_one_gpu(self):
        ram, found = PROFILES["47GB-2x16GB"]
        code, out, cfg, _ = run(ram, found, ["--family", "qwen-nvfp4-gptq", "--gpus", "0,1", "--no-start"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("runs on one GPU here", out)
        self.assertEqual(cfg["gpu"], 0)

    def test_the_disk(self):
        ram, found = PROFILES["128GB-1x24GB"]
        code, out, _, _ = run(ram, found, ["--family", "qwen-nvfp4-gptq", "--no-start"],
                              extra=[mock.patch.object(setup, "free_gb", lambda p: 100.0)])
        self.assertEqual(code, 1)
        self.assertRegex(out, r"not enough free disk space in .*: need ~12[0-9] GB")


class Files(unittest.TestCase):
    def test_pinned_urls_and_each_file_downloaded_once(self):
        ram, found = PROFILES["128GB-1x24GB"]
        with tempfile.TemporaryDirectory() as d:
            got = []

            def install_in(argv):
                # one models folder for every run (the harness makes a new temp folder per run; the last flag wins)
                return run(ram, found, [*argv, "--models-dir", d], downloads=got)

            code, out, cfg, _ = install_in(["--family", "qwen-nvfp4-gptq", "--no-start"])
            self.assertEqual(code, 0, out[-3000:])
            rev = FILLED[QG]["revision"]
            self.assertIn(f"https://huggingface.co/{QG}/resolve/{rev}/pack/experts.bin", got)
            self.assertEqual(len(got), sum(len(FILLED[QG][c]) for c in setup.NVFP4_COMPONENTS))
            got.clear()
            code, out, cfg, _ = install_in(["--family", "qwen-nvfp4-gptq", "--no-start"])
            self.assertEqual((code, got), (0, []))                           # every file already here and checked
            self.assertIn("pack/experts.bin already downloaded", out)

    def test_modelscope_is_said_not_tried(self):
        ram, found = PROFILES["128GB-1x24GB"]
        with mock.patch.dict(setup.os.environ, {"STRATA_SOURCE": "modelscope"}):
            code, out, cfg, _ = run(ram, found, ["--family", "qwen-nvfp4-gptq", "--no-start"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("ModelScope has no copy of", out)


class Fetch(unittest.TestCase):
    """nvfp4_fetch: a stale finish mark or .part of another version of a file, the pinned revision only, a link."""

    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.models = Path(t.name) / "models"
        self.dst = self.models / setup.repo_dir(QG) / "ple.gguf"
        self.dst.parent.mkdir(parents=True)
        self.got = []

    def fetch(self, data: bytes, size=None):
        def fake_download(url, dst, what=None, unpinned_ok=True):
            self.got.append((url, unpinned_ok))
            if not (dst.exists() and setup.done(dst)):
                dst.write_bytes(data)
                setup.mark(dst)
        files = [("ple", QG, "1" * 40, "ple.gguf", size or len(data), hashlib.sha256(data).hexdigest())]
        with mock.patch.object(setup, "download", fake_download), mock.patch.object(setup, "NVFP4_REPOS", {}):
            setup.nvfp4_fetch(files, {(QG, "ple.gguf"): self.dst}, [self.models])

    def test_a_mark_of_another_version_is_downloaded_again_at_once(self):
        # the table pins a new file: the old one, checked under its own SHA-256, goes now, not after a failed hash
        old = b"A" * 1000
        self.dst.write_bytes(old)
        self.dst.with_name("ple.gguf.done").write_text(f"sha256 {hashlib.sha256(old).hexdigest()}")
        self.fetch(b"B" * 1000)
        self.assertEqual(self.dst.read_bytes(), b"B" * 1000)
        self.assertEqual(self.got, [(f"https://huggingface.co/{QG}/resolve/{'1' * 40}/ple.gguf", False)])
        self.assertIn(f"sha256 {hashlib.sha256(b'B' * 1000).hexdigest()}", self.dst.with_name("ple.gguf.done").read_text())
        self.assertFalse(self.dst.with_name("ple.gguf.part.sha256").exists())

    def test_a_whole_file_whose_check_was_stopped_is_kept(self):
        # downloaded whole, the window closed while it was hashed (a mark without a SHA-256): hashed, not deleted
        new = b"B" * 1000
        self.dst.write_bytes(new)
        setup.mark(self.dst)
        self.fetch(new)
        self.assertEqual(self.dst.read_bytes(), new)
        self.assertIn(f"sha256 {hashlib.sha256(new).hexdigest()}", self.dst.with_name("ple.gguf.done").read_text())

    def test_a_part_of_another_version_is_dropped(self):
        part, want = self.dst.with_name("ple.gguf.part"), self.dst.with_name("ple.gguf.part.sha256")
        part.write_bytes(b"A" * 500)
        want.write_text("a" * 64)
        setup.nvfp4_stale(self.dst, "b" * 64)
        self.assertFalse(part.exists())
        part.write_bytes(b"B" * 500)
        want.write_text("b" * 64)
        setup.nvfp4_stale(self.dst, "b" * 64)
        self.assertTrue(part.exists())                    # the same version: resumed

    def test_the_pinned_revision_never_falls_back_to_main(self):
        asked = []

        def urlopen(req, timeout=None):
            asked.append(req.full_url)
            raise setup.urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
        url = f"https://huggingface.co/{QG}/resolve/{'1' * 40}/ple.gguf"
        with mock.patch.object(setup.urllib.request, "urlopen", urlopen), mock.patch.object(setup.time, "sleep"), \
                mock.patch("sys.stdout", io.StringIO()) as out, self.assertRaises(SystemExit):
            setup.download(url, self.dst, "ple.gguf", unpinned_ok=False)
        self.assertTrue(asked)
        self.assertFalse(any("/resolve/main/" in u for u in asked), asked)
        self.assertIn("not at its pinned revision any more", out.getvalue())

    def test_a_linked_file_drops_its_part(self):
        data = b"P" * (64 << 20)
        sha_ = hashlib.sha256(data).hexdigest()
        other = self.models / setup.repo_dir(OTHER) / "ple.gguf"
        other.parent.mkdir(parents=True)
        other.write_bytes(data)
        setup.mark(other, f"sha256 {sha_}")
        part, want = self.dst.with_name("ple.gguf.part"), self.dst.with_name("ple.gguf.part.sha256")
        part.write_bytes(b"P" * 1000)                      # begun before the other model had it
        want.write_text(sha_)
        repos = {OTHER: {"revision": "2" * 40, "ple": {"ple.gguf": (len(data), sha_)}}}
        files = [("ple", QG, "1" * 40, "ple.gguf", len(data), sha_)]
        with mock.patch.object(setup, "NVFP4_REPOS", repos), \
                mock.patch.object(setup, "download", mock.Mock(side_effect=AssertionError("downloaded"))), \
                mock.patch("sys.stdout", io.StringIO()):
            setup.nvfp4_fetch(files, {(QG, "ple.gguf"): self.dst}, [self.models])
        self.assertTrue(self.dst.exists())
        self.assertFalse(part.exists())
        self.assertFalse(want.exists())


class Engine(unittest.TestCase):
    def a(self, **kw):
        return types.SimpleNamespace(**{"build": False, "prebuilt": setup.PREBUILT_URL, "yes": True, **kw})

    def test_the_fork_s_release(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "ROOT", Path(d)):
            pip, built = [], []
            with mock.patch.object(setup, "get_prebuilt", fork_engine([])), \
                    mock.patch.object(setup, "pip_cuda_libs", lambda tk=13: pip.append(tk)), \
                    mock.patch.object(setup, "build_engine", lambda *x, **k: built.append(x)):
                eng, vision = setup.nvfp4_engine(self.a(), card(0, "RTX 5090", 31.8, "120"), "gpu", None)
            self.assertEqual(eng, Path(d) / "engine")
            self.assertEqual((vision, pip, built), ("cpu", [], []))         # its encoder: the CPU; its own cuBLAS

    def test_upstream_s_is_compiled_over(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "ROOT", Path(d)):
            eng = Path(d) / "engine"
            eng.mkdir()
            (eng / "BUILD.json").write_text(json.dumps({"version": "0.1.40.3", "source": "release", "archs": [120]}))
            built = []
            with mock.patch.object(setup, "get_prebuilt", lambda *x, **k: eng), \
                    mock.patch.object(setup, "build_engine", lambda *x, **k: built.append(k) or eng):
                got, _ = setup.nvfp4_engine(self.a(prebuilt="https://github.com/Niko1221/Strata/releases/latest/download/"),
                                            card(0, "RTX 5090", 31.8, "120"), "none", None)
            self.assertEqual(built, [{"toolkit": 13}])
            with mock.patch.object(setup, "get_prebuilt", mock.Mock(side_effect=AssertionError("asked"))), \
                    mock.patch.object(setup, "build_engine", lambda *x, **k: built.append(k) or eng):
                setup.nvfp4_engine(self.a(build=True), card(0, "RTX 5090", 31.8, "120"), "none", None)
            self.assertEqual(len(built), 2)                                 # --build: compiled, nothing downloaded


class Calibration(unittest.TestCase):
    def test_a_rerun_keeps_the_tuned_settings(self):
        ram, found = PROFILES["128GB-1x24GB"]
        cal = {"settings": {"--pcie-frac": "0.35", "--pool-workers": "6"}, "date": "2026-10-01"}
        code, out, cfg, _ = run(ram, found, ["--family", "qwen-nvfp4-gptq", "--no-start"],
                                extra=[mock.patch.object(setup, "saved_calibration", lambda c: cal)])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual((arg(cfg, "--pcie-frac"), arg(cfg, "--pool-workers")), ("0.35", "6"))
        self.assertIn("the settings tuned for this PC earlier are used (2026-10-01)", out)


class DryRun(unittest.TestCase):
    def test_the_plan_and_nothing_else(self):
        ram, found = PROFILES["128GB-1x24GB"]
        stop = mock.Mock(side_effect=AssertionError("did something in a dry run"))
        repos = {**FILLED, **{setup.NVFP4_FAMILIES[f]["sources"]["pack"]: table(f) for f in setup.NVFP4_FAMILIES
                              if setup.NVFP4_FAMILIES[f]["sources"]["pack"] not in FILLED}}   # every family published
        for family in setup.NVFP4_FAMILIES:
            with self.subTest(family):
                code, out, cfg, _ = run(ram, found, ["--family", family, "--dry-run", "--vision", "yes"], repos=repos,
                                        extra=[
                    mock.patch.object(setup, "data_folder", stop), mock.patch.object(setup, "pip_install", stop),
                    mock.patch.object(setup, "get_prebuilt", stop), mock.patch.object(setup, "build_engine", stop),
                    mock.patch.object(setup, "download", stop), mock.patch.object(setup, "free_gb", stop),
                    mock.patch.object(setup, "write_setup_config", stop), mock.patch.object(setup, "start", stop)])
                self.assertEqual(code, 0, out[-3000:])
                self.assertIsNone(cfg)
                self.assertIn("Plan (--dry-run: nothing is downloaded, installed or written):", out)
                for line in ("pack     9 file(s), 69.5 GB", "ple      1 file(s), 51.2 GB", "profile  data/expert-profile.bin",
                             "images   mmproj-Qwen3.8-Flash-Next-BF16.gguf from ISTA-DASLab", "disk     ~130 GB more",
                             f"engine   {setup.PREBUILT_ASSET} from sergqwer/strata-nvfp4's release" if setup.WIN else
                             "engine   compiled from this source (sergqwer/strata-nvfp4 publishes"):
                    self.assertIn(line, out)
                flat = " ".join(out.split())
                for words in ("--native-dense-gguf", "--spec 6 --spec-min-p 0.7", "--max-context 131072 --kv int8 "
                              "--vision"):
                    self.assertIn(words, flat)

    def test_never_with_update_rollback_or_calibrate(self):
        ram, found = PROFILES["128GB-1x24GB"]
        stop = mock.Mock(side_effect=AssertionError("ran"))
        for flag in ("--update", "--rollback-engine", "--calibrate"):
            with self.subTest(flag):
                code, out, _, _ = run(ram, found, ["--dry-run", flag], extra=[
                    mock.patch.object(setup, "update_install", stop), mock.patch.object(setup, "rollback_engine", stop),
                    mock.patch.object(setup, "calibrate_config", stop),
                    mock.patch("sys.stderr", io.StringIO())])
                self.assertEqual(code, 2)                  # argparse's error

    def test_it_plans_what_the_run_would_do(self):
        # a new copy of Strata: the dry run is set up like the earlier install, from the files of another data folder
        ram, found = PROFILES["128GB-1x24GB"]
        with tempfile.TemporaryDirectory() as d:
            other = Path(d) / "Strata-old"
            (other / "models").mkdir(parents=True)
            (other / "setup.py").write_text("")
            prev = other / "strata-qwen-gptq-nvfp4.json"
            prev.write_text(json.dumps({"exe": "x", "args": ["--max-context", "65536", "--kv", "int8"],
                                        "model_name": "qwen3.8-flash-next-gptq-nvfp4"}))
            m = other / "models" / setup.repo_dir(QG)
            for path, (size, digest) in FILLED[QG]["pack"].items():
                (m / path).parent.mkdir(parents=True, exist_ok=True)
                (m / path).write_bytes(b"")
                setup.mark(m / path, f"sha256 {digest}")
            settings = {"installs": [str(other)]}
            code, out, cfg, _ = run(ram, found, ["--dry-run"], extra=[
                mock.patch.object(setup, "load_settings", lambda: settings),
                mock.patch.object(setup, "other_installs", lambda s: [other])])
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("Found your earlier install", out)
        self.assertIn("[ok] model: Qwen3.8-Flash-Next (NVFP4, GPTQ)", out)
        self.assertIn("pack     9 file(s), 69.5 GB", out)
        self.assertIn("(9 already here)", out)             # found in the other folder's models
        self.assertIn("--max-context 65536", " ".join(out.split()))

    def test_a_link_is_not_counted_on_the_disk(self):
        ram, found = PROFILES["128GB-1x24GB"]
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / setup.repo_dir(HUIHUI)          # a withdrawn huihui install's PLE table, checked
            (m / "ple-fp8-e4m3.gguf").parent.mkdir(parents=True, exist_ok=True)
            (m / "ple-fp8-e4m3.gguf").write_bytes(b"")
            setup.mark(m / "ple-fp8-e4m3.gguf", f"sha256 {sha('ple')}")
            code, out, _, _ = run(ram, found, ["--family", "qwen-nvfp4-gptq", "--dry-run", "--models-dir", d],
                                  extra=[mock.patch.object(setup, "NVFP4_WITHDRAWN", WITHDRAWN)])
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("ple      1 file(s), 51.2 GB", out)
        self.assertIn("(1 linked from another model's copy)", out)
        self.assertIn("disk     ~78 GB more", out)        # 130 less the 51.2 GB linked

    def test_only_for_the_nvfp4_models(self):
        ram, found = PROFILES["128GB-1x24GB"]
        code, out, _, _ = run(ram, found, ["--family", "qwen", "--dry-run"])
        self.assertEqual(code, 1)
        self.assertIn("--dry-run shows the plan of this fork's NVFP4 models only", out)


class Check(unittest.TestCase):
    def test_check_lists_them(self):
        ram, found = PROFILES["64GB-1x32GB"]
        code, out, _, _ = run(ram, found, ["--check"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("qwen-nvfp4-gptq fits in the engine's low-RAM mode", " ".join(out.split()))
        code, out, _, _ = run(ram, found, ["--check"], repos=EMPTY)
        self.assertIn("qwen-nvfp4-gptq not offered: not published yet", " ".join(out.split()))


class RamRule(unittest.TestCase):
    """ram88: the engine's auto low-RAM rule (generate.cpp, platform::low_ram_auto) - below 92 GiB of ullTotalPhys -
    and setup's NVFP4_ALL_RAM_GB are one number; a 96 GB PC (93.4-95.6 GiB listed) gets every expert in RAM."""
    def test_the_threshold(self):
        self.assertEqual(setup.NVFP4_ALL_RAM_GB, 92)
        for ram, low in ((127.18, False), (95.6, False), (93.4, False), (92.0, False), (91.9, True), (63.7, True),
                         (0.0, False)):
            with self.subTest(ram):
                self.assertEqual(setup.engine_low_ram(ram), low)

    def test_check_and_install_on_a_96_gb_pc(self):
        found = [card(0, "NVIDIA GeForce RTX 5090", 31.8, "120")]
        code, out, _, _ = run(95.6, found, ["--check"])
        self.assertEqual(code, 0, out[-3000:])
        line = next(x for x in out.splitlines() if x.strip().startswith("qwen-nvfp4-gptq"))
        self.assertEqual(line.split(None, 1)[1], "fits (all 63 GiB of experts in RAM)")
        code, out, cfg, _ = run(95.6, found, ["--family", "qwen-nvfp4-gptq", "--no-start"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("RAM: 95.6 GiB - all 63 GiB of experts are loaded into RAM", out)
        for flag in ("--low-ram", "--no-low-ram", "--mmap-experts", "--resident-experts", "--ram-budget"):
            self.assertNotIn(flag, cfg["args"])                              # the engine decides the same way
        code, out, _, _ = run(63.7, found, ["--check"])
        self.assertIn("qwen-nvfp4-gptq fits in the engine's low-RAM mode (below 92 GiB", " ".join(out.split()))


def page_file_os(paging, existing=("\\??\\C:\\pagefile.sys",), volumes=None):
    """The OS query page_files() makes, mocked: the registry's PagingFiles / ExistingPageFiles and each drive's
    (volume MB, free MB, page file MB now)."""
    vols = volumes or {}
    return [mock.patch.object(setup, "WIN", True),
            mock.patch.object(setup, "_mm_multi_sz", lambda name: None if paging is None else list(paging)
                              if name == "PagingFiles" else list(existing)),
            mock.patch.object(setup, "_volume_mb", lambda d: vols.get(d, (1907726, 900000, 0)))]


class PageFile(unittest.TestCase):
    """A page file below 60000 MB in all FOR SURE is warned about (setup and the engine, never a stop), a fixed 64000
    advised; sizes in the Virtual memory dialog's MB, every drive's file summed.  A file Windows grows on demand
    (system-managed, or initial below maximum) counts at its size now, never at what it may grow to (upstream #60)."""
    def pf(self, *a, ram=95.6, **k):
        with contextlib.ExitStack() as st:
            for p in page_file_os(*a, **k):
                st.enter_context(p)
            st.enter_context(mock.patch.object(setup, "ram_gb", lambda: ram))
            return setup.page_files(), setup.page_file_warning()

    def test_this_pc_two_fixed_files(self):
        got, w = self.pf(["c:\\pagefile.sys 64000 64000", "d:\\pagefile.sys 64000 64000"],
                         volumes={"C": (1907726, 900000, 64000), "D": (3815453, 1000000, 64000)})
        self.assertEqual(got, (128000, "C: 64000 MB, D: 64000 MB", False))
        self.assertIsNone(w)

    def test_64000_typed_is_enough_and_59999_is_not(self):
        self.assertIsNone(self.pf(["C:\\pagefile.sys 64000 64000"], volumes={"C": (1907726, 900000, 64000)})[1])
        self.assertIsNone(self.pf(["C:\\pagefile.sys 60000 60000"], volumes={"C": (1907726, 900000, 60000)})[1])
        got, w = self.pf(["C:\\pagefile.sys 59999 59999"], volumes={"C": (1907726, 900000, 59999)})
        self.assertEqual(got[0], 59999)
        self.assertIn("INCREASE it to 64000 MB (64 GB), a fixed size", w[0])
        self.assertIn("59999 MB in all for sure (C: 59999 MB)", w[0])
        self.assertNotIn("grow on demand", w[0])

    def test_two_small_files_are_summed(self):
        got, w = self.pf(["C:\\pagefile.sys 32000 32000", "D:\\pagefile.sys 32000 32000"],
                         volumes={"C": (953869, 400000, 32000), "D": (953869, 400000, 32000)})
        self.assertEqual(got[0], 64000)
        self.assertIsNone(w)

    def test_a_growing_file_counts_its_size_now(self):
        got, w = self.pf(["C:\\pagefile.sys 16000 64000"], volumes={"C": (953869, 400000, 16000)})
        self.assertEqual(got, (16000, "C: 16000-64000 MB, grows on demand, now 16000 MB", True))
        self.assertIn("It is set to grow on demand", w[0])
        self.assertIn("a fixed 64 GB worked", w[0])
        self.assertIn("initial and maximum both 64000 MB", w[0])
        self.assertIn("16000 MB for sure, it grows on demand", w[1])
        got, w = self.pf(["C:\\pagefile.sys 16000 64000"], volumes={"C": (953869, 400000, 64000)})   # grown already
        self.assertEqual(got[0], 64000)
        self.assertIsNone(w)

    def test_an_initial_size_above_the_file_now(self):
        self.assertEqual(self.pf(["C:\\pagefile.sys 64000 64000"], volumes={"C": (953869, 400000, 16000)})[0][0], 64000)
        got, w = self.pf(["C:\\pagefile.sys 64000 64000"], volumes={"C": (953869, 30000, 16000)})
        self.assertEqual(got, (46000, "C: 64000 MB (only 46000 fit the free disk)", False))
        self.assertIsNotNone(w)

    def test_system_managed(self):
        # its size now, not 3 x RAM: it may not grow in time while WDDM charges the VRAM
        for paging in (["C:\\pagefile.sys 0 0"], ["?:\\pagefile.sys"], ["C:\\pagefile.sys"]):
            with self.subTest(paging):
                got, w = self.pf(paging, volumes={"C": (953869, 500000, 6000)})
                self.assertEqual(got, (6000, "C: system-managed, grows on demand, now 6000 MB", True))
                self.assertIn("It is set to grow on demand", w[0])
                self.assertIn("INCREASE it to 64000 MB (64 GB), a fixed size", w[0])
        got, w = self.pf(["?:\\pagefile.sys"], existing=("\\??\\D:\\pagefile.sys",),
                         volumes={"D": (1907726, 900000, 6000)})                       # where Windows keeps it now
        self.assertEqual(got, (6000, "D: system-managed, grows on demand, now 6000 MB", True))
        got, w = self.pf(["?:\\pagefile.sys"], volumes={"C": (1907726, 900000, 70000)})   # grown to 70000 now
        self.assertEqual(got[0], 70000)
        self.assertIsNone(w)

    def test_none_and_unknown(self):
        got, w = self.pf([])
        self.assertEqual(got, (0, "no page file", False))
        self.assertIn("0 MB in all for sure (no page file)", w[0])
        self.assertEqual(self.pf(None), (None, None))                                # the setting cannot be read
        with mock.patch.object(setup, "WIN", False):
            self.assertIsNone(setup.page_files())

    def test_the_message(self):
        _, (long, short) = self.pf(["C:\\pagefile.sys 16000 16000"], volumes={"C": (953869, 400000, 16000)})
        for words in ("WARNING", "INCREASE it to 64000 MB", "commit only RAM + page file", "~100 GiB",
                      "~30 GiB that WDDM charges", "significantly slower", "not start", "sysdm.cpl",
                      "Virtual memory", "initial and maximum both 64000 MB"):
            self.assertIn(words, long)
        self.assertIn("INCREASE it to 64000 MB, a fixed size", short)

    def test_check_and_install_say_it_framed_and_again_at_the_end(self):
        found = [card(0, "NVIDIA GeForce RTX 5090", 31.8, "120")]
        small = [mock.patch.object(setup, "page_files", lambda: (16000, "C: 16000 MB", False))]
        code, out, _, _ = run(95.6, found, ["--check"], extra=small)
        self.assertEqual(code, 0, out[-3000:])                               # a warning, never a stop
        self.assertIn("!" * 100, out)
        self.assertIn("[!]  WARNING: the page file is too small - INCREASE it to 64000 MB", out)
        tail = out[out.index("This PC can run Strata"):]
        self.assertIn("WARNING: the page file is too small (16000 MB for sure): INCREASE it to 64000 MB, a fixed size", tail)
        for argv in (["--family", "qwen-nvfp4-gptq", "--no-start"], ["--family", "qwen-nvfp4-gptq", "--dry-run"]):
            with self.subTest(argv):
                code, out, _, _ = run(95.6, found, argv, extra=small)
                self.assertEqual(code, 0, out[-3000:])
                self.assertEqual(out.count("WARNING: the page file is too small - INCREASE"), 1)
                self.assertIn("WARNING: the page file is too small (16000 MB for sure)", out[-1500:])
        growing = [mock.patch.object(setup, "page_files", lambda: (6000, "C: system-managed, grows on demand, now "
                                                                    "6000 MB", True))]
        code, out, _, _ = run(95.6, found, ["--check"], extra=growing)
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("It is set to grow on demand", " ".join(out.split()))
        self.assertIn("WARNING: the page file is too small (6000 MB for sure, it grows on demand): INCREASE it to "
                      "64000 MB, a fixed size", out[out.index("This PC can run Strata"):])
        code, out, _, _ = run(95.6, found, ["--check"])                      # the harness: 2 x 64000 MB
        self.assertNotIn("page file is too small", out)


class Withdrawn(unittest.TestCase):
    """0.1.41-nvfp4.3: huihui-nvfp4 is not offered any more (it loops in long thinking); an install of it keeps working,
    every run says so once and names the replacement (qwen-nvfp4-gptq --uncensored on), and nothing of it is
    deleted."""
    HCFG = {"exe": "<engine>", "args": ["--pack", "M/huihui/pack", "--max-context", "262144", "--kv", "int8"],
            "model_name": "huihui-qwen3.8-flash-next-abliterated-nvfp4", "port": 8080}

    def test_the_flag_names_the_replacement(self):
        ram, found = PROFILES["128GB-1x24GB"]
        code, out, cfg, _ = run(ram, found, ["--family", "huihui-nvfp4", "--no-start"])
        self.assertEqual((code, cfg), (1, None))
        self.assertIn("huihui-nvfp4 was withdrawn: it often loops in long thinking", out)
        self.assertIn("--family qwen-nvfp4-gptq", out)
        self.assertNotIn("huihui-nvfp4", " ".join(setup.NVFP4_FAMILIES))
        self.assertNotIn(HUIHUI, setup.NVFP4_REPOS)

    def test_an_install_of_it_keeps_starting(self):
        ram, found = PROFILES["128GB-1x24GB"]
        started = mock.Mock(return_value=0)
        code, out, cfg, _ = install(ram, found, [], configs=[("strata-huihui-nvfp4.json", self.HCFG)],
                                    extra=[mock.patch.object(setup, "start", started)])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(started.call_args[0][0].name, "strata-huihui-nvfp4.json")   # its own config, as it was
        self.assertEqual(out.count("huihui-nvfp4 was withdrawn"), 1)
        self.assertIn("keeps working as it is and nothing is deleted", out)
        self.assertIn("--family qwen-nvfp4-gptq", out)
        self.assertEqual(cfg["args"], self.HCFG["args"])

    def test_update_keeps_it(self):
        ram, found = PROFILES["128GB-1x24GB"]
        code, out, cfg, _ = install(ram, found, ["--update"], configs=[("strata-huihui-nvfp4.json", self.HCFG)])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(out.count("huihui-nvfp4 was withdrawn"), 1)
        self.assertIn("up to date", out)
        self.assertEqual(cfg["model_name"], self.HCFG["model_name"])
        self.assertEqual(cfg["args"], self.HCFG["args"])

    def test_a_rerun_offers_the_default(self):
        # --setup on a folder with huihui installed: the default is qwen-nvfp4-gptq, not upstream's GGUF default
        ram, found = PROFILES["128GB-1x24GB"]
        code, out, cfg, _ = install_with(ram, found, ["--setup", "--no-start"],
                                         configs=[("strata-huihui-nvfp4.json", self.HCFG)])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-gptq-nvfp4")

    def test_it_is_read_back_as_itself(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-huihui-nvfp4.json"
            p.write_text(json.dumps(self.HCFG))
            ch = setup.choices_from_config(p)
            self.assertEqual((ch["family"], ch["model"]), ("huihui-nvfp4", "NVFP4"))
            self.assertEqual(setup.withdrawn_family(p), "huihui-nvfp4")
            self.assertIsNone(setup.withdrawn_family(Path(d) / "strata-qwen-gptq-nvfp4.json"))

    def test_a_new_copy_is_not_set_up_like_it(self):
        # an earlier install of huihui in another folder: said, and this copy is set up anew (qwen-nvfp4-gptq, the default)
        ram, found = PROFILES["128GB-1x24GB"]
        with tempfile.TemporaryDirectory() as d:
            other = Path(d) / "Strata-old"
            other.mkdir()
            (other / "setup.py").write_text("")
            (other / "strata-huihui-nvfp4.json").write_text(json.dumps(self.HCFG))
            started = mock.Mock(return_value=0)                    # no --no-start: a new copy looks for one
            code, out, cfg, _ = run(ram, found, [], extra=[
                mock.patch.object(setup, "load_settings", lambda: {"installs": [str(other)]}),
                mock.patch.object(setup, "other_installs", lambda s: [other]),
                mock.patch.object(setup, "start", started)])
            self.assertTrue((other / "strata-huihui-nvfp4.json").is_file())
        self.assertEqual(code, 0, out[-3000:])
        self.assertNotIn("Found your earlier install", out)
        self.assertIn("Your earlier install of it keeps working there", out)
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-gptq-nvfp4")

    def test_a_withdrawn_install_s_copy_is_linked(self):
        # switching: the PLE table is the same file in both repositories - linked from the huihui install, not fetched
        ram, found = PROFILES["128GB-1x24GB"]
        with tempfile.TemporaryDirectory() as d:
            m = Path(d) / setup.repo_dir(HUIHUI)
            (m / "ple-fp8-e4m3.gguf").parent.mkdir(parents=True)
            (m / "ple-fp8-e4m3.gguf").write_bytes(b"")
            setup.mark(m / "ple-fp8-e4m3.gguf", f"sha256 {sha('ple')}")
            got = []
            code, out, cfg, _ = run(ram, found, ["--family", "qwen-nvfp4-gptq", "--no-start", "--models-dir", d],
                                    downloads=got, extra=[mock.patch.object(setup, "NVFP4_WITHDRAWN", WITHDRAWN)])
            self.assertTrue((m / "ple-fp8-e4m3.gguf").is_file())                     # the huihui copy stays
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("ple-fp8-e4m3.gguf: the same file as", out)
        self.assertFalse(any(u.endswith("/ple-fp8-e4m3.gguf") for u in got), got)
        self.assertIn(f"https://huggingface.co/{QG}/resolve/{FILLED[QG]['revision']}/pack/experts.bin", got)



class OrcaWithdrawn(unittest.TestCase):
    """0.1.41-nvfp4.5: OrcaRouter's abliteration made a much weaker agent; setup no longer offers it, an install of it
    keeps working and is told why, and the replacement is the original Qwen with --uncensored on."""
    OCFG = {"exe": "<engine>", "args": ["--pack", "M/orca/pack", "--max-context", "262144", "--kv", "int8"],
            "model_name": "orcarouter-qwen3.8-flash-next-uncensored-nvfp4", "port": 8080}

    def test_not_offered_and_the_flag_names_the_replacement(self):
        self.assertNotIn("orca-nvfp4", setup.NVFP4_FAMILIES)
        self.assertNotIn(ORCA, setup.NVFP4_REPOS)
        self.assertEqual(setup.NVFP4_DEFAULT, "qwen-nvfp4-gptq")
        ram, found = PROFILES["128GB-1x24GB"]
        code, out, cfg, _ = run(ram, found, ["--family", "orca-nvfp4", "--no-start"])
        self.assertEqual((code, cfg), (1, None))
        flat = " ".join(out.split())
        self.assertIn("orca-nvfp4 was withdrawn: its abliteration", flat)
        self.assertIn("10-70 steps against the original Qwen's 175", flat)
        self.assertIn("--family qwen-nvfp4-gptq --uncensored on", flat)

    def test_an_install_of_it_keeps_starting(self):
        ram, found = PROFILES["128GB-1x24GB"]
        started = mock.Mock(return_value=0)
        code, out, cfg, _ = install(ram, found, [], configs=[("strata-orca-nvfp4.json", self.OCFG)],
                                    extra=[mock.patch.object(setup, "start", started)])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(started.call_args[0][0].name, "strata-orca-nvfp4.json")
        self.assertEqual(out.count("orca-nvfp4 was withdrawn"), 1)
        self.assertIn("keeps working as it is and nothing is deleted", out)
        self.assertEqual(cfg["args"], self.OCFG["args"])

    def test_a_rerun_offers_the_original(self):
        ram, found = PROFILES["128GB-1x24GB"]
        code, out, cfg, _ = install_with(ram, found, ["--setup", "--no-start"],
                                         configs=[("strata-orca-nvfp4.json", self.OCFG)])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-gptq-nvfp4")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-orca-nvfp4.json"
            p.write_text(json.dumps(self.OCFG))
            self.assertEqual(setup.choices_from_config(p)["family"], "orca-nvfp4")
            self.assertEqual(setup.withdrawn_family(p), "orca-nvfp4")


class Uncensored(unittest.TestCase):
    """--uncensored on|off: this fork's refusal-direction projection (data/uncensor), off by default, offered for
    UNCENSOR_FAMILIES only; never for NVIDIA's model.  Upstream's --experimental-speed-projection on|off is its alias
    for the NVFP4 models."""
    FLAGS = ["--control-vector-scaled", "--control-vector-layer-range", "--cvec-mode", "--cvec-dir"]

    def cvec(self, cfg):
        a = cfg["args"]
        if "--control-vector-scaled" not in a:
            return None
        i = a.index("--control-vector-scaled")
        return a[i:i + 8]

    def test_the_vector_is_shipped(self):
        self.assertTrue(setup.UNCENSOR_VECTOR.is_file(), setup.UNCENSOR_VECTOR)
        self.assertEqual(setup.UNCENSOR_VECTOR.parent.name, "uncensor")
        self.assertTrue((setup.UNCENSOR_VECTOR.parent / "README.md").is_file())
        self.assertEqual(setup.UNCENSOR_LAYERS, (8, 33))
        self.assertEqual(setup.UNCENSOR_FAMILIES, ("qwen-nvfp4-gptq",))
        self.assertIn("qwen-nvidia-nvfp4", setup.UNCENSOR_REFUSED)

    def test_off_by_default(self):
        ram, found = PROFILES["128GB-1x24GB"]
        code, out, cfg, asked = run(ram, found, ["--no-start"], answers="")       # Enter on every question
        self.assertEqual(code, 0, out[-3000:])
        self.assertTrue(any("Disable censorship? [n]" in q for q in asked), asked)
        self.assertIsNone(self.cvec(cfg))
        self.assertNotIn("uncensored", cfg)
        self.assertIn("censorship: on (the original model)", out)
        code, out, cfg, _ = run(ram, found, ["--no-start"])                        # --yes: the same
        self.assertEqual(code, 0, out[-3000:])
        self.assertIsNone(self.cvec(cfg))

    def test_on_writes_the_flags_and_the_request_default(self):
        ram, found = PROFILES["128GB-1x24GB"]
        for argv, answers in ((["--uncensored", "on", "--no-start"], None),
                              (["--no-start"], {"Disable censorship": "y"}),
                              (["--experimental-speed-projection", "on", "--no-start"], None)):   # upstream's name
            with self.subTest(argv=argv, answers=answers):
                code, out, cfg, _ = run(ram, found, argv, answers=answers)
                self.assertEqual(code, 0, out[-3000:])
                self.assertEqual(cfg["model_name"], "qwen3.8-flash-next-gptq-nvfp4")
                self.assertEqual(self.cvec(cfg), [
                    "--control-vector-scaled", str(setup.UNCENSOR_VECTOR).replace(chr(92), "/") + ":1.0",
                    "--control-vector-layer-range", "8", "33", "--cvec-mode", "project", "--cvec-dir"])
                self.assertEqual(cfg["args"][cfg["args"].index("--cvec-dir") + 1], "per-layer")
                self.assertIs(cfg["uncensored"], True)
                self.assertIn("censorship: off (--uncensored on", out)
        code, out, cfg, _ = run(ram, found, ["--uncensored", "off", "--no-start"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertIsNone(self.cvec(cfg))

    def test_the_dry_run_shows_it(self):
        ram, found = PROFILES["128GB-1x24GB"]
        code, out, cfg, _ = run(ram, found, ["--family", "qwen-nvfp4-gptq", "--uncensored", "on", "--dry-run"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("--cvec-mode project", " ".join(out.split()))

    def test_never_for_nvidia_s_model(self):
        ram, found = PROFILES["128GB-1x24GB"]
        for flag in ("--uncensored", "--experimental-speed-projection"):
            with self.subTest(flag):
                code, out, cfg, _ = run(ram, found, ["--family", "qwen-nvidia-nvfp4", flag, "on", "--no-start"])
                self.assertEqual((code, cfg), (1, None))
                flat = " ".join(out.split())
                self.assertIn("--uncensored is not available for qwen-nvidia-nvfp4: the NVIDIA Open Model License does "
                              "not allow bypassing the model's safety guardrails", flat)
                self.assertIn("--family qwen-nvfp4-gptq --uncensored on", flat)
        code, out, cfg, asked = run(ram, found, ["--family", "qwen-nvidia-nvfp4", "--no-start"], answers="")
        self.assertEqual(code, 0, out[-3000:])
        self.assertFalse(any("censorship" in q for q in asked), asked)         # not even asked
        self.assertNotIn("--control-vector-scaled", cfg["args"])

    def test_not_for_the_other_weights_or_the_gguf_models(self):
        ram, found = PROFILES["128GB-1x24GB"]
        for family, why in (("swift", "Swift 1.5 has other weights"), ("coder", "the Coder has other weights"),
                            ("qwen", "measured on qwen-nvfp4-gptq only so far")):
            with self.subTest(family):
                code, out, cfg, _ = install_with(ram, found, ["--family", family, "--uncensored", "on", "--no-start"])
                self.assertEqual((code, cfg), (1, None))
                self.assertIn(f"--uncensored is not available for {family}", out)
                self.assertIn(why, " ".join(out.split()))

    def test_a_new_copy_keeps_it(self):
        ram, found = PROFILES["128GB-1x24GB"]
        with tempfile.TemporaryDirectory() as d:
            other = Path(d) / "Strata-old"
            other.mkdir()
            (other / "setup.py").write_text("")
            vec = str(other / "data" / "uncensor" / setup.UNCENSOR_VECTOR.name)
            (other / "strata-qwen-gptq-nvfp4.json").write_text(json.dumps({
                "exe": "x", "args": ["--max-context", "131072", "--kv", "int8", *setup.uncensor_args(Path(vec))],
                "uncensored": True}))
            ch = setup.choices_from_config(other / "strata-qwen-gptq-nvfp4.json")
            self.assertEqual((ch["family"], ch["uncensored"], ch["esp"]), ("qwen-nvfp4-gptq", "on", "off"))
            code, out, cfg, _ = run(ram, found, ["--dry-run"], extra=[
                mock.patch.object(setup, "load_settings", lambda: {"installs": [str(other)]}),
                mock.patch.object(setup, "other_installs", lambda s: [other])])
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("Found your earlier install", out)
        self.assertIn("censorship: off (--uncensored on", out)


if __name__ == "__main__":
    unittest.main()
