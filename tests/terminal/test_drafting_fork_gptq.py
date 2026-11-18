"""strata-nvfp4: Auto drafts with DFlash2 for this fork's NVFP4 GPTQ set on a CUDA card that holds as many experts as the RTX 5090 it was
measured on (strata_control/drafting.py FORK_GPTQ_SETS, FORK_GPTQ_MIN_SLOTS), and keeps the MTP everywhere else; data_dir_of knows the
fork's <data>/pack layout."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import drafting

NAMES = ("qwen3.8-flash-next-gptq-nvfp4", "qwen3.8-flash-next-nvfp4-gptq", "qwen3.8-flash-next-nvfp4-gptq-uncensored")
OTHERS = ("qwen3.8-flash-next-nvidia-nvfp4", "qwen3.8-flash-next-mix-nvfp4", "qwen3.8-flash-next-nvfp4-nvfp4")


def tray_args(data: Path) -> list[str]:
    return ["--pack", str(data / "pack"), "--native", str(data / "dense.gguf"), "--expert-cache", "auto", "--prefill", "auto",
            "--spec", "6", "--spec-min-p", "0.7", "--mtp", str(data / "mtp"), "--max-context", "524288", "--kv", "int8"]


class ForkGptq(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name) / "qwen-nvfp4-gptq"
        (self.data / drafting.FOLDER).mkdir(parents=True)
        (self.data / "mtp").mkdir()
        (self.data / "mtp" / "experts.bin").write_bytes(b"x")
        (self.data / drafting.FOLDER / drafting.VOCAB_NAME).write_bytes(b"\0" * 64)
        self.ready = mock.patch.object(drafting, "pack_ready", lambda d: d is not None and Path(d) == self.data)
        self.ready.start()

    def tearDown(self):
        self.ready.stop()
        self.tmp.cleanup()

    def start(self, name, vram_mib, backend="cuda", archs=("sm_120",), drafting_choice=None):
        cfg = {"args": tray_args(self.data), "model_name": name}
        if drafting_choice is not None:
            cfg["drafting"] = drafting_choice
        with mock.patch.object(drafting, "computer_language_signals", lambda: {}):
            decision = drafting.apply(cfg, data_dir=drafting.data_dir_of(cfg["args"]), backend=backend, archs=list(archs), vram_mib=vram_mib)
        return decision, cfg["args"]

    def test_names_supported(self):
        for name in NAMES:
            self.assertTrue(drafting.supported_model(name), name)
        for name in OTHERS:
            self.assertFalse(drafting.supported_model(name), name)

    def test_data_dir_of_fork_layout(self):
        self.assertEqual(drafting.data_dir_of(tray_args(self.data)), self.data)
        up = Path(self.tmp.name) / "up"
        self.assertEqual(drafting.data_dir_of(["--pack", str(up / "packs" / "IQ3_S")]), up)     # upstream's layout unchanged

    def test_auto_dflash2_on_the_measured_card(self):
        for name in NAMES:
            decision, args = self.start(name, 32607)
            self.assertEqual(decision.drafter, "dflash2", name)
            self.assertNotIn("--mtp", args)
            self.assertNotIn("--spec-min-p", args)
            self.assertEqual(args[args.index("--spec") + 1], "6")
            self.assertEqual(args[args.index("--draft-conf") + 1], "0.5")
            self.assertEqual(args[args.index("--draft-fallback") + 1], "none")

    def test_auto_mtp_where_not_measured(self):
        for vram in (12288, 16303, 24463):                            # 12-24 GB: fewer slots than the 5090
            decision, args = self.start(NAMES[1], vram)
            self.assertEqual(decision.drafter, "mtp", vram)
            self.assertIn("--mtp", args)
            self.assertNotIn("--draft", args)
        self.assertEqual(self.start(NAMES[1], None)[0].drafter, "mtp")                     # several cards / not known
        self.assertEqual(self.start(NAMES[1], 32607, backend="hip", archs=("gfx1201",))[0].drafter, "mtp")
        self.assertEqual(self.start(NAMES[1], 120000, archs=("sm_121",))[0].drafter, "mtp")   # the Spark

    def test_named_choice_wins(self):
        self.assertEqual(self.start(NAMES[1], 16303, drafting_choice="dflash2")[0].drafter, "dflash2")
        self.assertEqual(self.start(NAMES[1], 32607, drafting_choice="mtp")[0].drafter, "mtp")

    def test_other_nvfp4_sets_unchanged(self):
        for name in OTHERS:
            decision, args = self.start(name, 32607)
            self.assertEqual(decision.drafter, "mtp", name)
            self.assertIn("--mtp", args)

    def test_no_drafter_file_keeps_the_mtp(self):
        self.ready.stop()
        try:
            with mock.patch.object(drafting, "pack_ready", lambda d: False):
                decision, args = self.start(NAMES[1], 32607)
        finally:
            self.ready.start()
        self.assertEqual(decision.drafter, "mtp")
        self.assertEqual(decision.note, drafting.NOT_DOWNLOADED)
        self.assertIn("--mtp", args)


if __name__ == "__main__":
    unittest.main()
