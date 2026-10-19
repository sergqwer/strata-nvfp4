"""The model's recommended sampling (0.1.41-nvfp4.6 .. 0.1.44.1-nvfp4.2: a "sampling" block setup wrote into every run
config).  Since 0.2.0-beta-nvfp4.1 the server applies the model card's set itself, per thinking state (upstream 0.1.45,
serve/server.py MODEL_CARD_SAMPLING), so setup writes none: a new config has no block, a re-run keeps the block an older
install (or the user) has, and --update leaves configs as they are.

    python -m unittest tools.test_setup_sampling
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
from serve import server  # noqa: E402
from test_setup_golden import GOLDEN, PROFILES, install  # noqa: E402
from test_setup_nvfp4 import run  # noqa: E402

QWEN = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}   # Qwen/Qwen3.8-Flash-Next's generation_config.json


class Table(unittest.TestCase):
    def test_the_server_s_model_card_has_the_values(self):
        self.assertEqual(setup.QWEN_SAMPLING, QWEN)
        card = server.model_card_sampling("qwen4exp", True)            # a config without "arch": Flash-Next's card
        self.assertEqual({k: card[k] for k in QWEN}, QWEN)             # thinking: the values setup used to write

    def test_the_golden_configs_have_no_block(self):
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        written = [(k, v["config"]) for k, v in golden.items() if v["code"] == 0]
        self.assertTrue(written)
        for key, cfg in written:
            self.assertNotIn("sampling", cfg, key)

    def test_the_nvfp4_models_and_the_bundle(self):
        ram, found = PROFILES["128GB-1x24GB"]
        for family in setup.NVFP4_FAMILIES:                  # the committed tables (0.1.42-nvfp4.2: the Q8_0 packs too)
            with self.subTest(family):
                code, out, cfg, _ = run(ram, found, ["--family", family, "--no-start"], repos=setup.NVFP4_REPOS)
                self.assertEqual(code, 0, out[-3000:])
                self.assertNotIn("sampling", cfg)
        bundle = json.loads((ROOT / "release" / "windows" / "config" / "strata-qwen-nvfp4-gptq.json").read_text())
        self.assertNotIn("sampling", bundle)
        self.assertIs(bundle["uncensored"], False)                   # the switch's default stays where it is


class Rerun(unittest.TestCase):
    ARGV = ["--setup", "--family", "qwen", "--model", "Q2_0", "--context", "65536", "--no-start"]

    def test_an_older_install_s_block_is_kept(self):
        ram, found = PROFILES["64GB-1x32GB"]
        code, out, first, _ = install(ram, found, self.ARGV)
        self.assertEqual(code, 0, out[-3000:])
        self.assertNotIn("sampling", first)
        for block in (dict(QWEN), {"temperature": 0.3, "min_p": 0.05}):   # setup's old one, the user's own
            with self.subTest(block=block):
                code, out, cfg, _ = install(ram, found, self.ARGV, configs=[("strata-q2_0.json", {**first, "sampling": block})])
                self.assertEqual(code, 0, out[-3000:])
                self.assertEqual(cfg["sampling"], block)


class Update(unittest.TestCase):
    CFG = {"exe": "x", "args": ["--max-context", "65536"], "model_name": "qwen3.8-flash-next-q2_0", "port": 8080}

    def update(self, cfg):
        ram, found = PROFILES["64GB-1x32GB"]
        return install(ram, found, ["--update"], configs=[("strata-q2_0.json", cfg)],
                       extra=[mock.patch.object(setup, "engine_version", lambda exe: (0, 1, 41))])

    def test_a_config_without_it_gets_none(self):
        code, out, cfg, _ = self.update(self.CFG)
        self.assertEqual(code, 0, out[-3000:])
        self.assertNotIn("sampling", cfg)
        self.assertNotIn("recommended sampling", out)
        self.assertEqual(cfg["args"], self.CFG["args"])

    def test_a_block_is_left_alone(self):
        code, out, cfg, _ = self.update({**self.CFG, "sampling": {"temperature": 0}})
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["sampling"], {"temperature": 0})          # greedy, as the user set it


if __name__ == "__main__":
    unittest.main()
