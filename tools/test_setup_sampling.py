"""The model's recommended sampling in setup's run configs (0.1.41-nvfp4.6): every family setup writes gets its
"sampling" block (FAMILY_SAMPLING, from the models' generation_config.json), a re-run keeps the user's own values in
it, and --update adds the block to a config that has none.

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
from test_setup_golden import GOLDEN, PROFILES, install  # noqa: E402
from test_setup_nvfp4 import install_with  # noqa: E402

QWEN = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}   # Qwen/Qwen3.8-Flash-Next's generation_config.json


class Table(unittest.TestCase):
    def test_every_family_has_its_values(self):
        self.assertEqual(setup.QWEN_SAMPLING, QWEN)
        for f in [*setup.FAMILIES, *setup.NVFP4_FAMILIES, *setup.NVFP4_WITHDRAWN]:
            self.assertEqual(setup.FAMILY_SAMPLING[f], QWEN, f)       # each one's generation_config: Qwen's values

    def test_the_golden_configs_carry_it(self):
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        written = [(k, v["config"]) for k, v in golden.items() if v["code"] == 0]
        self.assertTrue(written)
        for key, cfg in written:
            self.assertEqual(cfg["sampling"], setup.FAMILY_SAMPLING[key.split()[1]], key)

    def test_the_nvfp4_models_and_the_bundle(self):
        ram, found = PROFILES["128GB-1x24GB"]
        for family in setup.NVFP4_FAMILIES:
            with self.subTest(family):
                code, out, cfg, _ = install_with(ram, found, ["--family", family, "--no-start"])
                self.assertEqual(code, 0, out[-3000:])
                self.assertEqual(cfg["sampling"], QWEN)
        bundle = json.loads((ROOT / "release" / "windows" / "config" / "strata-qwen-nvfp4-gptq.json").read_text())
        self.assertEqual(bundle["sampling"], QWEN)
        self.assertIs(bundle["uncensored"], False)                   # the switch's default stays where it is


class Rerun(unittest.TestCase):
    ARGV = ["--setup", "--family", "qwen", "--model", "Q2_0", "--context", "65536", "--no-start"]

    def test_the_user_s_values_win_the_missing_keys_are_added(self):
        ram, found = PROFILES["64GB-1x32GB"]
        code, out, first, _ = install(ram, found, self.ARGV)
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(first["sampling"], QWEN)
        old = {**first, "sampling": {"temperature": 0.3, "min_p": 0.05}}
        code, out, cfg, _ = install(ram, found, self.ARGV, configs=[("strata-q2_0.json", old)])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["sampling"], {"temperature": 0.3, "min_p": 0.05, "top_p": 0.95, "top_k": 20})
        self.assertIn("sampling temperature, sampling min_p", out)       # said as kept


class Update(unittest.TestCase):
    CFG = {"exe": "x", "args": ["--max-context", "65536"], "model_name": "qwen3.8-flash-next-q2_0", "port": 8080}

    def update(self, cfg):
        ram, found = PROFILES["64GB-1x32GB"]
        return install(ram, found, ["--update"], configs=[("strata-q2_0.json", cfg)],
                       extra=[mock.patch.object(setup, "engine_version", lambda exe: (0, 1, 41))])

    def test_a_config_without_it_gets_it(self):
        code, out, cfg, _ = self.update(self.CFG)
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["sampling"], QWEN)
        self.assertIn("the model's recommended sampling (temperature 1, top_p 0.95, top_k 20)", out)
        self.assertEqual(cfg["args"], self.CFG["args"])

    def test_a_block_of_the_user_s_is_left_alone(self):
        code, out, cfg, _ = self.update({**self.CFG, "sampling": {"temperature": 0}})
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg["sampling"], {"temperature": 0})          # greedy, as the user set it
        self.assertNotIn("recommended sampling", out)

    def test_an_unknown_model_is_left_alone(self):
        code, out, cfg, _ = install(*PROFILES["64GB-1x32GB"], ["--update"],
                                    configs=[("strata-mine.json", {**self.CFG, "model_name": "mine"})],
                                    extra=[mock.patch.object(setup, "engine_version", lambda exe: (0, 1, 41)),
                                           mock.patch.object(setup, "choices_from_config",
                                                             lambda p: {"family": "something-else"})])
        self.assertEqual(code, 0, out[-3000:])
        self.assertNotIn("sampling", cfg)


if __name__ == "__main__":
    unittest.main()
