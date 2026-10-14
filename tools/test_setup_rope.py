"""Tests for setup.py's automatic rope configuration (§4 + §6): the factor derives from the FINAL context
actually served - RAM reductions first, then final / trained 262144 clamped at 1 - explicit
--rope-scaling/--rope-scale choices win verbatim, an explicit none is refused past the trained range, and
nothing turns on by itself inside it.  Past it the method is dynamic where the engine runs it (one GPU, one
dialog at a time, 8-bit K/V), else yarn.  No GPU, no downloads, no prompts.

    python -m unittest tools.test_setup_rope
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class DerivedFactor(unittest.TestCase):
    """The formula itself: max(1.0, final / trained)."""

    def test_the_task_table(self):
        self.assertEqual(setup.derived_factor(131072), 1.0)        # below the trained range: clamped at 1
        self.assertEqual(setup.derived_factor(262144), 1.0)        # exactly the trained range: 1
        self.assertEqual(setup.derived_factor(393216), 1.5)        # 384K
        self.assertEqual(setup.derived_factor(524288), 2.0)        # 512K

    def test_free_form_contexts_get_the_exact_ratio(self):
        # /262144 = /2**18 is exact in binary floating point, so no dust
        self.assertEqual(setup.derived_factor(600000), 600000 / 262144)
        self.assertGreater(setup.derived_factor(600000), 2.0)

    def test_the_clamp_is_what_keeps_128k_at_one(self):
        self.assertEqual(setup.derived_factor(131072), 1.0)        # NOT the raw 0.5


class ResolveRope(unittest.TestCase):
    def test_invalid_explicit_factors_are_rejected(self):
        for ctx in (131072, 262144, 1048576):
            for method in (None, "none", "linear", "yarn"):
                for scale in (float("nan"), float("inf"), float("-inf"), -1.0, 0.0, 0.5):
                    with self.subTest(ctx=ctx, method=method, scale=scale):
                        with self.assertRaisesRegex(ValueError, r"--rope-scale.*finite.*>= 1"):
                            setup.resolve_rope(ctx, method, scale)

    def test_valid_explicit_factors_remain_verbatim(self):
        for ctx in (131072, 262144, 1048576):
            for method in ("linear", "yarn"):
                for scale in (1.0, 1.5, 4.0):
                    with self.subTest(ctx=ctx, method=method, scale=scale):
                        self.assertEqual(setup.resolve_rope(ctx, method, scale), (method, scale))

    # ---- the automatic flow the docs promise (an omitted flag, --yes or the interactive default)
    def test_omitted_384k_gets_dynamic_and_1_5(self):
        self.assertEqual(setup.resolve_rope(393216, None, None), ("dynamic", 1.5))

    def test_omitted_512k_gets_dynamic_and_2(self):
        self.assertEqual(setup.resolve_rope(524288, None, None), ("dynamic", 2.0))

    def test_deterministic_for_yes(self):
        # --yes never prompts, so the resolution must be a pure function of its inputs
        for _ in range(3):
            self.assertEqual(setup.resolve_rope(524288, None, None), ("dynamic", 2.0))

    def test_omitted_method_with_an_explicit_factor(self):
        # --rope-scale alone still expresses the intent to extend: the method defaults, the factor is kept
        self.assertEqual(setup.resolve_rope(393216, None, 1.8), ("dynamic", 1.8))

    # ---- dynamic where the engine runs it, yarn where it would refuse it
    def test_yarn_where_dynamic_cannot_run(self):
        for why in (setup.rope_dynamic_blocker(2), setup.rope_dynamic_blocker(1, 2),
                    setup.rope_dynamic_blocker(1, 1, "q4_0")):
            self.assertTrue(why)
            self.assertEqual(setup.resolve_rope(524288, None, None, blocker=why), ("yarn", 2.0))
        self.assertIsNone(setup.rope_dynamic_blocker(1, 1, "int8"))
        self.assertIsNone(setup.rope_dynamic_blocker(1, None, "fp16", {"source": "local", "version": "0.1.44"}))
        self.assertIn("engine", setup.rope_dynamic_blocker(1, None, "int8", {"version": "0.1.44"}))   # upstream's
        old = {"fork": "x", "version": "0.1.44-nvfp4.1"}                      # this fork's, before dynamic
        self.assertIn("engine", setup.rope_dynamic_blocker(1, None, "int8", old))
        self.assertIsNone(setup.rope_dynamic_blocker(1, None, "int8", {**old, "version": "0.1.44.1-nvfp4.1"}))

    def test_an_explicit_dynamic_where_it_cannot_run_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            setup.resolve_rope(524288, "dynamic", None, blocker=setup.rope_dynamic_blocker(1, 2))
        self.assertIn("one dialog at a time", str(cm.exception))
        self.assertIn("yarn", str(cm.exception))
        self.assertEqual(setup.resolve_rope(524288, "dynamic", None), ("dynamic", 2.0))
        self.assertEqual(setup.resolve_rope(131072, "dynamic", None), (None, None))   # within 262144: the stock model

    def test_amd_gets_yarn_unless_dynamic_is_asked_for(self):
        hip = setup.ROPE_DYNAMIC_HIP                                       # not measured there: no refusal
        self.assertEqual(setup.resolve_rope(524288, None, None, untested=hip), ("yarn", 2.0))
        self.assertEqual(setup.resolve_rope(524288, "dynamic", None, untested=hip), ("dynamic", 2.0))
        cfg = {"exe": "x", "backend": "hip", "rope_auto": True,
               "args": ["--max-context", "524288", "--rope-scaling", "dynamic", "--rope-scale", "2"]}
        self.assertIn("NVIDIA only", setup.fit_auto_rope(cfg))
        self.assertEqual(cfg["args"][3], "yarn")
        cfg["args"][3], cfg["rope_auto"] = "dynamic", False                # the user's own dynamic on AMD: kept
        self.assertIsNone(setup.fit_auto_rope(cfg))

    def test_the_parallel_tip_says_it_switches_rope(self):
        q2 = setup.MODELS["Q2_0"]["arena_gb"]
        self.assertIn("switches RoPE from dynamic to yarn", setup.parallel_note(None, 32, q2, 524288, "int8", True,
                                                                                 True)[0])
        self.assertNotIn("RoPE", setup.parallel_note(None, 32, q2, 524288, "int8", True)[0])

    # ---- explicit selections are respected, verbatim
    def test_explicit_method_gets_the_derived_factor(self):
        self.assertEqual(setup.resolve_rope(524288, "linear", None), ("linear", 2.0))

    def test_explicit_method_and_factor_are_kept_verbatim(self):
        self.assertEqual(setup.resolve_rope(393216, "linear", 2.0), ("linear", 2.0))
        self.assertEqual(setup.resolve_rope(524288, "yarn", 1.5), ("yarn", 1.5))

    def test_an_explicit_override_survives_a_context_reduction(self):
        # §4: the RAM check reduced the final context to 128K - a user-supplied factor is NOT recalculated
        self.assertEqual(setup.resolve_rope(131072, "yarn", 2.0), ("yarn", 2.0))
        self.assertEqual(setup.resolve_rope(131072, "linear", 1.75), ("linear", 1.75))

    # ---- an explicit none past the trained range is refused with an explanation, not overridden
    def test_explicit_none_past_trained_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            setup.resolve_rope(524288, "none", None)
        self.assertIn("262144", str(cm.exception))
        self.assertIn("yarn", str(cm.exception))                   # the message names the way out

    def test_explicit_none_within_trained_is_the_stock_model(self):
        self.assertEqual(setup.resolve_rope(131072, "none", None), (None, None))

    # ---- inside the trained range nothing turns on by itself; a chosen method gets factor 1
    def test_omitted_within_trained_adds_nothing(self):
        self.assertEqual(setup.resolve_rope(262144, None, None), (None, None))
        self.assertEqual(setup.resolve_rope(131072, None, None), (None, None))

    def test_a_scale_alone_within_trained_is_rejected(self):
        with self.assertRaises(ValueError):
            setup.resolve_rope(131072, None, 2.0)

    def test_a_chosen_method_within_trained_gets_factor_1(self):
        # no more unconditional factor 2: 1 = the trained angles, no automatic expansion
        self.assertEqual(setup.resolve_rope(131072, "yarn", None), ("yarn", 1.0))
        self.assertEqual(setup.resolve_rope(262144, "linear", None), ("linear", 1.0))

    # ---- §4 coordination: the RAM reduction lands BEFORE the rope config, so the factor matches the
    # context actually served - the 512K-on-IQ3_S-with-under-90-GB case from the task, end to end
    def test_512k_on_a_small_ram_machine_resolves_to_128k_factor_1(self):
        # asked 524288 with --rope-scaling yarn and no factor; the RAM check brings the final context to 131072
        self.assertEqual(setup.resolve_rope(131072, "yarn", None), ("yarn", 1.0))

    def test_no_reduction_384k_and_512k_resolve_as_documented(self):
        self.assertEqual(setup.resolve_rope(393216, "yarn", None), ("yarn", 1.5))
        self.assertEqual(setup.resolve_rope(524288, "yarn", None), ("yarn", 2.0))

    # ---- the short-context item's own table: no factor-2 fallback inside the trained window, for either
    # method, with the explicit override column kept verbatim
    def test_the_short_context_table(self):
        cases = [
            (131072, None, "yarn", 1.0),                   # --context 131072 --rope-scaling yarn
            (262144, None, "yarn", 1.0),                   # --context 262144 --rope-scaling yarn
            (393216, None, "yarn", 1.5),
            (524288, None, "yarn", 2.0),
            (131072, 2.0, "yarn", 2.0),                    # --rope-scale 2 is kept inside the window
        ]
        for ctx, scale, method, want in cases:
            self.assertEqual(setup.resolve_rope(ctx, method, scale), (method, want), f"ctx {ctx}")

    def test_linear_gets_the_same_rule_as_yarn(self):
        # the old fallback applied to both methods, so the fix must too
        self.assertEqual(setup.resolve_rope(131072, "linear", None), ("linear", 1.0))
        self.assertEqual(setup.resolve_rope(393216, "linear", None), ("linear", 1.5))


class DynamicInstall(unittest.TestCase):
    """setup end to end (tools/test_setup_golden.py's mocked PC): 512K writes dynamic on one GPU, yarn on two or with
    parallel dialogs; an explicit dynamic there is refused; the config says whose choice it was and keeps it in step."""

    def install(self, argv, profile="64GB-1x32GB", **kw):
        sys.path.insert(0, str(ROOT / "tools"))
        from test_setup_golden import PROFILES, install
        ram, found = PROFILES[profile]
        return install(ram, found, ["--family", "qwen", "--model", "Q2_0", "--no-start", "--context", "524288", *argv],
                       **kw)

    @staticmethod
    def rope(cfg):
        a = cfg["args"]
        return a[a.index("--rope-scaling") + 1], a[a.index("--rope-scale") + 1], cfg.get("rope_auto")

    def test_one_gpu_gets_dynamic(self):
        code, out, cfg, _ = self.install([])
        self.assertEqual(code, 0, out[-2000:])
        self.assertEqual(self.rope(cfg), ("dynamic", "2", True))
        self.assertIn("rope scaling: dynamic", out)

    def test_interactive_default_is_dynamic(self):
        code, out, cfg, asked = self.install([], answers="")
        self.assertEqual(code, 0, out[-2000:])
        self.assertTrue(any("RoPE extension method" in q and "[dynamic]" in q for q in asked), asked)
        self.assertEqual(self.rope(cfg), ("dynamic", "2", True))
        code, out, cfg, _ = self.install([], answers={"RoPE": "yarn"})
        self.assertEqual(self.rope(cfg), ("yarn", "2", False))           # a pick of the user's: kept as it is

    def test_yarn_on_two_gpus_and_with_parallel_dialogs(self):
        for argv, profile, says in ((["--gpus", "0,1"], "32GB-2x24GB", "one GPU"),
                                    (["--parallel", "2"], "64GB-1x32GB", "one dialog at a time")):
            with self.subTest(argv=argv):
                code, out, cfg, _ = self.install(argv, profile)
                self.assertEqual(code, 0, out[-2000:])
                self.assertEqual(self.rope(cfg), ("yarn", "2", True))
                self.assertIn(says, out)
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    code, out, cfg, _ = self.install(argv + ["--rope-scaling", "dynamic"], profile)
                self.assertEqual(code, 1)                                 # fail(): nothing written
                self.assertIn("--rope-scaling dynamic needs", out + err.getvalue())

    def test_kept_in_step_with_the_config(self):
        cfg = {"exe": "x", "args": ["--max-context", "524288", "--rope-scaling", "dynamic", "--rope-scale", "2",
                                    "--kv", "int8"], "rope_auto": True}
        self.assertIsNone(setup.fit_auto_rope(cfg))
        cfg["parallel"] = 2                                                # "parallel" kept from an earlier config
        self.assertIn("one dialog at a time", setup.fit_auto_rope(cfg))
        self.assertEqual(cfg["args"][3], "yarn")
        cfg.pop("parallel")
        self.assertIn("dynamic", setup.fit_auto_rope(cfg))                 # and back
        cfg["gpu"], cfg["rope_auto"] = [0, 1], False                       # the user's own dynamic: refused
        with self.assertRaises(ValueError):
            setup.fit_auto_rope(cfg)
        cfg["args"][3] = "yarn"
        self.assertIsNone(setup.fit_auto_rope(cfg))                        # the user's yarn: left alone

    def test_read_back(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-q2_0.json"
            for auto, method, want in ((False, "dynamic", "dynamic"), (False, "linear", "linear"),
                                       (True, "dynamic", None), (None, "yarn", None)):   # None: an older setup's
                c = {"args": ["--max-context", "524288", "--rope-scaling", method, "--rope-scale", "2"]}
                if auto is not None:
                    c["rope_auto"] = auto
                p.write_text(json.dumps(c))
                self.assertEqual(setup.choices_from_config(p)["rope_scaling"], want, c)


if __name__ == "__main__":
    unittest.main()
