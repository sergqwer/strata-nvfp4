"""A four-part version (0.1.40.2) cannot break setup: versions compare all four numbers (0.1.40 < 0.1.40.2, so a
0.1.40 engine is updated), this fork's 0.1.40.3-nvfp4.1 compares as (0, 1, 40, 3, 1), the engine zips are looked for
under the fork's own v<release> tag first and the latest release second, and a binary that says `engine=0.1.40.2` is
read as (0, 1, 40, 2).

    python -m unittest tools.test_setup_hotfix_tag
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402


class HotfixTag(unittest.TestCase):
    def test_a_four_part_version_compares_all_four_numbers(self):
        ver = tuple(int(x) for x in "0.1.40.3".split("."))   # a hotfix version has four numbers; MIN_ENGINE may have three
        self.assertEqual(len(ver), 4)
        self.assertLess(ver, setup.UPSTREAM_MIN_ENGINE)        # an installed hotfix of the older release is replaced
        self.assertGreaterEqual((0, 1, 42, 1), setup.UPSTREAM_MIN_ENGINE)   # a later hotfix of the minimum release is accepted
        self.assertLess((0, 1, 41, 1), setup.UPSTREAM_MIN_ENGINE)   # an engine from before 0.1.42 (the unreleased 0.1.41.1 numbering) is replaced
        self.assertEqual(setup.version_tuple("0.1.40.2"), (0, 1, 40, 2))
        self.assertLess((0, 1, 40), setup.MIN_ENGINE)       # an installed 0.1.40 engine is replaced
        self.assertEqual(setup.version_tuple(setup.version_text(setup.MIN_ENGINE)), setup.MIN_ENGINE)

    def test_the_fork_s_version_compares_above_its_upstream_one(self):
        # this fork's "0.1.40.3-nvfp4.1" read number by number was 0.1.40: older than MIN_ENGINE, replaced on every run
        self.assertEqual(setup.version_tuple("0.1.40.3-nvfp4.1"), (0, 1, 40, 3, 1))
        self.assertEqual(setup.version_tuple("0.1.40-nvfp4.3"), (0, 1, 40, 0, 3))   # an older fork release
        self.assertGreater(setup.version_tuple("0.1.40.3-nvfp4.1"), (0, 1, 40, 3))
        self.assertGreater(setup.version_tuple("0.1.40.3-nvfp4.2"), setup.version_tuple("0.1.40.3-nvfp4.1"))
        self.assertLess(setup.version_tuple("0.1.40.3-nvfp4.9"), (0, 1, 40, 4))
        self.assertEqual(setup.version_text((0, 1, 40, 3, 1)), "0.1.40.3-nvfp4.1")
        for tag in ("0.1.40-nvfp4.3", "0.1.41-nvfp4.1", "0.1.40.3-nvfp4.1"):   # written back as the tag spells it
            self.assertEqual(setup.version_text(setup.version_tuple(tag)), tag)
        # MIN_ENGINE is the fork's own release (release/make_windows_bundle.py's VERSION), never below upstream's
        self.assertEqual(setup.MIN_ENGINE, max(setup.UPSTREAM_MIN_ENGINE, setup.version_tuple(setup.release_version())))
        # a port to a new upstream release bumps VERSION with it: left behind, every installed fork engine would be
        # older than MIN_ENGINE and its own release "not published yet"
        self.assertGreaterEqual(setup.version_tuple(setup.release_version()), setup.UPSTREAM_MIN_ENGINE)
        self.assertRegex(setup.release_version(), r"^\d+\.\d+\.\d+(\.\d+)?-nvfp4\.\d+$")

    def test_the_engine_zips_are_found_for_a_hotfix_tag(self):
        # the fork's zips are under its release's tag first, the latest release second
        bases = setup.prebuilt_bases(setup.PREBUILT_URL)
        self.assertEqual(bases[0], setup.PREBUILT_TAG_URL.format(version=setup.release_version()))
        self.assertEqual(bases[-1], setup.PREBUILT_URL)
        with mock.patch.object(setup, "release_version", return_value="0.1.40.3-nvfp4.2"):
            self.assertEqual(setup.prebuilt_bases(setup.PREBUILT_URL),
                             ["https://github.com/sergqwer/strata-nvfp4/releases/download/v0.1.40.3-nvfp4.2/",
                              setup.PREBUILT_URL])

    def test_an_engine_that_says_four_parts_is_read_as_four(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "strata.exe"
            exe.write_bytes(b"MZ\x00engine=0.1.40.2\n\x00")
            self.assertEqual(setup.engine_version(exe), (0, 1, 40, 2))
            exe.write_bytes(b"MZ\x00engine=0.1.40\n\x00")
            self.assertEqual(setup.engine_version(exe), (0, 1, 40))


if __name__ == "__main__":
    unittest.main()
