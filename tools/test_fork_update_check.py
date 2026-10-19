"""strata_control/fork_release.py (this fork): the update notice's version order (the one setup.py uses), GitHub's
latest release asked at most once a day (the answer, a failure too, kept in <install>/.update/update-check.json or the
per-user folder), the installed version (a fork engine's BUILD.json, else the source's release), the folder kind and
its update command, the off switches and the lines it writes.  A mocked opener: nothing reaches the network.

    python -m unittest tools.test_fork_update_check
"""
from __future__ import annotations

import http.client
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from strata_control import fork_release as F  # noqa: E402

DAY = F.EVERY_S
T0 = 1_800_000_000.0
NOTES = "https://github.com/sergqwer/strata-nvfp4/releases/tag/"


def answer(tag="v0.2.0-beta-nvfp4.2", url=None, **extra) -> bytes:
    return json.dumps({"tag_name": tag, "html_url": url or NOTES + tag, "name": tag, "assets": [], **extra}).encode()


class Opener:
    """urlopen's stand-in: each call takes the next of `answers` (bytes, or an exception to raise)."""

    def __init__(self, *answers):
        self.answers, self.requests = list(answers), []

    def __call__(self, req, timeout=None):
        self.requests.append((req, timeout))
        if not self.answers:
            raise AssertionError("GitHub was asked again")
        a = self.answers.pop(0)
        if isinstance(a, BaseException):
            raise a
        return io.BytesIO(a)


def write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def bundle(root: Path, version="0.2.0-beta-nvfp4.1") -> Path:
    write(root / "engine" / "BUILD.json", json.dumps({"version": version, "source": "release",
                                                      "fork": "https://github.com/sergqwer/strata-nvfp4"}))
    write(root / "tools" / "bundle_update.py", "")
    write(root / "start-server.cmd", "")
    return root


def clone(root: Path, version="0.2.0-beta-nvfp4.1", engine=None) -> Path:
    (root / ".git").mkdir(parents=True, exist_ok=True)
    write(root / "release" / "make_windows_bundle.py", 'import os\nVERSION = "%s"\n' % version)
    if engine is not None:
        write(root / "engine" / "BUILD.json", json.dumps(engine))
    return root


class Case(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.root = Path(d.name) / "strata-nvfp4"
        self.root.mkdir()
        self.user = Path(d.name) / "user-state"
        p = mock.patch("strata_control.paths.state_base", return_value=self.user)
        p.start()
        self.addCleanup(p.stop)


class Order(unittest.TestCase):
    def test_setup_and_the_notice_share_one_order(self):
        import setup
        self.assertIs(setup.version_tuple, F.version_tuple)
        self.assertEqual(setup.release_version(), F.release_version(ROOT))
        self.assertEqual(F.version_tuple("0.2.0-beta-nvfp4.1"), (0, 2, 0, 0, 1))
        self.assertEqual(F.version_tuple("v0.2.0"), ())                     # a tag's "v" is stripped by the caller

    def test_newer_same_older(self):
        self.assertTrue(F.is_newer("0.2.0-beta-nvfp4.2", "0.2.0-beta-nvfp4.1"))
        self.assertFalse(F.is_newer("0.2.0-beta-nvfp4.1", "0.2.0-beta-nvfp4.1"))
        self.assertFalse(F.is_newer("0.2.0-beta-nvfp4.0", "0.2.0-beta-nvfp4.1"))
        self.assertTrue(F.is_newer("0.2.0-beta-nvfp4.1", "0.1.44.1-nvfp4.2"))
        self.assertFalse(F.is_newer("0.1.44.1-nvfp4.2", "0.2.0-beta-nvfp4.1"))
        self.assertTrue(F.is_newer("0.2.1-nvfp4.1", "0.2.0.9-nvfp4.9"))
        self.assertFalse(F.is_newer("garbage", "0.2.0-beta-nvfp4.1"))
        self.assertFalse(F.is_newer("0.2.0-beta-nvfp4.2", None))

    def test_a_final_release_comes_after_its_pre_release(self):
        # version_tuple skips the word: equal numbers, so the notice breaks the tie (final after beta, never back)
        self.assertEqual(F.version_tuple("0.2.0-nvfp4.1"), F.version_tuple("0.2.0-beta-nvfp4.1"))
        self.assertTrue(F.is_newer("0.2.0-nvfp4.1", "0.2.0-beta-nvfp4.1"))
        self.assertFalse(F.is_newer("0.2.0-beta-nvfp4.1", "0.2.0-nvfp4.1"))
        self.assertFalse(F.is_newer("0.2.0-rc1-nvfp4.1", "0.2.0-beta-nvfp4.1"))
        self.assertTrue(F.is_newer("0.2.0-beta-nvfp4.2", "0.2.0-nvfp4.1"))           # the numbers decide first


class Query(unittest.TestCase):
    def test_one_plain_request(self):
        op = Opener(answer())
        self.assertEqual(F.query_latest(op), {"tag": "v0.2.0-beta-nvfp4.2", "version": "0.2.0-beta-nvfp4.2",
                                              "url": NOTES + "v0.2.0-beta-nvfp4.2"})
        (req, timeout), = op.requests
        self.assertEqual((req.full_url, req.get_method(), timeout), (F.LATEST_API, "GET", 5.0))
        self.assertEqual(req.get_header("User-agent"), F.UA)
        self.assertEqual(sorted(h.lower() for h in req.headers), ["accept", "user-agent"])   # no token, no id
        self.assertIsNone(req.data)

    def test_what_github_cannot_say(self):
        cases = [urllib.error.URLError("offline"), TimeoutError("timed out"),
                 urllib.error.HTTPError(F.LATEST_API, 403, "rate limit exceeded", {}, None),
                 http.client.IncompleteRead(b"{"), b"<html>not json", b"[1, 2]", answer(tag="latest"),
                 answer(tag="v0.2.0-beta-nvfp4.2<script>")]
        for c in cases:
            with self.subTest(c=c), self.assertRaises((OSError, ValueError)):
                F.query_latest(Opener(c))
        with self.assertRaisesRegex(OSError, "HTTP 403"):
            F.query_latest(Opener(urllib.error.HTTPError(F.LATEST_API, 403, "rate limit", {}, None)))

    def test_a_link_elsewhere_becomes_the_release_page(self):
        rel = F.query_latest(Opener(answer(url="https://example.com/x")))
        self.assertEqual(rel["url"], NOTES + "v0.2.0-beta-nvfp4.2")


class Cache(Case):
    def test_asked_once_a_day(self):
        op = Opener(answer(), answer("v0.2.0-beta-nvfp4.3"))
        rel, source, err = F.latest(self.root, now=T0, opener=op)
        self.assertEqual((rel["tag"], source, err), ("v0.2.0-beta-nvfp4.2", "github", None))
        kept = json.loads((self.root / ".update" / F.CACHE).read_text(encoding="utf-8"))
        self.assertEqual((kept["checked_at"], kept["latest"]["tag"]), (T0, "v0.2.0-beta-nvfp4.2"))
        for later in (1, DAY / 2, DAY - 1):
            self.assertEqual(F.latest(self.root, now=T0 + later, opener=op)[:2], (rel, "cache"))
        self.assertEqual(len(op.requests), 1)
        rel, source, _ = F.latest(self.root, now=T0 + DAY + 1, opener=op)
        self.assertEqual((rel["tag"], source, len(op.requests)), ("v0.2.0-beta-nvfp4.3", "github", 2))

    def test_a_clock_set_back_asks_again(self):
        op = Opener(answer(), answer())
        F.latest(self.root, now=T0, opener=op)
        F.latest(self.root, now=T0 - 60, opener=op)
        self.assertEqual(len(op.requests), 2)

    def test_a_failure_counts_and_keeps_the_last_answer(self):
        op = Opener(answer(), urllib.error.HTTPError(F.LATEST_API, 403, "rate limit", {}, None))
        F.latest(self.root, now=T0, opener=op)
        rel, source, err = F.latest(self.root, now=T0 + DAY + 1, opener=op)
        self.assertEqual((rel["tag"], source), ("v0.2.0-beta-nvfp4.2", "github"))
        self.assertIn("HTTP 403", err)
        rel, source, err = F.latest(self.root, now=T0 + DAY + 2, opener=op)     # not asked again for a day
        self.assertEqual((rel["tag"], source), ("v0.2.0-beta-nvfp4.2", "cache"))
        self.assertIn("HTTP 403", err)

    def test_offline_from_the_start(self):
        op = Opener(urllib.error.URLError("no route"))
        self.assertEqual(F.latest(self.root, now=T0, opener=op)[:2], (None, "github"))
        self.assertEqual(F.latest(self.root, now=T0 + 5, opener=op)[:2], (None, "cache"))
        self.assertEqual(len(op.requests), 1)

    def test_a_read_only_folder_keeps_it_per_user(self):
        write(self.root / ".update", "a file where the folder would go")
        op = Opener(answer())
        F.latest(self.root, now=T0, opener=op)
        self.assertTrue((self.user / F.USER_CACHE).is_file())
        self.assertEqual(F.latest(self.root, now=T0 + 1, opener=op)[1], "cache")

    def test_a_damaged_or_edited_cache_is_asked_again_or_cleaned(self):
        write(self.root / ".update" / F.CACHE, "{not json")
        op = Opener(answer())
        self.assertEqual(F.latest(self.root, now=T0, opener=op)[1], "github")
        write(self.root / ".update" / F.CACHE, json.dumps({"checked_at": T0, "latest": {"tag": "v1<b>", "url": "x"}}))
        self.assertEqual(F.latest(self.root, now=T0 + 1, opener=Opener())[:2], (None, "cache"))
        write(self.root / ".update" / F.CACHE, json.dumps({"checked_at": T0, "latest": {
            "tag": "v0.2.0-beta-nvfp4.2", "url": "javascript:alert(1)"}}))
        self.assertEqual(F.latest(self.root, now=T0 + 1, opener=Opener())[0]["url"], NOTES + "v0.2.0-beta-nvfp4.2")

    def test_beside_bundle_update_s_files(self):
        import bundle_update
        self.assertNotIn(F.CACHE, (bundle_update.JOURNAL, bundle_update.MANIFEST))
        self.assertFalse(F.CACHE.startswith("unpacked"))             # unpack() deletes .update/unpacked*
        self.assertFalse(F.CACHE.endswith((".zip", ".part")))        # the download and its partial file
        bundle(self.root)
        F.latest(self.root, now=T0, opener=Opener(answer()))
        self.assertEqual(bundle_update.load_json(self.root / ".update" / bundle_update.JOURNAL), {})


class Notice(Case):
    def test_a_release_folder(self):
        bundle(self.root)
        self.assertEqual(F.install_kind(self.root), "bundle")
        n = F.check(self.root, now=T0, opener=Opener(answer()))
        self.assertEqual((n["status"], n["installed"], n["installed_from"], n["latest"], n["kind"]),
                         ("available", "0.2.0-beta-nvfp4.1", "engine BUILD.json", "0.2.0-beta-nvfp4.2", "bundle"))
        line = F.log_line(n)
        self.assertEqual(line, "[strata] a newer strata-nvfp4 release is published: v0.2.0-beta-nvfp4.2 (this is "
                               "0.2.0-beta-nvfp4.1) - close the server and run update.cmd in %s; notes: %s"
                         % (self.root, NOTES + "v0.2.0-beta-nvfp4.2"))
        self.assertEqual(F.web_view(n), {"latest": "0.2.0-beta-nvfp4.2", "tag": "v0.2.0-beta-nvfp4.2",
                                         "url": NOTES + "v0.2.0-beta-nvfp4.2", "installed": "0.2.0-beta-nvfp4.1",
                                         "kind": "bundle", "command": n["command"]})

    def test_up_to_date_and_older_say_nothing(self):
        bundle(self.root)
        for tag in ("v0.2.0-beta-nvfp4.1", "v0.2.0-beta-nvfp4.0", "v0.1.44.1-nvfp4.2"):
            with self.subTest(tag=tag):
                (self.root / ".update" / F.CACHE).unlink(missing_ok=True)
                n = F.check(self.root, now=T0, opener=Opener(answer(tag)))
                self.assertEqual(n["status"], "current")
                self.assertIsNone(F.log_line(n))
                self.assertIsNone(F.web_view(n))

    def test_the_engine_beside_the_exe_counts(self):
        bundle(self.root)
        other = self.root / "elsewhere" / "strata.exe"
        write(other.parent / "BUILD.json", json.dumps({"version": "0.2.0-beta-nvfp4.0", "fork": "x"}))
        self.assertEqual(F.installed_version(self.root, other), ("0.2.0-beta-nvfp4.0", "engine BUILD.json"))

    def test_a_clone_with_an_engine_compiled_here(self):
        # a local build's BUILD.json says CMake's upstream version (0.2.0): the source's release counts instead
        clone(self.root, engine={"source": "local", "version": "0.2.0"})
        self.assertEqual(F.install_kind(self.root), "clone")
        self.assertEqual(F.installed_version(self.root), ("0.2.0-beta-nvfp4.1", "release/make_windows_bundle.py"))
        n = F.check(self.root, now=T0, opener=Opener(answer()))
        self.assertEqual(n["status"], "available")
        self.assertIn("UPDATE.bat" if os.name == "nt" else "./update.sh", F.log_line(n))
        self.assertIn("`strata update`", F.log_line(n))
        clone(self.root, engine={"source": "release", "version": "0.2.0-beta-nvfp4.2", "fork": "x"})
        self.assertEqual(F.installed_version(self.root), ("0.2.0-beta-nvfp4.2", "engine BUILD.json"))

    def test_without_a_version_github_is_not_asked(self):
        write(self.root / "engine" / "BUILD.json", json.dumps({"version": "0.2.0", "source": "prebuilt"}))
        n = F.check(self.root, now=T0, opener=Opener())
        self.assertEqual((n["status"], n["installed"], n["source"]), ("unknown", None, None))
        self.assertIsNone(F.log_line(n))
        self.assertFalse((self.root / ".update").exists())

    def test_a_failed_ask_is_one_quiet_line_once(self):
        bundle(self.root)
        n = F.check(self.root, now=T0, opener=Opener(urllib.error.URLError("offline")))
        self.assertEqual(n["status"], "unknown")
        self.assertRegex(F.log_line(n), r"^\[strata\] update check: could not ask GitHub \(.*offline.*\); asked again "
                                        r"in a day \(STRATA_UPDATE_CHECK=0 turns it off\)$")
        self.assertIsNone(F.log_line(F.check(self.root, now=T0 + 60, opener=Opener())))   # from the cache: silent

    def test_a_source_copy_without_git(self):
        write(self.root / "release" / "make_windows_bundle.py", 'VERSION = "0.2.0-beta-nvfp4.1"\n')
        write(self.root / "engine" / "BUILD.json", json.dumps({"source": "release", "version": "0.2.0-beta-nvfp4.1",
                                                               "fork": "x"}))
        write(self.root / "tools" / "bundle_update.py", "")           # a source copy has it, but no start-server.cmd
        self.assertEqual(F.install_kind(self.root), "source")


class Switches(unittest.TestCase):
    def test_off(self):
        for env in ("0", "false", "No", " off "):
            self.assertIn("STRATA_UPDATE_CHECK", F.disabled({}, {"STRATA_UPDATE_CHECK": env}))
        self.assertIn("update_check", F.disabled({"update_check": False}, {}))
        self.assertEqual(F.disabled({}, {}, offline=True), "--offline")

    def test_on(self):
        for env in ({}, {"STRATA_UPDATE_CHECK": "1"}, {"STRATA_UPDATE_CHECK": ""}):
            self.assertIsNone(F.disabled({"update_check": True}, env))
            self.assertIsNone(F.disabled(None, env))


if __name__ == "__main__":
    unittest.main()
