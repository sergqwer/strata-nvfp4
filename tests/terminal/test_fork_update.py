"""strata-nvfp4: `strata update` / `strata components update` without a named feed run this fork's own updater for the
folder strata.py is in (strata_cli/fork_update.py): tools/bundle_update.py in a release folder, UPDATE.bat / update.sh
in a clone.  Flags that only mean something for upstream's signed feed are refused (exit 2) naming --feed; a named
feed (--feed, STRATA_UPDATE_FEED, one chosen before) still goes to upstream's updater; rollback is upstream's.
The updater, git and GitHub are stand-ins: nothing runs, nothing reaches the network.

    python -m unittest tests.terminal.test_fork_update -v
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from strata_cli import fork_update, update_cmd
from strata_control import fork_release

WINDOWS = os.name == "nt"
LATEST = {"tag": "v0.2.0-beta-nvfp4.2", "version": "0.2.0-beta-nvfp4.2",
          "url": "https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.2.0-beta-nvfp4.2"}


def write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


class Runner:
    """subprocess.run's stand-in: records each command; git answers from `git`, anything else with `rc`."""

    def __init__(self, rc=0, git=None, out="", effect=None):
        self.calls, self.rc, self.git, self.out, self.effect = [], rc, git or {}, out, effect

    def __call__(self, cmd, **kw):
        self.calls.append((list(cmd), kw))
        if cmd[0] == "git":
            rc, out = self.git.get(cmd[3], (0, ""))
            return SimpleNamespace(returncode=rc, stdout=out, stderr="")
        if self.effect:
            self.effect()
        return SimpleNamespace(returncode=self.rc, stdout=self.out, stderr="")


class ForkUpdate(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.root = Path(d.name) / "strata-nvfp4"
        self.components = Path(d.name) / "components"
        for p in (mock.patch.dict(os.environ, {"STRATA_COMPONENTS_DIR": str(self.components)}),
                  mock.patch.object(fork_update, "ROOT", self.root)):
            p.start()
            self.addCleanup(p.stop)
        for name in ("STRATA_UPDATE_FEED", "STRATA_NO_ACQUIRE", "STRATA_UPDATE_TRUSTED_KEYS"):
            os.environ.pop(name, None)
        self.runner = Runner()
        self.asked = []

        def query(timeout=5.0):
            self.asked.append(timeout)
            if isinstance(self.latest, BaseException):
                raise self.latest
            return self.latest
        self.latest = LATEST
        for p in (mock.patch.object(fork_update.subprocess, "run", self.runner),
                  mock.patch.object(fork_release, "query_latest", query)):
            p.start()
            self.addCleanup(p.stop)

    def bundle(self, version="0.2.0-beta-nvfp4.1"):
        write(self.root / "engine" / "BUILD.json", json.dumps({"version": version, "source": "release", "fork": "x"}))
        write(self.root / "tools" / "bundle_update.py", "")
        write(self.root / "start-server.cmd", "")

    def clone(self, version="0.2.0-beta-nvfp4.1"):
        (self.root / ".git").mkdir(parents=True)
        write(self.root / "release" / "make_windows_bundle.py", 'VERSION = "%s"\n' % version)

    def cli(self, *args):
        from strata_cli.cli import main
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(args))
        return code, out.getvalue(), err.getvalue()

    # ---------------------------------------------------------------- routing
    def test_a_release_folder_runs_bundle_update(self):
        self.bundle()
        bu = str(self.root / "tools" / "bundle_update.py")
        for args, tail in ((["update"], []), (["update", "--force"], ["--force"]), (["components", "update"], []),
                           (["update", "--check"], ["--check"])):
            with self.subTest(args=args):
                self.runner.calls.clear()
                code, out, err = self.cli(*args)
                self.assertEqual(code, 0, out + err)
                (cmd, kw), = self.runner.calls
                self.assertEqual(cmd, [sys.executable, bu, *tail])
                self.assertEqual(kw["cwd"], str(self.root))
                self.assertIn("tools/bundle_update.py", out)
        self.assertEqual(self.asked, [])                 # bundle_update asks GitHub itself

    def test_a_clone_runs_update_bat_or_update_sh(self):
        self.clone()
        for args in (["update"], ["components", "update"]):
            with self.subTest(args=args):
                self.runner.calls.clear()
                code, out, err = self.cli(*args)
                self.assertEqual(code, 0, out + err)
                (cmd, kw), = self.runner.calls
                if WINDOWS:
                    self.assertEqual((cmd, kw["stdin"]), (["cmd", "/c", "UPDATE.bat"], subprocess.DEVNULL))
                else:
                    self.assertEqual(cmd, ["sh", "./update.sh"])
                self.assertEqual(kw["cwd"], str(self.root))
                self.assertIn("git pull, then setup.py --update", out)
        self.runner.rc = 1
        self.assertEqual(self.cli("update")[0], 1)       # the updater's exit code

    def test_a_named_feed_still_goes_to_upstream(self):
        self.clone()
        feed = self.root.parent / "feed"
        feed.mkdir()
        with mock.patch.object(fork_update, "run", side_effect=AssertionError("fork path")):
            code, _, err = self.cli("update", "--check", "--feed", str(feed))
            self.assertNotIn(code, (0, 2))
            self.assertNotIn("fork path", err)
            with mock.patch.dict(os.environ, {"STRATA_UPDATE_FEED": str(feed)}):
                self.assertNotIn(self.cli("update", "--check")[0], (0, 2))
            write(self.components / ".updates" / "state.json", json.dumps({"feed": {"source": str(feed)}}))
            self.assertNotIn(self.cli("update", "--check")[0], (0, 2))
            code, _, err = self.cli("components", "rollback")          # rollback is upstream's, feed or not
            self.assertNotIn("fork path", err)
        ours = [c for c, _ in self.runner.calls if "UPDATE.bat" in c or "./update.sh" in c]
        self.assertEqual(ours, [])

    def test_the_default_feed_remembered_is_no_named_feed(self):
        self.clone()
        write(self.components / ".updates" / "state.json", json.dumps({"feed": {"source": update_cmd.DEFAULT_FEED}}))
        code, out, _ = self.cli("update")
        self.assertEqual(code, 0)
        self.assertEqual(len(self.runner.calls), 1)

    # ---------------------------------------------------------------- refused
    def test_flags_of_the_feed_are_refused_naming_it(self):
        self.bundle()
        for flag in (["--app"], ["--app-version", "1.0"], ["--no-engine"], ["--no-control"], ["--backend", "cuda"],
                     ["--components-dir", str(self.components)], ["--model-config", "x.json"]):
            with self.subTest(flag=flag):
                code, _, err = self.cli("update", *flag)
                self.assertEqual(code, 2)
                self.assertIn(flag[0], err)
                self.assertIn("--feed", err)
        code, _, err = self.cli("update", "--wait", "60")
        self.assertEqual(code, 2)
        self.assertIn("does not wait", err)
        self.assertEqual(self.runner.calls, [])

    def test_force_in_a_clone_is_refused(self):
        self.clone()
        code, _, err = self.cli("update", "--force")
        self.assertEqual(code, 2)
        self.assertIn("nothing to force", err)
        self.assertEqual(self.runner.calls, [])

    def test_acquiring_forbidden_is_exit_5(self):
        self.bundle()
        with mock.patch.dict(os.environ, {"STRATA_NO_ACQUIRE": "1"}):
            code, _, err = self.cli("update")
        self.assertEqual(code, 5)
        self.assertIn("STRATA_NO_ACQUIRE", err)
        self.assertEqual(self.runner.calls, [])

    # ---------------------------------------------------------------- --check and --output json
    def clone_git(self, behind=0):
        self.runner.git = {"symbolic-ref": (0, "main\n"), "rev-parse": (0, "origin/main\n"), "fetch": (0, ""),
                           "rev-list": (0, "0\t%d\n" % behind)}

    def test_check_in_a_clone_changes_nothing(self):
        self.clone()
        self.clone_git(behind=0)
        with mock.patch.object(fork_update.shutil, "which", return_value="git"):
            code, out, _ = self.cli("update", "--check")
            self.assertEqual(code, 0)
            self.assertIn("v0.2.0-beta-nvfp4.2 is the newest published", out)
            self.assertIn("A newer release is published (v0.2.0-beta-nvfp4.2, this is 0.2.0-beta-nvfp4.1)", out)
            self.assertIn("Nothing was changed", out)
            git = [c[1] for c, _ in self.runner.calls]
            self.assertEqual([c[0] for c, _ in self.runner.calls], ["git"] * 4)
            self.assertEqual(git, ["-C"] * 4)
            self.assertEqual([c[3] for c, _ in self.runner.calls], ["symbolic-ref", "rev-parse", "fetch", "rev-list"])
            self.latest = dict(LATEST, tag="v0.2.0-beta-nvfp4.1", version="0.2.0-beta-nvfp4.1")
            self.runner.calls.clear()
            code, out, _ = self.cli("update", "--check", "--output", "json")
            r = json.loads(out)
            self.assertEqual((code, r["status"], r["installed"], r["latest"]["tag"], r["git"]["behind"]),
                             (0, "current", "0.2.0-beta-nvfp4.1", "v0.2.0-beta-nvfp4.1", 0))
            self.clone_git(behind=3)
            r = json.loads(self.cli("update", "--check", "--output", "json")[1])
            self.assertEqual((r["status"], r["git"]["behind"]), ("available", 3))   # new commits, same release

    def test_check_json_in_a_release_folder(self):
        self.bundle()
        code, out, _ = self.cli("update", "--check", "--output", "json")
        r = json.loads(out)
        self.assertEqual((code, r["ok"], r["kind"], r["status"], r["installed"], r["latest"]["version"]),
                         (0, True, "bundle", "available", "0.2.0-beta-nvfp4.1", "0.2.0-beta-nvfp4.2"))
        self.assertEqual(self.runner.calls, [])
        self.latest = OSError("GitHub answered HTTP 403")
        code, out, _ = self.cli("update", "--check", "--json")
        r = json.loads(out)
        self.assertEqual((code, r["ok"], r["status"], r["error"]), (6, False, "unknown", "GitHub answered HTTP 403"))

    def test_update_json_says_what_changed(self):
        self.bundle()
        self.runner.out = "Installed: 0.2.0-beta-nvfp4.1. Published: 0.2.0-beta-nvfp4.2\n"
        self.runner.effect = lambda: self.bundle("0.2.0-beta-nvfp4.2")
        code, out, err = self.cli("update", "--output", "json")
        r = json.loads(out)
        self.assertEqual((code, r["ok"], r["status"], r["installed_before"], r["installed"], r["latest"]["tag"]),
                         (0, True, "updated", "0.2.0-beta-nvfp4.1", "0.2.0-beta-nvfp4.2", "v0.2.0-beta-nvfp4.2"))
        self.assertIn("Published: 0.2.0-beta-nvfp4.2", err)        # the updater's own lines: stderr
        self.runner.rc, self.runner.effect = 1, None
        r = json.loads(self.cli("update", "--output", "jsonl")[1])
        self.assertEqual((r["ok"], r["status"], r["exit_code"]), (False, "failed", 1))


if __name__ == "__main__":
    unittest.main()
