"""tools/bundle_update.py (the release bundle's update.cmd): the newest release is found, checked against GitHub's
SHA-256 and installed over the program by renames (a journal undoes a stopped swap), keeping config/, data/, models/
and the virtual environments; a file of config/ or data/ still as a release shipped it is updated.  Also
release/publish.py's check that the bundle and setup.py's asset hold one engine.  Mocked network and a bundle in a
temp folder: nothing is downloaded.

    python -m unittest tools.test_bundle_update
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bundle_update as U  # noqa: E402

OLD, NEW = "0.1.40.3-nvfp4.1", "0.1.40.3-nvfp4.2"
ZIP_URL = "https://github.com/sergqwer/strata-nvfp4/releases/download/v%s/strata-nvfp4-v%s-windows-x64.zip" % (NEW, NEW)


def write(p: Path, data) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())


OLD_CONFIG = '{"release": 1}'
OLD_PROFILE = b"shipped profile 1"


def bundle(root: Path, manifest=False) -> None:
    """An unpacked older release, used: an edited config, the user's own profile, a model, a venv, a log.  manifest:
    the older release had a release-manifest.json (with its config and profile as shipped)."""
    write(root / "engine" / "BUILD.json", json.dumps({"version": OLD, "fork": "x"}))
    write(root / "tools" / "a.py", "old a")
    write(root / "tools" / "b.py", "old b")
    if manifest:
        write(root / "release-manifest.json", json.dumps({"version": OLD, "files": {
            "config/strata-nvfp4.json": hashlib.sha256(OLD_CONFIG.encode()).hexdigest(),
            "data/expert-profile.bin": hashlib.sha256(OLD_PROFILE).hexdigest()}}))
    write(root / "engine" / "strata.exe", b"old engine")
    write(root / "serve" / "server.py", "old server")
    write(root / "start-server.cmd", "old start")
    write(root / "requirements-serve.txt", "numpy\n")
    write(root / "config" / "strata-nvfp4.json", '{"edited": true}')
    write(root / "data" / "expert-profile.bin", b"my own profile")
    write(root / "models" / "pack" / "experts.bin", b"63 GiB")
    write(root / ".venv-serve" / "pyvenv.cfg", "x")
    write(root / "strata.log", "log")


def release_zip(earlier=None) -> bytes:
    """The new release; earlier: its manifest's "earlier" hashes (the config and profile versions git has)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        if earlier is not None:
            z.writestr("strata-nvfp4/release-manifest.json", json.dumps({"version": NEW, "files": {}, "earlier": earlier}))
        z.writestr("strata-nvfp4/engine/BUILD.json", json.dumps({"version": NEW, "fork": "x"}))
        z.writestr("strata-nvfp4/engine/strata.exe", b"new engine")
        z.writestr("strata-nvfp4/tools/a.py", "new a")
        z.writestr("strata-nvfp4/tools/b.py", "new b")
        z.writestr("strata-nvfp4/serve/server.py", "new server")
        z.writestr("strata-nvfp4/start-server.cmd", "new start")
        z.writestr("strata-nvfp4/update.cmd", "new update")
        z.writestr("strata-nvfp4/requirements-serve.txt", "numpy\n")
        z.writestr("strata-nvfp4/config/strata-nvfp4.json", '{"release": true}')
        z.writestr("strata-nvfp4/data/expert-profile.bin", b"shipped profile")
        z.writestr("strata-nvfp4/data/draft_vocab.bin", b"vocab")
    return buf.getvalue()


class Response(io.BytesIO):
    def __init__(self, body: bytes, status=200):
        super().__init__(body)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def opener(body: bytes, digest="auto", seen=None):
    """GitHub: the release JSON (with the asset's digest) and the zip, honoring Range."""
    def open_(req, timeout=None):
        url = req.full_url
        if seen is not None:
            seen.append((url, req.headers.get("Range")))
        if url.startswith(U.API):
            d = ("sha256:" + hashlib.sha256(body).hexdigest()) if digest == "auto" else digest
            return Response(json.dumps({"tag_name": "v" + NEW, "name": "a fix", "assets": [
                {"name": "strata-nvfp4-v%s-windows-x64.zip" % NEW, "size": len(body), "digest": d,
                 "browser_download_url": ZIP_URL}]}).encode())
        if url == ZIP_URL:
            rng = req.headers.get("Range")
            if rng:
                start = int(rng.split("=")[1].rstrip("-"))
                return Response(body[start:], status=206)
            return Response(body)
        raise AssertionError("unexpected URL " + url)
    return open_


def run(root, argv=(), body=None, **kw):
    out = io.StringIO()
    with redirect_stdout(out), mock.patch.object(U.time, "sleep"):
        code = U.main(list(argv), root=root, opener=opener(body or release_zip(), **kw))
    return code, out.getvalue()


def old_program(test, root):
    for p, text in (("engine/strata.exe", "old engine"), ("serve/server.py", "old server"),
                    ("start-server.cmd", "old start"), ("tools/a.py", "old a"), ("tools/b.py", "old b")):
        test.assertEqual((root / p).read_bytes(), text.encode(), p)
    test.assertFalse((root / "tools" / "tools").exists())               # never moved into a folder still there
    test.assertFalse((root / ".update" / U.JOURNAL).exists())


class Update(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.root = Path(t.name)
        bundle(self.root)

    def test_installs_the_program_and_keeps_yours(self):
        code, out = run(self.root)
        self.assertEqual(code, 0, out)
        r = self.root
        self.assertEqual(json.loads((r / "engine" / "BUILD.json").read_text())["version"], NEW)
        self.assertEqual((r / "engine" / "strata.exe").read_bytes(), b"new engine")
        self.assertEqual((r / "serve" / "server.py").read_text(), "new server")
        self.assertEqual((r / "update.cmd").read_text(), "new update")
        self.assertEqual((r / ".previous" / "engine" / "strata.exe").read_bytes(), b"old engine")   # one generation
        self.assertEqual((r / "config" / "strata-nvfp4.json").read_text(), '{"edited": true}')      # yours stays
        self.assertEqual((r / "config" / "strata-nvfp4.json.new").read_text(), '{"release": true}')
        self.assertEqual((r / "data" / "expert-profile.bin").read_bytes(), b"my own profile")
        self.assertEqual((r / "data" / "draft_vocab.bin").read_bytes(), b"vocab")                 # only what is new
        self.assertEqual((r / "models" / "pack" / "experts.bin").read_bytes(), b"63 GiB")
        self.assertTrue((r / ".venv-serve" / "pyvenv.cfg").exists())
        self.assertTrue((r / "strata.log").exists())
        self.assertFalse((r / ".update" / "unpacked").exists())
        self.assertFalse(any((r / ".update").glob("*.zip")))
        self.assertIn("Checked: the size and SHA-256 GitHub publishes", out)

    def test_check_and_up_to_date_change_nothing(self):
        code, out = run(self.root, ["--check"])
        self.assertEqual(code, 0)
        self.assertIn("Published: %s" % NEW, out)
        self.assertEqual((self.root / "engine" / "strata.exe").read_bytes(), b"old engine")
        (self.root / "engine" / "BUILD.json").write_text(json.dumps({"version": NEW}))
        seen = []
        with redirect_stdout(io.StringIO()):
            self.assertEqual(U.main([], root=self.root, opener=opener(release_zip(), seen=seen)), 0)
        self.assertEqual([u for u, _ in seen], [U.API + "/latest"])          # the zip is not fetched

    def test_a_wrong_or_missing_sha256_changes_nothing(self):
        for digest, why in (("sha256:" + "0" * 64, "wrong SHA-256"), (None, "publishes no SHA-256")):
            with self.subTest(digest=digest):
                code, out = run(self.root, digest=digest)
                self.assertEqual(code, 1)
                self.assertIn(why, out)
                self.assertEqual((self.root / "engine" / "strata.exe").read_bytes(), b"old engine")
                self.assertFalse(any((self.root / ".update").glob("*.zip")))   # deleted: the next run fetches afresh
        with mock.patch.dict(os.environ, {"STRATA_SKIP_SHA256": "1"}):
            code, out = run(self.root, digest=None)
        self.assertEqual(code, 0, out)
        self.assertEqual((self.root / "engine" / "strata.exe").read_bytes(), b"new engine")

    def test_a_stopped_download_resumes(self):
        body = release_zip()
        part = self.root / ".update" / ("strata-nvfp4-v%s-windows-x64.zip.part" % NEW)
        write(part, body[:1000])
        seen = []
        with redirect_stdout(io.StringIO()):
            self.assertEqual(U.main([], root=self.root, opener=opener(body, seen=seen)), 0)
        self.assertIn((ZIP_URL, "bytes=1000-"), seen)
        self.assertEqual((self.root / "engine" / "strata.exe").read_bytes(), b"new engine")

    def test_not_while_the_engine_runs_and_not_outside_a_bundle(self):
        with mock.patch.object(U, "engine_running", lambda root: True):
            code, out = run(self.root)
        self.assertEqual(code, 1)
        self.assertIn("close the server window", out)
        self.assertEqual((self.root / "engine" / "strata.exe").read_bytes(), b"old engine")
        (self.root / "start-server.cmd").unlink()
        code, out = run(self.root)
        self.assertEqual(code, 1)
        self.assertIn("UPDATE.bat", out)

    def test_a_failure_part_way_puts_the_old_program_back(self):
        for n in range(1, 12):                       # the swap stopped at each of its renames in turn
            with self.subTest(failing_rename=n):
                real, calls = U.os.rename, []

                def flaky(src, dst):
                    calls.append(src)
                    if len(calls) == n:
                        raise OSError("in use")
                    return real(src, dst)

                with mock.patch.object(U.os, "rename", flaky), mock.patch.object(U.shutil, "move",
                                                                                 side_effect=AssertionError("copied")):
                    code, out = run(self.root)
                self.assertEqual(code, 1, out)
                self.assertIn("the program in this folder is the one it was", out)
                old_program(self, self.root)
                self.assertFalse((self.root / ".previous").exists())
                self.assertTrue(any((self.root / ".update").glob("*.zip")))   # checked: reused by the next run

    @unittest.skipUnless(os.name == "nt", "Windows: a folder with an open file in it cannot be renamed")
    def test_a_file_in_use_in_the_program(self):
        held = open(self.root / "tools" / "b.py", "rb")               # an editor or a console open in tools
        try:
            code, out = run(self.root)
        finally:
            held.close()
        self.assertEqual(code, 1, out)
        old_program(self, self.root)
        code, out = run(self.root)                                     # closed: the next run installs it
        self.assertEqual(code, 0, out)
        self.assertIn("Using strata-nvfp4-v%s-windows-x64.zip, downloaded and checked before" % NEW, out)
        self.assertEqual((self.root / "tools" / "b.py").read_text(), "new b")

    @unittest.skipUnless(os.name == "nt", "Windows: a folder with an open file in it cannot be renamed")
    def test_an_antivirus_scanning_the_unpacked_release(self):
        held, real = [], zipfile.ZipFile.extractall

        def extractall(zf, path=None, *a, **k):
            real(zf, path, *a, **k)
            held.append(open(Path(path) / "strata-nvfp4" / "tools" / "b.py", "rb"))

        with mock.patch.object(zipfile.ZipFile, "extractall", extractall):
            code, out = run(self.root)
        for h in held:
            h.close()
        self.assertEqual(code, 1, out)
        old_program(self, self.root)

    def test_a_swap_that_cannot_be_undone_is_finished_by_the_next_run(self):
        real = U.os.rename
        state = {"n": 0}

        def stuck(src, dst):                         # the last forward rename fails (update.cmd placed), and so does putting tools back
            state["n"] += 1
            if state["n"] == 11 or (Path(src).name == "tools" and ".previous" not in str(src)
                                   and "unpacked" in str(dst)):
                raise OSError("in use")
            return real(src, dst)

        with mock.patch.object(U.os, "rename", stuck):
            code, out = run(self.root)
        self.assertEqual(code, 1)
        self.assertIn("could not be put back whole (still in use: tools)", out)
        self.assertNotIn("is the one it was", out)
        self.assertTrue((self.root / ".update" / U.JOURNAL).exists())
        self.assertEqual((self.root / ".previous" / "tools" / "a.py").read_text(), "old a")   # the only full copy
        code, out = run(self.root, ["--check"])      # the next run: the old program back first, .previous not deleted
        self.assertIn("putting the old program back first", out)
        old_program(self, self.root)
        self.assertEqual(code, 0, out)
        code, out = run(self.root)
        self.assertEqual(code, 0, out)
        self.assertEqual((self.root / "tools" / "b.py").read_text(), "new b")

    def test_a_previous_that_cannot_be_deleted_changes_nothing(self):
        write(self.root / ".previous" / "engine" / "strata.exe", b"older engine")
        with mock.patch.object(U.shutil, "rmtree", lambda p, ignore_errors=False: None):
            code, out = run(self.root)
        self.assertEqual(code, 1)
        self.assertIn(".previous cannot be deleted", out)
        old_program(self, self.root)
        self.assertEqual((self.root / ".previous" / "engine" / "strata.exe").read_bytes(), b"older engine")

    def test_the_server_or_converter_running_and_checked_again_before_installing(self):
        busy = self.root / ".venv-serve" / "Scripts" / "python.exe"
        write(busy, b"python")
        real = open

        def locked(p, mode="r", *a, **k):
            if Path(p) == busy and "+" in mode:
                raise PermissionError("in use")
            return real(p, mode, *a, **k)

        with mock.patch("builtins.open", locked):
            self.assertEqual(U.engine_running(self.root), str(Path(".venv-serve", "Scripts", "python.exe")))
        answers = iter([None, "engine\\strata.exe"])            # free before the download, started meanwhile
        with mock.patch.object(U, "engine_running", lambda root: next(answers)):
            code, out = run(self.root)
        self.assertEqual(code, 1)
        self.assertIn("was started meanwhile", out)
        old_program(self, self.root)
        self.assertTrue(any((self.root / ".update").glob("*.zip")))   # kept: no second download
        seen = []
        with redirect_stdout(io.StringIO()):
            self.assertEqual(U.main([], root=self.root, opener=opener(release_zip(), seen=seen)), 0)
        self.assertEqual([u for u, _ in seen], [U.API + "/latest"])

    def test_an_unfinished_download_keeps_its_part(self):
        body = release_zip()

        def cut(req, timeout=None):                  # every attempt ends after 100 bytes
            if req.full_url.startswith(U.API):
                return opener(body)(req)
            rng = req.headers.get("Range")
            start = int(rng.split("=")[1].rstrip("-")) if rng else 0
            r = Response(body[start:start + 100], status=206 if rng else 200)
            real_read = r.read

            def read(n=-1):
                b = real_read(n)
                if not b:
                    raise OSError("connection reset")
                return b
            r.read = read
            return r

        out = io.StringIO()
        with redirect_stdout(out), mock.patch.object(U.time, "sleep"):
            code = U.main([], root=self.root, opener=cut)
        self.assertEqual(code, 1)
        self.assertIn("run update.cmd again to go on with the download", out.getvalue())
        part = self.root / ".update" / ("strata-nvfp4-v%s-windows-x64.zip.part" % NEW)
        self.assertEqual(part.stat().st_size, 1000)               # 10 attempts of 100 bytes, kept
        old_program(self, self.root)
        code, out = run(self.root)                                     # resumed from byte 1000
        self.assertEqual(code, 0, out)


class Merge(unittest.TestCase):
    """config/ and data/: a file still as a release shipped it is updated; one the user changed stays (.new beside)."""

    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.root = Path(t.name)

    def test_an_unchanged_file_is_updated_a_changed_one_kept(self):
        for manifest in (True, False):               # the older release had a manifest / only the new one's history
            with self.subTest(manifest=manifest):
                root = self.root / str(manifest)
                bundle(root, manifest=manifest)
                write(root / "config" / "strata-nvfp4.json", OLD_CONFIG)         # as the older release shipped it
                write(root / "data" / "expert-profile.bin", OLD_PROFILE)
                write(root / "config" / "mine.json", "{}")
                earlier = {} if manifest else {
                    "config/strata-nvfp4.json": [hashlib.sha256(OLD_CONFIG.encode()).hexdigest()],
                    "data/expert-profile.bin": [hashlib.sha256(OLD_PROFILE).hexdigest()]}
                code, out = run(root, body=release_zip(earlier))
                self.assertEqual(code, 0, out)
                self.assertEqual((root / "config" / "strata-nvfp4.json").read_text(), '{"release": true}')
                self.assertFalse((root / "config" / "strata-nvfp4.json.new").exists())
                self.assertEqual((root / "data" / "expert-profile.bin").read_bytes(), b"shipped profile")
                self.assertIn("updated data/expert-profile.bin", out)
                self.assertEqual((root / "config" / "mine.json").read_text(), "{}")
        root = self.root / "edited"                  # the user's edits: kept, the release's beside them
        bundle(root, manifest=True)
        code, out = run(root, body=release_zip({}))
        self.assertEqual(code, 0, out)
        self.assertEqual((root / "config" / "strata-nvfp4.json").read_text(), '{"edited": true}')
        self.assertEqual((root / "config" / "strata-nvfp4.json.new").read_text(), '{"release": true}')
        self.assertEqual((root / "data" / "expert-profile.bin").read_bytes(), b"my own profile")
        self.assertEqual((root / "data" / "expert-profile.bin.new").read_bytes(), b"shipped profile")

    def test_the_manifest_and_git_s_versions(self):
        repo = Path(__file__).resolve().parents[1]
        for path in ("data/expert-profile.bin", "release/windows/config/strata-qwen-nvfp4-gptq.json"):
            with self.subTest(path):
                versions = U.git_versions(repo, path)
                if U.subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", path]).returncode:
                    continue                         # changed in this working tree: not a committed version
                self.assertIn(U.sha256(repo / path), versions)
        write(self.root / "top" / "engine" / "strata.exe", b"x")
        write(self.root / "top" / U.MANIFEST, "{}")
        m = U.build_manifest(self.root / "top", NEW, {"data/a": {"b", "a"}})
        self.assertEqual(m, {"version": NEW, "files": {"engine/strata.exe": hashlib.sha256(b"x").hexdigest()},
                             "earlier": {"data/a": ["a", "b"]}})


class Publish(unittest.TestCase):
    """release/publish.py's check_zips: the bundle and setup.py's asset hold one engine folder, byte for byte."""

    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.d = Path(t.name)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "release"))
        import publish
        self.P = publish
        self.build = {"version": NEW, "commit": "abc1234", "dirty": False, "vision": "cpu", "cuda_libs": "bundled",
                      "engine_sha256": hashlib.sha256(b"exe").hexdigest()}
        self.engine = {"BUILD.json": json.dumps(self.build), "strata.exe": b"exe", "strata-vision.exe": b"vis",
                       "cublas64_13.dll": b"c1", "cublasLt64_13.dll": b"c2"}

    def zips(self, setup_engine=None, manifest=True):
        b, s = self.d / "bundle.zip", self.d / "setup.zip"
        rest = {"serve/server.py": b"srv"}
        files = {**{"engine/" + n: d for n, d in self.engine.items()}, **rest}
        with zipfile.ZipFile(b, "w") as z:
            for n, d in files.items():
                z.writestr("strata-nvfp4/" + n, d)
            if manifest:
                z.writestr("strata-nvfp4/release-manifest.json", json.dumps({"version": NEW, "files": {
                    n: hashlib.sha256(d if isinstance(d, bytes) else d.encode()).hexdigest() for n, d in files.items()}}))
        with zipfile.ZipFile(s, "w") as z:
            for n, d in (setup_engine or self.engine).items():
                z.writestr(n, d)
        return b, s

    def check(self, **kw):
        b, s = self.zips(**kw)
        with mock.patch("sys.stderr", io.StringIO()):
            try:
                self.P.check_zips(b, s, NEW, "abc1234def")
            except SystemExit as e:
                return str(e)
        return None

    def test_one_engine_in_both(self):
        self.assertIsNone(self.check())
        no_dll = {k: v for k, v in self.engine.items() if k != "cublasLt64_13.dll"}
        self.assertIn("cublasLt64_13.dll", self.check(setup_engine=no_dll))
        other = {**self.engine, "strata-vision.exe": b"another build"}
        self.assertIn("strata-vision.exe is not the bundle's", self.check(setup_engine=other))
        self.assertIn("no release-manifest.json", self.check(manifest=False))
        self.engine = no_dll                              # missing from both: BUILD.json promises it
        self.assertIn("lacks cublasLt64_13.dll", self.check())

    def test_the_bundle_ships_it(self):
        repo = Path(__file__).resolve().parents[1]
        cmd = (repo / "release" / "windows" / "update.cmd").read_bytes()
        self.assertIn(b"python tools\\bundle_update.py %*", cmd)
        self.assertNotIn(b"\n", cmd.replace(b"\r\n", b""))                 # CRLF, or cmd.exe mangles it
        last = [x for x in cmd.split(b"\r\n") if x.strip()][-1]
        self.assertIn(b"exit /b", last)                                      # replaced while it runs: one line


if __name__ == "__main__":
    unittest.main()
