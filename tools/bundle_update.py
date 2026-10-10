"""tools/bundle_update.py - update an unpacked release bundle (strata-nvfp4) in place to this fork's newest release.

    update.cmd                         (the bundle's root: runs this with the Python on PATH)
    python tools\\bundle_update.py [--check] [--force] [--tag vX]

1. Asks GitHub for sergqwer/strata-nvfp4's latest release (or --tag) and finds its bundle zip,
   strata-nvfp4-v<version>-windows-x64.zip, with the size and SHA-256 GitHub publishes for it.
2. Stops when engine/BUILD.json already is that version (--force installs it again); --check only says.
3. Refuses while the engine or the server runs (Windows keeps their programs locked): close the server first.
4. Downloads the zip into .update/ (a stopped download resumes; a zip already downloaded and checked is reused) and
   checks its size and SHA-256; a wrong file is deleted and nothing is changed.  No published SHA-256: it stops
   (STRATA_SKIP_SHA256=1 installs it unchecked).
5. Replaces the program - engine/, serve/, tools/, third_party/, docs/ and the top-level files - with the release's,
   keeping the replaced one in .previous/ (one generation).  Each item is renamed, never copied, and every step is
   written to .update/journal.json first: a failure puts the old program back, and one that cannot be put back (a
   file in use) is finished by the next run before anything else.
6. config/ and data/: a file still as an earlier release shipped it (release-manifest.json's hashes) is replaced by
   the release's; one you changed stays, with the release's beside it as <name>.new; a new one is added.  models/,
   the .venv-* folders, logs and anything else of yours stay.  A .venv-* whose requirements file changed gets pip.
Standard library only (the bundle's Python may have no packages yet).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

SLUG = "sergqwer/strata-nvfp4"
API = "https://api.github.com/repos/%s/releases" % SLUG
TOP = "strata-nvfp4"                       # the zip's top-level folder (release/make_windows_bundle.py)
KEEP = {"config", "data", "models"}        # the user's: never replaced
MERGE = ("config", "data")                 # the release's files merged into the user's
MANIFEST = "release-manifest.json"         # every shipped file's SHA-256 (release/make_windows_bundle.py)
JOURNAL = "journal.json"                   # in .update/: the swap's steps, so a stopped one can be undone
UA = {"User-Agent": "strata-bundle-update"}


class UpdateError(RuntimeError):
    """The update stopped; the message says in what state the folder is."""


def say(msg=""):
    print(msg, flush=True)


def release(tag=None, opener=urllib.request.urlopen) -> dict:
    """The release's JSON: the latest, or `tag`."""
    url = API + ("/tags/" + tag if tag else "/latest")
    with opener(urllib.request.Request(url, headers={**UA, "Accept": "application/vnd.github+json"}), timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def bundle_asset(rel: dict) -> dict | None:
    """The bundle zip of a release: {"name", "url", "size", "sha256" or None}."""
    want = "%s-%s-windows-x64.zip" % (TOP, rel.get("tag_name", ""))
    for a in rel.get("assets") or []:
        if a.get("name") == want:
            d = str(a.get("digest") or "")
            return {"name": want, "url": a.get("browser_download_url"), "size": int(a.get("size") or 0),
                    "sha256": d.split(":", 1)[1] if d.startswith("sha256:") else None}
    return None


def installed_version(root: Path) -> str | None:
    try:
        return json.loads((root / "engine" / "BUILD.json").read_text(encoding="utf-8")).get("version")
    except (OSError, ValueError):
        return None


def download(url: str, dst: Path, size: int, opener=urllib.request.urlopen, tries=10, pause=10) -> None:
    """Resumable: a .part file is continued with a Range request.  It becomes `dst` only when whole (the published
    size); a download that cannot be finished keeps its .part for the next run and raises OSError."""
    part = dst.with_name(dst.name + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(tries):
        have = part.stat().st_size if part.exists() else 0
        if size and have >= size:
            break
        try:
            req = urllib.request.Request(url, headers={**UA, **({"Range": "bytes=%d-" % have} if have else {})})
            with opener(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
                if have and getattr(r, "status", 206) != 206:      # the range was ignored: from the start
                    f.seek(0)
                    f.truncate()
                last = 0.0
                while True:
                    b = r.read(8 << 20)
                    if not b:
                        break
                    f.write(b)
                    if time.time() - last > 2:
                        last = time.time()
                        print("\r  %s: %.0f MB" % (dst.name, f.tell() / 1e6), end="", flush=True)
            print()
            if not size:
                break
        except OSError as e:
            say("  download interrupted (%s)%s" % (e, "; retrying in %d s ..." % pause if attempt + 1 < tries else ""))
            if attempt + 1 < tries:
                time.sleep(pause)
    have = part.stat().st_size if part.exists() else 0
    if size and have < size:
        raise OSError("%s: %d of %d bytes downloaded (kept: the next run resumes it)" % (dst.name, have, size))
    part.replace(dst)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check(z: Path, size: int, sha: str | None) -> str | None:
    """Why the downloaded zip is not the published one, or None."""
    if size and z.stat().st_size != size:
        return "%s is %d bytes, GitHub says %d" % (z.name, z.stat().st_size, size)
    if sha is None:
        if os.environ.get("STRATA_SKIP_SHA256") == "1":
            say("  [!] STRATA_SKIP_SHA256=1: %s is NOT checked against a SHA-256" % z.name)
            return None
        return "GitHub publishes no SHA-256 for %s (STRATA_SKIP_SHA256=1 installs it unchecked)" % z.name
    got = sha256(z)
    return None if got == sha else "%s has the wrong SHA-256 (%s, expected %s)" % (z.name, got, sha)


# programs that run from this folder: Windows keeps a running program's file locked (it cannot be opened for writing)
RUNNING = (("engine", "strata.exe"), ("engine", "strata-vision.exe"),
           (".venv-serve", "Scripts", "python.exe"), (".venv-serve", "Scripts", "pythonw.exe"),
           (".venv-convert", "Scripts", "python.exe"), (".venv-convert", "Scripts", "pythonw.exe"))


def engine_running(root: Path) -> str | None:
    """The program of this folder that is running (the engine, the image encoder, the server's or the converter's
    Python), or None.  This script's own Python is not counted."""
    me = os.path.normcase(os.path.abspath(sys.executable))
    for parts in RUNNING:
        p = root.joinpath(*parts)
        if p.exists() and os.path.normcase(os.path.abspath(p)) != me:
            try:
                with open(p, "r+b"):
                    pass
            except OSError:
                return str(Path(*parts))
    return None


def same(a: Path, b: Path) -> bool:
    return a.stat().st_size == b.stat().st_size and a.read_bytes() == b.read_bytes()


def load_json(p: Path) -> dict:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_journal(path: Path, journal: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(journal, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def undo(root: Path, journal: dict) -> list[str]:
    """Puts the old program back from a journal, last step first: a placed item goes back to the unpacked release, a
    moved one returns from .previous - only into a free place, never into a folder that is still there.  What could
    not be put back, as names (empty: the folder holds the old program again)."""
    new, prev = Path(journal["new"]), root / ".previous"
    stuck = []
    try:
        new.mkdir(parents=True, exist_ok=True)          # where a placed item goes back to (deleted by hand: again)
    except OSError:
        pass
    for op, name in reversed(journal["steps"]):
        dst = root / name
        try:
            if op == "placed":
                if (dst.exists() or dst.is_symlink()) and not (new / name).exists():
                    os.rename(dst, new / name)
            elif name not in stuck and (prev / name).exists() and not (dst.exists() or dst.is_symlink()):
                os.rename(prev / name, dst)
        except OSError:
            pass
        if op == "moved" and (prev / name).exists() and name not in stuck:
            stuck.append(name)                          # still in .previous: its place is taken or it is in use
        elif op == "placed" and dst.exists() and not (new / name).exists() and name not in stuck:
            stuck.append(name)                          # the release's is still in its place
    return stuck


def finish_undo(root: Path) -> None:
    """A swap a previous run could not finish: undone now, before anything else (UpdateError when it still cannot)."""
    jp = root / ".update" / JOURNAL
    journal = load_json(jp)
    if not journal:
        return
    say("A previous update stopped half way: putting the old program back first ...")
    stuck = undo(root, journal)
    if stuck:
        raise UpdateError("still in use: %s. The old program is in .previous; close every window and program using "
                          "files in this folder (an Explorer window, an editor, the antivirus scan) and run update.cmd "
                          "again - nothing else is changed until this is done" % ", ".join(stuck))
    jp.unlink()
    remove_empty(root / ".previous")
    say("  done: the folder holds the program it had before that update.")


def remove_empty(d: Path) -> None:
    try:
        d.rmdir()
    except OSError:
        pass


def unpack(z: Path, upd: Path) -> Path:
    """The zip into a new folder of .update/ (one an earlier run could not delete is left alone)."""
    for old in upd.glob("unpacked*"):
        shutil.rmtree(old, ignore_errors=True)
    tmp = upd / ("unpacked-%d" % time.time_ns())
    with zipfile.ZipFile(z) as f:
        f.extractall(tmp)
    new = tmp / TOP
    if not (new / "engine" / "strata.exe").exists():
        shutil.rmtree(tmp, ignore_errors=True)
        raise UpdateError("%s has no %s/engine/strata.exe; nothing was changed" % (z.name, TOP))
    return new


def build_manifest(top: Path, version: str, earlier: dict) -> dict:
    """release-manifest.json (release/make_windows_bundle.py): every file of the bundle folder `top` with its SHA-256,
    and `earlier`: {config/ or data/ path: the SHA-256s of every version of it a release may have shipped}."""
    files = {p.relative_to(top).as_posix(): sha256(p) for p in sorted(top.rglob("*"))
             if p.is_file() and p.name != MANIFEST}
    return {"version": version, "files": files, "earlier": {k: sorted(v) for k, v in earlier.items()}}


def git_versions(repo: Path, path: str) -> set:
    """The SHA-256 of every committed version of `path` in `repo`'s history; a text file both with LF and with CRLF
    line ends (a Windows checkout ships it with CRLF)."""
    log = subprocess.run(["git", "-C", str(repo), "log", "--format=%H", "--", path], capture_output=True, text=True)
    shas = set()
    for c in log.stdout.split():
        r = subprocess.run(["git", "-C", str(repo), "show", "%s:%s" % (c, path)], capture_output=True)
        if r.returncode != 0:                           # deleted in that commit
            continue
        b = r.stdout
        shas.add(hashlib.sha256(b).hexdigest())
        if b"\0" not in b:
            lf = b.replace(b"\r\n", b"\n")
            shas.update(hashlib.sha256(x).hexdigest() for x in (lf, lf.replace(b"\n", b"\r\n")))
    return shas


def known_hashes(*manifests: dict) -> dict:
    """{path: SHA-256s an earlier release shipped it with}: the installed release's files and every earlier version the
    new release's manifest lists."""
    known = {}
    for m in manifests:
        for rel, sha in (m.get("files") or {}).items():
            known.setdefault(rel, set()).add(sha)
        for rel, shas in (m.get("earlier") or {}).items():
            known.setdefault(rel, set()).update(shas)
    return known


def merge(root: Path, new: Path, known: dict) -> list[str]:
    """config/ and data/: the release's file where yours is missing or still an earlier release's; yours kept (with the
    release's beside it as <name>.new) where you changed it."""
    done = []
    for name in MERGE:
        for f in sorted(p for p in (new / name).rglob("*") if p.is_file()) if (new / name).is_dir() else []:
            rel = f.relative_to(new).as_posix()
            dst = root / rel
            try:
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst)
                    done.append("added %s" % rel)
                elif same(f, dst):
                    continue
                elif sha256(dst) in known.get(rel, ()):     # as a release shipped it: not yours
                    tmp = dst.with_name(dst.name + ".update-tmp")
                    shutil.copy2(f, tmp)
                    os.replace(tmp, dst)
                    done.append("updated %s (it was as an earlier release shipped it)" % rel)
                else:
                    shutil.copy2(f, dst.with_name(dst.name + ".new"))
                    done.append("kept your %s (you changed it); the release's is beside it as %s.new" % (rel, dst.name))
            except OSError as e:
                done.append("[!] could not update %s (%s)" % (rel, e))
    return done


def install(root: Path, z: Path) -> list[str]:
    """The release's program replaces this one (the replaced one goes to .previous/), all or nothing: renames only,
    each written to the journal before it is done, undone on a failure.  Then config/ and data/ are merged.  What was
    done, as lines.  UpdateError says in what state a failure left the folder."""
    upd, prev = root / ".update", root / ".previous"
    upd.mkdir(exist_ok=True)
    jp = upd / JOURNAL
    finish_undo(root)
    new = unpack(z, upd)
    if prev.exists():                                   # the generation before: gone for good before the swap starts
        shutil.rmtree(prev, ignore_errors=True)
        if prev.exists():
            shutil.rmtree(new.parent, ignore_errors=True)
            raise UpdateError(".previous cannot be deleted (a file in it is in use); nothing was changed")
    known = known_hashes(load_json(root / MANIFEST), load_json(new / MANIFEST))   # read before the manifest moves
    journal = {"new": str(new), "steps": []}
    items = [p.name for p in sorted(new.iterdir()) if p.name not in KEEP]
    try:
        prev.mkdir()
        for name in items:
            dst = root / name
            if dst.exists() or dst.is_symlink():
                journal["steps"].append(["moved", name])
                save_journal(jp, journal)
                os.rename(dst, prev / name)
            journal["steps"].append(["placed", name])
            save_journal(jp, journal)
            os.rename(new / name, dst)
    except OSError as e:
        stuck = undo(root, journal)
        if stuck:
            raise UpdateError("%s; the old program could not be put back whole (still in use: %s). It is in "
                              ".previous: close every window and program using files in this folder and run "
                              "update.cmd again - it puts the old program back first" % (e, ", ".join(stuck)))
        jp.unlink(missing_ok=True)
        remove_empty(prev)
        shutil.rmtree(new.parent, ignore_errors=True)
        raise UpdateError("%s; the program in this folder is the one it was" % e)
    jp.unlink(missing_ok=True)
    done = ["program replaced (%d items; the previous one is in .previous)" % len(items)]
    done += merge(root, new, known)
    shutil.rmtree(new.parent, ignore_errors=True)
    return done


def main(argv=None, root: Path | None = None, opener=urllib.request.urlopen) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="only say whether a newer release is published")
    ap.add_argument("--force", action="store_true", help="install the release even when it is the installed version")
    ap.add_argument("--tag", help="a release tag instead of the latest (e.g. v0.1.40.3-nvfp4.1)")
    a = ap.parse_args(argv)
    root = root or Path(__file__).resolve().parents[1]
    have = installed_version(root)
    if (have is None or not (root / "start-server.cmd").exists()) and not (root / ".update" / JOURNAL).exists():
        say("This is not an unpacked strata-nvfp4 release (no engine\\BUILD.json or start-server.cmd here).")
        say("A git clone updates with UPDATE.bat (git pull + setup.py --update).")
        return 1
    if (root / ".update" / JOURNAL).exists():           # a swap stopped half way: undone before anything else
        busy = engine_running(root)
        if busy:
            say("%s is running: close the server window (start-server.cmd) and run update.cmd again." % busy)
            return 1
        try:
            finish_undo(root)
        except UpdateError as e:
            say("Not finished: %s." % e)
            return 1
        have = installed_version(root)
    try:
        rel = release(a.tag, opener)
    except (OSError, ValueError) as e:
        say("Cannot ask GitHub for the release (%s): check the internet connection and try again." % e)
        return 1
    asset = bundle_asset(rel)
    tag = str(rel.get("tag_name", "?"))
    if asset is None:
        say("Release %s has no %s-%s-windows-x64.zip." % (tag, TOP, tag))
        return 1
    say("Installed: %s. Published: %s (%s)." % (have, tag.lstrip("v"), rel.get("name") or tag))
    if tag.lstrip("v") == have and not a.force:
        say("Already up to date.")
        return 0
    if a.check:
        say("Run update.cmd to install it.")
        return 0
    busy = engine_running(root)
    if busy:
        say("%s is running: close the server window (start-server.cmd) and run update.cmd again." % busy)
        return 1
    z = root / ".update" / asset["name"]
    if z.exists() and check(z, asset["size"], asset["sha256"]) is None:
        say("Using %s, downloaded and checked before." % asset["name"])
    else:
        z.unlink(missing_ok=True)
        say("Downloading %s (%.0f MB) ..." % (asset["name"], asset["size"] / 1e6))
        try:
            download(asset["url"], z, asset["size"], opener)
        except OSError as e:
            say("Not installed: %s. Nothing was changed; run update.cmd again to go on with the download." % e)
            return 1
        why = check(z, asset["size"], asset["sha256"])
        if why:
            z.unlink(missing_ok=True)
            say("Not installed: %s. The download was deleted; run update.cmd again to fetch it afresh." % why)
            return 1
        say("Checked: the size and SHA-256 GitHub publishes for it.")
    busy = engine_running(root)                          # again: the download can take minutes
    if busy:
        say("%s was started meanwhile: close it and run update.cmd again (the download is kept)." % busy)
        return 1
    try:
        lines = install(root, z)
    except (OSError, UpdateError, zipfile.BadZipFile) as e:
        if isinstance(e, zipfile.BadZipFile):
            z.unlink(missing_ok=True)
        say("Not installed: %s. A file in use (an open window in a folder here, an antivirus scan) can stop it: run "
            "update.cmd again%s." % (e, "" if isinstance(e, zipfile.BadZipFile) else " (the checked download is kept)"))
        return 1
    for line in lines:
        say("  " + line)
    z.unlink(missing_ok=True)
    for req, venv in (("requirements-serve.txt", ".venv-serve"), ("requirements-model.txt", ".venv-model"),
                      ("requirements-convert.txt", ".venv-convert")):
        old, new, py = root / ".previous" / req, root / req, root / venv / "Scripts" / "python.exe"
        if py.exists() and new.exists() and not (old.exists() and same(old, new)):
            say("  %s changed: updating %s ..." % (req, venv))
            if subprocess.run([str(py), "-m", "pip", "install", "--quiet", "-r", str(new)]).returncode != 0:
                say("  [!] pip failed: delete %s and the start scripts make it again" % venv)
    say("Updated to %s. Start it with start-server.cmd." % tag.lstrip("v"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
