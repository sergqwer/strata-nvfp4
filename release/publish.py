"""release/publish.py - publish the zip release/make_windows_bundle.py made as GitHub release v<VERSION>.

    python release/publish.py --title "Strata NVFP4 v... - what changed" --notes notes.md [--dry-run]
    python release/publish.py --replace [--notes notes.md] [--dry-run]

--replace re-publishes an existing release after a fix to it: the tag moves to HEAD, the zip is replaced and the
SHA-256 line of the notes (or the new notes) updated.  The checks below still hold, except the one on the tag.
Refuses unless:
  - `origin` is this fork (sergqwer/strata-nvfp4), and every gh call names it with -R - in a clone whose gh
    default is unset, gh picks the `upstream` remote (Niko1221/Strata) and would publish there;
  - the working tree is clean and HEAD is what origin/main holds (the tag points at pushed code);
  - the zip's engine/BUILD.json names HEAD, this VERSION and a clean tree (not a stale or local bundle);
  - the tag does not exist yet.
The zip's SHA-256 is appended to the notes.
"""
import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import zipfile

REPO = pathlib.Path(__file__).resolve().parents[1]
SLUG = "sergqwer/strata-nvfp4"


def run(*args, check=True):
    r = subprocess.run(list(args), capture_output=True, text=True, cwd=REPO)
    if check and r.returncode != 0:
        sys.exit("publish: `%s` failed:\n%s%s" % (" ".join(args), r.stdout, r.stderr))
    return r


def fail(msg):
    sys.exit("publish: " + msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--title", help="required for a new release")
    ap.add_argument("--notes", help="a Markdown file; required for a new release")
    ap.add_argument("--replace", action="store_true", help="re-publish the existing release of this version")
    ap.add_argument("--dry-run", action="store_true", help="run every check, publish nothing")
    a = ap.parse_args()
    if not a.replace and (not a.title or not a.notes):
        fail("a new release needs --title and --notes")

    version = re.search(r'^VERSION = "([^"]+)"', (REPO / "release" / "make_windows_bundle.py").read_text(), re.M).group(1)
    tag = "v" + version
    zip_path = REPO / "dist" / ("strata-nvfp4-%s-windows-x64.zip" % tag)

    origin = run("git", "remote", "get-url", "origin").stdout.strip()
    if SLUG not in origin:
        fail("origin is %s, not %s" % (origin, SLUG))
    if run("git", "status", "--porcelain").stdout.strip():
        fail("the working tree has uncommitted changes")
    run("git", "fetch", "origin", "main")
    head = run("git", "rev-parse", "HEAD").stdout.strip()
    if head != run("git", "rev-parse", "origin/main").stdout.strip():
        fail("HEAD %s is not origin/main - push first (git push origin HEAD:main)" % head[:7])

    if not zip_path.exists():
        fail("%s is missing - run release/make_windows_bundle.py" % zip_path.name)
    with zipfile.ZipFile(zip_path) as z:
        build = json.loads(z.read("strata-nvfp4/engine/BUILD.json"))
    if build.get("version") != version:
        fail("the zip is version %s, the release is %s" % (build.get("version"), version))
    if not head.startswith(build.get("commit", "-")):
        fail("the zip was built from %s, HEAD is %s - rebuild it" % (build.get("commit"), head[:7]))
    if build.get("dirty", True):
        fail("the zip was built from a tree with uncommitted changes - rebuild it from a clean tree")

    exists = run("gh", "release", "view", tag, "-R", SLUG, check=False).returncode == 0
    if exists and not a.replace:
        fail("%s already exists on %s (--replace re-publishes it)" % (tag, SLUG))
    if a.replace and not exists:
        fail("%s does not exist on %s - nothing to replace" % (tag, SLUG))

    h = hashlib.sha256()
    with open(zip_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    sha_line = "SHA-256 `%s`" % h.hexdigest()
    if a.notes:
        notes = pathlib.Path(a.notes).read_text(encoding="utf-8").rstrip() + "\n\n" + sha_line + "\n"
    else:
        body = run("gh", "release", "view", tag, "-R", SLUG, "--json", "body", "--jq", ".body").stdout.rstrip()
        notes, n = re.subn(r"SHA-256 `[0-9a-f]{64}`", sha_line, body)
        if n != 1:
            fail("the notes of %s have %d SHA-256 lines, expected 1 - pass --notes" % (tag, n))
        notes += "\n"
    print("publish: %s %s on %s, commit %s, %s (%.1f MB), SHA-256 %s" % (
        "replacing" if a.replace else "creating", tag, SLUG, head[:7], zip_path.name, zip_path.stat().st_size / 1e6,
        h.hexdigest()))
    if a.dry_run:
        print("publish: --dry-run, every check passed; nothing published")
        return
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as nf:
        nf.write(notes)
    if a.replace:
        run("git", "tag", "-f", tag, head)
        run("git", "push", "-f", "origin", "refs/tags/" + tag)
        run("gh", "release", "upload", tag, str(zip_path), "--clobber", "-R", SLUG)
        edit = ["gh", "release", "edit", tag, "-R", SLUG, "--notes-file", nf.name]
        if a.title:
            edit += ["--title", a.title]
        run(*edit)
        print("publish: %s now at %s with the new zip" % (tag, head[:7]))
    else:
        r = run("gh", "release", "create", tag, "-R", SLUG, "--target", head, "--title", a.title,
                "--notes-file", nf.name, str(zip_path))
        print(r.stdout.strip())


if __name__ == "__main__":
    main()
