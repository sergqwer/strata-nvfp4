"""tools/nvfp4_table.py - one repository's entry of setup.py's NVFP4_REPOS, ready to paste.

    python tools/nvfp4_table.py Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata [--revision main]
    python tools/nvfp4_table.py REPO --local DIR       (before the upload: the files as they are in DIR)

From the Hub: the commit the revision points at, and each file's size and SHA-256 - the LFS pointer's for the big
ones, a download of the small ones (a few MB in all).  --local hashes the folder's files instead (minutes for the
~130 GB); the revision is then left for the commit of the upload.  Each file goes to its component by its path:
pack/... the experts pack, mtp/... the draft head's runtime folder, *ple*.gguf the n-gram table, *embd*.gguf the
token embedding, expert-profile*.bin the profile, any other .gguf the dense weights; the rest (README, LICENSE,
manifests) is listed as a comment.  Check the result against what the upload holds before committing it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

COMPONENTS = ("pack", "dense", "ple", "embd", "mtp", "profile")
SMALL = 64 << 20                       # files below this without an LFS pointer are downloaded and hashed


def component(path: str) -> str | None:
    name = path.split("/")[-1].lower()
    if path.startswith("pack/"):
        return "pack"
    if path.startswith("mtp/"):
        return "mtp"
    if name.endswith(".gguf"):
        if name.startswith("mmproj"):
            return None
        return "ple" if "ple" in name else "embd" if "embd" in name else "dense"
    if name.startswith("expert-profile") and name.endswith(".bin"):
        return "profile"
    return None


def sha256_of(read) -> str:
    h = hashlib.sha256()
    for b in iter(lambda: read(16 << 20), b""):
        h.update(b)
    return h.hexdigest()


def hub_files(repo: str, revision: str, endpoint: str, opener=urllib.request.urlopen) -> tuple[str, dict]:
    """(commit, {path: (bytes, sha256)}) from the Hub's API."""
    url = f"{endpoint}/api/models/{repo}/revision/{revision}?blobs=true"
    with opener(urllib.request.Request(url, headers={"User-Agent": "strata-nvfp4-table"}), timeout=60) as r:
        info = json.loads(r.read().decode("utf-8"))
    commit, files = info["sha"], {}
    for s in info.get("siblings") or []:
        path, lfs = s["rfilename"], s.get("lfs") or {}
        if component(path) is None:                    # README, LICENSE, manifests: named, not fetched
            files[path] = (int(s.get("size") or 0), "")
        elif lfs.get("sha256"):
            files[path] = (int(lfs.get("size") or s.get("size")), lfs["sha256"])
        elif int(s.get("size") or 0) < SMALL:
            with opener(urllib.request.Request(f"{endpoint}/{repo}/resolve/{commit}/{path}",
                                               headers={"User-Agent": "strata-nvfp4-table"}), timeout=60) as r:
                data = r.read()
            files[path] = (len(data), hashlib.sha256(data).hexdigest())
        else:
            raise SystemExit(f"{path}: {s.get('size')} bytes without an LFS SHA-256 - hash it with --local")
    return commit, files


def local_files(folder: Path) -> dict:
    files = {}
    for p in sorted(x for x in folder.rglob("*") if x.is_file() and ".cache" not in x.parts and ".git" not in x.parts):
        rel = p.relative_to(folder).as_posix()
        print(f"  hashing {rel} ...", file=sys.stderr, flush=True)
        with open(p, "rb") as f:
            files[rel] = (p.stat().st_size, sha256_of(f.read))
    return files


def entry(repo: str, commit: str, files: dict) -> str:
    """setup.py's NVFP4_REPOS entry for the repository, as Python source."""
    by = {c: {} for c in COMPONENTS}
    rest = []
    for path in sorted(files):
        c = component(path)
        (by[c].__setitem__(path, files[path]) if c else rest.append(path))
    out = [f'    "{repo}": {{', f'        "revision": "{commit}",' + ("" if commit else "   # the upload's commit")]
    for c in COMPONENTS:
        items = [f'"{p}": ({s}, "{h}")' for p, (s, h) in by[c].items()]
        out.append(f'        "{c}": {{' + (", ".join(items) if len(items) < 2 else
                                         "\n            " + ",\n            ".join(items) + ",\n        ") + "},")
    out.append("    },")
    if rest:
        out.append("    # not a component: " + ", ".join(rest))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("repo")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--local", type=Path, help="hash this folder's files instead of asking the Hub")
    a = ap.parse_args(argv)
    endpoint = (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/") or "https://huggingface.co"
    commit, files = ("", local_files(a.local)) if a.local else hub_files(a.repo, a.revision, endpoint)
    print(entry(a.repo, commit, files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
