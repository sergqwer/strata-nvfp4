"""tools/bundle_model.py - the Windows bundle's prepare-model.cmd: this fork's qwen-nvfp4-gptq (the original
Qwen3.8-Flash-Next, every expert NVFP4 by GPTQ) ready-made from Hugging Face at the revision setup.py pins, and the
image encoder setup installs beside it, each file checked against its SHA-256.  Nothing is converted.

    prepare-model.cmd                                  (runs: python tools\\bundle_model.py)
    python tools\\bundle_model.py --check               what is there and checked; downloads nothing
    python tools\\bundle_model.py --write FILE          the maintainer, in a clone: FILE from setup.py's tables

model-files.json (the bundle's root) lists every file: repository, revision, path in it, bytes, SHA-256 and where it
goes under models\\.  A file is done once "<name>.sha256" beside it holds the SHA-256 it was checked against; a
stopped download resumes (huggingface_hub keeps the partial file); a file that does not match is deleted, and the
next run fetches it again.  HF_ENDPOINT (a mirror) is honoured by huggingface_hub; the checks stay the same.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "model-files.json"
FAMILY = "qwen-nvfp4-gptq"
MODEL_DIR = "models/qwen-nvfp4-gptq"
CONFIG = "config/strata-qwen-nvfp4-gptq.json"


def drafter(root: Path = ROOT, check_only: bool = False) -> int:
    """The DFlash2 drafter (incoai's, pinned and checked, packed here: strata_control/drafting_fetch.py) into models/qwen-nvfp4-gptq/dflash2,
    where Auto drafts with it: one CUDA card that holds as many experts as the RTX 5090 it was measured on (strata_control/drafting.py
    FORK_GPTQ_SETS).  Anywhere else the MTP drafts and nothing is fetched.  A failure is a message, not an error: the MTP drafts until a
    later run gets it.  -> 1 only for --check with the drafter missing."""
    sys.path.insert(0, str(root))
    from strata_control import drafting
    try:
        cfg = json.loads((root / CONFIG).read_text(encoding="utf-8"))
        backend, archs = drafting.detect_hardware()
        vram = drafting.card_vram_mib(cfg, backend, archs)
        wanted = drafting.choice_of("auto", model_name=cfg.get("model_name"), backend=backend, archs=archs,
                                    args=drafting.strip_mtp_flags(cfg.get("args") or []), vram_mib=vram) == "dflash2"
    except Exception as exc:  # noqa: BLE001 - no config or no card answer: the MTP drafts
        print(f"DFlash2 drafter: not looked for ({exc}); the MTP drafts")
        return 0
    data = root / MODEL_DIR
    if not wanted:
        print("DFlash2 drafter: not used on this card (the MTP drafts)")
        return 0
    if drafting.pack_ready(data):
        print("DFlash2 drafter: here")
        return 0
    if check_only:
        print(f"  missing: the DFlash2 drafter ({MODEL_DIR}/{drafting.FOLDER}, {drafting.DOWNLOAD_GB:g} GB to download)")
        return 1
    from strata_control import drafting_fetch
    print(f"DFlash2 drafter: {drafting.REPO} ({drafting.DOWNLOAD_GB:g} GB), packed for the engine here", flush=True)
    try:
        drafting_fetch.ensure(data, progress=lambda msg, done, total: None)
        if drafting.engine_reads_selector_q4(root / "engine" / "strata.exe"):
            drafting_fetch.convert_selector(data)          # the Q4_0 selector: 174 MiB less VRAM
    except Exception as exc:  # noqa: BLE001 - the MTP drafts until a later run gets it
        print(f"The DFlash2 drafter could not be prepared ({exc}). The model drafts with the MTP; run prepare-model.cmd again to retry.")
        return 0
    print(f"DFlash2 drafter: ready in {data / drafting.FOLDER}")
    return 0


def tuned_mtp(root: Path = ROOT, check_only: bool = False) -> int:
    """strata-nvfp4 0.2.15-nvfp4.4: Basalt's self-distilled draft layer into models/qwen-nvfp4-gptq/mtp/rt-tuned (pinned, checked:
    strata_control/drafting_fetch.py ensure_fork_tuned_mtp), which Auto's MTP drafts with on CUDA.  A failure is a message: the model's
    own layer drafts until a later run gets it.  -> 1 only for --check with the layer missing."""
    sys.path.insert(0, str(root))
    from strata_control import drafting
    mtp = root / MODEL_DIR / "mtp"
    if drafting.fork_tuned_mtp_complete(mtp / drafting.FORK_MTP_TUNED) and drafting.sync_fork_tuned_mtp(mtp):
        print("Tuned draft layer: here")
        return 0
    if check_only:
        print(f"  missing: the tuned draft layer ({MODEL_DIR}/mtp/{drafting.FORK_MTP_TUNED}, {drafting.FORK_MTP_TUNED_GB:g} GB to download)")
        return 1
    from strata_control import drafting_fetch
    print(f"Tuned draft layer: Basalt's self-distilled layer from {drafting.FORK_MTP_TUNED_REPO} ({drafting.FORK_MTP_TUNED_GB:g} GB)", flush=True)
    try:
        out = drafting_fetch.ensure_fork_tuned_mtp(mtp)
    except Exception as exc:  # noqa: BLE001 - the model's own layer drafts until a later run gets it
        print(f"The tuned draft layer could not be prepared ({exc}). The model drafts with its own layer; run prepare-model.cmd again to retry.")
        return 0
    print(f"Tuned draft layer: ready in {out}")
    return 0


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def mark_of(dst: Path) -> Path:
    return dst.with_name(dst.name + ".sha256")


def done(dst: Path, size: int, sha: str) -> bool:
    """Downloaded whole and checked against this SHA-256 (its mark), and still that size."""
    try:
        return dst.is_file() and dst.stat().st_size == size and mark_of(dst).read_text(encoding="utf-8").strip() == sha
    except OSError:
        return False


def load(table: Path = TABLE) -> dict:
    t = json.loads(table.read_text(encoding="utf-8"))
    for f in t["files"]:
        for k in ("repo", "revision", "path", "size", "sha256", "dir"):
            if k not in f:
                raise SystemExit(f"{table.name}: a file without {k!r}: {f}")
    return t


def fetch_one(f: dict, root: Path, download) -> Path:
    """Download one file (huggingface_hub, resumable) into root/<dir>/<path>, check it, mark it."""
    local = root / f["dir"]
    dst = local / f["path"]
    got = Path(download(repo_id=f["repo"], filename=f["path"], revision=f["revision"], local_dir=str(local)))
    if got.resolve() != dst.resolve():                  # huggingface_hub put it elsewhere: keep our layout
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(got), str(dst))
    size = dst.stat().st_size
    if size != f["size"]:
        dst.unlink(missing_ok=True)
        raise SystemExit(f"{f['path']}: {size} bytes, expected {f['size']} - deleted; run prepare-model.cmd again")
    print(f"  checking {f['path']} (SHA-256, {size / 1e9:.1f} GB) ...", flush=True)
    sha = sha256_file(dst)
    if sha != f["sha256"]:
        dst.unlink(missing_ok=True)
        raise SystemExit(f"{f['path']}: SHA-256 {sha}, expected {f['sha256']} - deleted (corrupt or not the pinned "
                         "file); run prepare-model.cmd again")
    mark_of(dst).write_text(sha + "\n", encoding="utf-8")
    return dst


def prepare(root: Path = ROOT, table: Path = TABLE, download=None, check_only: bool = False) -> int:
    t = load(table)
    todo = [f for f in t["files"] if not done(root / f["dir"] / f["path"], f["size"], f["sha256"])]
    have = len(t["files"]) - len(todo)
    total = sum(f["size"] for f in t["files"])
    print(f"{t['title']}: {len(t['files'])} files, {total / 1e9:.1f} GB ({have} already here and checked)")
    if check_only or not todo:
        for f in todo:
            print(f"  missing or not checked: {f['dir']}/{f['path']}")
        return 0 if not todo else 1
    need = sum(f["size"] - ((root / f["dir"] / f["path"]).stat().st_size
                            if (root / f["dir"] / f["path"]).is_file() else 0) for f in todo)
    (root / "models").mkdir(exist_ok=True)
    free = shutil.disk_usage(root / "models").free
    if free < need + (1 << 30):
        raise SystemExit(f"not enough free space on this drive: ~{need / 1e9:.0f} GB more needed, {free / 1e9:.0f} GB free")
    if download is None:
        try:
            from huggingface_hub import hf_hub_download as download
        except ImportError:
            raise SystemExit("huggingface_hub is missing: run prepare-model.cmd (it installs requirements-model.txt)")
    for i, f in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {f['repo']}: {f['path']} ({f['size'] / 1e9:.1f} GB)", flush=True)
        fetch_one(f, root, download)
    print(f"All {len(t['files'])} files are here and checked.")
    return 0


def write_table(out: Path) -> None:
    """The maintainer: model-files.json from setup.py's NVFP4_REPOS (the family's pinned files) and the image encoder
    setup installs (its pinned repository; size and SHA-256 from the Hub's LFS record at that revision)."""
    import urllib.request
    sys.path.insert(0, str(ROOT))
    import setup                                        # a clone: setup.py is beside tools/
    files, why = setup.nvfp4_files(FAMILY)
    if why:
        raise SystemExit(f"{FAMILY}: {why}")
    rows = [{"repo": repo, "revision": rev, "path": path, "size": size, "sha256": sha, "dir": MODEL_DIR}
            for _comp, repo, rev, path, size, sha in files]
    base = setup.FAMILIES["qwen"]["mmproj_hf"]          # https://huggingface.co/<repo>/resolve/<revision>/
    parts = base.rstrip("/").split("/")
    repo, rev = "/".join(parts[3:5]), parts[6]
    api = f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true"
    with urllib.request.urlopen(urllib.request.Request(api, headers={"User-Agent": "strata-bundle"}), timeout=60) as r:
        info = json.loads(r.read().decode("utf-8"))
    s = next(x for x in info["siblings"] if x["rfilename"] == setup.MMPROJ)
    rows.append({"repo": repo, "revision": info["sha"], "path": setup.MMPROJ, "size": int(s["lfs"]["size"]),
                 "sha256": s["lfs"]["sha256"], "dir": "models"})
    table = {"family": FAMILY, "title": setup.NVFP4_FAMILIES[FAMILY]["title"],
             "note": "tools/bundle_model.py --write from setup.py's NVFP4_REPOS: do not edit by hand", "files": rows}
    out.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(rows)} files, {sum(x['size'] for x in rows) / 1e9:.1f} GB")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="say what is there and checked; download nothing")
    ap.add_argument("--write", type=Path, metavar="FILE", help="the maintainer: write the table from setup.py")
    a = ap.parse_args(argv)
    if a.write:
        write_table(a.write)
        return 0
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    rc = prepare(check_only=a.check)
    if rc == 0 or a.check:                          # the model's files first: the drafter is for a model that is here
        rc = max(rc, drafter(check_only=a.check))
        rc = max(rc, tuned_mtp(check_only=a.check))
    return rc


if __name__ == "__main__":
    sys.exit(main())
