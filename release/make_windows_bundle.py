"""release/make_windows_bundle.py - the Windows release: dist/strata-nvfp4 and its zip.

    python release/make_windows_bundle.py [--allow-dirty]

1. Refuses a working tree with uncommitted changes (BUILD.json names the commit the engine was built from;
   --allow-dirty for a local try, marked "dirty" there and refused by release/publish.py).
2. Builds the portable engine itself (release/build-release.cmd: STRATA_PORTABLE=ON, sm_75/86/89/120 (the W4A4 unit 120a),
   build-release/) and the image encoder (release/build-vision.cmd: CPU only, build-vision-cpu/), and checks with
   ninja dry runs that both are current - a bundle once shipped the previous release's engine.
3. Assembles dist/strata-nvfp4: that engine, cuBLAS from %CUDA_PATH%, the part of llama.cpp the converter imports,
   and the bundle's own files (README, scripts, config) from release/windows/.
4. Zips it to dist/strata-nvfp4-v<VERSION>-windows-x64.zip and writes its SHA-256 beside it.
Publish with release/publish.py.
"""
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import zipfile

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "dist" / "strata-nvfp4"
CUDA = pathlib.Path(os.environ["CUDA_PATH"])
CUDA_BIN = CUDA / "bin" / "x64" if (CUDA / "bin" / "x64" / "cublas64_13.dll").exists() else CUDA / "bin"
LLAMA = REPO / "third_party" / "llama.cpp"
VERSION = "0.1.37-nvfp4.2"
ZIP = REPO / "dist" / ("strata-nvfp4-v%s-windows-x64.zip" % VERSION)
ENGINE = REPO / "build-release" / "strata.exe"
VISION = REPO / "build-vision-cpu" / "bin" / "strata-vision.exe"
allow_dirty = "--allow-dirty" in sys.argv[1:]


def git(*args, cwd=REPO):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True).stdout.strip()


# 1. a clean tree, so the commit in BUILD.json is what the bundle holds
dirty = git("status", "--porcelain")
if dirty and not allow_dirty:
    sys.exit("make_windows_bundle: the working tree has uncommitted changes - commit them first "
             "(or --allow-dirty for a local try):\n" + dirty)

# 2. the engine and the image encoder built from this tree, now
for script, build, target, exe in (("build-release.cmd", "build-release", "strata", ENGINE),
                                   ("build-vision.cmd", "build-vision-cpu", "strata-vision", VISION)):
    print("building %s (release/%s) ..." % (target, script), flush=True)
    if subprocess.run(["cmd", "/c", str(REPO / "release" / script)]).returncode != 0:
        sys.exit("make_windows_bundle: release/%s failed" % script)
    # ninja itself says whether the target is current (a dry run after the build must have nothing to do)
    dry = subprocess.run(["ninja", "-C", str(REPO / build), "-n", target], capture_output=True, text=True)
    if not exe.exists() or dry.returncode != 0 or "no work to do" not in dry.stdout:
        sys.exit("make_windows_bundle: %s is not current after the build:\n%s" % (exe, dry.stdout[-2000:]))

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
skip = shutil.ignore_patterns("__pycache__", "*.pyc", "test_*.py", "*_fake_server.py", "chat_golden.json", "tests")


def cp(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# engine: the portable build and the cuBLAS it links (the CUDA runtime is static), and the image encoder
cp(ENGINE, OUT / "engine" / "strata.exe")
cp(VISION, OUT / "engine" / "strata-vision.exe")
for dll in ("cublas64_13.dll", "cublasLt64_13.dll"):
    cp(CUDA_BIN / dll, OUT / "engine" / dll)
(OUT / "engine" / "BUILD.json").write_text(json.dumps({
    "version": VERSION, "source": "release", "archs": [75, 86, 89, 120], "ptx": False, "cuda": CUDA.name.lstrip("v"),
    "vision": "cpu", "portable": True, "fork": "https://github.com/sergqwer/strata-nvfp4",
    "commit": git("rev-parse", "--short", "HEAD"), "dirty": bool(dirty),
    "engine_sha256": hashlib.sha256(ENGINE.read_bytes()).hexdigest()}, indent=1) + "\n")

# the server (with its chat page) and the tools
shutil.copytree(REPO / "serve", OUT / "serve", ignore=skip)
shutil.copytree(REPO / "tools", OUT / "tools", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "test_*.py", "vision"))

# the part of llama.cpp the converter imports, at its pinned commit, with its license
for name in ("convert_hf_to_gguf.py", "LICENSE"):
    cp(LLAMA / name, OUT / "third_party" / "llama.cpp" / name)
shutil.copytree(LLAMA / "conversion", OUT / "third_party" / "llama.cpp" / "conversion", ignore=skip)
shutil.copytree(LLAMA / "gguf-py", OUT / "third_party" / "llama.cpp" / "gguf-py",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "examples"))
(OUT / "third_party" / "llama.cpp" / "COMMIT").write_text(git("rev-parse", "HEAD", cwd=LLAMA) + "\n")

# data, docs, license
for name in ("expert-profile.bin", "draft_vocab.bin"):
    cp(REPO / "data" / name, OUT / "data" / name)
cp(REPO / "docs" / "NVFP4.md", OUT / "docs" / "NVFP4.md")
cp(REPO / "LICENSE", OUT / "LICENSE")

# the bundle's own files; .cmd with CRLF, or cmd.exe mangles them
for src in (REPO / "release" / "windows").rglob("*"):
    if src.is_file():
        dst = OUT / src.relative_to(REPO / "release" / "windows")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".cmd":
            dst.write_bytes(src.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        else:
            shutil.copy2(src, dst)

files = [f for f in OUT.rglob("*") if f.is_file()]
print("assembled %s: %d files, %.1f MB" % (OUT, len(files), sum(f.stat().st_size for f in files) / 1e6))

# 4. the zip (a top-level strata-nvfp4 folder, as the README's steps expect) and its SHA-256
if ZIP.exists():
    ZIP.unlink()
seven = shutil.which("7z")
if seven:
    subprocess.run([seven, "a", "-tzip", "-mx=7", str(ZIP), str(OUT)], check=True, stdout=subprocess.DEVNULL)
else:
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=7) as z:
        for f in sorted(files):
            z.write(f, f.relative_to(OUT.parent).as_posix())
h = hashlib.sha256()
with open(ZIP, "rb") as f:
    for block in iter(lambda: f.read(1 << 24), b""):
        h.update(block)
(ZIP.parent / (ZIP.name + ".sha256")).write_text(h.hexdigest() + "  " + ZIP.name + "\n")
print("zipped %s: %.1f MB, SHA-256 %s" % (ZIP, ZIP.stat().st_size / 1e6, h.hexdigest()))
