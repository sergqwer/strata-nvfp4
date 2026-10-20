"""release/make_windows_bundle.py - the Windows release: dist/strata-nvfp4 and its zip.

    python release/make_windows_bundle.py [--allow-dirty] [--engine-from CHECKOUT]

1. Refuses a working tree with uncommitted changes (BUILD.json names the commit the engine was built from;
   --allow-dirty for a local try, marked "dirty" there and refused by release/publish.py).
2. Builds the portable engine itself (release/build-release.cmd: STRATA_PORTABLE=ON, sm_75/86/89/120 (the W4A4 unit 120a),
   build-release/) and the image encoder (release/build-vision.cmd: CPU only, build-vision-cpu/), and checks with
   ninja dry runs that both are current - a bundle once shipped the previous release's engine.
   --engine-from CHECKOUT takes them from another checkout's build-release/ and build-vision-cpu/ instead (a worktree
   of the same commit's engine): only when every engine source there is this commit's (git trees and the llama.cpp
   submodule) and unchanged, and ninja there has no work to do - so the exe stays byte for byte the one already built.
3. Assembles dist/strata-nvfp4: that engine, cuBLAS from %CUDA_PATH%, the part of llama.cpp the converter imports,
   and the bundle's own files (README, scripts, config, model-files.json: what prepare-model.cmd downloads) from
   release/windows/.
   release-manifest.json lists every file's SHA-256 and, for config/ and data/, every version git has of their
   source: update.cmd (tools/bundle_update.py) replaces a file still as a release shipped it, never one the user changed.
   strata-nvfp4: the new interface - app/host (apps/host/server and the page built from the lockfile: npm ci, npm run
   build) and node/node.exe (the pinned official Node LTS zip, its SHA-256 checked against nodejs.org's SHASUMS256.txt
   and this file, cached outside the repository in %LOCALAPPDATA%/strata-nvfp4-build or STRATA_BUILD_CACHE).
4. Zips it to dist/strata-nvfp4-v<VERSION>-windows-x64.zip and writes its SHA-256 beside it.
5. Writes setup.py's asset from the same engine folder: dist/strata-windows-x64.zip, its files at the top level
   (strata.exe, strata-vision.exe, strata_kq.dll, the cuBLAS DLLs, BUILD.json with "cuda_libs": "bundled"), the layout get_prebuilt
   unpacks into engine/ - so a clone's START-HERE / UPDATE.bat installs this fork's engine, never upstream's.
Publish both with release/publish.py.
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
VERSION = "0.2.0-beta-nvfp4.2"
ZIP = REPO / "dist" / ("strata-nvfp4-v%s-windows-x64.zip" % VERSION)
SETUP_ZIP = REPO / "dist" / "strata-windows-x64.zip"   # setup.py's PREBUILT_ASSET
allow_dirty = "--allow-dirty" in sys.argv[1:]
ENGINE_FROM = pathlib.Path(sys.argv[sys.argv.index("--engine-from") + 1]).resolve() if "--engine-from" in sys.argv else None
BUILT = ENGINE_FROM or REPO                            # the checkout whose build-release/ and build-vision-cpu/ are taken
ENGINE = BUILT / "build-release" / "strata.exe"
VISION = BUILT / "build-vision-cpu" / "bin" / "strata-vision.exe"
# strata-nvfp4: the new interface's runtime, an official Node.js LTS build (https://nodejs.org/dist/)
NODE_VERSION = "24.22.0"
NODE_ZIP = "node-v%s-win-x64.zip" % NODE_VERSION
NODE_SHA256 = "f9683a19877e12a9b167305c8db71040d812876359de21f0d2e97c3f321f80c3"   # SHASUMS256.txt of v24.22.0
NODE_CACHE = pathlib.Path(os.environ.get("STRATA_BUILD_CACHE") or pathlib.Path(os.environ.get("LOCALAPPDATA", pathlib.Path.home())) / "strata-nvfp4-build") / "node"
HOST = REPO / "apps" / "host"
# the page's runtime libraries (bundled into app/host/dist by vite); their licences go into THIRD-PARTY-NOTICES.txt
HOST_LIBS = ("react", "react-dom", "scheduler", "pdfjs-dist", "prismjs", "html-to-image")


def git(*args, cwd=REPO):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True).stdout.strip()


# 1. a clean tree, so the commit in BUILD.json is what the bundle holds
dirty = git("status", "--porcelain")
if dirty and not allow_dirty:
    sys.exit("make_windows_bundle: the working tree has uncommitted changes - commit them first "
             "(or --allow-dirty for a local try):\n" + dirty)

# 2. the engine and the image encoder built from this tree, now (or, --engine-from, the same sources built in another checkout)
ENGINE_SOURCES = ("src", "include", "cmake", "CMakeLists.txt", "third_party", "tools/vision")
if ENGINE_FROM:
    for path in ENGINE_SOURCES:
        here, there = git("rev-parse", "HEAD:" + path), git("rev-parse", "HEAD:" + path, cwd=ENGINE_FROM)
        if not here or here != there:
            sys.exit("make_windows_bundle: %s differs between this checkout and %s - build the engine here" % (path, ENGINE_FROM))
    changed = git("status", "--porcelain", "--", *ENGINE_SOURCES, cwd=ENGINE_FROM)
    if changed:
        sys.exit("make_windows_bundle: %s has uncommitted engine source changes:\n%s" % (ENGINE_FROM, changed))
    if git("rev-parse", "HEAD", cwd=LLAMA) != git("rev-parse", "HEAD", cwd=ENGINE_FROM / "third_party" / "llama.cpp"):
        sys.exit("make_windows_bundle: the llama.cpp submodule differs from %s's" % ENGINE_FROM)
    print("engine: the build in %s (the same engine sources as this commit)" % ENGINE_FROM, flush=True)
for script, build, target, exe in (("build-release.cmd", "build-release", "strata", ENGINE),
                                   ("build-vision.cmd", "build-vision-cpu", "strata-vision", VISION)):
    if not ENGINE_FROM:
        print("building %s (release/%s) ..." % (target, script), flush=True)
        if subprocess.run(["cmd", "/c", str(REPO / "release" / script)]).returncode != 0:
            sys.exit("make_windows_bundle: release/%s failed" % script)
    # ninja itself says whether the target is current (a dry run after the build must have nothing to do)
    dry = subprocess.run(["ninja", "-C", str(BUILT / build), "-n", target], capture_output=True, text=True)
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
# upstream 0.1.44: the K-quant prompt kernels are a library of their own, opened only for a model with those experts and
# STRATA_MMQ_KQ=1 (setup writes it for Unsloth's UD-Q5_K_XL / UD-Q6_K_XL); the release build makes it beside the engine
KQ = BUILT / "build-release" / "strata_kq.dll"
if not KQ.exists():
    sys.exit("make_windows_bundle: %s is missing (the K-quant prompt library the strata target builds)" % KQ)
cp(KQ, OUT / "engine" / "strata_kq.dll")
for dll in ("cublas64_13.dll", "cublasLt64_13.dll"):
    cp(CUDA_BIN / dll, OUT / "engine" / dll)
(OUT / "engine" / "BUILD.json").write_text(json.dumps({
    "version": VERSION, "source": "release", "archs": [75, 86, 89, 120], "ptx": False, "cuda": CUDA.name.lstrip("v"),
    "vision": "cpu", "portable": True, "fork": "https://github.com/sergqwer/strata-nvfp4",
    "commit": git("rev-parse", "--short", "HEAD"), "dirty": bool(dirty),
    "engine_sha256": hashlib.sha256(ENGINE.read_bytes()).hexdigest(), "cuda_libs": "bundled"}, indent=1) + "\n")

# the server (with its chat page) and the tools
shutil.copytree(REPO / "serve", OUT / "serve", ignore=skip)
# upstream 0.2.0: serve imports strata_control (engine_component, vision_reader, procs, presets, private_files) at start,
# and /v1/telemetry strata_cli's client (stdlib only).  Not the rest of strata_cli: no `strata update` in this bundle
# (update.cmd is its updater - tools/bundle_update.py, which `strata update` would only run; the server's update
# notice is strata_control/fork_release.py).
shutil.copytree(REPO / "strata_control", OUT / "strata_control", ignore=skip)
for name in ("__init__.py", "client.py"):
    cp(REPO / "strata_cli" / name, OUT / "strata_cli" / name)
shutil.copytree(REPO / "tools", OUT / "tools", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "test_*.py", "vision"))

# strata-nvfp4: the new interface - the App Host (apps/host/server, no tests or docs) and its page, built from the lockfile
npm = shutil.which("npm.cmd") or shutil.which("npm")
if not npm:
    sys.exit("make_windows_bundle: npm is not on PATH (the page is built with npm ci && npm run build)")
for step in (["ci", "--no-audit", "--no-fund"], ["run", "build"]):
    print("apps/host: npm %s ..." % " ".join(step), flush=True)
    if subprocess.run([npm, *step], cwd=HOST).returncode != 0:
        sys.exit("make_windows_bundle: npm %s failed in apps/host" % step[0])
shutil.copytree(HOST / "server", OUT / "app" / "host" / "server", ignore=shutil.ignore_patterns("*.md", "__pycache__"))
# the page; not its source maps, and not upstream's GT Walsheim Pro webfonts (a commercial typeface: the page falls back
# to the system font, "Segoe UI")
shutil.copytree(HOST / "dist", OUT / "app" / "host" / "dist", ignore=shutil.ignore_patterns("*.map", "GTWalsheimPro-*"))
cp(HOST / "package.json", OUT / "app" / "host" / "package.json")


def node_zip() -> pathlib.Path:
    """The pinned Node.js zip in the cache, downloaded when missing; its SHA-256 must be both nodejs.org's and NODE_SHA256."""
    import urllib.request
    NODE_CACHE.mkdir(parents=True, exist_ok=True)
    base = "https://nodejs.org/dist/v%s/" % NODE_VERSION
    with urllib.request.urlopen(base + "SHASUMS256.txt", timeout=60) as r:
        listed = {line.split()[1]: line.split()[0] for line in r.read().decode().splitlines() if len(line.split()) == 2}
    if listed.get(NODE_ZIP) != NODE_SHA256:
        sys.exit("make_windows_bundle: nodejs.org's SHASUMS256.txt says %s for %s, this script %s" % (listed.get(NODE_ZIP), NODE_ZIP, NODE_SHA256))
    z = NODE_CACHE / NODE_ZIP
    if not z.exists() or hashlib.sha256(z.read_bytes()).hexdigest() != NODE_SHA256:
        print("downloading %s ..." % (base + NODE_ZIP), flush=True)
        tmp = z.with_suffix(".part")
        with urllib.request.urlopen(base + NODE_ZIP, timeout=600) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        got = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if got != NODE_SHA256:
            tmp.unlink()
            sys.exit("make_windows_bundle: %s has SHA-256 %s, expected %s" % (NODE_ZIP, got, NODE_SHA256))
        os.replace(tmp, z)
    return z


with zipfile.ZipFile(node_zip()) as z:
    top = "node-v%s-win-x64/" % NODE_VERSION
    (OUT / "node").mkdir(parents=True, exist_ok=True)
    for name in ("node.exe", "LICENSE"):
        (OUT / "node" / name).write_bytes(z.read(top + name))
    node_license = z.read(top + "LICENSE").decode("utf-8")

# the part of llama.cpp the converter imports, at its pinned commit, with its license
for name in ("convert_hf_to_gguf.py", "LICENSE"):
    cp(LLAMA / name, OUT / "third_party" / "llama.cpp" / name)
shutil.copytree(LLAMA / "conversion", OUT / "third_party" / "llama.cpp" / "conversion", ignore=skip)
shutil.copytree(LLAMA / "gguf-py", OUT / "third_party" / "llama.cpp" / "gguf-py",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "examples"))
(OUT / "third_party" / "llama.cpp" / "COMMIT").write_text(git("rev-parse", "HEAD", cwd=LLAMA) + "\n")

# data, docs, license
for name in ("expert-profile.bin", "draft_vocab.bin",
             # the censorship switch the config loads (off by default: "uncensored": false)
             "uncensor/qwen-nvfp4-gptq-uncensored.gguf", "uncensor/qwen-nvfp4-gptq-uncensored.json",
             "uncensor/README.md"):
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

# THIRD-PARTY-NOTICES.txt: the new interface's runtime (Node.js) and the page's libraries, with their licence texts
notices = ["", "", "The new interface (app\\host\\, node\\)", "=" * 39, "",
           "Node.js %s (node\\node.exe), the official win-x64 build from https://nodejs.org/dist/v%s/ - MIT License and the"
           % (NODE_VERSION, NODE_VERSION), "licences of the components it includes; the full text follows and is node\\LICENSE.", "",
           "The App Host (app\\host\\server) and the page are Strata's (see LICENSE). The page bundles these libraries:"]
texts = []
for lib in HOST_LIBS:
    meta = json.loads((HOST / "node_modules" / lib / "package.json").read_text(encoding="utf-8"))
    notices.append("  %s %s - %s" % (lib, meta["version"], meta.get("license")))
    lic = next((q for q in sorted((HOST / "node_modules" / lib).iterdir()) if q.name.upper().startswith("LICEN")), None)
    if lic is None:
        sys.exit("make_windows_bundle: node_modules/%s has no licence file" % lib)
    texts += ["", "----- %s %s (%s) -----" % (lib, meta["version"], lic.name), lic.read_text(encoding="utf-8").strip()]
notices += texts + ["", "----- Node.js %s (node\\LICENSE) -----" % NODE_VERSION, node_license.strip(), ""]
with open(OUT / "THIRD-PARTY-NOTICES.txt", "a", encoding="utf-8", newline="\r\n") as f:
    f.write("\n".join(notices))

# release-manifest.json: what update.cmd needs to tell a file of config/ or data/ the user changed from one a release
# shipped (the earlier versions: their source's git history, so a bundle made before the manifest is covered too)
sys.path.insert(0, str(REPO / "tools"))
import bundle_update  # noqa: E402

earlier = {}
for f in sorted(p for d in bundle_update.MERGE for p in (OUT / d).rglob("*") if p.is_file()):
    rel = f.relative_to(OUT).as_posix()
    src = "release/windows/" + rel if (REPO / "release" / "windows" / rel).is_file() else rel
    earlier[rel] = bundle_update.git_versions(REPO, src)
    if not earlier[rel]:
        sys.exit("make_windows_bundle: %s has no git history (%s) - commit it first" % (rel, src))
manifest = bundle_update.build_manifest(OUT, VERSION, earlier)
(OUT / bundle_update.MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")

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

# 5. setup.py's asset: the same engine folder, its files at the top level (get_prebuilt's layout)
if SETUP_ZIP.exists():
    SETUP_ZIP.unlink()
with zipfile.ZipFile(SETUP_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=7) as z:
    for f in sorted((OUT / "engine").iterdir()):
        z.write(f, f.name)
h = hashlib.sha256()
with open(SETUP_ZIP, "rb") as f:
    for block in iter(lambda: f.read(1 << 24), b""):
        h.update(block)
(SETUP_ZIP.parent / (SETUP_ZIP.name + ".sha256")).write_text(h.hexdigest() + "  " + SETUP_ZIP.name + "\n")
print("zipped %s (setup.py's engine): %.1f MB, SHA-256 %s" % (SETUP_ZIP, SETUP_ZIP.stat().st_size / 1e6, h.hexdigest()))
