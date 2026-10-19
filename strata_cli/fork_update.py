"""strata-nvfp4 (this fork): `strata update` and `strata components update` without a named feed.

Upstream's `strata update` installs upstream's engine from its signed feed (update_cmd.py); this fork uses that feed
only when it is named (--feed, STRATA_UPDATE_FEED, or a non-default one chosen before).  Without one, this runs the
fork's own updater for the folder strata.py is in:

    a release folder (the unpacked zip)       tools/bundle_update.py, what update.cmd runs (--check, --force)
    a git clone (or the source without git)   UPDATE.bat on Windows, ./update.sh elsewhere: git pull, then
                                              setup.py --update (the engine this version needs, the packages)

--check changes nothing: in a release folder bundle_update's --check; in a clone `git fetch` (the remote's branches
only), whether the branch's remote is ahead, and whether a newer release of this fork is published.
--output json / jsonl: one JSON object at the end - status, installed, latest (the updater's own lines go to stderr).
--wait is not supported (close the server first); --app, --app-version, --no-engine, --no-control, --backend,
--components-dir and --model-config are for upstream's feed only: refused, naming --feed.
Exit codes: 0 done or nothing to do, 1 the updater did not finish (its lines say why), 2 usage, 5 acquiring forbidden
(STRATA_NO_ACQUIRE), 6 --check could not ask GitHub.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # the folder strata.py is in
#: (attribute, flag): meaningful for upstream's signed feed only
FEED_ONLY = (("app", "--app"), ("app_version", "--app-version"), ("no_engine", "--no-engine"),
             ("no_control", "--no-control"), ("backend", "--backend"), ("components_dir", "--components-dir"),
             ("model_config", "--model-config"))
KIND_TEXT = {"bundle": "release folder", "clone": "git clone", "source": "source folder (not a git clone)"}


def updater_cmd(root: Path, kind: str, force: bool = False) -> tuple[list, str, str]:
    """The updater for a folder of `kind`: its command line, its name and what it does."""
    if kind == "bundle":
        return ([sys.executable, str(root / "tools" / "bundle_update.py"), *(["--force"] if force else [])],
                "tools/bundle_update.py", "update.cmd's updater: the newest release's zip, checked against GitHub's SHA-256")
    what = ("git pull, then setup.py --update" if kind == "clone" else
            "it names the newest source zip (no git here), then setup.py --update")
    if os.name == "nt":
        return ["cmd", "/c", "UPDATE.bat"], "UPDATE.bat", what
    return ["sh", "./update.sh"], "./update.sh", what


def _latest(query, timeout: float):
    try:
        return query(timeout=timeout), None
    except (OSError, ValueError) as e:
        return None, str(e) or type(e).__name__


def run(a, root: Path | None = None, runner=None, query=None) -> int:
    """`strata update` without a feed (`a`: update_cmd's arguments); the exit code."""
    from strata_control import acquisition, fork_release as fr
    root = Path(root or ROOT)
    runner = runner or subprocess.run
    kind = fr.install_kind(root)
    _, name, what = updater_cmd(root, kind)
    machine = a.output in ("json", "jsonl")

    def fail(message: str, code: int) -> int:
        if machine:
            print(json.dumps({"schema_version": 1, "ok": False, "error": {"code": code, "message": message}}))
        else:
            print(f"strata: {message}", file=sys.stderr)
        return code

    flags = [flag for attr, flag in FEED_ONLY if getattr(a, attr, None)]
    if flags:
        return fail(f"{' '.join(flags)}: only for upstream's signed engine feed, which this strata-nvfp4 install uses "
                    "only when it is named: add --feed URL (or set STRATA_UPDATE_FEED). Without a feed `strata update` "
                    f"runs this fork's own updater ({name})", 2)
    if getattr(a, "wait", None):
        return fail(f"--wait: this fork's updater ({name}) does not wait for a running server: close it, then run "
                    "`strata update` again (--wait is for upstream's feed, with --feed)", 2)
    if getattr(a, "force", False) and kind != "bundle":
        return fail(f"--force installs a release folder's release again; this {KIND_TEXT[kind]} has nothing to force "
                    f"(`strata update` runs {name}: {what})", 2)
    try:
        acquisition.require("the update", "`strata update` from a terminal")
    except acquisition.AcquisitionRefused as exc:
        return fail(f"{exc} (the update is the one explicit step that downloads; run it from a shell where "
                    "STRATA_NO_ACQUIRE is not set)", 5)
    query = query or fr.query_latest
    if a.check:
        return _check(root, kind, machine, runner, query)
    return _update(a, root, kind, machine, runner, query)


def _git_state(root: Path, runner) -> dict:
    """`git fetch`, then how far the branch and its remote are apart: {"branch", "upstream", "behind", "ahead"} or
    {"error"}.  Only the remote's branches change."""
    if not shutil.which("git"):
        return {"error": "git is not on PATH"}

    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}      # a remote that wants a password fails, never waits

    def git(*args, timeout=30):
        return runner(["git", "-C", str(root), *args], capture_output=True, text=True, encoding="utf-8",
                      errors="replace", stdin=subprocess.DEVNULL, timeout=timeout, env=env)
    out = {}
    try:
        out["branch"] = git("symbolic-ref", "--short", "-q", "HEAD").stdout.strip() or None
        up = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        if up.returncode != 0:
            out["error"] = "this branch follows no remote branch, so git pull has nothing to pull"
            return out
        out["upstream"] = up.stdout.strip()
        f = git("fetch", "--quiet", timeout=180)
        if f.returncode != 0:
            out["error"] = "git fetch failed: " + ((f.stderr or "").strip().splitlines() or ["?"])[-1]
        n = git("rev-list", "--left-right", "--count", "HEAD...@{u}")
        if n.returncode == 0 and len(n.stdout.split()) == 2:
            out["ahead"], out["behind"] = (int(x) for x in n.stdout.split())
    except (OSError, subprocess.SubprocessError) as e:
        out["error"] = f"git did not answer ({type(e).__name__})"
    return out


def _check(root: Path, kind: str, machine: bool, runner, query) -> int:
    from strata_control import fork_release as fr
    if kind == "bundle" and not machine:
        print("strata update --check: a strata-nvfp4 release folder - tools/bundle_update.py --check (update.cmd's)",
              flush=True)
        return runner(updater_cmd(root, kind)[0] + ["--check"], cwd=str(root)).returncode
    _, name, what = updater_cmd(root, kind)
    installed, origin = fr.installed_version(root)
    engine = fr.engine_build(root)
    git = _git_state(root, runner) if kind == "clone" else None
    rel, err = _latest(query, 15)
    newer = bool(rel and installed and fr.is_newer(rel["version"], installed))
    behind = (git or {}).get("behind") or 0
    status = "available" if newer or behind else "current" if rel else "unknown"
    if machine:
        print(json.dumps({"schema_version": 1, "ok": rel is not None, "kind": kind, "updater": name,
                          "status": status, "installed": installed, "installed_from": origin,
                          "engine": engine.get("version"), "latest": rel, "error": err,
                          **({"git": git} if git is not None else {})}, ensure_ascii=False))
        return 0 if rel is not None else 6
    print(f"strata-nvfp4 in {root} (a {KIND_TEXT[kind]}; `strata update` runs {name}: {what})")
    if kind != "bundle":
        where = f" (branch {git['branch']})" if git and git.get("branch") else ""
        print(f"  Code     {fr.release_version(root) or '?'}{where}")
        if git and git.get("error"):
            print(f"           {git['error']}")
        if git and git.get("behind") is not None:
            print(f"           {git['upstream']}: " + (f"{git['behind']} new commit{'s' * (git['behind'] != 1)}, "
                                                       f"{name} pulls them" if git["behind"] else "nothing new"))
    print(f"  Engine   {engine.get('version') or 'none installed'}" +
          (" (this fork's release engine)" if engine.get("fork") else
           " (compiled here)" if engine.get("source") == "local" else " (not this fork's)" if engine else ""))
    if rel:
        print(f"  Release  {rel['tag']} is the newest published: {rel['url']}")
    else:
        print(f"  Release  GitHub could not be asked: {err}")
    if newer:
        print(f"A newer release is published ({rel['tag']}, this is {installed}): run `strata update` to install it.")
    elif behind:
        print("New commits are on the remote: run `strata update` to pull them.")
    elif rel:
        print(f"Up to date ({installed or '?'}).")
    print("Nothing was changed." + (" (git fetch updated only the remote's branches)" if git and "upstream" in git else ""))
    return 0 if rel is not None else 6


def _update(a, root: Path, kind: str, machine: bool, runner, query) -> int:
    from strata_control import fork_release as fr
    cmd, name, what = updater_cmd(root, kind, getattr(a, "force", False))
    # UPDATE.bat ends with `pause`: with no input it goes on at once (setup.py --update asks nothing)
    stdin = subprocess.DEVNULL if machine or (os.name == "nt" and kind != "bundle") else None
    if not machine:
        print(f"strata update: this strata-nvfp4 {KIND_TEXT[kind]} runs {name} - {what} (upstream's signed feed "
              "only with --feed)", flush=True)
        return runner(cmd, cwd=str(root), stdin=stdin).returncode
    before = fr.installed_version(root)[0]
    p = runner(cmd, cwd=str(root), stdin=stdin, capture_output=True, text=True, encoding="utf-8", errors="replace")
    sys.stderr.write((p.stdout or "") + (p.stderr or ""))
    after = fr.installed_version(root)[0]
    rel, _ = _latest(query, 10)
    print(json.dumps({"schema_version": 1, "ok": p.returncode == 0, "kind": kind, "updater": name,
                      "status": "failed" if p.returncode else "updated" if after != before else "done",
                      "installed_before": before, "installed": after, "latest": rel, "exit_code": p.returncode},
                     ensure_ascii=False))
    return p.returncode
