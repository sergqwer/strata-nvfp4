"""strata-nvfp4 (this fork): its releases on GitHub - their version order, the kind of folder Strata runs from and the
command that updates it, and the server's update-available notice.

The notice only reads.  At most once a day it asks GitHub which release of sergqwer/strata-nvfp4 is the newest (one
small JSON request: no token, no install id, nothing about this PC) and says so; it never downloads or installs
anything.  Updating stays an explicit step: update.cmd in a release folder, UPDATE.bat / update.sh in a clone, or
`strata update`, which runs them.  It is apart from upstream's acquire line (acquisition.py: downloads and installs)
and from its signed engine feed (updater.py).  Off: STRATA_UPDATE_CHECK=0, "update_check": false in the run config,
or serve --offline.  Standard library only: setup.py imports it before any package is installed.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

SLUG = "sergqwer/strata-nvfp4"
LATEST_API = "https://api.github.com/repos/%s/releases/latest" % SLUG
RELEASES = "https://github.com/%s/releases" % SLUG
ENV = "STRATA_UPDATE_CHECK"                   # 0 / false / no / off: no check
OFF = ("0", "false", "no", "off")
EVERY_S = 24 * 3600                           # GitHub is asked at most this often (a failed ask counts too)
CACHE = "update-check.json"                   # in <install>/.update/, beside bundle_update's zip, journal and unpacked-*
USER_CACHE = "strata-nvfp4-update-check.json"  # the per-user fallback (paths.state_base()) when the folder is read-only
UA = "strata-nvfp4-update-check"

# "0.2.0-beta-nvfp4.1": the upstream version, a pre-release word, this fork's number
_VERSION = re.compile(r"\s*(\d+(?:\.\d+)*)(?:-((?!nvfp4\.)[A-Za-z]+\d*))?(?:-nvfp4\.(\d+))?")
_TAG = re.compile(r"v?\d+(?:\.\d+){1,4}(?:-[A-Za-z]+\d*)?(?:-nvfp4\.\d+)?")
_URL = re.compile(re.escape(RELEASES) + r"/tag/[A-Za-z0-9._-]{1,64}")


def version_tuple(v) -> tuple:
    """An engine version as numbers: "0.1.40.3" -> (0, 1, 40, 3); this fork's "0.1.40.3-nvfp4.1" -> (0, 1, 40, 3, 1),
    above the upstream version it is built on (read number by number it was 0.1.40, an engine to replace).  A
    pre-release word between them is skipped: "0.2.0-beta-nvfp4.1" -> (0, 2, 0, 0, 1), "0.2.0-beta" -> (0, 2, 0)."""
    m = _VERSION.match(str(v or ""))
    if not m:
        return ()
    nums = tuple(int(x) for x in m.group(1).split("."))[:5]
    if m.group(3) is not None:
        nums = (nums + (0, 0, 0, 0))[:4] + (int(m.group(3)),)
    return nums


def is_newer(latest, installed) -> bool:
    """`latest` is a later release than `installed`: version_tuple's order; where that is equal, a release without a
    pre-release word is later than one with it (0.2.0-nvfp4.1 after 0.2.0-beta-nvfp4.1)."""
    a, b = version_tuple(latest), version_tuple(installed)
    if not a or not b:
        return False
    if a != b:
        return a > b
    pre = [bool(m and m.group(2)) for m in (_VERSION.match(str(latest)), _VERSION.match(str(installed)))]
    return pre == [False, True]


def _json(p: Path) -> dict:
    try:
        d = json.loads(Path(p).read_text(encoding="utf-8-sig"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def release_version(root) -> str | None:
    """The release of this fork the source in `root` belongs to: release/make_windows_bundle.py's VERSION (its tag is
    v<this>); None without that file (a release folder has none)."""
    try:
        text = (Path(root) / "release" / "make_windows_bundle.py").read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r'^VERSION = "([^"]+)"', text, re.M)
    return m.group(1) if m else None


def install_kind(root) -> str:
    """"bundle": an unpacked release zip (a release engine's BUILD.json, tools/bundle_update.py and start-server.cmd,
    no .git); "clone": a git clone; "source": the source without git (GitHub's zip of main)."""
    root = Path(root)
    if (root / ".git").exists():
        return "clone"
    if _json(root / "engine" / "BUILD.json").get("source") == "release" and \
            (root / "tools" / "bundle_update.py").is_file() and (root / "start-server.cmd").is_file():
        return "bundle"
    return "source"


def updater_name(kind: str) -> str:
    return "update.cmd" if kind == "bundle" else "UPDATE.bat" if os.name == "nt" else "./update.sh"


def update_command(root, kind: str | None = None) -> str:
    """What updates the folder `root`, as a sentence's end."""
    kind = kind or install_kind(root)
    if kind == "bundle":
        return "close the server and run update.cmd in %s" % root
    return "run %s in %s (or `strata update`)" % (updater_name(kind), root)


def engine_build(root, exe=None) -> dict:
    """The engine's BUILD.json (beside `exe`, else <root>/engine/), {} when there is none."""
    exe = Path(exe) if exe else Path(root) / "engine" / ("strata.exe" if os.name == "nt" else "strata")
    return _json(exe.parent / "BUILD.json")


def installed_version(root, exe=None) -> tuple[str | None, str]:
    """(the installed release's version, where it was read).  The engine's BUILD.json when it is this fork's (a release
    folder's engine, or setup's ready-made one in a clone); else - an engine compiled here says only CMake's upstream
    version, an engine of upstream's is not this fork's - the release the source belongs to (release_version: what a
    `git pull` moves); (None, "") when neither says."""
    meta = engine_build(root, exe)
    if meta.get("fork") and version_tuple(meta.get("version")):
        return str(meta["version"]), "engine BUILD.json"
    v = release_version(root)
    return (v, "release/make_windows_bundle.py") if version_tuple(v) else (None, "")


def _release(d) -> dict | None:
    """{"tag", "version", "url"} from GitHub's answer or the cache, or None when it is not a release of this fork."""
    if not isinstance(d, dict):
        return None
    tag = d.get("tag_name", d.get("tag"))
    if not isinstance(tag, str) or not _TAG.fullmatch(tag):
        return None
    url = d.get("html_url", d.get("url"))
    return {"tag": tag, "version": tag.lstrip("v"),
            "url": url if isinstance(url, str) and _URL.fullmatch(url) else RELEASES + "/tag/" + tag}


def query_latest(opener=None, timeout: float = 5.0) -> dict:
    """GitHub's latest release of this fork, {"tag", "version", "url"}; OSError or ValueError when it cannot say.
    The serving side's one URL of its own (tests/terminal/test_no_acquisition.py lists it): a version, read only."""
    req = urllib.request.Request(LATEST_API, headers={"User-Agent": UA, "Accept": "application/vnd.github+json"})
    try:
        with (opener(req, timeout=timeout) if opener else urllib.request.urlopen(req, timeout=timeout)) as r:
            body = r.read(4 << 20)
    except urllib.error.HTTPError as e:
        raise OSError("GitHub answered HTTP %s" % e.code) from None
    except http.client.HTTPException as e:
        raise OSError("GitHub's answer broke off (%s)" % type(e).__name__) from None
    rel = _release(json.loads(body.decode("utf-8")))
    if rel is None:
        raise ValueError("GitHub's answer names no release of %s" % SLUG)
    return rel


def cache_files(root) -> list[Path]:
    """Where the last ask is kept: the install folder's .update/, else the per-user state folder."""
    files = [Path(root) / ".update" / CACHE]
    try:
        from strata_control import paths
        files.append(paths.state_base() / USER_CACHE)
    except Exception:  # noqa: BLE001 - no home folder: the install folder only
        pass
    return files


def _read_cache(root) -> dict | None:
    best = None
    for p in cache_files(root):
        d = _json(p)
        t = d.get("checked_at")
        if isinstance(t, (int, float)) and not isinstance(t, bool) and (best is None or t > best["checked_at"]):
            best = d
    return best


def _write_cache(root, data: dict) -> Path | None:
    for p in cache_files(root):
        tmp = p.with_name("%s.%d.tmp" % (p.name, os.getpid()))
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
            os.replace(tmp, p)
            return p
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass
    return None


def latest(root, *, now: float | None = None, opener=None, timeout: float = 5.0,
           every: float = EVERY_S) -> tuple[dict | None, str, str | None]:
    """(the latest release or None, "cache" or "github", why the last ask failed or None).  Within `every` seconds of
    the last ask the cache answers; else GitHub is asked once and the answer (a failure too) is kept."""
    now = time.time() if now is None else now
    cached = _read_cache(root)
    if cached is not None and 0 <= now - cached["checked_at"] < every:
        err = cached.get("error")
        return _release(cached.get("latest")), "cache", err if isinstance(err, str) else None
    try:
        rel, err = query_latest(opener, timeout), None
    except (OSError, ValueError) as e:            # offline, a timeout, HTTP 403 (rate limit), not JSON
        rel, err = (_release(cached.get("latest")) if cached else None), (str(e) or type(e).__name__)[:200]
    _write_cache(root, {"checked_at": now, "latest": rel, "error": err})
    return rel, "github", err


def check(root, exe=None, **kw) -> dict:
    """The notice for the install in `root` (its engine `exe`): {"status": "available" | "current" | "unknown",
    "installed", "installed_from", "latest", "tag", "url", "kind", "command", "source", "error"}.  Without an installed
    version GitHub is not asked."""
    installed, origin = installed_version(root, exe)
    kind = install_kind(root)
    out = {"status": "unknown", "installed": installed, "installed_from": origin, "latest": None, "tag": None,
           "url": RELEASES, "kind": kind, "command": update_command(root, kind), "source": None, "error": None}
    if installed is None:
        return out
    rel, out["source"], out["error"] = latest(root, **kw)
    if rel is not None:
        out.update(latest=rel["version"], tag=rel["tag"], url=rel["url"],
                   status="available" if is_newer(rel["version"], installed) else "current")
    return out


def disabled(cfg: dict | None = None, environ=None, offline: bool = False) -> str | None:
    """Why the server does not check, or None: STRATA_UPDATE_CHECK=0, "update_check": false, --offline."""
    environ = os.environ if environ is None else environ
    if str(environ.get(ENV, "")).strip().lower() in OFF:
        return "%s=%s" % (ENV, environ.get(ENV))
    if (cfg or {}).get("update_check") is False:
        return '"update_check": false in the run config'
    if offline:
        return "--offline"
    return None


def log_line(n: dict | None) -> str | None:
    """The server's line for a check: the newer release when there is one, a quiet one when GitHub could not be
    asked just now, else None (up to date, or the answer of a check within the last day)."""
    if not n:
        return None
    if n["status"] == "available":
        return ("[strata] a newer strata-nvfp4 release is published: %s (this is %s) - %s; notes: %s"
                % (n["tag"], n["installed"], n["command"], n["url"]))
    if n["source"] == "github" and n["error"]:
        return ("[strata] update check: could not ask GitHub (%s); asked again in a day (%s=0 turns it off)"
                % (n["error"], ENV))
    return None


def web_view(n: dict | None) -> dict | None:
    """What /metrics and /v1/status carry: the newer release, or None."""
    if not n or n.get("status") != "available":
        return None
    return {k: n[k] for k in ("latest", "tag", "url", "installed", "kind", "command")}
