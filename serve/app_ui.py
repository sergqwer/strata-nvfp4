"""serve/app_ui.py - strata-nvfp4: the new interface (upstream 0.2.0's app page) beside a running server.

start-server.cmd sets STRATA_APP_UI=1; once the server answers, `start()` runs the App Host (app/host/server/index.mjs on
the release's own node/node.exe) on http://127.0.0.1:8791, attached to this server: the host starts no engine, it
uses this one (apps/host/server/attach.mjs). The key reaches it in its environment, never on a command line.

  * The host is in this server's job object (winjob.contain) and watches this process (STRATA_EXIT_WITH_PARENT): it
    ends with the server, however the server ends - Ctrl+C, the window closed, a crash.
  * No node.exe (or no app/ folder): one line says so and the server's own page stays the interface.
  * The port is fixed (8791; STRATA_APP_UI_PORT), because the page keeps its chats in the browser per address.
  * STRATA_APP_UI_OPEN=0: the browser is not opened.
Standard library only.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORT = 8791


def wanted(environ=None) -> bool:
    environ = os.environ if environ is None else environ
    return str(environ.get("STRATA_APP_UI", "")).strip().lower() not in ("", "0", "false", "no", "off")


def config_environ(cfg: dict | None, environ=None) -> dict:
    """The environment start() reads, with the run config's "app_ui": true / "app_ui_open": false where the environment
    says nothing: a tray's per-model config turns the new interface on with its model, without a browser tab at every
    load (the tray starts the server; the interface ends with it)."""
    env = dict(os.environ if environ is None else environ)
    cfg = cfg or {}
    if "STRATA_APP_UI" not in env and cfg.get("app_ui") is True:
        env["STRATA_APP_UI"] = "1"
    if "STRATA_APP_UI_OPEN" not in env and cfg.get("app_ui_open") is False:
        env["STRATA_APP_UI_OPEN"] = "0"
    return env


def find(root: Path = ROOT, environ=None):
    """(node, host dir) or (None, why)."""
    environ = os.environ if environ is None else environ
    for host in (root / "app" / "host", root / "apps" / "host"):           # a release folder, a clone
        if (host / "server" / "index.mjs").is_file():
            break
    else:
        return None, "app\\host is missing"
    if not (host / "dist" / "index.html").is_file():
        return None, f"the page is not built ({host / 'dist'} is missing)"
    node = Path(environ.get("STRATA_NODE") or root / "node" / ("node.exe" if os.name == "nt" else "node"))
    if not node.is_file() and host.parent.name == "apps" and not environ.get("STRATA_NODE"):
        import shutil                                 # a clone: the Node on PATH (24.12 or newer, apps/host/package.json)
        node = Path(shutil.which("node") or node)
    if not node.is_file():
        return None, f"{node.relative_to(root) if node.is_relative_to(root) else node} is missing"
    return (node, host), None


def port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def data_home(environ=None) -> Path:
    environ = os.environ if environ is None else environ
    if os.name == "nt":
        base = Path(environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "strata-nvfp4" / "app-ui"


def host_env(url: str, key: str, name: str, *, port: int, home: Path, version: str | None, update_off: bool, environ=None) -> dict:
    """The host's environment: attached to `url`, loopback only, nothing downloaded, this fork's update check."""
    environ = os.environ if environ is None else environ
    env = {k: v for k, v in environ.items() if not k.startswith("STRATA_ATTACH_") and k not in ("STRATA_API_KEY", "NODE_OPTIONS")}
    env.update({
        "STRATA_ATTACH_URL": url, "STRATA_ATTACH_KEY": key, "STRATA_ATTACH_NAME": name,
        "STRATA_PORT": str(port), "STRATA_BIND": "127.0.0.1", "STRATA_UI_MODE": "browser", "STRATA_UI_REASON": "chosen",
        "STRATA_REAL_ENGINE": "1", "STRATA_NO_ACQUIRE": "1", "STRATA_NO_BUNDLED_ENGINE": "1", "STRATA_EXIT_WITH_PARENT": "1",
        "STRATA_HOME": str(home), "STRATA_MODELS_DIR": str(home / "models"),
        "STRATA_FORK_ROOT": str(ROOT), "STRATA_FORK_PYTHON": sys.executable,
    })
    if version:
        env["STRATA_APP_VERSION"] = version
    if update_off:
        env["STRATA_UPDATE_CHECK"] = "0"
    return env


def start(svc, here: str, bound_port: int, *, version: str | None = None, update_off: bool = False, environ=None, popen=subprocess.Popen,
          open_browser=None, contain=None, say=None):
    """Start the host beside this server; returns the process or None. Prints what happened, never the key."""
    environ = os.environ if environ is None else environ
    say = say or (lambda line: print(line, flush=True))
    if not wanted(environ):
        return None
    found, why = find(ROOT, environ)
    old = f"http://{here}:{bound_port}/"
    if not found:
        say(f"       new interface: not started ({why}); the page above is the interface")
        return None
    node, host = found
    port = int(environ.get("STRATA_APP_UI_PORT") or DEFAULT_PORT)
    if port_busy(port):
        say(f"       new interface: port {port} is already in use (another Strata interface?); not started - the page above works")
        return None
    keys = svc.api_key if isinstance(svc.api_key, (list, tuple)) else [k.strip() for k in str(svc.api_key or "").split(",") if k.strip()]
    home = data_home(environ)
    home.mkdir(parents=True, exist_ok=True)
    env = host_env(f"http://{'127.0.0.1' if here in ('0.0.0.0', '::', '') else here}:{bound_port}/v1", keys[0] if keys else "",
                   str(getattr(svc, "model", "") or ""), port=port, home=home, version=version, update_off=update_off, environ=environ)
    log = open(home / "app-ui.log", "ab")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = popen([str(node), str(host / "server" / "index.mjs"), "--dist", str(host / "dist")], cwd=str(host), env=env,
                     stdin=subprocess.DEVNULL, stdout=log, stderr=log, creationflags=flags)
    except OSError as e:
        say(f"       new interface: could not start {node} ({e}); the page above works")
        return None
    finally:
        log.close()
    if contain is None:
        from serve import winjob
        contain = winjob.contain
    contain(proc)                                      # ends with this server, however it ends

    url = f"http://127.0.0.1:{port}"

    def announce():
        deadline = time.time() + 60
        while time.time() < deadline and proc.poll() is None:
            try:
                with urllib.request.urlopen(url + "/api/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except OSError:
                time.sleep(0.5)
        else:
            say(f"       new interface: did not start (see {home / 'app-ui.log'}); the page above works")
            return
        say(f"New interface: {url}  (the old page stays at {old})")
        if str(environ.get("STRATA_APP_UI_OPEN", "1")).strip().lower() not in ("0", "false", "no", "off"):
            (open_browser or _open)(url + "/")

    threading.Thread(target=announce, daemon=True, name="strata-app-ui").start()
    return proc


def _open(url: str) -> None:
    import webbrowser
    webbrowser.open(url)
