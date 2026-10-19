"""This fork's update notice in the server: start_update_check runs on its own thread (the start and the requests never
wait for it), writes one line when a newer strata-nvfp4 release is published and puts it in GET /metrics and GET
/v1/status, which the web page shows; STRATA_UPDATE_CHECK=0, "update_check": false and --offline turn it off.  The
check itself is strata_control/fork_release.py (tools/test_fork_update_check.py); here it is a stand-in: no network.

    python -m unittest serve.test_update_notice -v
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve import runconfig
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve, start_update_check

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "serve" / "web"
NEWER = {"status": "available", "installed": "0.2.0-beta-nvfp4.1", "installed_from": "engine BUILD.json",
         "latest": "0.2.0-beta-nvfp4.2", "tag": "v0.2.0-beta-nvfp4.2",
         "url": "https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.2.0-beta-nvfp4.2", "kind": "bundle",
         "command": "close the server and run update.cmd in C:\\strata-nvfp4", "source": "github", "error": None}


def service() -> Service:
    tok = ByteTokenizer()
    return Service(MockEngine(tok, "hi"), tok, ChatTemplate(Path(__file__).with_name("chat_template.jinja")))


class StartCheck(unittest.TestCase):
    def setUp(self):
        p = mock.patch.dict(os.environ)
        p.start()
        self.addCleanup(p.stop)
        os.environ.pop("STRATA_UPDATE_CHECK", None)

    def run_check(self, result, cfg=None, offline=False):
        svc, seen, out = service(), [], io.StringIO()

        def check(root, exe):
            seen.append((root, exe))
            if isinstance(result, BaseException):
                raise result
            return result
        with contextlib.redirect_stdout(out):
            t = start_update_check(svc, cfg or {}, "C:/x/engine/strata.exe", offline, check=check)
            if t is not None:
                t.join(10)
        return svc, seen, out.getvalue(), t

    def test_a_newer_release_is_one_line_and_a_field(self):
        svc, seen, out, t = self.run_check(NEWER)
        self.assertTrue(t.daemon)
        self.assertEqual(seen, [(ROOT, "C:/x/engine/strata.exe")])
        self.assertEqual(out, "[strata] a newer strata-nvfp4 release is published: v0.2.0-beta-nvfp4.2 (this is "
                              "0.2.0-beta-nvfp4.1) - close the server and run update.cmd in C:\\strata-nvfp4; notes: "
                              "https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.2.0-beta-nvfp4.2\n")
        self.assertEqual(svc.update_notice["tag"], "v0.2.0-beta-nvfp4.2")
        self.assertNotIn("error", svc.update_notice)

    def test_up_to_date_says_nothing(self):
        svc, _, out, _ = self.run_check({**NEWER, "status": "current", "latest": "0.2.0-beta-nvfp4.1"})
        self.assertEqual((out, svc.update_notice), ("", None))

    def test_a_failing_check_never_stops_the_server(self):
        svc, _, out, t = self.run_check(RuntimeError("boom"))
        self.assertEqual((out, svc.update_notice), ("", None))
        self.assertFalse(t.is_alive())

    def test_the_start_does_not_wait_for_it(self):
        gate, svc = threading.Event(), service()
        t = start_update_check(svc, {}, "x", check=lambda root, exe: gate.wait(10) and NEWER)
        self.assertTrue(t.is_alive())                     # returned while GitHub "answers"
        with contextlib.redirect_stdout(io.StringIO()):
            gate.set()
            t.join(10)
        self.assertEqual(svc.update_notice["latest"], "0.2.0-beta-nvfp4.2")

    def test_off_switches(self):
        for cfg, env, offline in (({"update_check": False}, None, False), ({}, "0", False), ({}, "off", False),
                                  ({}, None, True)):
            with self.subTest(cfg=cfg, env=env, offline=offline), mock.patch.dict(os.environ):
                if env is not None:
                    os.environ["STRATA_UPDATE_CHECK"] = env
                svc, seen, out, t = self.run_check(NEWER, cfg, offline)
                self.assertEqual((t, seen, out, svc.update_notice), (None, [], "", None))

    def test_an_owned_server_does_not_ask(self):
        # --owner-fd: the Terminal's or the app's runtime, an engine update's test start - never a network ask
        svc, seen = service(), []
        self.assertIsNone(start_update_check(svc, {}, "x", check=lambda root, exe: seen.append(1), owned=True))
        self.assertEqual(seen, [])


class Endpoints(unittest.TestCase):
    def setUp(self):
        self.svc = service()
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read())

    def test_metrics_and_status_carry_the_notice(self):
        self.assertIsNone(self.get("/metrics")["update"])
        self.assertIsNone(self.get("/v1/status")["update"])
        from strata_control import fork_release
        self.svc.update_notice = fork_release.web_view(NEWER)
        for path in ("/metrics", "/v1/status"):
            u = self.get(path)["update"]
            self.assertEqual((u["tag"], u["installed"], u["url"]),
                             ("v0.2.0-beta-nvfp4.2", "0.2.0-beta-nvfp4.1", NEWER["url"]))
        self.assertNotIn("update", self.get("/health"))   # the unauthenticated route says nothing about versions


class Config(unittest.TestCase):
    def test_update_check_is_true_or_false(self):
        import serve.server as S

        class Reached(Exception):
            pass

        def fake_serve(svc, host=None, port=None):
            raise Reached
        for value, ok in ((False, True), (True, True), ("no", False), (0, False)):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as d, \
                    mock.patch.object(S, "serve", fake_serve), contextlib.redirect_stdout(io.StringIO()):
                p = Path(d) / "cfg.json"
                p.write_text(json.dumps({"update_check": value}), encoding="utf-8")
                with mock.patch.object(sys, "argv", ["server.py", "--engine", "mock", "--port", "0", "--config", str(p)]):
                    if ok:
                        with self.assertRaises(Reached):
                            S.main()
                    else:
                        with self.assertRaisesRegex(SystemExit, "update_check"):
                            S.main()

    def test_the_settings_view_lists_it(self):
        got = {k["key"]: k for k in runconfig.view({"args": []}, "x/strata-q.json")["keys"]}
        self.assertEqual((got["update_check"]["kind"], got["update_check"]["default"]), ("bool", "default: on"))
        new, changed = runconfig.apply({"args": []}, {"update_check": False})
        self.assertEqual((changed, new["update_check"]), (["update_check"], False))


class WebPage(unittest.TestCase):
    def test_the_page_has_the_notice(self):
        html = (WEB / "index.html").read_text(encoding="utf-8")
        for el in ('id="update-note"', 'id="update-link"', 'id="update-hide"', 'rel="noopener noreferrer"'):
            self.assertIn(el, html)
        js = (WEB / "app.js").read_text(encoding="utf-8")
        self.assertIn("renderUpdate(m.update)", js)
        self.assertIn('store.get("update-hidden"', js)

    @unittest.skipUnless(shutil.which("node"), "node is not installed")
    def test_app_js_parses(self):
        r = subprocess.run(["node", "--check", str(WEB / "app.js")], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
