"""strata-nvfp4: serve/app_ui.py - the new interface beside a running server (python -m unittest serve.test_app_ui)."""
from __future__ import annotations

import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from serve import app_ui


class FakeSvc:
    api_key = "k-secret,k2"
    model = "qwen3.8-flash-next-gptq-nvfp4"


class AppUiTest(unittest.TestCase):
    def test_wanted(self):
        self.assertFalse(app_ui.wanted({}))
        self.assertFalse(app_ui.wanted({"STRATA_APP_UI": "0"}))
        self.assertTrue(app_ui.wanted({"STRATA_APP_UI": "1"}))

    def test_find_names_what_is_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(app_ui.find(root, {})[1], "app\\host is missing")
            (root / "app" / "host" / "server").mkdir(parents=True)
            (root / "app" / "host" / "server" / "index.mjs").write_text("")
            self.assertIn("not built", app_ui.find(root, {})[1])
            (root / "app" / "host" / "dist").mkdir()
            (root / "app" / "host" / "dist" / "index.html").write_text("")
            self.assertIn("is missing", app_ui.find(root, {})[1])
            (root / "node").mkdir()
            (root / "node" / "node.exe").write_bytes(b"")
            (root / "node" / "node").write_bytes(b"")
            found, why = app_ui.find(root, {})
            self.assertIsNone(why)
            self.assertEqual(found[1], root / "app" / "host")

    def test_host_env_attached_loopback_no_downloads_key_only_in_env(self):
        env = app_ui.host_env("http://127.0.0.1:8080/v1", "k-secret", "m", port=8791, home=Path("H"), version="0.2.0-beta-nvfp4.2",
                              update_off=True, environ={"STRATA_API_KEY": "x", "STRATA_ATTACH_KEY_FILE": "f", "PATH": "p"})
        self.assertEqual(env["STRATA_ATTACH_URL"], "http://127.0.0.1:8080/v1")
        self.assertEqual(env["STRATA_ATTACH_KEY"], "k-secret")
        self.assertNotIn("STRATA_API_KEY", env)
        self.assertNotIn("STRATA_ATTACH_KEY_FILE", env)
        self.assertEqual((env["STRATA_BIND"], env["STRATA_NO_ACQUIRE"], env["STRATA_REAL_ENGINE"], env["STRATA_EXIT_WITH_PARENT"]), ("127.0.0.1", "1", "1", "1"))
        self.assertEqual(env["STRATA_UPDATE_CHECK"], "0")
        self.assertEqual(env["STRATA_APP_VERSION"], "0.2.0-beta-nvfp4.2")

    def test_start_without_node_says_why_and_starts_nothing(self):
        said = []
        with mock.patch.object(app_ui, "find", return_value=(None, "node\\node.exe is missing")):
            self.assertIsNone(app_ui.start(FakeSvc(), "127.0.0.1", 8080, environ={"STRATA_APP_UI": "1"}, say=said.append,
                                           popen=lambda *a, **k: self.fail("started")))
        self.assertEqual(len(said), 1)
        self.assertIn("node\\node.exe is missing", said[0])

    def test_start_spawns_the_host_in_the_job_key_never_on_the_command_line(self):
        calls, contained, said = [], [], []

        class Proc:
            def poll(self):
                return 0                                   # ends at once: the announcer says it did not start

        def popen(argv, **kw):
            calls.append((argv, kw))
            return Proc()
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(app_ui, "find", return_value=((Path("node.exe"), Path("host")), None)), \
                mock.patch.object(app_ui, "port_busy", return_value=False), \
                mock.patch.object(app_ui, "data_home", return_value=Path(d)):
            proc = app_ui.start(FakeSvc(), "0.0.0.0", 8080, environ={"STRATA_APP_UI": "1", "STRATA_APP_UI_OPEN": "0"}, popen=popen,
                                contain=contained.append, say=said.append)
            self.assertIsNotNone(proc)
            argv, kw = calls[0]
            self.assertFalse(any("k-secret" in str(x) for x in argv))
            self.assertEqual(kw["env"]["STRATA_ATTACH_KEY"], "k-secret")
            self.assertEqual(kw["env"]["STRATA_ATTACH_URL"], "http://127.0.0.1:8080/v1")
            self.assertEqual(contained, [proc])

    def test_busy_port_is_not_taken(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(16)
        try:
            port = s.getsockname()[1]
            self.assertTrue(app_ui.port_busy(port))
            said = []
            with mock.patch.object(app_ui, "find", return_value=((Path("node.exe"), Path("host")), None)):
                self.assertIsNone(app_ui.start(FakeSvc(), "127.0.0.1", 8080, environ={"STRATA_APP_UI": "1", "STRATA_APP_UI_PORT": str(port)},
                                               say=said.append, popen=lambda *a, **k: self.fail("started")))
            self.assertIn("already in use", said[0])
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()
