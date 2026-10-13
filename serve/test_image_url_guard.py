"""serve/test_image_url_guard.py - an image URL is fetched only from a public address, on every redirect too, and a
local image file that is missing answers what an unreadable one answers.

From shoftee's fork (07abd70a): the URL is the caller's, so the server fetched whatever it was pointed at - the
link-local metadata address a cloud instance keeps its credentials on, a service on localhost, a host on the LAN -
and the model read the bytes back to the caller.  Local file paths, data: URLs and public URLs work as before.

    python -m unittest serve.test_image_url_guard -v
"""
import contextlib
import http.server
import socket
import sys
import tempfile
import threading
import unittest
import urllib.parse
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve.server as server  # noqa: E402
from serve.server import IMAGE_UNREADABLE, Vision, _is_global, _require_public  # noqa: E402


def resolves_to(*addresses):
    """socket.getaddrinfo answering `addresses` for any name (no DNS is asked in these tests)."""
    def fake(host, port, *args, **kwargs):
        return [(socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, 0)) for a in addresses]
    return mock.patch.object(server.socket, "getaddrinfo", side_effect=fake)


class PublicOnly(unittest.TestCase):
    def test_private_and_local_addresses_are_refused(self):
        for url in ("http://169.254.169.254/latest/meta-data/", "http://127.0.0.1:8081/v1/models",
                    "http://10.0.0.5/x.png", "http://192.168.1.1/x.png", "http://172.16.0.1/x.png",
                    "http://100.64.0.1/x.png", "http://0.0.0.0/x.png", "http://[::1]/x.png", "http://[fd00::1]/x.png",
                    "http://[fe80::1]/x.png", "http://[::ffff:127.0.0.1]/x.png", "http://[64:ff9b::a9fe:a9fe]/x",
                    "http://[2002:a00:1::]/x.png"):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "not a public address"):
                _require_public(url)

    def test_a_name_is_judged_by_every_address_it_resolves_to(self):
        with resolves_to("127.0.0.1", "::1"), self.assertRaisesRegex(ValueError, "not a public address"):
            _require_public("http://localhost:8095/x.png")
        with resolves_to("93.184.215.14", "10.1.2.3"), self.assertRaisesRegex(ValueError, "not a public address"):
            _require_public("https://mixed.example/x.png")      # one public and one private address: refused
        with resolves_to("93.184.215.14", "2606:2800:21f:cb07:6820:80da:af6b:8b2c"):
            _require_public("https://example.com/x.png")
        with mock.patch.object(server.socket, "getaddrinfo", side_effect=socket.gaierror("no such host")), \
                self.assertRaisesRegex(ValueError, "does not resolve"):
            _require_public("https://nowhere.invalid/x.png")

    def test_only_http_and_https(self):
        for url in ("file:///etc/hostname", "ftp://8.8.8.8/x.png", "gopher://8.8.8.8/"):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "http or https"):
                _require_public(url)
        with self.assertRaisesRegex(ValueError, "no host"):
            _require_public("http:///x.png")

    def test_public_addresses_pass(self):
        for url in ("http://8.8.8.8/a.png", "https://[2001:4860:4860::8888]/a.png"):
            with self.subTest(url=url):
                _require_public(url)
        self.assertTrue(_is_global("8.8.8.8"))
        self.assertFalse(_is_global("169.254.169.254"))
        self.assertFalse(_is_global("not an address"))

    def test_nothing_is_fetched_from_a_refused_url(self):
        with mock.patch.object(server.urllib.request.OpenerDirector, "open") as opened:
            with self.assertRaisesRegex(ValueError, "not a public address"):
                Vision.load("http://169.254.169.254/latest/meta-data/iam/security-credentials/")
        opened.assert_not_called()


class Redirects(unittest.TestCase):
    """The test server is on loopback; it stands in for a public host here (_is_global answers True for 127.0.0.1
    only), so what is tested is the redirect check alone."""

    def setUp(self):
        class Files(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                path = urllib.parse.urlsplit(self.path).path
                if path.startswith("/to/"):
                    self.send_response(302)
                    self.send_header("Location", urllib.parse.unquote(path[len("/to/"):]))
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                elif path == "/x.png":
                    self.send_response(200)
                    self.send_header("Content-Length", "3")
                    self.end_headers()
                    self.wfile.write(b"png")
                else:
                    self.send_error(404)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Files)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        real = server._is_global
        self.public = mock.patch.object(server, "_is_global", side_effect=lambda a: a == "127.0.0.1" or real(a))

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def to(self, target):
        return self.base + "/to/" + urllib.parse.quote(target, safe="")

    def test_a_redirect_to_a_private_address_is_refused(self):
        with self.public:
            for target in ("http://169.254.169.254/latest/meta-data/", "http://10.0.0.5/x.png", "http://[::1]/x.png"):
                with self.subTest(target=target), self.assertRaisesRegex(ValueError, "not a public address"):
                    Vision.load(self.to(target))

    def test_a_redirect_to_a_public_host_is_followed(self):
        with self.public:
            self.assertEqual(Vision.load(self.to(self.base + "/x.png")), b"png")
            self.assertEqual(Vision.load(self.base + "/x.png"), b"png")


class LocalFiles(unittest.TestCase):
    def test_a_missing_file_answers_like_an_unreadable_one(self):
        """Two answers would let a caller map this computer's files one path at a time."""
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "x.png"
            f.write_bytes(b"x")
            answers = []
            for src, denied in ((str(Path(d) / "no-such-file.png"), False), (str(f), True), (d, False),
                                ("not a path at all", False)):
                with mock.patch.object(Path, "read_bytes", side_effect=PermissionError("denied")) if denied else \
                        contextlib.nullcontext():
                    with self.assertRaises(ValueError) as e:
                        Vision.load(src)
                    answers.append(str(e.exception))
        self.assertEqual(set(answers), {IMAGE_UNREADABLE})

    def test_local_files_and_data_urls_load_as_before(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "x.png"
            f.write_bytes(b"\x89PNG\r\n\x1a\n")
            for src in (str(f), "file://" + str(f), "data:image/png;base64,iVBORw0KGgo="):
                with self.subTest(src=src):
                    self.assertEqual(Vision.load(src), b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
