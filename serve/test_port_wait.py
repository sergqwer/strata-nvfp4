"""strata-nvfp4: the start's port check waits for a port a stopping server still holds (serve/server.py wait_port_free)."""
import errno
import socket
import threading
import time
import unittest

from serve import server


def held_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    return s, s.getsockname()[1]


class PortWait(unittest.TestCase):
    def test_free_port_at_once(self):
        s, port = held_port()
        s.close()
        t0 = time.monotonic()
        self.assertIsNone(server.wait_port_free("127.0.0.1", port, 5))
        self.assertLess(time.monotonic() - t0, 1.0)

    def test_port_let_go_while_waiting(self):
        s, port = held_port()
        threading.Timer(1.0, s.close).start()
        t0 = time.monotonic()
        self.assertIsNone(server.wait_port_free("127.0.0.1", port, 10))
        self.assertGreaterEqual(time.monotonic() - t0, 0.9)

    def test_port_kept_is_refused_after_the_wait(self):
        s, port = held_port()
        try:
            t0 = time.monotonic()
            e = server.wait_port_free("127.0.0.1", port, 0.6)
            self.assertIsNotNone(e)
            self.assertTrue(server.port_in_use(e))
            self.assertGreaterEqual(time.monotonic() - t0, 0.5)
            self.assertIn("already in use", server.listen_problem("127.0.0.1", port, e))
        finally:
            s.close()

    def test_other_errors_at_once(self):
        calls = []

        def probe():
            calls.append(1)
            raise OSError(errno.EACCES, "denied")
        t0 = time.monotonic()
        e = server.wait_port_free("127.0.0.1", 1, 10, bind=probe)
        self.assertEqual(e.errno, errno.EACCES)
        self.assertEqual(len(calls), 1)
        self.assertLess(time.monotonic() - t0, 0.5)


if __name__ == "__main__":
    unittest.main()
