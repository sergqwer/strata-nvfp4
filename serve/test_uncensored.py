"""serve/test_uncensored.py - this fork's "uncensored": the honest name of upstream's experimental_speed_projection (the
engine's control vector; setup --uncensored on loads the refusal-direction projection).  Per request, in the run
config (top level, as setup writes it, or the sampling block) and in the shared Chat settings; the engine gets
upstream's cvec key either way.

    python -m unittest serve.test_uncensored -v
"""
from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import (ByteTokenizer, MockEngine, Service, StrataEngine, clean_shared_defaults,  # noqa: E402
                          sampling_defaults_from_config, uncensored_alias)
from serve.frontend import ChatTemplate  # noqa: E402

TEMPLATE = Path(__file__).resolve().parents[1] / "serve" / "chat_template.jinja"


class Alias(unittest.TestCase):
    def test_the_request_field(self):
        for value in (True, False):
            got = uncensored_alias({"uncensored": value, "temperature": 0})
            self.assertIs(got["experimental_speed_projection"], value)
            self.assertEqual(StrataEngine.projection_key(got), f" cvec={int(value)}")
        # upstream's key wins when both are given; anything but a bool is not mapped
        self.assertIs(uncensored_alias({"uncensored": True, "experimental_speed_projection": False})
                      ["experimental_speed_projection"], False)
        self.assertNotIn("experimental_speed_projection", uncensored_alias({"uncensored": "yes"}))
        self.assertEqual(StrataEngine.projection_key(uncensored_alias({})), "")         # absent: the engine's default

    def test_the_config(self):
        self.assertEqual(sampling_defaults_from_config({"uncensored": True}), {"experimental_speed_projection": True})
        self.assertEqual(sampling_defaults_from_config({"sampling": {"uncensored": False, "temperature": 0.6}}),
                         {"experimental_speed_projection": False, "temperature": 0.6})
        # the sampling block's own key wins over the top-level one
        self.assertEqual(sampling_defaults_from_config({"uncensored": True, "sampling": {"uncensored": False}}),
                         {"experimental_speed_projection": False})
        with self.assertRaises(SystemExit):
            sampling_defaults_from_config({"uncensored": "on"})

    def test_the_shared_chat_settings(self):
        self.assertEqual(clean_shared_defaults({"uncensored": False}), {"experimental_speed_projection": False})
        with self.assertRaises(ValueError):
            clean_shared_defaults({"uncensored": 1})


class Run(unittest.TestCase):
    """Service.run: the engine sees upstream's key whichever name the request or the config used."""

    def service(self, defaults=None):
        tok = ByteTokenizer()
        seen = []
        eng = MockEngine(tok, ["ok"])
        real = eng.generate

        def generate(ids, max_new, sampling, cancel, embeddings=None):
            seen.append(dict(sampling or {}))
            return real(ids, max_new, sampling, cancel)
        eng.generate = generate
        svc = Service(eng, tok, ChatTemplate(TEMPLATE), "strata", sampling_defaults=defaults)
        return svc, seen

    def go(self, svc, sampling):
        for _ in svc.run(list(b"hi"), False, None, 4, sampling, threading.Event()):
            pass

    def test_request_and_config(self):
        svc, seen = self.service({"experimental_speed_projection": True})
        self.go(svc, {"uncensored": False})
        self.go(svc, {})
        self.go(svc, {"experimental_speed_projection": False})
        self.assertEqual([s.get("experimental_speed_projection") for s in seen], [False, True, False])


class Metrics(unittest.TestCase):
    """The web app's switch starts at the server's default: /metrics says it (engine.uncensored_default)."""

    def default(self, defaults=None, shared=None):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, ["ok"]), tok, ChatTemplate(TEMPLATE), "strata", sampling_defaults=defaults)
        if shared:
            svc.shared = dict(shared)
        return svc.metrics()["engine"]["uncensored_default"]

    def test_it_follows_the_config(self):
        self.assertIs(self.default(sampling_defaults_from_config({"uncensored": False})), False)
        self.assertIs(self.default(sampling_defaults_from_config({"uncensored": True})), True)
        self.assertIs(self.default(), True)                       # no default: a loaded vector is on (upstream's)
        self.assertIs(self.default({"experimental_speed_projection": True},
                                   clean_shared_defaults({"uncensored": False})), False)   # the shared Chat settings win


if __name__ == "__main__":
    unittest.main()
