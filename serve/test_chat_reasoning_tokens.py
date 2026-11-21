"""usage.completion_tokens_details.reasoning_tokens on /v1/chat/completions (no GPU or model pack).

    python -m unittest serve.test_chat_reasoning_tokens -v
"""
import json
import unittest
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve
from serve.test_server import CTX, ThinkingEngine

ROOT = Path(__file__).resolve().parents[1]
THOUGHT_TOKENS = len(ThinkingEngine.THOUGHT + "</think>")         # one token a byte; the closing tag is thinking


class ReasoningTokens(unittest.TestCase):
    def start(self, engine):
        tok = engine.tok
        self.svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def chat(self, stream, **extra):
        body = {"model": "x", "max_tokens": 400, "stream": stream,
                "messages": [{"role": "user", "content": "two plus two?"}], **extra}
        req = urllib.request.Request(f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/chat/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
        if not stream:
            return json.loads(raw)["usage"]
        chunks = [json.loads(line[6:]) for line in raw.decode().splitlines() if line.startswith("data: {")]
        return chunks[-1]["usage"]

    def test_counted_in_both_forms(self):
        self.start(ThinkingEngine(ByteTokenizer()))
        for stream in (False, True):
            usage = self.chat(stream)
            reasoning = usage["completion_tokens_details"]["reasoning_tokens"]
            self.assertEqual(reasoning, THOUGHT_TOKENS)
            self.assertLess(reasoning, usage["completion_tokens"])

    def test_counted_across_a_thinking_budget(self):
        # the budget closes the thinking and the answer continues: the count stops at the budget, not at zero
        self.start(ThinkingEngine(ByteTokenizer()))
        reasoning = self.chat(False, reasoning_budget_tokens=50)["completion_tokens_details"]["reasoning_tokens"]
        self.assertGreater(reasoning, 0)
        self.assertLess(reasoning, THOUGHT_TOKENS)

    def test_zero_when_thinking_is_off(self):
        self.start(MockEngine(ByteTokenizer(), "ok", max_context=CTX))
        usage = self.chat(False, reasoning_effort="none")
        self.assertEqual(usage["completion_tokens_details"]["reasoning_tokens"], 0)
        self.assertGreater(usage["completion_tokens"], 0)

if __name__ == "__main__":
    unittest.main()
