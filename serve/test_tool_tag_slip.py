"""serve/test_tool_tag_slip.py - fork #24: a parameter tag that lost its `>` before its line break.

Qwen3.8-Flash-Next at long contexts sometimes writes `<parameter=NAME` or `</parameter` straight into a line break.
The parser took everything up to the next `>` as the name, or never ended the value; with stream_tools it had already
sent that much JSON and never closed it, so a client that joins the streamed arguments got text that is not JSON.  A
value left open also turned the model's end of turn into STOP_TEXT_MAX copies of its text (EOS literal).  Inside a
`<tool_call>` such a tag is now the tag; outside a call the text stays as written, and a call that never closes still
fails loudly (#211).  The shapes are the four failures of the report (A-D), rebuilt with filler text.

    python -m unittest serve.test_tool_tag_slip -v
"""
import json
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve.frontend import OutputParser  # noqa: E402


def tool(name, *params):
    return {"name": name, "parameters": {"type": "object", "properties": {p: {"type": "string"} for p in params},
                                         "required": [params[0]]}}


TOOLS = [tool("write_file", "content", "overwrite", "path"), tool("think", "thought"),
         tool("write_document", "doc", "do", "heading")]
HEAD = "ok\n</think>\n\n<tool_call>\n"
TAIL = "\n</function>\n</tool_call>"

SHAPES = {
    # A: the opener of a third parameter lost its `>`; its value is empty
    "A": (HEAD + "<function=write_file>\n<parameter=content>\nline one\nline two\n</parameter>\n<parameter=path>\n"
          "notes/a-file.txt\n</parameter>\n<parameter=content_bk\n</parameter>" + TAIL,
          {"content": "line one\nline two", "path": "notes/a-file.txt", "content_bk": ""}),
    # B: the only parameter's opener lost its `>`
    "B": (HEAD + "<function=think>\n<parameter=thought\nfirst: second: third: a longer piece of thought\n</parameter>"
          + TAIL, {"thought": "first: second: third: a longer piece of thought"}),
    # C: the last parameter's closer lost its `>`
    "C": (HEAD + "<function=write_document>\n<parameter=doc>\nreport\n</parameter>\n<parameter=do>\nappend\n"
          "</parameter>\n<parameter=heading>\nA heading here\n</parameter" + TAIL,
          {"doc": "report", "do": "append", "heading": "A heading here"}),
    "well formed": (HEAD + "<function=think>\n<parameter=thought>\nsome thought\n</parameter>" + TAIL,
                    {"thought": "some thought"}),
}
# D: no slip - the model wrote `</think>` and its reply inside the value, then ended the turn there: left loud
D = HEAD + "<function=think>\n<parameter=thought>\nthe thought itself\n</think>\n\nThe reply the model meant to give."
PROSE = ("ok\n</think>\n\nThe format has `<parameter=NAME>` openers. Written badly it reads <parameter=name\nand "
         "</parameter\nwhich is wrong.")


def pieces(text, how, rng):
    if how == "whole":
        return [text]
    if how == "chars":
        return list(text)
    out, i = [], 0
    while i < len(text):
        n = rng.randint(1, 4)
        out.append(text[i:i + n])
        i += n
    return out


def run(text, how, rng, stream=True):
    p = OutputParser(thinking=True, tools=TOOLS, stream_tools=stream, recover=True)
    ev = [e for c in pieces(text, how, rng) for e in p.feed(c)] + p.finish("stop")
    return p, ev


class Slip(unittest.TestCase):
    def test_the_report_shapes(self):
        rng = random.Random(24)
        for name, (text, want) in SHAPES.items():
            for how in ("whole", "chars", "chunks", "chunks"):
                for stream in (True, False):
                    with self.subTest(shape=name, how=how, stream=stream):
                        p, ev = run(text, how, rng, stream)
                        calls = [e.call for e in ev if e.kind == "tool_call"]
                        self.assertEqual(len(calls), 1)
                        self.assertEqual(calls[0].arguments, want)
                        if stream:
                            self.assertEqual(json.loads("".join(e.text for e in ev if e.kind == "tool_args")), want)
                        self.assertFalse([e for e in ev if e.kind == "content" and e.text.strip()])
                        self.assertEqual(bool(p.notes), name != "well formed")

    def test_a_call_that_never_closes_stays_loud(self):
        rng = random.Random(211)
        for how in ("whole", "chars", "chunks"):
            with self.subTest(how=how):
                _, ev = run(D, how, rng)
                self.assertFalse([e for e in ev if e.kind == "tool_call"])
                with self.assertRaises(ValueError):
                    json.loads("".join(e.text for e in ev if e.kind == "tool_args"))

    def test_prose_outside_a_call_is_left_as_written(self):
        rng = random.Random(5)
        for how in ("whole", "chars", "chunks"):
            with self.subTest(how=how):
                p, ev = run(PROSE, how, rng)
                self.assertEqual("".join(e.text for e in ev if e.kind == "content"),
                                 PROSE.split("</think>\n\n", 1)[1])
                self.assertFalse(p.notes)

    def test_the_end_of_turn_after_a_slipped_closer_ends_the_reply(self):
        # C up to its `</tool_call>`: the value is closed now, so EOS literal does not write the end of turn as text
        text = SHAPES["C"][0]
        p = OutputParser(thinking=True, tools=TOOLS, stream_tools=True, recover=True)
        for c in text:
            p.feed(c)
        self.assertIsNone(p.literal_context())

    def test_eos_literal_stops_past_the_calls_own_closers(self):
        # a closer slipped in a way the repair does not cover (a space before the line break): the model has left the
        # call; its end of turn ends the reply instead of becoming text inside the argument
        p = OutputParser(thinking=True, tools=TOOLS, stream_tools=True, recover=True)
        p.feed(HEAD + "<function=think>\n<parameter=thought>\nx\n</parameter \n</function>\n</tool_call>")
        self.assertIsNone(p.literal_context())
        # still open: a value that quotes a whole call (a file that documents the format), and a plain open value
        quoted = (HEAD + "<function=write_file>\n<parameter=content>\nExample:\n<tool_call>\n<function=think>\n"
                  "<parameter=thought>\nhi\n</parameter>\n</function>\n</tool_call>\nThe token ")
        for text in (quoted, HEAD + "<function=think>\n<parameter=thought>\nsome "):
            with self.subTest(text=text[-30:]):
                p = OutputParser(thinking=True, tools=TOOLS, stream_tools=True, recover=True)
                for c in text:
                    p.feed(c)
                self.assertEqual(p.literal_context(), "in a tool call's argument")


class Server(unittest.TestCase):
    """Through the HTTP API: the end of turn right after a call whose closer slipped ends the reply (one engine pass),
    where EOS literal used to write it as text inside the argument up to STOP_TEXT_MAX times."""

    def setUp(self):
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, Service, serve
        from serve.test_stop_token_text import PassesEngine
        self.tok = ByteTokenizer()
        self.engine = PassesEngine(self.tok, ["ok"])
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def test_a_slipped_closer_then_the_end_of_turn(self):
        import urllib.request
        from serve.test_stop_token_text import BASH
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.engine.__init__(self.tok, ["<tool_call>\n<function=Bash>\n<parameter=command>\nls -la\n"
                                                "</parameter\n</function>\n</tool_call>"])
                body = {"model": "m", "messages": [{"role": "user", "content": "list"}], "max_tokens": 400,
                        "stream": stream, "tools": [BASH], "chat_template_kwargs": {"enable_thinking": False}}
                req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    raw = r.read().decode()
                if stream:
                    args, finish = "", None
                    for line in raw.splitlines():
                        if line.startswith("data: {"):
                            for c in json.loads(line[6:])["choices"]:
                                for t in c["delta"].get("tool_calls") or []:
                                    args += (t.get("function") or {}).get("arguments") or ""
                                finish = c.get("finish_reason") or finish
                else:
                    c = json.loads(raw)["choices"][0]
                    args, finish = c["message"]["tool_calls"][0]["function"]["arguments"], c["finish_reason"]
                self.assertEqual(finish, "tool_calls")
                self.assertEqual(json.loads(args), {"command": "ls -la"})
                self.assertEqual(len(self.engine.prompts), 1)      # no EOS-literal pass


if __name__ == "__main__":
    unittest.main()
