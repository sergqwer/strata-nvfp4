"""serve/test_tool_python_bools.py - a tool call's `True` / `False` (Qwen writes Python-style booleans) is a JSON
boolean where the parameter's schema allows one, in the final call and in the streamed arguments alike.

From jarydkrish's fork (2b111433): the value failed json.loads and went out as the string "True", which a client
that checks its schema (Claude Code) refuses.  A declared string, number, object, ... keeps the text, as do a quoted
"True", `yes`, `TRUE` and `1`.

    python -m unittest serve.test_tool_python_bools -v
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import OutputParser, bool_allowed, parse_tool_call  # noqa: E402

TOOL = {"name": "store", "parameters": {"type": "object", "properties": {
    "active": {"type": "boolean"},
    "maybe": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
    "listed": {"type": ["boolean", "null"]},
    "anything": {"description": "no type"},
    "literal": {"type": "string"},
    "count": {"type": "integer"},
    "text_or_null": {"type": ["string", "null"]},
}}}


def call_text(params: dict[str, str]) -> str:
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function=store>\n{body}</function>\n</tool_call>"


def run(text: str, step: int, tools=(TOOL,)):
    """-> (final arguments, the streamed tool_args joined and decoded)."""
    p = OutputParser(thinking=False, tools=list(tools), stream_tools=True)
    evs = []
    for i in range(0, len(text), step):
        evs += p.feed(text[i:i + step])
    evs += p.finish("stop")
    calls = [e.call.arguments for e in evs if e.kind == "tool_call"]
    streamed = "".join(e.text for e in evs if e.kind == "tool_args")
    return calls, json.loads(streamed) if streamed else None


class PythonBooleans(unittest.TestCase):
    def check(self, params, want):
        text = call_text(params)
        self.assertEqual(parse_tool_call(text[len("<tool_call>\n"):-len("\n</tool_call>")], TOOL).arguments, want)
        for step in (1, 3, 1000):
            with self.subTest(step=step):
                calls, streamed = run(text, step)
                self.assertEqual(calls, [want])
                self.assertEqual(streamed, want)

    def test_declared_booleans_take_python_spellings(self):
        for spelling, value in (("True", True), ("False", False), ("true", True), ("false", False)):
            with self.subTest(spelling=spelling):
                self.check({"active": spelling, "maybe": spelling, "listed": spelling, "anything": spelling},
                           {"active": value, "maybe": value, "listed": value, "anything": value})

    def test_strings_and_other_types_keep_the_text(self):
        self.check({"literal": "True", "count": "True", "text_or_null": "False"},
                   {"literal": "True", "count": "True", "text_or_null": "False"})

    def test_other_spellings_are_not_booleans(self):
        for spelling, value in (('"True"', "True"), ("yes", "yes"), ("TRUE", "TRUE"), ("1", 1), ("None", "None")):
            with self.subTest(spelling=spelling):
                self.check({"active": spelling}, {"active": value})

    def test_an_undeclared_tool_decodes_them_too(self):
        text = "<tool_call>\n<function=other>\n<parameter=flag>\nTrue\n</parameter>\n</function>\n</tool_call>"
        p = OutputParser(thinking=False, tools=[TOOL], stream_tools=True)
        evs = p.feed(text) + p.finish("stop")
        self.assertEqual([e.call.arguments for e in evs if e.kind == "tool_call"], [{"flag": True}])
        self.assertEqual(json.loads("".join(e.text for e in evs if e.kind == "tool_args")), {"flag": True})

    def test_bool_allowed(self):
        self.assertTrue(bool_allowed({"type": "boolean"}))
        self.assertTrue(bool_allowed({"type": ["null", "boolean"]}))
        self.assertTrue(bool_allowed({"oneOf": [{"type": "integer"}, {"type": "boolean"}]}))
        self.assertTrue(bool_allowed({}))
        self.assertTrue(bool_allowed(None))
        self.assertFalse(bool_allowed({"type": "string"}))
        self.assertFalse(bool_allowed({"type": "integer"}))
        self.assertFalse(bool_allowed({"anyOf": [{"type": "string"}, {"type": "null"}]}))


if __name__ == "__main__":
    unittest.main()
