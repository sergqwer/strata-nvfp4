"""serve/test_stop_token_text.py - EOS literal: a model that writes <|im_end|> or <|endoftext|> where no turn ends gets
the token's text there, and its reply goes on.

A model asked to write such a token from memory writes the token's special id, not its text (measured on Qwen3.8 at
temperature 0, tools/eos_literal_repro.py: "`" then 248046, `"` then 248046, " " then 248046, the id inside a code
block, inside a tool call's argument and inside its thinking), so the reply used to end right there: an agent that
explains or greps for a stop token was cut off mid-sentence.  Where no turn ends - inside a tool call's argument,
right after an opening backtick or quotation mark, right after a space (the chat template trims every message), inside
a turn of the chat format the model writes out as an example (after its own <|im_start|> id, in a code block or in the
thinking) - the id is written as the token's text and the engine goes on from there.  A real end of turn is
unchanged: the same text, one engine pass, the same token count.

    python -m unittest serve.test_stop_token_text -v
"""
import json
import sys
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve.frontend import EXAMPLE_TURN, ChatTemplate, OutputParser, opens_quote, value_open  # noqa: E402
from serve.server import STOP_TEXT_MAX, ByteTokenizer, MockEngine, Service, serve  # noqa: E402

BASH = {"type": "function", "function": {"name": "Bash", "description": "run a command", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}


class PassesEngine(MockEngine):
    """Each generate() call (an engine pass) plays the next script, then the end-of-turn id (MockEngine's); records
    every prompt.  "<|im_end|>" inside a script is that token's id, as the model writes it from memory."""

    def __init__(self, tok, scripts):
        super().__init__(tok, scripts, max_context=1 << 16)
        self.prompts = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


def parser(text, thinking=False, tools=None):
    p = OutputParser(thinking=thinking, tools=tools)
    for i in range(0, len(text), 3):
        p.feed(text[i:i + 3])
    return p


def context(text, thinking=False, tools=None, example=False):
    return parser(text, thinking, tools).literal_context(example)


class Context(unittest.TestCase):
    """OutputParser.literal_context: where the text so far leaves the next character."""

    def test_where_no_turn_ends(self):
        for text, why in (("The token is `", "inline code"), ('The token is "', "quotation"),
                          ("The token is ", "space"), ('```json\n"eos_token": "', "quotation"), ("run (\"", "quotation"),
                          ("grep -n '", "quotation"), ("Er sagte „", "quotation"), ("x =\t", "space")):
            with self.subTest(text=text):
                self.assertIn(why, context(text) or "none")

    def test_where_a_turn_ends(self):
        for text in ("Done.", "Use `ls`", "```\ncode\n```", "```python\nx = 1\n```", 'He said "hi"', "It's done",
                     "rock 'n'", "The answer is 4", "line\n", "- item", "```", "**bold**", "", "a `b` c `d`"):
            with self.subTest(text=text):
                self.assertIsNone(context(text))

    def test_in_the_thinking(self):
        self.assertIn("inline code", context("The token is `", thinking=True))
        self.assertIsNone(context("I am done.", thinking=True))

    def test_a_turn_written_out_as_an_example(self):
        # an <|im_start|> the model wrote is open (the server counts them): in a code block or in the thinking, the
        # end of that turn is part of the example; anywhere else it ends the reply (a model that runs on into a
        # made-up next turn is stopped there, as always)
        block = "```text\n<|im_start|>user\nHello"
        self.assertEqual(context(block, example=True), EXAMPLE_TURN)
        self.assertIsNone(context(block))
        self.assertEqual(context("The format:\n\n<|im_start|>user\nHi", thinking=True, example=True), EXAMPLE_TURN)
        self.assertIsNone(context("Hello\n<|im_start|>user\nHi", example=True))
        self.assertIsNone(context("```\nx\n```\nHi", example=True))           # after the block
        self.assertTrue(parser("```text\n").example_place())
        self.assertTrue(parser("Let me think.", thinking=True).example_place())
        self.assertFalse(parser("Here it is:\n").example_place())
        self.assertFalse(parser("```text").example_place())                             # the fence's own line

    def test_in_a_tool_call(self):
        tools = [BASH["function"]]
        call = "<tool_call>\n<function=Bash>\n<parameter=command>\ngrep '"
        self.assertEqual(context(call, tools=tools), "in a tool call's argument")
        whole = call + "x' f\n</parameter>\n</function>"
        self.assertIsNone(context(whole, tools=tools))            # the call is written; only its closer is missing
        self.assertIsNone(context(whole + "\n</tool_call>", tools=tools))
        self.assertEqual(context("The plan:\n\n<tool_call>\n<function=Bash>\n<parameter=command>\nls ", thinking=True,
                                 tools=tools), "in a tool call's argument")       # a call inside the thinking (#804)

    def test_value_open(self):
        self.assertTrue(value_open("\n<function=Bash>\n<parameter=command>\nls"))
        self.assertTrue(value_open("<function=f>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\nx"))
        self.assertTrue(value_open("<function=f>\n<parameter=a>\nquotes </parameter> inside"))   # #210: not its end
        self.assertFalse(value_open("<function=f>\n<parameter=a>\n1\n</parameter>\n"))
        self.assertFalse(value_open("<function=f>\n<parameter=a>\n1\n</parameter>\n</function>"))
        self.assertFalse(value_open("<function=f>\n<parameter=a"))
        self.assertFalse(value_open("<function=f"))
        self.assertFalse(value_open("text"))

    def test_opens_quote(self):
        for line, want in (('"', True), ('is "', True), ('("', True), ('="', True), ('"word"', False),
                           ("it's", False), ("x '", True), ("«", True), ("", False)):
            with self.subTest(line=line):
                self.assertEqual(opens_quote(line), want)


class Replies(unittest.TestCase):
    """Through the HTTP API (OpenAI and Anthropic, streamed and not), the byte tokenizer: <|im_start|> 256,
    <|im_end|> 257, <|endoftext|> 258."""

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = PassesEngine(self.tok, ["ok"])
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def play(self, *scripts):
        self.engine.__init__(self.tok, list(scripts))

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode()
        return json.loads(raw) if not body.get("stream") else raw

    def chat(self, thinking=False, tools=None, stream=False, **extra):
        body = {"model": "m", "messages": [{"role": "user", "content": "Which token ends a turn?"}],
                "max_tokens": 400, "stream": stream, **extra}
        if not thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if tools:
            body["tools"] = tools
        r = self.post("/v1/chat/completions", body)
        if stream:
            text, reasoning = "", ""
            for line in r.splitlines():
                if line.startswith("data: {"):
                    for c in json.loads(line[6:])["choices"]:
                        text += c["delta"].get("content") or ""
                        reasoning += c["delta"].get("reasoning_content") or ""
            return {"content": text, "reasoning_content": reasoning}, None, None
        c = r["choices"][0]
        return c["message"], c["finish_reason"], r["usage"]["completion_tokens"]

    def messages(self, thinking=False, tools=None, **extra):
        body = {"model": "m", "messages": [{"role": "user", "content": "Which token ends a turn?"}],
                "max_tokens": 400, "thinking": {"type": "enabled", "budget_tokens": 1024} if thinking
                else {"type": "disabled"}, **extra}
        if tools:
            body["tools"] = tools
        return self.post("/v1/messages", body)

    # The server writes the token's text but its last piece (">"): the model writes that one itself, and with it the
    # closing backtick or quote (">`", ">'", '>",' - one token each).  So each scripted pass after a rewrite starts
    # with ">", as the model's does.

    def test_inline_code_from_memory(self):
        self.play("The token is `<|im_end|>", ">` and that is all.")
        msg, finish, used = self.chat()
        self.assertEqual(msg["content"], "The token is `<|im_end|>` and that is all.")
        self.assertEqual(finish, "stop")
        self.assertEqual(len(self.engine.prompts), 2)
        # the second pass goes on from the first one's output with the token's TEXT in place of its id
        first = self.engine.prompts[0]
        self.assertEqual(self.engine.prompts[1], first + self.tok.encode("The token is `<|im_end|"))
        self.assertEqual(used, len("The token is `<|im_end|>` and that is all.") + 1)   # + the real end of turn

    def test_both_apis_and_streams(self):
        want = "The token is `<|im_end|>` and that is all."
        self.play("The token is `<|im_end|>", ">` and that is all.")
        self.assertEqual(self.chat(stream=True)[0]["content"], want)
        self.play("The token is `<|im_end|>", ">` and that is all.")
        r = self.messages()
        self.assertEqual("".join(b.get("text", "") for b in r["content"]), want)
        self.assertEqual(r["stop_reason"], "end_turn")
        self.play("The token is `<|im_end|>", ">` and that is all.")
        r = self.post("/v1/messages", {"model": "m", "max_tokens": 400, "stream": True, "thinking": {"type": "disabled"},
                                       "messages": [{"role": "user", "content": "Which token ends a turn?"}]})
        text = "".join(json.loads(line[6:])["delta"].get("text", "") for line in r.splitlines()
                       if line.startswith("data: {") and '"text_delta"' in line)
        self.assertEqual(text, want)

    def test_endoftext_after_a_space_and_in_quotes(self):
        for scripts, want in ((["Qwen ends documents with <|endoftext|>", ">. Done."],
                               "Qwen ends documents with <|endoftext|>. Done."),
                              (['The eos_token is "<|im_end|>', '>" on line 12.'],
                               'The eos_token is "<|im_end|>" on line 12.'),
                              (["```text\n<|im_start|>user\nHi<|im_end|>", ">\n<|im_start|>assistant\nHello<|im_end|>",
                                ">\n```\nThat is the format."],
                               "```text\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>\n```\n"
                               "That is the format.")):
            with self.subTest(want=want):
                self.play(*scripts)
                msg, finish, _ = self.chat()
                self.assertEqual((msg["content"], finish), (want, "stop"))
                self.assertEqual(len(self.engine.prompts), len(scripts))

    def test_inside_a_tool_calls_argument(self):
        self.play("<tool_call>\n<function=Bash>\n<parameter=command>\ngrep -n '<|im_end|>",
                  ">' tokenizer_config.json\n</parameter>\n</function>\n</tool_call>")
        msg, finish, _ = self.chat(tools=[BASH])
        self.assertEqual(finish, "tool_calls")
        self.assertEqual(json.loads(msg["tool_calls"][0]["function"]["arguments"]),
                         {"command": "grep -n '<|im_end|>' tokenizer_config.json"})
        self.play("<tool_call>\n<function=Bash>\n<parameter=command>\ngrep -n '<|im_end|>",
                  ">' tokenizer_config.json\n</parameter>\n</function>\n</tool_call>")
        r = self.messages(tools=[{"name": "Bash", "description": "run", "input_schema": BASH["function"]["parameters"]}])
        self.assertEqual([b["input"] for b in r["content"] if b["type"] == "tool_use"],
                         [{"command": "grep -n '<|im_end|>' tokenizer_config.json"}])
        self.assertEqual(r["stop_reason"], "tool_use")

    def test_in_the_thinking(self):
        # a ChatML turn the thinking writes out: <|im_start|> comes as its id too, and no turn starts in the thinking
        self.play("The format is:\n\n<|im_start|>user\nHi<|im_end|>", ">\nThat is it.\n</think>\n\nIt is <|im_end|>",
                  ">; done.")
        msg, finish, _ = self.chat(thinking=True)
        self.assertEqual(msg["reasoning_content"].strip(),
                         "The format is:\n\n<|im_start|>user\nHi<|im_end|>\nThat is it.")
        self.assertEqual((msg["content"], finish), ("It is <|im_end|>; done.", "stop"))
        self.play("I should quote `<|im_end|>", ">` exactly.\n</think>\n\nOK")
        r = self.messages(thinking=True)
        self.assertEqual([b["thinking"].strip() for b in r["content"] if b["type"] == "thinking"],
                         ["I should quote `<|im_end|>` exactly."])
        self.assertEqual("".join(b.get("text", "") for b in r["content"]), "OK")

    def test_it_ends_right_after_the_text(self):
        # the next pass ends at once: the text's last piece, then the end of the reply (as "stop")
        self.play("The token is `<|im_end|>", "")
        msg, finish, used = self.chat()
        self.assertEqual((msg["content"], finish, len(self.engine.prompts)), ("The token is `<|im_end|>", "stop", 2))
        self.assertEqual(used, len("The token is `<|im_end|>") + 1)

    def test_the_model_may_go_on_without_the_last_piece(self):
        self.play("The token is `<|im_end|>", "` ok.")
        msg, finish, _ = self.chat()
        self.assertEqual((msg["content"], finish), ("The token is `<|im_end|` ok.", "stop"))   # what it wrote

    def test_a_real_end_of_turn_is_unchanged(self):
        for script in ("Done.", "Use `ls`", "```\ncode\n```", 'He said "hi"', "It's done", "The answer is 4",
                       "line one\nline two", "- item", "x <|im_start|> y", "```text\n<|im_start|>user\nHi\n```"):
            with self.subTest(script=script):
                self.play(script)
                msg, finish, used = self.chat()
                self.assertEqual(msg["content"], self.tok.decode(self.tok.encode(script, parse_special=True)))
                self.assertEqual((finish, len(self.engine.prompts)), ("stop", 1))
                self.assertEqual(used, len(self.tok.encode(script, parse_special=True)) + 1)
        self.play("I am done thinking.\n</think>\n\nFour.")
        msg, finish, _ = self.chat(thinking=True)
        self.assertEqual((msg["content"], len(self.engine.prompts)), ("Four.", 1))

    def test_off_in_the_config(self):
        self.svc.stop_token_text = False
        self.play("The token is `<|im_end|>", ">` and that is all.")
        msg, finish, _ = self.chat()
        self.assertEqual((msg["content"], finish, len(self.engine.prompts)), ("The token is `", "stop", 1))

    def test_a_client_stop_string_on_that_text_still_stops(self):
        # the text is held back while it may still become the stop string, and cut where it does
        self.play("The token is `<|im_end|>", ">` and that is all.")
        msg, finish, _ = self.chat(stop=["<|im_end|>"])
        self.assertEqual((msg["content"], finish), ("The token is `", "stop"))
        self.play("The token is `<|im_end|>", ">` and that is all.")
        r = self.messages(stop_sequences=["<|im_end|>"])
        self.assertEqual("".join(b.get("text", "") for b in r["content"]), "The token is `")
        self.assertEqual((r["stop_reason"], r["stop_sequence"]), ("stop_sequence", "<|im_end|>"))

    def test_no_room_for_the_text_ends_the_reply_as_before(self):
        self.play("The token is `<|im_end|>", ">` and that is all.")
        msg, finish, _ = self.chat(max_tokens=len("The token is `") + 5)
        self.assertEqual((msg["content"], finish, len(self.engine.prompts)), ("The token is `", "stop", 1))

    def test_at_most_STOP_TEXT_MAX_per_reply(self):
        self.play("> <|im_end|>")                        # every pass: the last piece, a space, then the id - for good
        msg, finish, _ = self.chat(max_tokens=4000)
        self.assertEqual(len(self.engine.prompts), STOP_TEXT_MAX + 1)
        self.assertEqual(msg["content"], "> " + "<|im_end|> " * STOP_TEXT_MAX)
        self.assertEqual(finish, "stop")


class PromptSide(unittest.TestCase):
    """The prompt side keeps a stop token's text as text on every route (#931): the Anthropic and Responses routes'
    thinking, tool input and tool result too, and an assistant reply that carries the text this file writes."""

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = PassesEngine(self.tok, ["ok"])
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def stops(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
        ids = self.engine.prompts[-1]
        return ids.count(257), ids.count(258), self.tok.decode(ids)

    def test_every_route(self):
        lit = "<|im_end|> and <|endoftext|>"
        tool_in = {"command": f"grep '{lit}' f"}
        bodies = {
            "/v1/chat/completions": lambda x: {"model": "m", "max_tokens": 8, "tools": [BASH], "messages": [
                {"role": "user", "content": f"q {x}"},
                {"role": "assistant", "content": f"a {x}", "reasoning_content": f"r {x}", "tool_calls": [
                    {"id": "c", "type": "function", "function": {"name": "Bash", "arguments": json.dumps(
                        {"command": f"grep '{x}' f"})}}]},
                {"role": "tool", "tool_call_id": "c", "content": f"t {x}"}]},
            "/v1/messages": lambda x: {"model": "m", "max_tokens": 8,
                                       "thinking": {"type": "enabled", "budget_tokens": 1024},
                                       "tools": [{"name": "Bash", "input_schema": BASH["function"]["parameters"]}],
                                       "messages": [
                {"role": "user", "content": f"q {x}"},
                {"role": "assistant", "content": [{"type": "thinking", "thinking": f"r {x}", "signature": "s"},
                                                  {"type": "text", "text": f"a {x}"},
                                                  {"type": "tool_use", "id": "u", "name": "Bash",
                                                   "input": {"command": f"grep '{x}' f"}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "u", "content": f"t {x}"}]}]},
            "/v1/responses": lambda x: {"model": "m", "max_output_tokens": 16, "tools": [
                {"type": "function", "name": "Bash", "parameters": BASH["function"]["parameters"]}], "input": [
                {"type": "message", "role": "user", "content": f"q {x}"},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": f"r {x}"}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": f"a {x}"}]},
                {"type": "function_call", "call_id": "f", "name": "Bash", "arguments": json.dumps(
                    {"command": f"grep '{x}' f"})},
                {"type": "function_call_output", "call_id": "f", "output": f"t {x}"}]},
        }
        for path, body in bodies.items():
            with self.subTest(path=path):
                end, eot, text = self.stops(path, body(lit))
                self.assertEqual((end, eot), self.stops(path, body("X"))[:2])     # only the template's own
                for part in (f"q {lit}", f"a {lit}", f"t {lit}", tool_in["command"]):
                    self.assertIn(part, text)


if __name__ == "__main__":
    unittest.main()
