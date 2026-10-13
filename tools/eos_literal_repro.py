"""tools/eos_literal_repro.py - a reply that writes a stop token's TEXT (<|im_end|>, <|endoftext|>, </s>, <|eot_id|>,
<end_of_turn>, ...) must not end there.  End to end, against a running server (the real model), temperature 0.

    python tools/eos_literal_repro.py --url http://127.0.0.1:8080 [--api-key KEY] [--server-log server.out]
                                      [--only NAME,...] [--out result.json]

Every case asks the model to write such a token - from the prompt or from memory, as text, in backticks, in a code
block, inside a tool call's argument, inside its thinking, and after a tool result that contains one - and to end
with the word FINISHED.  A case passes when FINISHED (or the tool call it asks for) arrives.  Both APIs: OpenAI
/v1/chat/completions and Anthropic /v1/messages (Claude Code's).

`--normal` runs ordinary requests instead (code, quotes, lists, a tool call, with and without thinking) and
`--compare A.json B.json` checks two such runs reply for reply: a server with the fix must answer them byte for byte
as the one without it.

`--server-log`: the server's console output (started with STRATA_DEBUG=1); each case then shows the ids the model
generated around the end of its reply (248046 = <|im_end|>, 248044 = <|endoftext|>): an answer cut by a stop token
whose text the model meant to write ends in that id, right where the text should have been.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request

LITS = ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>", "<end_of_turn>"]
END = "FINISHED"
BASH_OA = {"type": "function", "function": {
    "name": "Bash", "description": "Run a shell command and return its output.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "the command"}},
                   "required": ["command"]}}}
BASH_AN = {"name": "Bash", "description": "Run a shell command and return its output.",
           "input_schema": BASH_OA["function"]["parameters"]}

RECALL = ("What exact special token string does Qwen's ChatML chat format put at the end of every message? Write the "
          f"token in backticks, then on the next line write {END}.")
COPY = ("Repeat this line exactly, then on the next line write the word " + END + ":\n"
        "Qwen ends a turn with <|im_end|> and a document with <|endoftext|>.")
EXPLAIN = ("In two short sentences explain what <|im_end|> and <|endoftext|> do in Qwen models, writing each token "
           f"literally (not in a code block). End your answer with the word {END}.")
CODE = ("Write a one-line Python statement, in a ```python code block, that removes every '<|im_end|>' from the "
        f"string s. After the code block write {END}.")
OTHERS = ("List these end-of-sequence tokens one per line, each in backticks: </s>, <|eot_id|>, <end_of_turn>, "
          f"<|im_end|>, <|endoftext|>. Then write {END}.")
TOOL = ("Use the Bash tool to search tokenizer_config.json for the literal string <|im_end|> with grep -n (put the "
        "string in single quotes). Call the tool now; do not explain.")
# from memory: the token's text is nowhere in the prompt, the model writes it from what it knows
RECALL_PLAIN = ("What exact special token string does Qwen's ChatML chat format put at the end of every message? Answer "
                f"in one plain sentence that contains the token itself (no backticks, no code), then write {END}.")
RECALL_QUOTE = ("What exact special token string does Qwen's ChatML chat format put at the end of every message? Answer "
                f'with: The token is "TOKEN". (with the real token in the double quotes), then write {END}.')
RECALL_FENCE = ("Show a minimal ChatML conversation with one user message and one assistant reply, written out raw with "
                f"all its special tokens, inside a ```text code block. After the block write {END}.")
RECALL_JSON = ("Write the two lines of a Qwen tokenizer_config.json that set eos_token and pad_token, inside a ```json "
               f"code block. After the block write {END}.")
RECALL_TOOL = ("Use the Bash tool to grep tokenizer_config.json for Qwen's end-of-turn special token (write the token "
               "itself in the grep pattern, in single quotes). Call the tool now; do not explain.")
RECALL_AGENT = ("llama.cpp never stops generating with my Qwen GGUF: it runs until the context is full. In a few "
                "sentences, explain which token should stop it and how to check that GGUF's eos_token_id, writing the "
                f"token's exact text. End with {END}.")
THINK = ("Before answering, think about which literal special token string ends an assistant turn in ChatML and "
         f"quote it in your thinking. Then answer with that token in backticks and the word {END}.")


def oa_case(prompt, thinking, tools=None, history=None, max_tokens=None):
    body = {"messages": (history or []) + ([{"role": "user", "content": prompt}] if prompt else []),
            "temperature": 0, "max_tokens": max_tokens or (3000 if thinking else 300), "stream": False}
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        body["reasoning_effort"] = "low"
    if tools:
        body["tools"] = tools
    return "/v1/chat/completions", body


def an_case(prompt, thinking, tools=None, history=None, max_tokens=None):
    body = {"messages": (history or []) + ([{"role": "user", "content": prompt}] if prompt else []),
            "temperature": 0, "max_tokens": max_tokens or (3000 if thinking else 300)}
    if thinking:
        body["thinking"] = {"type": "enabled", "budget_tokens": 2000}
    else:
        body["thinking"] = {"type": "disabled"}
    if tools:
        body["tools"] = tools
    return "/v1/messages", body


# an agent loop: the model's own earlier turn quoted the token, ran grep for it, and the result holds it
GREP = "grep -n 'eos_token\\|<|im_end|>' tokenizer_config.json"
GREPPED = '12:  "eos_token": "<|im_end|>",\n31:      "content": "<|im_end|>",\n40:  "pad_token": "<|endoftext|>",'
OA_LOOP = [
    {"role": "user", "content": "llama.cpp never stops generating with my model. Find out which token ends a turn."},
    {"role": "assistant", "content": "Qwen models end a turn with `<|im_end|>`. Let me check the tokenizer config.",
     "tool_calls": [{"id": "call_1", "type": "function", "function": {
         "name": "Bash", "arguments": json.dumps({"command": GREP})}}]},
    {"role": "tool", "tool_call_id": "call_1",
     "content": GREPPED},
    {"role": "user", "content": "Which token is the eos_token in that output? Quote it exactly in backticks, say which "
                                f"line it is on, then write {END}."},
]
AN_LOOP = [
    {"role": "user", "content": "llama.cpp never stops generating with my model. Find out which token ends a turn."},
    {"role": "assistant", "content": [
        {"type": "text", "text": "Qwen models end a turn with `<|im_end|>`. Let me check the tokenizer config."},
        {"type": "tool_use", "id": "toolu_1", "name": "Bash",
         "input": {"command": GREP}}]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_1",
         "content": GREPPED},
        {"type": "text", "text": "Which token is the eos_token in that output? Quote it exactly in backticks, say which "
                                 f"line it is on, then write {END}."}]},
]

CASES = {
    "oa-recall": (oa_case(RECALL, False), "end"),
    "oa-copy": (oa_case(COPY, False), "end"),
    "oa-explain": (oa_case(EXPLAIN, False), "end"),
    "oa-code": (oa_case(CODE, False), "end"),
    "oa-others": (oa_case(OTHERS, False), "end"),
    "oa-tool": (oa_case(TOOL, False, [BASH_OA]), "tool"),
    "oa-loop": (oa_case(None, False, [BASH_OA], OA_LOOP), "end"),
    "oa-recall-plain": (oa_case(RECALL_PLAIN, False), "end"),
    "oa-recall-quote": (oa_case(RECALL_QUOTE, False), "end"),
    "oa-recall-fence": (oa_case(RECALL_FENCE, False), "end"),
    "oa-recall-json": (oa_case(RECALL_JSON, False), "end"),
    "oa-recall-tool": (oa_case(RECALL_TOOL, False, [BASH_OA]), "tool"),
    "oa-recall-agent": (oa_case(RECALL_AGENT, False, max_tokens=600), "end"),
    "oa-think-recall": (oa_case(THINK, True), "end"),
    "oa-think-fence": (oa_case(RECALL_FENCE, True), "end"),
    "oa-think-rtool": (oa_case(RECALL_TOOL, True, [BASH_OA]), "tool"),
    "oa-think-explain": (oa_case(EXPLAIN, True), "end"),
    "oa-think-tool": (oa_case(TOOL, True, [BASH_OA]), "tool"),
    "an-recall": (an_case(RECALL, False), "end"),
    "an-explain": (an_case(EXPLAIN, False), "end"),
    "an-code": (an_case(CODE, False), "end"),
    "an-tool": (an_case(TOOL, False, [BASH_AN]), "tool"),
    "an-loop": (an_case(None, False, [BASH_AN], AN_LOOP), "end"),
    "an-recall-fence": (an_case(RECALL_FENCE, False), "end"),
    "an-recall-tool": (an_case(RECALL_TOOL, False, [BASH_AN]), "tool"),
    "an-recall-agent": (an_case(RECALL_AGENT, False, max_tokens=600), "end"),
    "an-think-recall": (an_case(THINK, True), "end"),
    "an-think-rtool": (an_case(RECALL_TOOL, True, [BASH_AN]), "tool"),
    "an-think-tool": (an_case(TOOL, True, [BASH_AN]), "tool"),
}

# --normal: ordinary requests whose replies must not change (an A/B of two servers: --compare A.json B.json)
NORMAL_PROMPTS = [
    "What is 17*23? Answer with just the number.",
    "Write a Python function that reverses a string, in a ```python code block, and nothing else.",
    "Give three tips for writing clear commit messages, as a bulleted list.",
    "Translate 'good morning' into French, German and Spanish.",
    "Explain in two sentences what a mutex is.",
    "Quote the first line of the Zen of Python in double quotes, and nothing else.",
    "Show a bash one-liner that counts the lines of all .py files, as inline code in one sentence.",
    "What does `ls -la` show? One sentence.",
    "Return a JSON object with keys name and age for Alice, 30, in a ```json code block.",
    "Say hello.",
    "Write a haiku about rain.",
    "End your reply with a single word in backticks: the name of Python's package installer.",
]
NORMAL = {}
for k, q in enumerate(NORMAL_PROMPTS):
    NORMAL[f"n-oa-{k}"] = (oa_case(q, False, max_tokens=400), "any")
    NORMAL[f"n-an-{k}"] = (an_case(q, False, max_tokens=400), "any")
for k in (0, 1, 4, 5):
    NORMAL[f"n-oa-think-{k}"] = (oa_case(NORMAL_PROMPTS[k], True, max_tokens=4000), "any")
NORMAL["n-oa-tool"] = (oa_case("List the files in /tmp with the Bash tool.", False, [BASH_OA]), "any")
NORMAL["n-an-tool"] = (an_case("List the files in /tmp with the Bash tool.", False, [BASH_AN]), "any")
NORMAL["n-an-think-tool"] = (an_case("List the files in /tmp with the Bash tool.", True, [BASH_AN]), "any")


def post(url, path, body, key):
    headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url.rstrip("/") + path, json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=900) as f:
        return json.loads(f.read())


def reply_of(path, r):
    """-> (content text, reasoning text, [tool call arguments as text], finish, output tokens)"""
    if path == "/v1/messages":
        text = "".join(b.get("text", "") for b in r["content"] if b["type"] == "text")
        think = "".join(b.get("thinking", "") for b in r["content"] if b["type"] == "thinking")
        calls = [json.dumps(b["input"]) for b in r["content"] if b["type"] == "tool_use"]
        return text, think, calls, r.get("stop_reason"), (r.get("usage") or {}).get("output_tokens")
    m = r["choices"][0]["message"]
    calls = [c["function"]["arguments"] for c in m.get("tool_calls") or []]
    return (m.get("content") or "", m.get("reasoning_content") or "", calls, r["choices"][0].get("finish_reason"),
            (r.get("usage") or {}).get("completion_tokens"))


def compare(a_path, b_path) -> int:
    """Two result files of the same cases: every reply byte for byte (text, thinking, calls, finish, token count)."""
    a = {r["case"]: r for r in json.load(open(a_path, encoding="utf-8"))}
    b = {r["case"]: r for r in json.load(open(b_path, encoding="utf-8"))}
    keys = ("content", "reasoning", "tool_args", "finish", "tokens")
    same = [c for c in a if c in b and all(a[c].get(k) == b[c].get(k) for k in keys)]
    for c in a:
        if c in b and c not in same:
            print(f"DIFF {c}: " + ", ".join(f"{k}: {a[c].get(k)!r:.120} -> {b[c].get(k)!r:.120}" for k in keys
                                             if a[c].get(k) != b[c].get(k)))
    print(f"{len(same)} of {len(set(a) & set(b))} replies identical")
    return 0 if len(same) == len(set(a) & set(b)) else 1


def ids_since(log, pos):
    """The last `raw ids:` line the server printed after `pos` (STRATA_DEBUG=1)."""
    if not log:
        return None
    with open(log, encoding="utf-8", errors="replace") as f:
        f.seek(pos)
        got = re.findall(r"\[strata\] raw ids: (\[[^\]]*\])", f.read())
    return json.loads(got[-1]) if got else None


def log_end(log):
    if not log:
        return 0
    with open(log, "rb") as f:
        return f.seek(0, 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--server-log")
    ap.add_argument("--only", help="comma-separated case names (default: all)")
    ap.add_argument("--out")
    ap.add_argument("--normal", action="store_true", help="the ordinary requests (an A/B: record, then --compare)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two --out files of the same cases")
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)
    if not a.url:
        ap.error("--url is required (or --compare A B)")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")    # replies in any language, on any console
    cases = NORMAL if a.normal else CASES
    names = a.only.split(",") if a.only else list(cases)
    results, failed = [], 0
    for name in names:
        (path, body), want = cases[name]
        pos = log_end(a.server_log)
        t0 = time.time()
        r = post(a.url, path, body, a.api_key)
        text, think, calls, finish, tokens = reply_of(path, r)
        time.sleep(0.3)                               # the server's debug lines land before its reply; to be sure
        ids = ids_since(a.server_log, pos)
        if want == "tool":
            ok = any("im_end" in c for c in calls)
        elif want == "any":
            ok = True
        else:
            ok = END in text
        failed += not ok
        rec = {"case": name, "ok": ok, "finish": finish, "seconds": round(time.time() - t0, 1),
               "content": text, "reasoning": think, "tool_args": calls, "tokens": tokens,
               "last_ids": ids[-12:] if ids else None, "n_ids": len(ids) if ids else None,
               "ended_by_id": ids[-1] if ids and ids[-1] in (248044, 248046) else None,
               "special_ids_inside": [t for t in (ids or [])[:-1] if t >= 248000]}
        results.append(rec)
        print(f"{'PASS' if ok else 'FAIL'} {name:17s} finish={finish} ids={rec['n_ids']} last={rec['last_ids']} "
              f"content={text[-160:]!r} tools={calls}", flush=True)
    print(f"{len(names) - failed} of {len(names)} passed")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1, ensure_ascii=False)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
