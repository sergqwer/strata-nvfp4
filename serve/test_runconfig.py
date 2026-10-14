"""serve/test_runconfig.py - #564: the web page's Settings view of the run config (GET / POST /config), against the
mock engine: only the listed keys change, every other key of the file stays, a bad value changes nothing, and only
Strata's own page may write it (JSON, no foreign Origin, the API key when one is set).

    python -m unittest serve.test_runconfig
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve import runconfig
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, engine_args, serve

ROOT = Path(__file__).resolve().parents[1]
CFG = {"exe": "engine/strata.exe", "args": ["--pack", "data/packs/q2_0", "--kv", "int8", "--spec-min-p", "0.5"],
       "model_name": "qwen3.8-flash-next-q2_0", "port": 8080, "host": "0.0.0.0", "api_key": "",
       "sampling": {"temperature": 0.7, "presence_penalty": 1.5},
       "mcp_servers": {"files": {"command": "npx", "args": ["-y", "server-filesystem", "."]}},
       "cors_origins": ["http://localhost:3000"]}


class Apply(unittest.TestCase):
    def test_only_the_named_keys_change(self):
        new, changed = runconfig.apply(CFG, {"sampling.top_p": 0.95, "reasoning_budget_tokens": 4096,
                                             "vram_reserve_mib": 2048, "aliases": "qwen, local", "lazy_load": True})
        self.assertEqual(changed, ["sampling.top_p", "reasoning_budget_tokens", "vram_reserve_mib", "aliases",
                                   "lazy_load"])
        self.assertEqual(new["sampling"], {"temperature": 0.7, "presence_penalty": 1.5, "top_p": 0.95})
        self.assertEqual(new["args"][-2:], ["--vram-reserve-mib", "2048"])
        self.assertEqual(new["aliases"], ["qwen", "local"])
        for k in ("exe", "model_name", "port", "host", "api_key", "mcp_servers", "cors_origins"):
            self.assertEqual(new[k], CFG[k], k)
        self.assertEqual(CFG["sampling"], {"temperature": 0.7, "presence_penalty": 1.5})   # the input is left alone

    def test_null_is_the_default_again(self):
        cfg, _ = runconfig.apply(CFG, {"vram_reserve_mib": 2048, "fit_max_tokens": True})
        new, changed = runconfig.apply(cfg, {"vram_reserve_mib": None, "fit_max_tokens": None,
                                             "sampling.temperature": None, "idle_unload_s": None})
        self.assertEqual(changed, ["vram_reserve_mib", "fit_max_tokens", "sampling.temperature"])
        self.assertEqual(new["args"], CFG["args"])
        self.assertNotIn("fit_max_tokens", new)
        self.assertEqual(new["sampling"], {"presence_penalty": 1.5})
        new, _ = runconfig.apply({"sampling": {"top_k": 20}}, {"sampling.top_k": None})
        self.assertNotIn("sampling", new)                                 # an empty block goes

    def test_refused(self):
        for changes in ({"api_key": "x"}, {"host": "0.0.0.0"}, {"mcp_servers": {}}, {"before_load": "calc.exe"},
                        {"args": []}, {"exe": "evil.exe"}, {"trusted_origins": ["*"]}, {"allowed_hosts": ["*"]},
                        {"sampling.temperature": -1}, {"sampling.top_p": 0}, {"sampling.top_k": 65},
                        {"sampling.min_p": 2}, {"reasoning_budget_tokens": 1.5}, {"fit_max_tokens": "yes"},
                        {"effort_position": "middle"}, {"vram_reserve_mib": -5}, {"idle_unload_s": True},
                        {"aliases": [1]}, {}, None, ["x"]):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runconfig.apply(CFG, changes)
        with self.assertRaises(ValueError):                              # one bad value: nothing is applied
            runconfig.apply(CFG, {"fit_max_tokens": True, "sampling.top_k": 0})
        with self.assertRaises(ValueError):
            runconfig.apply({**CFG, "vision": {"gpu": True}}, {"lazy_load": True})

    def test_view(self):
        v = runconfig.view({**CFG, "args": CFG["args"] + ["--vram-reserve-mib", "1500"]}, "x/strata-q2_0.json")
        self.assertEqual(v["file"], "strata-q2_0.json")
        got = {k["key"]: k for k in v["keys"]}
        self.assertEqual(got["sampling.temperature"]["value"], 0.7)
        self.assertEqual(got["vram_reserve_mib"]["value"], 1500)
        self.assertIsNone(got["fit_max_tokens"]["value"])
        self.assertEqual(got["anthropic_thinking"]["choices"], ["model", "on_request"])
        self.assertNotIn("api_key", got)
        self.assertNotIn("mcp_servers", got)

    def test_sections_and_the_grid(self):
        keys = runconfig.view(CFG, "x/strata-q2_0.json")["keys"]
        self.assertEqual(keys[0]["section"], "Sampling defaults")
        self.assertEqual(sum("section" in k for k in keys), len(runconfig.SECTIONS))
        css = (ROOT / "serve" / "web" / "app.css").read_text(encoding="utf-8")
        rule = css[css.index(".cfg-form {"):].split("}", 1)[0]
        self.assertNotIn("max-content", rule)   # a long help took the whole row and pushed the controls out of the card


class Uncensored(unittest.TestCase):
    """The censorship switch's default ("uncensored", 0.1.41-nvfp4.7): settable where setup loaded the refusal
    projection; refused where it did not (NVIDIA's model, Swift, the Coder); null is the default (off) again."""
    VEC = ["--control-vector-scaled", "data/uncensor/qwen-nvfp4-gptq-uncensored.gguf:1.0",
           "--control-vector-layer-range", "8", "33", "--cvec-mode", "project", "--cvec-dir", "per-layer"]
    WITH = {**CFG, "args": CFG["args"] + VEC, "uncensored": False}

    def test_listed_after_the_sampling_keys(self):
        keys = [k for k, _, _ in runconfig.EDITABLE]
        self.assertEqual(keys.index("uncensored"), keys.index("sampling.min_p") + 1)
        got = {k["key"]: k for k in runconfig.view(self.WITH, "strata-qwen-gptq-nvfp4.json")["keys"]}
        self.assertEqual((got["uncensored"]["kind"], got["uncensored"]["value"]), ("bool", False))
        self.assertNotIn("available", got["uncensored"])
        self.assertIn("Disable censorship by default", got["uncensored"]["help"])

    def test_true_and_false_round_trip(self):
        new, changed = runconfig.apply(self.WITH, {"uncensored": True})
        self.assertEqual((changed, new["uncensored"]), (["uncensored"], True))
        self.assertEqual(new["args"], self.WITH["args"])                   # the vector stays loaded
        new, changed = runconfig.apply(new, {"uncensored": False})
        self.assertEqual((changed, new["uncensored"]), (["uncensored"], False))
        with self.assertRaises(ValueError):
            runconfig.apply(self.WITH, {"uncensored": "yes"})

    def test_refused_without_a_projection(self):
        for cfg in (CFG, {**CFG, "model_name": "qwen3.8-flash-next-nvidia-nvfp4"}):
            with self.assertRaises(ValueError) as e:
                runconfig.apply(cfg, {"uncensored": True})
            self.assertIn("no refusal projection", str(e.exception))
            self.assertIn("NVIDIA", str(e.exception))
            new, _ = runconfig.apply({**cfg, "uncensored": True}, {"uncensored": False})   # off is always allowed
            self.assertIs(new["uncensored"], False)
        got = {k["key"]: k for k in runconfig.view(CFG, "strata-qwen-nvidia-nvfp4.json")["keys"]}
        self.assertIs(got["uncensored"]["available"], False)
        self.assertIn("not available", got["uncensored"]["help"])

    def test_null_removes_the_key(self):
        new, changed = runconfig.apply(self.WITH, {"uncensored": None})
        self.assertEqual(changed, ["uncensored"])
        self.assertNotIn("uncensored", new)
        new, changed = runconfig.apply({**CFG, "uncensored": True}, {"uncensored": None})   # without a vector too
        self.assertNotIn("uncensored", new)

    def test_the_page_renders_it_like_the_other_bools(self):
        js = (ROOT / "serve" / "web" / "app.js").read_text(encoding="utf-8")
        self.assertIn('k.available === false ? " disabled" : ""', js)    # a model without the projection: greyed out


class Dialogs(unittest.TestCase):
    """Kept dialogs: the conversation cache (--conversation-cache-slots / -mib) and its disk tier (--conversation-disk-dir,
    -mib, --conversation-disk-persist without a value), each set and cleared alone, and refused without what they need."""
    ON = {"conversation_cache_slots": 8, "conversation_cache_mib": 16384, "conversation_disk_dir": "D:/strata dialogs",
          "conversation_disk_mib": 131072, "conversation_disk_persist": True}

    def test_on_and_off(self):
        new, changed = runconfig.apply(CFG, self.ON)
        self.assertEqual(changed, list(self.ON))
        self.assertEqual(new["args"][len(CFG["args"]):],
                         ["--conversation-cache-slots", "8", "--conversation-cache-mib", "16384",
                          "--conversation-disk-dir", "D:/strata dialogs", "--conversation-disk-mib", "131072",
                          "--conversation-disk-persist"])
        got = {k["key"]: k for k in runconfig.view(new, "x.json")["keys"]}
        self.assertEqual({k: got[k]["value"] for k in self.ON}, self.ON)
        self.assertEqual([got[k]["kind"] for k in self.ON], ["number", "number", "text", "number", "bool"])
        self.assertEqual(got["conversation_cache_mib"]["default"], "off")   # the empty field's meaning
        again, changed = runconfig.apply(new, self.ON)                   # the same values: nothing changes
        self.assertEqual((again, changed), (new, []))
        off, changed = runconfig.apply(new, {k: None for k in self.ON})
        self.assertEqual((off["args"], changed), (CFG["args"], list(self.ON)))
        off, _ = runconfig.apply(new, {"conversation_disk_persist": False})   # off = the option left out
        self.assertNotIn("--conversation-disk-persist", off["args"])
        self.assertIn("--conversation-disk-dir", off["args"])

    def test_a_folder_named_by_digits_stays_text(self):
        new, _ = runconfig.apply(CFG, {"conversation_cache_mib": 4096, "conversation_disk_dir": "2026"})
        self.assertEqual(runconfig.value_of(new, "conversation_disk_dir"), "2026")

    def test_refused_without_what_they_need(self):
        for changes, says in (({"conversation_cache_slots": 8}, "conversation_cache_mib"),
                              ({"conversation_disk_dir": "D:/d"}, "conversation_cache_mib"),
                              ({"conversation_cache_mib": 0, "conversation_disk_dir": "D:/d"}, "conversation_cache_mib"),
                              ({"conversation_cache_mib": 4096, "conversation_disk_persist": True}, "conversation_disk_dir"),
                              ({"conversation_cache_mib": 4096, "conversation_disk_mib": 1024}, "conversation_disk_dir"),
                              ({"conversation_disk_persist": "yes"}, "true or false"),
                              ({"conversation_disk_dir": "a\nb"}, "folder"), ({"conversation_disk_dir": " "}, "folder"),
                              ({"conversation_cache_mib": -1}, "at least 0")):
            with self.subTest(changes=changes), self.assertRaises(ValueError) as e:
                runconfig.apply(CFG, changes)
            self.assertIn(says, str(e.exception))
        new, _ = runconfig.apply(CFG, self.ON)
        with self.assertRaises(ValueError):                              # the RAM goes, the disk would stay unused
            runconfig.apply(new, {"conversation_cache_mib": None})


class Parallel(unittest.TestCase):
    """Dialogs at once ("parallel" -> --batch N): a list of 1-8, empty = the engine's one at a time; the choice that is
    one dialog per GPU says so (never chosen for the user); greyed out where the args set --batch themselves."""
    def test_choices_name_one_per_gpu(self):
        for gpu, per in ((None, 1), (0, 1), ([0, 1], 2), ("0,1,2,3", 4)):
            cfg = {**CFG, **({} if gpu is None else {"gpu": gpu})}
            got = {k["key"]: k for k in runconfig.view(cfg, "x.json")["keys"]}["parallel"]
            self.assertEqual((got["kind"], got["value"], got["section"]), ("choice", None, "Dialogs"))
            self.assertEqual([n for n, _ in got["choices"]], list(range(1, 9)))
            marked = [n for n, label in got["choices"] if "1 per GPU" in label]
            self.assertEqual(marked, [per], gpu)
            self.assertIn("1 at a time", got["default"])
            self.assertIn("95.6", got["help"])                          # the 2-GPU split's cost is said
            self.assertNotIn("available", got)

    def test_set_and_cleared(self):
        new, changed = runconfig.apply(CFG, {"parallel": 2})
        self.assertEqual((changed, new["parallel"], new["args"]), (["parallel"], 2, CFG["args"]))
        self.assertEqual(engine_args({**new, "args": list(new["args"])})[-2:], ["--batch", "2"])
        one, _ = runconfig.apply(new, {"parallel": 1})
        self.assertNotIn("--batch", engine_args(one))                   # 1: one at a time, as the default
        off, changed = runconfig.apply(new, {"parallel": None})
        self.assertEqual((changed, "parallel" in off), (["parallel"], False))
        for bad in (0, 9, 1.5, True, "2"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                runconfig.apply(CFG, {"parallel": bad})

    def test_a_hand_written_value_is_shown(self):
        got = {k["key"]: k for k in runconfig.view({**CFG, "parallel": 12}, "x.json")["keys"]}["parallel"]
        self.assertEqual((got["value"], got["choices"][-1][0]), (12, 12))

    def test_greyed_out_under_batch_in_the_args(self):
        got = {k["key"]: k for k in runconfig.view({**CFG, "args": CFG["args"] + ["--batch", "4"]},
                                                   "x.json")["keys"]}["parallel"]
        self.assertIs(got["available"], False)
        self.assertIn("--batch", got["help"])


class Context(unittest.TestCase):
    """--max-context as setup's sizes, never empty; past the trained 262144 the RoPE extension setup writes (empty:
    dynamic on one GPU and one dialog at a time, else yarn; the factor context / 262144), taken out again within it;
    none past it refused, as by setup."""
    BASE = {**CFG, "args": CFG["args"] + ["--max-context", "262144"]}

    def rope(self, cfg):
        a = cfg["args"]
        return tuple(a[a.index(f) + 1] if f in a else None for f in ("--max-context", "--rope-scaling", "--rope-scale"))

    def test_past_the_trained_context_and_back(self):
        new, changed = runconfig.apply(self.BASE, {"max_context": 524288})
        self.assertEqual((changed, self.rope(new)), (["max_context", "rope_scaling"], ("524288", "dynamic", "2")))
        self.assertIs(new["rope_auto"], True)
        new, changed = runconfig.apply(new, {"max_context": 393216})
        self.assertEqual((changed, self.rope(new)), (["max_context"], ("393216", "dynamic", "1.5")))
        new, changed = runconfig.apply(new, {"rope_scaling": "linear"})
        self.assertEqual((changed, self.rope(new), new["rope_auto"]), (["rope_scaling"], ("393216", "linear", "1.5"),
                                                                       False))
        new, changed = runconfig.apply(new, {"rope_scaling": None})     # the default past 262144: dynamic again
        self.assertEqual((changed, self.rope(new)), (["rope_scaling"], ("393216", "dynamic", "1.5")))
        new, changed = runconfig.apply(new, {"max_context": 131072})
        self.assertEqual((changed, self.rope(new)), (["max_context", "rope_scaling"], ("131072", None, None)))
        self.assertNotIn("rope_auto", new)
        self.assertEqual(new["args"][:len(CFG["args"])], CFG["args"])    # the other args stay
        both, changed = runconfig.apply(self.BASE, {"max_context": 524288, "rope_scaling": "linear"})
        self.assertEqual((changed, self.rope(both)), (["max_context", "rope_scaling"], ("524288", "linear", "2")))

    def test_view(self):
        new, _ = runconfig.apply(self.BASE, {"max_context": 524288})
        got = {k["key"]: k for k in runconfig.view(new, "x.json")["keys"]}
        ctx, rope = got["max_context"], got["rope_scaling"]
        self.assertEqual((ctx["kind"], ctx["value"], ctx["required"], ctx["section"]), ("choice", 524288, True, "Context"))
        self.assertEqual([n for n, _ in ctx["choices"]], runconfig.CONTEXTS)
        self.assertIn("trained", dict(ctx["choices"])[262144])
        self.assertIn("RoPE", dict(ctx["choices"])[524288])
        self.assertEqual((rope["kind"], rope["value"], rope["choices"], rope["default"]),
                         ("enum", None, ["dynamic", "yarn", "linear"], "default: dynamic past 262,144"))
        self.assertIn("now: dynamic, factor 2", rope["help"])
        odd = {k["key"]: k for k in runconfig.view({**CFG, "args": ["--max-context", "100000"]}, "x")["keys"]}
        self.assertIn(100000, [n for n, _ in odd["max_context"]["choices"]])   # the file's own size is listed

    def test_refused(self):
        for changes, says in (({"max_context": None}, "4096"), ({"max_context": 512}, "at least 1024"),
                              ({"max_context": 4 * 262144 + 1}, "at most 1048576"), ({"max_context": "262144"}, "whole"),
                              ({"rope_scaling": "yarn"}, "nothing to extend"), ({"rope_scaling": "none"}, "yarn or linear")):
            with self.subTest(changes=changes), self.assertRaises(ValueError) as e:
                runconfig.apply(self.BASE, changes)
            self.assertIn(says, str(e.exception))
        hand = {**CFG, "args": ["--max-context", "262144", "--rope-scaling", "none"]}
        with self.assertRaises(ValueError) as e:                          # none past 262144, as setup refuses it
            runconfig.apply(hand, {"max_context": 524288})
        self.assertIn("pick dynamic, yarn or linear", str(e.exception))
        self.assertEqual(runconfig.apply(hand, {"fit_max_tokens": True})[0]["args"], hand["args"])   # left alone

    def test_dynamic_follows_the_gpus_and_the_dialogs_at_once(self):
        new, _ = runconfig.apply(self.BASE, {"max_context": 524288})
        two, changed = runconfig.apply(new, {"parallel": 2})            # an automatic dynamic: yarn, and back
        self.assertEqual((changed, self.rope(two), two["rope_auto"]), (["parallel", "rope_scaling"],
                                                                       ("524288", "yarn", "2"), True))
        got = {k["key"]: k for k in runconfig.view(two, "x.json")["keys"]}["rope_scaling"]
        self.assertEqual((got["value"], got["default"]), (None, "default: yarn past 262,144"))
        self.assertIn("now: yarn, factor 2; dynamic needs one dialog at a time", got["help"])
        for back in (1, None):
            one, changed = runconfig.apply(two, {"parallel": back})
            self.assertEqual((changed, self.rope(one)), (["parallel", "rope_scaling"], ("524288", "dynamic", "2")))
        split, _ = runconfig.apply({**self.BASE, "gpu": [0, 1]}, {"max_context": 524288})
        self.assertEqual(self.rope(split), ("524288", "yarn", "2"))      # a layer split over two GPUs
        own, _ = runconfig.apply(new, {"rope_scaling": "yarn"})          # the user's yarn stays yarn
        self.assertEqual(self.rope(runconfig.apply(own, {"parallel": 1})[0]), ("524288", "yarn", "2"))
        self.assertEqual(runconfig.dynamic_blocker({**CFG, "args": ["--kv", "q4_0"]}), "needs 8-bit or fp16 K/V "
                                                                                       "(--kv q4_0)")

    def test_a_picked_dynamic_that_cannot_run_is_refused(self):
        dyn, changed = runconfig.apply(self.BASE, {"max_context": 524288, "rope_scaling": "dynamic"})
        self.assertEqual((changed, self.rope(dyn), dyn["rope_auto"]), (["max_context", "rope_scaling"],
                                                                       ("524288", "dynamic", "2"), False))
        for base, changes, says in ((dyn, {"parallel": 2}, "one dialog at a time"),
                                    ({**dyn, "gpu": [0, 1]}, {"max_context": 393216}, "one GPU"),
                                    ({**dyn, "parallel": 3}, {"rope_scaling": "yarn"}, None)):
            with self.subTest(changes=changes):
                if says is None:                                          # yarn is fine there
                    self.assertEqual(self.rope(runconfig.apply(base, changes)[0]), ("524288", "yarn", "2"))
                    continue
                with self.assertRaises(ValueError) as e:
                    runconfig.apply(base, changes)
                self.assertIn(f"rope_scaling: dynamic needs {says}", str(e.exception))

    def test_amd_gets_yarn_unless_dynamic_is_picked(self):
        hip = {**self.BASE, "backend": "hip"}                             # not measured on AMD: yarn, said why
        new, _ = runconfig.apply(hip, {"max_context": 524288})
        self.assertEqual((self.rope(new), new["rope_auto"]), (("524288", "yarn", "2"), True))
        got = {k["key"]: k for k in runconfig.view(new, "x.json")["keys"]}["rope_scaling"]
        self.assertIn("now: yarn, factor 2; dynamic is measured on NVIDIA only so far", got["help"])
        mine, _ = runconfig.apply(new, {"rope_scaling": "dynamic"})      # picked: kept, no refusal
        self.assertEqual(self.rope(mine), ("524288", "dynamic", "2"))
        rope = lambda a: a[a.index("--rope-scaling") + 1]   # noqa: E731
        self.assertEqual(rope(engine_args(mine)), "dynamic")
        with contextlib.redirect_stdout(io.StringIO()):                   # an automatic dynamic on a HIP build
            self.assertEqual(rope(engine_args({**new, "args": list(mine["args"])})), "yarn")

    def test_the_engine_gets_yarn_where_dynamic_cannot_run(self):
        new, _ = runconfig.apply(self.BASE, {"max_context": 524288})
        rope = lambda a: a[a.index("--rope-scaling") + 1]   # noqa: E731
        self.assertEqual(rope(engine_args(new)), "dynamic")
        for hand in ({"gpu": [0, 1]}, {"parallel": 2}):                   # set by hand or at start, not on the page
            with self.subTest(hand=hand), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(rope(engine_args({**new, **hand})), "yarn")
        mine = {**new, "rope_auto": False, "parallel": 2}                 # the user's own: the engine says why
        self.assertEqual(rope(engine_args(mine)), "dynamic")

    def test_the_page_keeps_a_required_choice(self):
        js = (ROOT / "serve" / "web" / "app.js").read_text(encoding="utf-8")
        self.assertIn("k.required && v != null", js)                       # no empty choice for the context


class Http(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "strata-q2_0.json"
        self.path.write_text(json.dumps(CFG, indent=1), encoding="utf-8")
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.config_path = str(self.path)
        self.httpd = serve(self.svc, port=0)
        self.host = f"127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.dir.cleanup()

    def call(self, method, body=None, headers=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        req = urllib.request.Request(f"http://{self.host}/config", method=method, headers=h,
                                     data=None if body is None else (body if isinstance(body, bytes)
                                                                     else json.dumps(body).encode()))
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def saved(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_read_and_write_from_the_page(self):
        status, v = self.call("GET")
        self.assertEqual(status, 200, v)
        self.assertEqual({k["key"]: k["value"] for k in v["keys"]}["sampling.temperature"], 0.7)
        with contextlib.redirect_stdout(io.StringIO()):
            status, v = self.call("POST", {"set": {"sampling.temperature": 1.0, "open_browser": False}},
                                  {"Origin": f"http://{self.host}"})
        self.assertEqual(status, 200, v)
        self.assertEqual(v["changed"], ["sampling.temperature", "open_browser"])
        cfg = self.saved()
        self.assertEqual(cfg["sampling"]["temperature"], 1.0)
        self.assertIs(cfg["open_browser"], False)
        for k in ("mcp_servers", "cors_origins", "host", "args", "exe"):
            self.assertEqual(cfg[k], CFG[k], k)
        self.assertEqual(json.loads((self.path.parent / "strata-q2_0.json.bak").read_text(encoding="utf-8")), CFG)
        status, v = self.call("POST", {"set": {"open_browser": False}})   # no Origin (a script on this PC): allowed
        self.assertEqual((status, v["changed"]), (200, []))

    def test_only_from_strata_s_own_page(self):
        before = self.path.read_bytes()
        status, _ = self.call("POST", {"set": {"fit_max_tokens": True}}, {"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        status, _ = self.call("POST", b'{"set": {"fit_max_tokens": true}}', {"Content-Type": "text/plain"})
        self.assertEqual(status, 415)
        for body in ({"set": {"api_key": ""}}, {"set": {"mcp_servers": {"x": {"command": "calc"}}}},
                     {"set": {"sampling.top_k": 0}}, {"nothing": 1}, b"not json"):
            status, b = self.call("POST", body, {"Origin": f"http://{self.host}"})
            self.assertEqual(status, 400, (body, b))
        self.assertEqual(self.path.read_bytes(), before)                 # nothing was written
        self.assertFalse((self.path.parent / "strata-q2_0.json.bak").exists())

    def test_the_api_key(self):
        self.svc.api_key = "secret"
        self.assertEqual(self.call("GET")[0], 401)
        self.assertEqual(self.call("POST", {"set": {"fit_max_tokens": True}})[0], 401)
        with contextlib.redirect_stdout(io.StringIO()):
            status, _ = self.call("POST", {"set": {"fit_max_tokens": True}}, {"Authorization": "Bearer secret"})
        self.assertEqual(status, 200)
        self.assertIs(self.saved()["fit_max_tokens"], True)

    def test_without_a_run_config(self):
        self.svc.config_path = None
        self.assertEqual(self.call("GET")[0], 404)
        self.assertEqual(self.call("POST", {"set": {"fit_max_tokens": True}})[0], 404)

    def test_the_page_has_the_view(self):
        web = ROOT / "serve" / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("cfg-card", "cfg-form", "cfg-save", "cfg-msg"):
            self.assertIn(f'id="{el}"', html)
        self.assertIn('fetch("config", {method: "POST", headers: headers(true)', js)


if __name__ == "__main__":
    unittest.main()
