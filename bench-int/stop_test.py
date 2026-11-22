"""stop_test.py - serve: request 1 (struct, 2000 greedy) is STOPped after ~1 s, request 2 (prose, 300 greedy) follows at
once; then a fresh engine runs request 2 alone.  Prints whether request 2's ids are the same both ways (the STOPped
window's adaptive round must not leak into the next request).  PF_EXE / PF_ENV / args as pf_client_int.py (Auto)."""
import os, subprocess, sys, time, json, threading
sys.path.insert(0, r"D:\Projects\strata-dec-int")
from strata_control import drafting
EXE = os.environ["PF_EXE"]
W = r"D:\Projects\basalt-work\fixtures"
j = json.load(open(r"D:\Projects\Strata-data\strata-qwen-nvfp4-gptq.tray.json", encoding="utf-8"))
DROP2 = {"--conversation-cache-mib", "--conversation-cache-slots", "--conversation-cache-min-free-mib", "--control-vector-scaled",
         "--cvec-mode", "--cvec-dir", "--prompt-cache", "--max-context"}
args, it = [], iter([str(a) for a in j["args"]])
for a in it:
    if a == "--vision": continue
    if a in DROP2: next(it); continue
    if a == "--control-vector-layer-range": next(it); next(it); continue
    args.append(a)
cfg = {"args": args, "model_name": "qwen3.8-flash-next-nvfp4-gptq"}
drafting.apply(cfg, data_dir=drafting.data_dir_of(args), backend="cuda", archs=["sm_120"], vram_mib=32607, engine=EXE)
cmd = [EXE, "--serve"] + cfg["args"] + ["--prompt-cache", "0", "--max-context", "66661"]
env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
for kv in os.environ.get("PF_ENV", "").split():
    k, _, v = kv.partition("="); env[k] = v
s1 = open(os.path.join(W, "struct.txt")).read().strip()
s2 = open(os.path.join(W, "prose.txt")).read().strip()
logdir = os.environ.get("ST_LOG", ".")

def run(seq, tag):
    log = open(os.path.join(logdir, tag + ".engine.log"), "w", encoding="utf-8", errors="replace")
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, env=env, bufsize=1)
    for line in p.stdout:
        if line.startswith("READY"): break
    outs = []
    for kind, ids, n in seq:
        p.stdin.write("GEN %d %s\n" % (n, ids)); p.stdin.flush()
        toks, t0, stopped = [], time.time(), False
        for line in p.stdout:
            if line.startswith("T "): toks.append(int(line.split()[1]))
            if kind == "stop" and not stopped and len(toks) >= 300:
                p.stdin.write("STOP\n"); p.stdin.flush(); stopped = True
            if line.startswith("DONE") or line.startswith("ERR"):
                outs.append((toks, line.strip()[:60])); break
    p.stdin.write("QUIT\n"); p.stdin.flush(); p.wait(timeout=120)
    return outs

import hashlib
tag = os.environ.get("ST_TAG", "stop")
a = run([("stop", s1, 2000), ("plain", s2, 300)], tag)
h = lambda t: hashlib.md5(",".join(map(str, t)).encode()).hexdigest()[:10]
print("%s | request 1: %d tokens (STOP sent at 300) %s | request 2: %d tokens md5 %s" % (tag, len(a[0][0]), a[0][1], len(a[1][0]), h(a[1][0])))
