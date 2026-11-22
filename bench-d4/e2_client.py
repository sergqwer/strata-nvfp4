"""e2_client.py (E2): r4/dec_client4.py + the reply's tokens and several requests in one engine start.
e2_client.py PROFILE TREE REQ [REQ...] - REQ = PROMPT_FILE:GEN, or "cont<k>:GEN" = request k's prompt + its reply (the
parked conversation comes back from the dialog cache).  Each request prints: idx, n_prompt, n_gen, md5 of the reply ids,
the DONE line.  env as dec_client4 (PF_EXE, PF_ENV, PF_LOG, PF_ADD, PF_CTX); PF_CONV=1 keeps the profile's dialog-cache
args (--conversation-cache-*); PF_IDS=DIR writes each reply's ids there.  No key or header is printed."""
import hashlib
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
prof, tree, reqs = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path.insert(0, tree)
from strata_control import drafting           # noqa: E402

TRAY = {"gptq": r"D:\Projects\Strata-data\strata-qwen-nvfp4-gptq.tray.json",
        "unc": r"D:\Projects\Strata-data\strata-qwen-nvfp4-gptq-uncensored.tray.json"}[prof]
j = json.load(open(TRAY, encoding="utf-8"))
CVEC = prof == "unc"
DROP2 = {"--prompt-cache"}
if os.environ.get("PF_CONV") != "1":
    DROP2 |= {"--conversation-cache-mib", "--conversation-cache-slots", "--conversation-cache-min-free-mib"}
if not CVEC:
    DROP2 |= {"--control-vector-scaled", "--cvec-mode", "--cvec-dir"}
args, it = [], iter([str(a) for a in j["args"]])
for a in it:
    if a == "--vision":
        continue
    if a in DROP2:
        next(it); continue
    if a == "--control-vector-layer-range" and not CVEC:
        next(it); next(it); continue
    args.append(a)
if os.environ.get("PF_CTX"):
    args[args.index("--max-context") + 1] = os.environ["PF_CTX"]
exe = os.environ["PF_EXE"]
env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
for kv in os.environ.get("PF_ENV", "").split():
    k, _, v = kv.partition("=")
    env[k] = v
    os.environ[k] = v
cfg = {"args": args, "model_name": j["model_name"]}
d = drafting.apply(cfg, data_dir=drafting.data_dir_of(args), backend="cuda", archs=["sm_120"], vram_mib=32607, engine=exe)
args = cfg["args"] + ([] if os.environ.get("PF_CONV") == "1" else ["--prompt-cache", "0"])   # PF_CONV: the tray's default prompt cache
args += os.environ.get("PF_ADD", "").split()
cmd = [exe, "--serve"] + args
log = open(os.environ.get("PF_LOG", os.devnull), "w", encoding="utf-8", errors="replace")
log.write("env: " + os.environ.get("PF_ENV", "") + " add: " + os.environ.get("PF_ADD", "") + "\n"); log.flush()
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, env=env, bufsize=1)
for line in p.stdout:
    if line.startswith("READY"):
        break
done = []   # (prompt ids, reply ids)
for k, r in enumerate(reqs):
    src, _, gen = r.rpartition(":")
    if src.startswith("cont"):
        a = done[int(src[4:])]
        ids = a[0] + a[1]
    else:
        ids = open(src).read().strip().split(",")
    t0 = time.perf_counter()
    p.stdin.write("GEN %s %s\n" % (gen, ",".join(ids)))
    p.stdin.flush()
    out = []
    end = ""
    for line in p.stdout:
        if line.startswith("T "):
            out.append(line.split()[1])
        elif line.startswith("DONE") or line.startswith("ERR"):
            end = line.strip()
            break
    if os.environ.get("PF_IDS"):
        open(os.path.join(os.environ["PF_IDS"], "reply%d.txt" % k), "w").write(",".join(out))
    md5 = hashlib.md5(",".join(out).encode()).hexdigest()[:10]
    print("req %d prompt %d gen %d md5 %s | %s | wall %.0f ms" % (k, len(ids), len(out), md5, end[:120],
                                                               (time.perf_counter() - t0) * 1000), flush=True)
    done.append((ids, out))
    if end.startswith("ERR"):
        break
p.stdin.write("QUIT\n")
p.stdin.flush()
try:
    p.wait(timeout=300)
except subprocess.TimeoutExpired:
    p.kill()
