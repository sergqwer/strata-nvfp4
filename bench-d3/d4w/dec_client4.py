"""dec_client4.py (D4 copy of rel04/dec_client.py + PF_ADD extra args)
dec_client.py PROFILE TREE PROMPT_IDS - one engine --serve start with a tray profile's args, the given tree's serve Auto
(drafting.apply), one greedy GEN request, quit.  rel04 (0.2.15-nvfp4.4 release checks), after bench-int/pf_client_int.py.
PROFILE: gptq | unc (the two *.tray.json; unc keeps the cvec projection).  TREE: the folder whose strata_control decides
Auto (the .3 tray install read-only, or the .4 bundle).  env: PF_EXE engine, PF_GEN tokens, PF_CTX (default: the profile's
--max-context), PF_MTP (replace the profile's --mtp folder), PF_ENV (STRATA_* vars), PF_LOG.  No key or header is printed."""
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True                 # never write __pycache__ into the tray install
prof, tree, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, tree)
from strata_control import drafting           # noqa: E402

TRAY = {"gptq": r"D:\Projects\Strata-data\strata-qwen-nvfp4-gptq.tray.json",
        "unc": r"D:\Projects\Strata-data\strata-qwen-nvfp4-gptq-uncensored.tray.json"}[prof]
j = json.load(open(TRAY, encoding="utf-8"))
ids = open(prompt).read().strip()
CVEC = prof == "unc"
DROP2 = {"--conversation-cache-mib", "--conversation-cache-slots", "--conversation-cache-min-free-mib", "--prompt-cache"}
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
if os.environ.get("PF_MTP"):
    args[args.index("--mtp") + 1] = os.environ["PF_MTP"]
if os.environ.get("PF_CTX"):
    args[args.index("--max-context") + 1] = os.environ["PF_CTX"]
exe = os.environ["PF_EXE"]
env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
for kv in os.environ.get("PF_ENV", "").split():
    k, _, v = kv.partition("=")
    env[k] = v
    os.environ[k] = v                          # STRATA_AUTO_NEW_MTP=0 reaches apply() too
cfg = {"args": args, "model_name": j["model_name"]}
d = drafting.apply(cfg, data_dir=drafting.data_dir_of(args), backend="cuda", archs=["sm_120"], vram_mib=32607, engine=exe)
args = cfg["args"] + ["--prompt-cache", "0"]
args += os.environ.get("PF_ADD", "").split()   # D4: extra engine args (e.g. --pcie-mode dma)
dr = [a for a in args if a.startswith("--mtp") or a.startswith("--draft") or a in ("--spec",)]
auto_line = "auto: %s note=%s | %s" % (d.drafter, d.note, " ".join(
    "%s %s" % (a, args[i + 1]) if i + 1 < len(args) and not args[i + 1].startswith("--") else a
    for i, a in enumerate(args) if a in dr))
cmd = [exe, "--serve"] + args
log = open(os.environ.get("PF_LOG", os.devnull), "w", encoding="utf-8", errors="replace")
log.write(auto_line + "\n" + "env: " + os.environ.get("PF_ENV", "") + "\n"); log.flush()
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, env=env, bufsize=1)
for line in p.stdout:
    if line.startswith("READY"):
        break
t0 = time.perf_counter()
p.stdin.write("GEN %s %s\n" % (os.environ.get("PF_GEN", "1"), ids))
p.stdin.flush()
for line in p.stdout:
    if line.startswith("DONE") or line.startswith("ERR"):
        print(auto_line)
        print(line.strip(), "| wall %.0f ms" % ((time.perf_counter() - t0) * 1000), flush=True)
        break
p.stdin.write("QUIT\n")
p.stdin.flush()
try:
    p.wait(timeout=300)
except subprocess.TimeoutExpired:
    p.kill()
