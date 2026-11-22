"""venc_client.py EXE OUTDIR [PICS...] - one engine --serve start with the GPTQ tray profile's args (as rel04/dec_client.py: no
conversation cache, the profile's drafter args as written) plus --vision-gpu, then VENC per picture (each twice: a cold and a warm
read), an optional GEN on a token file (VC_GEN=N VC_IDS=file) after the pictures, QUIT.  Prints each answer with the wall time; the
engine's stderr goes to OUTDIR/venc.log.  env: VC_ENV="K=V ..." (STRATA_* for the engine), VC_PRELOAD=1 (--vision-gpu-preload)."""
import json
import os
import subprocess
import sys
import time

exe, outdir = sys.argv[1], sys.argv[2]
pics = sys.argv[3:]
os.makedirs(outdir, exist_ok=True)
j = json.load(open(r"D:\Projects\Strata-data\strata-qwen-nvfp4-gptq.tray.json", encoding="utf-8"))
DROP2 = {"--conversation-cache-mib", "--conversation-cache-slots", "--conversation-cache-min-free-mib", "--prompt-cache"}
args, it = [], iter([str(a) for a in j["args"]])
for a in it:
    if a in DROP2:
        next(it)
        continue
    args.append(a)
mm = j["vision"]["mmproj"]
args += ["--prompt-cache", "0", "--vision-gpu", mm, "--vision-max-tokens", str(j["vision"].get("max_tokens", 1024))]
if os.environ.get("VC_PRELOAD") == "1":
    args.append("--vision-gpu-preload")
env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
for kv in os.environ.get("VC_ENV", "").split():
    k, _, v = kv.partition("=")
    env[k] = v
log = open(os.path.join(outdir, "venc.log"), "w", encoding="utf-8", errors="replace")
t_start = time.perf_counter()
p = subprocess.Popen([exe, "--serve"] + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, env=env, bufsize=1,
                     cwd=j.get("cwd"))
print("pid", p.pid, flush=True)
for line in p.stdout:
    if line.startswith("READY"):
        break
print("ready in %.1f s" % (time.perf_counter() - t_start), flush=True)


def ask(cmd, ends=("VOK", "ERR", "DONE")):
    t0 = time.perf_counter()
    p.stdin.write(cmd + "\n")
    p.stdin.flush()
    for line in p.stdout:
        if line.startswith(ends):
            return line.strip(), (time.perf_counter() - t0) * 1000
    return "EOF", 0


try:
    for k, pic in enumerate(pics):
        tag = os.path.splitext(os.path.basename(pic))[0]
        for rep in ("a", "b"):
            out = os.path.join(outdir, "gpu_%s_%s.sve" % (tag, rep))
            ans, ms = ask("VENC %s %s" % (os.path.abspath(pic), os.path.abspath(out)))
            print("%s %s: %s | wall %.0f ms" % (tag, rep, ans, ms), flush=True)
    if os.environ.get("VC_GEN"):
        ids = open(os.environ["VC_IDS"]).read().strip()
        ans, ms = ask("GEN %s %s" % (os.environ["VC_GEN"], ids), ends=("DONE", "ERR"))
        print("gen: %s | wall %.0f ms" % (ans[:200], ms), flush=True)
finally:
    try:
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        p.wait(timeout=120)
    except Exception:
        p.kill()
