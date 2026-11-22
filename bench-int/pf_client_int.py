"""pf_client.py strata|basalt PROMPT_IDS [extra engine args...] - start one engine with --serve (Strata: the GPTQ tray
profile's args as score_strata.py builds them, DFlash2, ctx 66661; Basalt: our model's DQ8 pack, ctx 66661, int8 KV, as
the community preset), wait for READY, send one GEN 1 request with the prompt, print the DONE line, quit.
Meant as nsys's target (through hidden_run.py), so only the prompt read and one token run on the GPU."""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, r"D:\Projects\strata-dec-int")   # the worktree's strata_control (PF_DRAFT=auto runs its apply())
which, prompt = sys.argv[1], sys.argv[2]
extra = sys.argv[3:]
ids = open(prompt).read().strip()
if which == "strata":
    from strata_control import drafting
    j = json.load(open(r"D:\Projects\Strata-data\strata-qwen-nvfp4-gptq.tray.json", encoding="utf-8"))
    DROP2 = {"--conversation-cache-mib", "--conversation-cache-slots", "--conversation-cache-min-free-mib",
             "--control-vector-scaled", "--cvec-mode", "--cvec-dir", "--prompt-cache", "--max-context"}
    CVEC = os.environ.get("PF_CVEC") == "1"   # PF_CVEC=1: keep the uncensoring projection (the uncensored profile)
    if CVEC:
        DROP2 -= {"--control-vector-scaled", "--cvec-mode", "--cvec-dir"}
    args, it = [], iter([str(a) for a in j["args"]])
    for a in it:
        if a == "--vision":
            continue
        if a in DROP2:
            next(it); continue
        if a == "--control-vector-layer-range" and not CVEC:
            next(it); next(it); continue
        args.append(a)
    if os.environ.get("PF_DRAFT") == "auto":   # the worktree's serve Auto (drafting.apply) on this card
        exe = os.environ.get("PF_EXE", r"C:\LLM_models\strata-nvfp4\engine\strata.exe")
        cfg = {"args": args, "model_name": "qwen3.8-flash-next-nvfp4-gptq" + ("-uncensored" if CVEC else "")}
        d = drafting.apply(cfg, data_dir=drafting.data_dir_of(args), backend="cuda", archs=["sm_120"], vram_mib=32607, engine=exe)
        args = cfg["args"] + ["--prompt-cache", "0", "--max-context", os.environ.get("PF_CTX", "66661")]
        auto_line = "auto: %s %s %s" % (d.drafter, d.note, " ".join(a for a in args if a.startswith("--mtp") or a in ("proxy", "auto")))
    elif os.environ.get("PF_DRAFT", "dflash2") == "mtp":   # PF_DRAFT=mtp: the profile's own MTP flags, no DFlash2
        args += ["--prompt-cache", "0", "--max-context", os.environ.get("PF_CTX", "66661")]
    else:
        args = drafting.strip_mtp_flags(args)
        DD = r"C:\LLM_models\Strata-data\hf\qwen-nvfp4-gptq\dflash2"
        args += ["--draft", "dflash2", "--draft-model", os.path.join(DD, "qwen3.8-flash-next-dflash2.q4_0-selq4.pack"),
                 "--draft-fallback", "none", "--mtp-draft-vocab", os.path.join(DD, "draft_vocab.bin"), "--draft-conf", "0.5",
                 "--prompt-cache", "0", "--max-context", os.environ.get("PF_CTX", "66661")]
    cmd = [os.environ.get("PF_EXE", r"C:\LLM_models\strata-nvfp4\engine\strata.exe"), "--serve"] + args + extra   # PF_EXE: a dev build
    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
    for kv in os.environ.get("PF_ENV", "").split():
        k, _, v = kv.partition("=")
        env[k] = v
else:
    cmd = [r"D:\Projects\basalt\bin\basalt.exe", "--serve", "--model", r"D:\Projects\basalt-work\models\Qwen3.8-FN-oursGPTQ-DQ8.basalt",
           "--gpu", "0", "--max-context", os.environ.get("PF_CTX", "66661"), "--kv", "int8", "--prompt-cache", "0"] + extra
    env = dict(os.environ)
log = open(os.environ.get("PF_LOG", os.devnull), "w", encoding="utf-8", errors="replace")
log.write((globals().get("auto_line", "") + chr(10)) + "cmd: " + " ".join(cmd[2:]) + chr(10)); log.flush()
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, env=env, bufsize=1,
                     cwd=r"D:\Projects\basalt" if which == "basalt" else None)
for line in p.stdout:
    if line.startswith("READY"):
        break
t0 = time.perf_counter()
p.stdin.write("GEN %s %s\n" % (os.environ.get("PF_GEN", "1"), ids))   # PF_GEN: tokens to generate (greedy)
p.stdin.flush()
for line in p.stdout:
    if line.startswith("DONE") or line.startswith("ERR"):
        print(line.strip(), "| wall %.0f ms" % ((time.perf_counter() - t0) * 1000), flush=True)
        break
p.stdin.write("QUIT\n")
p.stdin.flush()
p.wait(timeout=300)
