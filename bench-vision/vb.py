"""vb.py - vision GPU bench: one serve start with the GPTQ tray profile's args, then text / image / text turns.

  python -B vb.py TAG TREE [--exe ENGINE] [--vision k=v ...] [--top k=v ...] [--env K=V ...] [--seq SEQ]
TREE: the folder whose serve/ runs (cwd).  SEQ: comma list of t (text decode turn), s / b / c (image turns: shapes ~240,
big1024 ~1024, code4217 ~350 tokens; each gets a fresh tEXt chunk so the server cache never hits), wN (sleep N s).
Writes bench-vision/out/TAG.json + TAG.log (server log) + TAG.smi (nvidia-smi memory.used every 100 ms).
Never prints the tray config's key (the profile json has none; the request uses no key)."""
import base64, json, os, subprocess, sys, threading, time, urllib.request, zlib, struct

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
PY = r"C:\LLM_models\strata-nvfp4\.venv-serve\Scripts\python.exe"
TRAY = r"D:\Projects\Strata-data\strata-qwen-nvfp4-gptq.tray.json"
PICS = {"s": r"D:\Projects\strata-vision-gpu\tools\qwen35\pictures\shapes.png",
        "b": r"D:\Projects\strata-vision-gpu\tools\qwen35\pictures\big1024.png",
        "c": r"D:\Projects\strata-vision-gpu\tools\qwen35\pictures\code4217.png"}
QUEST = {"s": "Describe this picture in one sentence: which shapes, which colours, where.",
         "b": "Describe this picture in one sentence.",
         "c": "Read the text in this image exactly."}
TEXT = "Explain how a hash table works: hashing, collisions, load factor and resizing. About 400 words."
PORT = int(os.environ.get("VB_PORT", "8795"))


def uniq_png(path, salt):
    d = open(path, "rb").read()
    i = d.rfind(b"IEND") - 4
    body = b"tEXt" + b"vb\x00" + salt.encode()
    chunk = struct.pack(">I", len(body) - 4) + body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)
    return d[:i] + chunk + d[i:]


def main():
    tag, tree = sys.argv[1], sys.argv[2]
    a = sys.argv[3:]
    exe, vis, top, env_add, seq = None, {}, {}, {}, "t,s,t,b,t"
    k = 0
    while k < len(a):
        if a[k] == "--exe": exe = a[k + 1]; k += 2
        elif a[k] == "--seq": seq = a[k + 1]; k += 2
        elif a[k] in ("--vision", "--top", "--env"):
            key, _, v = a[k + 1].partition("=")
            v = json.loads(v) if v[:1] in "[{0123456789tfn-\"" and v not in ("",) else v
            {"--vision": vis, "--top": top, "--env": env_add}[a[k]][key] = v if a[k] != "--env" else a[k + 1].partition("=")[2]
            k += 2
        else: raise SystemExit("unknown " + a[k])
    cfg = json.load(open(TRAY, encoding="utf-8"))
    for key in list(cfg):
        if "key" in key.lower(): cfg.pop(key)
    if exe: cfg["exe"] = exe
    cfg["vision"].update(vis)
    for key, v in top.items():
        if v is None: cfg.pop(key, None)
        else: cfg[key] = v
    log = os.path.join(OUT, tag + ".log")
    cfg["log"] = log
    cfg["app_ui"] = False
    cfg["host"], cfg["port"] = "127.0.0.1", PORT
    cpath = os.path.join(OUT, tag + ".cfg.json")
    json.dump(cfg, open(cpath, "w", encoding="utf-8"), indent=1)
    open(log, "w").close()
    smi = open(os.path.join(OUT, tag + ".smi"), "w")
    sp = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,memory.used", "--format=csv,noheader,nounits", "-lms", "100"],
                          stdout=smi, stderr=subprocess.DEVNULL)
    env = {kk: vv for kk, vv in os.environ.items() if not kk.startswith("STRATA_")}
    env.update(env_add)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    srv = subprocess.Popen([PY, "-B", "-m", "serve.server", "--engine", "strata", "--config", cpath, "--host", "127.0.0.1",
                            "--port", str(PORT)], cwd=tree, env=env, stdout=open(log + ".out", "w"), stderr=subprocess.STDOUT)
    res = {"tag": tag, "tree": tree, "exe": cfg["exe"], "vision": cfg["vision"], "events": [], "pid": srv.pid}
    t_start = time.time()

    def mark(name, **kw):
        ev = {"t": round(time.time() - t_start, 3), "wall": time.time(), "ev": name, **kw}
        res["events"].append(ev)
        print(json.dumps(ev, ensure_ascii=False)[:400], flush=True)

    def chat(content, max_tokens):
        body = {"model": cfg.get("model_name", "m"), "max_tokens": max_tokens, "temperature": 0, "stream": True,
                "messages": [{"role": "user", "content": content}]}
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter(); ttft = None; text = ""; timings = None
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]": continue
                x = json.loads(line[5:])
                if x.get("timings"): timings = x["timings"]
                for ch in x.get("choices") or []:
                    dl = ch.get("delta") or {}
                    piece = (dl.get("content") or "") + (dl.get("reasoning_content") or "")
                    if piece and ttft is None: ttft = time.perf_counter() - t0
                    text += piece
        return {"ttft_ms": round(1000 * (ttft or 0), 1), "wall_ms": round(1000 * (time.perf_counter() - t0), 1),
                "tok_s": (timings or {}).get("predicted_per_second"), "prompt_ms": (timings or {}).get("prompt_ms"),
                "prompt_n": (timings or {}).get("prompt_n"), "gen": (timings or {}).get("predicted_n"),
                "md5": __import__("hashlib").md5(text.encode()).hexdigest()[:10], "text": text[-160:].replace(chr(10), " ")}

    try:
        dl = time.time() + 600
        while time.time() < dl:
            if srv.poll() is not None: raise SystemExit("server ended")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=5).read(); break
            except Exception:
                time.sleep(2)
        mark("http_up")
        r = chat("Say OK.", 4)           # the engine's load (lazy engines start here)
        mark("warm", **r)
        n = 0
        for step in seq.split(","):
            n += 1
            if step == "t":
                mark("text", **chat(TEXT, 400))
            elif step[0] == "w":
                time.sleep(float(step[1:])); mark("slept", s=float(step[1:]))
            else:
                img = base64.b64encode(uniq_png(PICS[step], f"{tag}-{n}")).decode()
                content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + img}},
                           {"type": "text", "text": QUEST[step]}]
                mark("image_" + step, **chat(content, 64))
    finally:
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(srv.pid)], capture_output=True, timeout=60)
            srv.wait(timeout=60)
        except Exception:
            srv.kill()
        time.sleep(3)
        sp.terminate()
        json.dump(res, open(os.path.join(OUT, tag + ".json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)


main()
