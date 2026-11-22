"""make_128k.py - two 131,000-token teacher-forcing items (English Wikipedia; the en and uk pools + source files joined)
+ 256 forced tokens, from basalt-work/corpus/long's sources (make_long.py) -> set_128k.json"""
import glob, json, os, sys
sys.path.insert(0, r"D:\Projects\basalt\tools")
from tokenizer import Tokenizer
D = r"D:\Projects\basalt-work\corpus\long"
tk = Tokenizer.from_gguf(r"C:\LLM_models\Strata-data\hf\qwen-nvfp4-gptq\qwen-nvfp4-gptq-dense.gguf")
def wiki(p):
    pages = json.load(open(p, encoding="utf-8"))["query"]["pages"]
    return next(iter(pages.values()))["extract"]
en = tk.encode("\n\n".join(wiki(p) for p in sorted(glob.glob(os.path.join(D, "src", "en_*.json")))))
uk = tk.encode("\n\n".join(wiki(p) for p in sorted(glob.glob(os.path.join(D, "src", "uk_*.json")))))
REL = r"D:\Projects\strata-rel0215"
code = tk.encode(open(os.path.join(REL, r"src\program\generate.cpp"), encoding="utf-8", errors="replace").read())
print("pools en %d uk %d code %d" % (len(en), len(uk), len(code)))
N, C = 131000, 256
items = []
mix1 = en + uk          # English then Ukrainian text, continuation in Ukrainian
mix2 = code + en        # a source file, then English text
for name, src in (("enuk", mix1), ("codeen", mix2)):
    assert len(src) >= N + C, (name, len(src))
    s = len(src) - (N + C)
    w = src[s:s + N + C]
    items.append({"id": "%s-%d" % (name, N), "kind": name, "prompt": w[:N], "cont": w[N:]})
json.dump({"items": items}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "set_128k.json"), "w"))
print(len(items), "items")
