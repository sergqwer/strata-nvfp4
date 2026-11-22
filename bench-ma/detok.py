"""detok.py LOG... - the greedy output of engine logs as text (the Qwen3.5-family byte-level BPE vocab, 248,320 ids)"""
import json, sys
VJ = r"C:\LLM_models\Strata-data\hf\edge0-145b-nvfp4\tokenizer\vocab.json"
inv = {v: k for k, v in json.load(open(VJ, encoding="utf-8")).items()}
bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b); cs.append(256 + n); n += 1
dec = {chr(c): b for b, c in zip(bs, cs)}


def text(ids):
    out = bytearray()
    for i in ids:
        s = inv.get(i, "<%d>" % i)
        out += s.encode() if s.startswith("<|") else bytes(dec.get(ch, 63) for ch in s)
    return out.decode("utf-8", "replace")


for p in sys.argv[1:]:
    for line in open(p, encoding="utf-8", errors="replace"):
        if "output  :" in line:
            print("==", p.replace("\\", "/").split("/")[-1])
            print(text([int(x) for x in line.split("output  :")[1].split()]))
            break
