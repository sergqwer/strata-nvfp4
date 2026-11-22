"""sum.py GLOB... - one line per serve log: tok/s, avg T, tokens/window, ms/window, verify, GPU-reach wait, per-layer
CPU, draft, CPU experts / PCIe per layer-window, mtp per-step time."""
import glob
import re
import sys

R_DONE = re.compile(r"(\d+) generated in (\d+) ms \(([0-9.]+) tok/s\)")
R_DT = re.compile(r"decode timing: (\d+) windows, avg T ([0-9.]+), ([0-9.]+) tokens/window, ([0-9.]+) ms/window = .*?verify ([0-9.]+) "
                  r"\(GPU-reach wait ([0-9.]+) \+ per-layer host ([0-9.]+) \[plan [0-9.]+ actq [0-9.]+ jobs [0-9.]+ CPU ([0-9.]+)\].*?"
                  r"draft ([0-9.]+).*?CPU experts ([0-9.]+) \(([0-9.]+) entries\), VRAM hits ([0-9.]+), PCIe ([0-9.]+)")
R_STEP = re.compile(r"\(([0-9.]+) per step\)")
print("%-34s %6s %5s %5s %6s %6s %6s %6s %5s %5s %5s %5s %6s" % ("log", "tok/s", "T", "tpw", "ms/w", "verify", "reach",
                                                                "CPU", "draft", "cpuE", "ent", "pcie", "step"))
files = []
for g in sys.argv[1:]:
    files += sorted(glob.glob(g))
for f in files:
    t = open(f, encoding="utf-8", errors="replace").read()
    d = R_DONE.findall(t)
    m = R_DT.findall(t)
    st = R_STEP.findall(t)
    name = f.replace("\\", "/").split("/")[-1].replace(".log", "")
    if not d or not m:
        print("%-34s (incomplete)" % name)
        continue
    m = m[-1]
    print("%-34s %6.1f %5.2f %5.2f %6.2f %6.2f %6.2f %6.2f %5.2f %5.2f %5.2f %5.2f %6s" % (
        name[:34], float(d[-1][2]), float(m[1]), float(m[2]), float(m[3]), float(m[4]), float(m[5]), float(m[7]),
        float(m[8]), float(m[9]), float(m[10]), float(m[12]), st[-1] if st else "-"))
