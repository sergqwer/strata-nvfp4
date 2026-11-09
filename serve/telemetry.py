"""serve/telemetry.py - hardware readings for the web app's Monitor tab (idea from PR #22 by code-martin).

A background thread samples once a second and keeps the last 60 readings of each series for the sparklines:
- GPU: NVIDIA's own NVML library (nvml.dll / libnvidia-ml.so.1, installed with every driver) through ctypes, so no
  pip package is needed: load, VRAM, temperature, power, PCIe link and throughput.  With the AMD backend (#301): the
  amdgpu driver's Linux sysfs files - load, VRAM, temperature and power.  On macOS (Metal backend): IOKit's
  IOAccelerator registry - load and VRAM; the temperature and power sensors need root (`powermetrics`), and an Apple
  Silicon GPU sits on the die, so there is no PCIe link.
- CPU, RAM, disk: `psutil` when it is installed (setup installs it); without it the CPU and RAM readings fall back to
  the OS (Windows GlobalMemoryStatusEx / GetSystemTimes, Linux /proc) and the disk rate is absent.
Anything that cannot be read is None; nothing here can stop the server.
"""
from __future__ import annotations

import collections
import ctypes
import os
from strata_control import sysinfo
import re
import subprocess
import sys
import threading
import time

HISTORY = 60
OTHER_CARDS_S = 5.0     # the cards the engine does not use are read this much less often (they only fill the GPU list)


def gpu_poll_s() -> float:
    """STRATA_GPU_POLL_S: seconds between GPU readings (default 1; 0 = no GPU readings after the cards' names).  #905:
    on Linux with GSP, an NVML query between requests stalled CUDA-graph branches (STRATA_DF_BRANCH=1, opt-in since):
    this turns the queries down or off without touching the CPU, RAM and disk readings."""
    try:
        return max(0.0, float(os.environ.get("STRATA_GPU_POLL_S", "1") or 1))
    except ValueError:
        return 1.0


# ------------------------------------------------------------------------------------------------ NVML
_nvml_loaded = None
_nvml_lock = threading.Lock()


def _nvml_lib():
    """NVIDIA's NVML (nvml.dll / libnvidia-ml.so.1, installed with every driver), loaded and initialised once; None
    without it (no NVIDIA driver) - asked again next time, so a driver that comes up later is found."""
    global _nvml_loaded
    with _nvml_lock:
        if _nvml_loaded is not None:
            return _nvml_loaded
        names = ["nvml.dll", os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                          "NVIDIA Corporation", "NVSMI", "nvml.dll")] if os.name == "nt" \
            else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
        lib = None
        for n in names:
            try:
                lib = ctypes.CDLL(n)
                break
            except OSError:
                continue
        if lib is None:
            return None
        try:
            init = getattr(lib, "nvmlInit_v2", None) or lib.nvmlInit
            if init() != 0:
                return None
        except (AttributeError, OSError):
            return None
        _nvml_loaded = lib
        return lib


def nvml_count() -> int:
    """How many NVIDIA cards NVML sees (0 without the driver)."""
    lib = _nvml_lib()
    if lib is None:
        return 0
    n = ctypes.c_uint()
    try:
        get = getattr(lib, "nvmlDeviceGetCount_v2", None) or lib.nvmlDeviceGetCount
        return n.value if get(ctypes.byref(n)) == 0 else 0
    except (AttributeError, OSError):
        return 0


class _Nvml:
    class Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    class Pci(ctypes.Structure):                                       # nvmlPciInfo_t (v2 and v3 share it)
        _fields_ = [("busIdLegacy", ctypes.c_char * 16), ("domain", ctypes.c_uint), ("bus", ctypes.c_uint),
                    ("device", ctypes.c_uint), ("pciDeviceId", ctypes.c_uint), ("pciSubSystemId", ctypes.c_uint),
                    ("busId", ctypes.c_char * 32)]

    def __init__(self, index=0):
        self.lib = self.dev = None
        self.lib = _nvml_lib()
        if self.lib is None:
            return
        try:
            h = ctypes.c_void_p()
            get = getattr(self.lib, "nvmlDeviceGetHandleByIndex_v2", None) or self.lib.nvmlDeviceGetHandleByIndex
            if get(ctypes.c_uint(index), ctypes.byref(h)) != 0:
                self.lib = None
                return
            self.dev = h
        except (AttributeError, OSError):
            self.lib = None

    def ok(self):
        return self.lib is not None and self.dev is not None

    def _uint(self, fn, *args):
        v = ctypes.c_uint()
        try:
            return v.value if getattr(self.lib, fn)(self.dev, *args, ctypes.byref(v)) == 0 else None
        except (AttributeError, OSError):
            return None

    def name(self):
        buf = ctypes.create_string_buffer(96)
        try:
            if self.lib.nvmlDeviceGetName(self.dev, buf, ctypes.c_uint(96)) == 0:
                return buf.value.decode(errors="replace")
        except (AttributeError, OSError):
            pass
        return None

    def info(self):
        """The card's fixed facts, read once: name, UUID and PCI bus id (what a CUDA ordinal is matched by), and the
        best PCIe link this card gets in this slot (pcie_gen_max / pcie_width_max: "possible with this device and
        system", NVML says) and the card's own best generation (pcie_gen_gpu_max: a Gen5 card in a Gen4 slot)."""
        from serve.gpus import pci_key
        out = {"name": self.name(), "uuid": None, "pci": None,
               "pcie_gen_max": self._uint("nvmlDeviceGetMaxPcieLinkGeneration"),
               "pcie_width_max": self._uint("nvmlDeviceGetMaxPcieLinkWidth"),
               "pcie_gen_gpu_max": self._uint("nvmlDeviceGetGpuMaxPcieLinkGeneration")}
        buf = ctypes.create_string_buffer(96)
        try:
            if self.lib.nvmlDeviceGetUUID(self.dev, buf, ctypes.c_uint(96)) == 0:
                out["uuid"] = buf.value.decode(errors="replace")
        except (AttributeError, OSError):
            pass
        p = self.Pci()
        try:
            get = getattr(self.lib, "nvmlDeviceGetPciInfo_v3", None) or getattr(self.lib, "nvmlDeviceGetPciInfo_v2")
            if get(self.dev, ctypes.byref(p)) == 0:
                out["pci"] = pci_key(p.busId.decode(errors="replace")) or pci_key(p.busIdLegacy.decode(errors="replace"))
        except (AttributeError, OSError):
            pass
        return out

    def read(self):
        out = {}
        u = self.Util()
        try:
            if self.lib.nvmlDeviceGetUtilizationRates(self.dev, ctypes.byref(u)) == 0:
                out["util"] = u.gpu
        except (AttributeError, OSError):
            pass
        m = self.Mem()
        try:
            if self.lib.nvmlDeviceGetMemoryInfo(self.dev, ctypes.byref(m)) == 0:
                out["mem_used"], out["mem_total"] = m.used, m.total
        except (AttributeError, OSError):
            pass
        out["temp"] = self._uint("nvmlDeviceGetTemperature", ctypes.c_uint(0))          # NVML_TEMPERATURE_GPU
        mw = self._uint("nvmlDeviceGetPowerUsage")
        out["power"] = mw / 1000.0 if mw is not None else None
        lim = self._uint("nvmlDeviceGetEnforcedPowerLimit")
        out["power_limit"] = lim / 1000.0 if lim is not None else None
        out["pcie_gen"] = self._uint("nvmlDeviceGetCurrPcieLinkGeneration")        # drops at idle (power saving)
        out["pcie_gen_max"] = self._uint("nvmlDeviceGetMaxPcieLinkGeneration")
        out["pcie_width"] = self._uint("nvmlDeviceGetCurrPcieLinkWidth")
        rx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(1))                 # NVML_PCIE_UTIL_RX_BYTES, KB/s
        tx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(0))
        out["pcie_rx_mb"] = rx / 1024.0 if rx is not None else None
        out["pcie_tx_mb"] = tx / 1024.0 if tx is not None else None
        return out


# ------------------------------------------------------------------------------------------------ AMD (Linux sysfs)
SYSFS = "/sys"


def amd_device_dir(index, sysfs=None):
    """The amdgpu sysfs folder (/sys/class/drm/renderD<N>/device) of the AMD GPU that HIP numbers `index`: the KFD
    topology's GPU nodes in order, the CPU nodes skipped, linked to their render node by drm_render_minor - the
    numbering setup's amd_gpus() and HIP_VISIBLE_DEVICES use.  None when there is no such card (or no amdgpu)."""
    base = os.path.join(sysfs or SYSFS, "class", "kfd", "kfd", "topology", "nodes")
    try:
        nodes = sorted((n for n in os.listdir(base) if n.isdigit()), key=int)
    except OSError:
        return None
    gpus = []
    for n in nodes:
        try:
            with open(os.path.join(base, n, "properties"), encoding="utf-8") as f:
                props = dict(line.strip().partition(" ")[::2] for line in f if line.strip())
            if int(props.get("gfx_target_version") or 0) == 0 or int(props.get("simd_count") or 0) == 0:
                continue
            gpus.append(props)
        except (OSError, ValueError):
            continue
    if not 0 <= index < len(gpus) or not gpus[index].get("drm_render_minor"):
        return None
    dev = os.path.join(sysfs or SYSFS, "class", "drm", "renderD" + gpus[index]["drm_render_minor"].strip(), "device")
    return dev if os.path.isdir(dev) else None


class _Amd:
    """#301: an AMD card's readings from the amdgpu driver's sysfs files (Linux; no ROCm library needed), with _Nvml's
    interface: load (gpu_busy_percent), VRAM (mem_info_vram_used / _total), and from its hwmon folder the temperature
    (temp1_input, the edge sensor, m°C), power (power1_average or power1_input, µW) and its cap (power1_cap)."""

    def __init__(self, index=0, sysfs=None):
        self.dev = amd_device_dir(index, sysfs)
        self.hwmon = None
        if self.dev:
            try:
                hw = sorted(os.listdir(os.path.join(self.dev, "hwmon")))
                self.hwmon = os.path.join(self.dev, "hwmon", hw[0]) if hw else None
            except OSError:
                pass

    def ok(self):
        return self.dev is not None

    @staticmethod
    def _int(path):
        try:
            with open(path, encoding="utf-8") as f:
                return int(f.read().strip())
        except (OSError, ValueError, TypeError):
            return None

    def name(self):
        try:
            with open(os.path.join(self.dev, "product_name"), encoding="utf-8") as f:
                return f.read().strip() or "AMD Radeon"
        except (OSError, TypeError):
            return "AMD Radeon"

    @staticmethod
    def _gen(text):
        """The PCIe generation of a link-speed file's contents ("16.0 GT/s PCIe" -> 4), or None."""
        try:
            return {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}[float(str(text).split()[0])]
        except (ValueError, TypeError, IndexError, KeyError):
            return None

    @staticmethod
    def _bdf(s):
        """True for a sysfs pci device name ("0000:03:00.0"); nothing is imported for it."""
        return (len(s) == 12 and s[4] == ":" and s[7] == ":" and s[10] == "." and s[11] in "01234567"
                and all(c in "0123456789abcdef" for c in s[0:4] + s[5:7] + s[8:10]))

    @staticmethod
    def _read(d, name):
        try:
            with open(os.path.join(d, name), encoding="utf-8") as f:
                return f.read().strip()
        except (OSError, TypeError):
            return None

    def _hops(self):
        """The PCIe devices between the root port and this card, the card last: the sysfs path names every
        hop (a card behind a bridge chain has more than one; a directly attached card has one)."""
        out, prefix = [], []
        for part in os.path.realpath(self.dev or "").split("/"):
            prefix.append(part)
            if self._bdf(part):
                out.append("/".join(prefix))
        return out

    def link(self):
        """The PCIe link the card actually gets: the **narrowest/slowest hop** between the root port and the card,
        from each hop's `max_link_speed` / `max_link_width` (the capability, so a power-saving downgrade or a Gen3
        slot cannot make it read low).  A card that is Gen4 on its own hop but sits behind a Gen3 root port really
        runs at Gen3, and that is the number a PCIe bandwidth budget needs.
        `pcie_own_gen` keeps the card's own hop aside: the gap between the two is what a user has to see.

        The bottleneck is only claimed when **every** hop of the path could be read; `pcie_path` reports how many
        of them were ("4/4").  A kernel or a container that hides part of /sys/devices would otherwise drop the
        unreadable hops in silence and report the card's own Gen4 hop as the whole path - i.e. be optimistic
        exactly where it matters.  When the walk is incomplete the reading falls back to the card's own negotiated
        link and pcie_path says so."""
        hops = self._hops()
        gens, widths, read = [], [], 0
        for d in hops:
            g = self._gen(_Amd._read(d, "max_link_speed"))
            w = self._int(os.path.join(d, "max_link_width"))
            read += 1 if (g is not None or w is not None) else 0
            if g:
                gens.append(g)
            if w:
                widths.append(w)
        own = self.dev or ""
        own_gen = self._gen(_Amd._read(own, "current_link_speed"))
        whole = bool(hops) and read == len(hops)
        gen = min(gens) if (whole and gens) else own_gen
        width = min(widths) if (whole and widths) else self._int(os.path.join(own, "current_link_width"))
        return {"pcie_gen": gen, "pcie_gen_max": gen, "pcie_own_gen": own_gen, "pcie_width": width,
                "pcie_path": "%d/%d" % (read, len(hops))}


    def read(self):
        out = {"util": self._int(os.path.join(self.dev, "gpu_busy_percent")),
               "mem_used": self._int(os.path.join(self.dev, "mem_info_vram_used")),
               "mem_total": self._int(os.path.join(self.dev, "mem_info_vram_total"))}
        if self.hwmon:
            t = self._int(os.path.join(self.hwmon, "temp1_input"))
            out["temp"] = t / 1000.0 if t is not None else None
            p = self._int(os.path.join(self.hwmon, "power1_average"))
            if p is None:
                p = self._int(os.path.join(self.hwmon, "power1_input"))
            out["power"] = p / 1e6 if p is not None else None
            cap = self._int(os.path.join(self.hwmon, "power1_cap"))
            out["power_limit"] = cap / 1e6 if cap is not None else None
        out.update(self.link())
        return out

    def info(self):
        """Name and PCI bus id (the sysfs folder's own name, "0000:03:00.0"); amdgpu has no UUID file to match by."""
        from serve.gpus import pci_key
        own = self.dev or ""
        path = self.link()                 # the slowest / narrowest hop to the root port (#1733): what the slot gives
        return {"name": self.name(), "uuid": None, "pci": pci_key(os.path.basename(os.path.realpath(own))),
                "pcie_gen_max": path.get("pcie_gen"), "pcie_width_max": path.get("pcie_width"),
                "pcie_gen_gpu_max": self._gen(_Amd._read(own, "max_link_speed"))}


class _NoGpu:
    """A card that is not there (a fixture index past its last card)."""
    lib = dev = None

    def ok(self):
        return False

    def name(self):
        return None

    def info(self):
        return {"name": None, "uuid": None, "pci": None}

    def read(self):
        return {}


def metal_working_set():
    """The Apple GPU's memory budget in bytes: MTLCreateSystemDefaultDevice's recommendedMaxWorkingSetSize, through
    the Metal framework and objc_msgSend (no PyObjC) - the same property metal/src/mtl_rt.mm reads for the engine
    (its "GPU working set" log line).  None off macOS, or when it cannot be read: a selector the device does not
    answer would raise an Objective-C exception no ctypes call can catch, so the class is asked first."""
    if sys.platform != "darwin":
        return None
    try:
        metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
        objc = ctypes.CDLL("/usr/lib/libobjc.dylib")
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.object_getClass.argtypes = [ctypes.c_void_p]
        objc.object_getClass.restype = ctypes.c_void_p
        objc.class_respondsToSelector.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        objc.class_respondsToSelector.restype = ctypes.c_ubyte
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        objc.objc_msgSend.restype = ctypes.c_ulonglong
        dev = metal.MTLCreateSystemDefaultDevice()
        if not dev:
            return None
        sel = objc.sel_registerName(b"recommendedMaxWorkingSetSize")
        if not objc.class_respondsToSelector(objc.object_getClass(dev), sel):
            return None
        return int(objc.objc_msgSend(ctypes.c_void_p(dev), ctypes.c_void_p(sel))) or None
    except (OSError, AttributeError, ValueError):
        return None


class _Metal:
    """macOS: an Apple GPU's readings from the IOKit registry (`ioreg -l -w0 -r -c IOAccelerator`, 22 ms), with
    _Nvml's interface: the load and the memory are the driver's PerformanceStatistics of the index'th accelerator
    node - Device Utilization % and the in-use system memory in bytes.  The total is the device's Metal working set,
    the budget one app gets of the unified memory.  No temperature or power reading exists without root
    (`powermetrics`), and an Apple Silicon GPU sits on the die, so the PCIe cards have nothing to show."""
    note = ("no GPU temperature or power reading without root (powermetrics), and an Apple Silicon GPU sits on the "
            "die, so there is no PCIe link to show; the engine's own figures are in its log")
    cmd = ("ioreg", "-l", "-w0", "-r", "-c", "IOAccelerator")

    def __init__(self, index=0, run=None):
        self.index = index
        self._run = run or self._ioreg
        self.nodes = len(self._blocks())
        self._name = None
        b = self._block()
        if b:
            m = re.search(r'"model" = "([^"]+)"', b)
            self._name = m.group(1) if m else None

    @staticmethod
    def _ioreg():
        try:
            return subprocess.run(list(_Metal.cmd), capture_output=True, text=True, timeout=10).stdout or ""
        except (OSError, subprocess.SubprocessError):
            return ""

    def _blocks(self):
        """The accelerator nodes, in the registry's order: every node that carries a PerformanceStatistics."""
        try:
            out = self._run() or ""
        except OSError:                     # ioreg missing, or a caller's runner that raises
            return []
        return [b for b in re.split(r"^\s*\+-o ", out, flags=re.M) if '"PerformanceStatistics"' in b]

    def _block(self):
        b = self._blocks()
        return b[self.index] if 0 <= self.index < len(b) else None

    def ok(self):
        return self.index < self.nodes

    def name(self):
        return self._name

    def read(self):
        b = self._block()
        if b is None:
            return {}

        def num(key):
            m = re.search(r'"%s"=([0-9.]+)' % re.escape(key), b)
            return float(m.group(1)) if m else None

        out = {}
        u = num("Device Utilization %")
        if u is None:                      # older chips report only the renderer and the tiler
            r, t = num("Renderer Utilization %"), num("Tiler Utilization %")
            u = max(v for v in (r, t) if v is not None) if (r is not None or t is not None) else None
        out["util"] = int(u) if u is not None else None
        used = num("In use system memory")
        out["mem_used"] = int(used) if used is not None else None
        out["mem_total"] = metal_working_set()
        return out


def gpu_reader(index=0, amd=False):
    """The card's readings: NVML (NVIDIA), the amdgpu sysfs files with the AMD backend (#301), or IOKit's
    IOAccelerator registry on macOS (Metal backend); a fixture's card in mock mode (STRATA_MOCK_GPUS, serve/gpus.py).
    `amd` comes first: no Mac runs the HIP backend, and the tests' AMD trees are read on a Mac too."""
    from serve.gpus import MockGpu, mock_fixture
    fx = mock_fixture()
    if fx is not None:
        cards = fx.get("cards") or []
        return MockGpu(index, cards[index]) if 0 <= index < len(cards) else _NoGpu()
    if amd:
        return _Amd(index)
    return _Metal(index) if sys.platform == "darwin" else _Nvml(index)


_DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"


def windows_display_adapters(vendor=None) -> list:
    """#1587: the display adapters Windows has drivers for, from the registry's display class (no subprocess, no vendor
    library): [{"name", "provider", "driver", "vram_bytes"}] in the class's order.  `vendor`: "amd", "nvidia" or
    "intel" keeps that vendor's cards.  [] off Windows or when nothing can be read."""
    if os.name != "nt":
        return []
    try:
        import winreg
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS)
    except OSError:
        return []
    out = []
    i = 0
    while True:
        try:
            sub = winreg.EnumKey(root, i)
        except OSError:
            break
        i += 1
        if not sub.isdigit():
            continue
        try:
            k = winreg.OpenKey(root, sub)
        except OSError:
            continue

        def val(name, k=k):
            try:
                return winreg.QueryValueEx(k, name)[0]
            except OSError:
                return None
        name = val("DriverDesc")
        if not isinstance(name, str) or not name.strip():
            continue
        mem = val("HardwareInformation.qwMemorySize")
        if mem is None:
            mem = val("HardwareInformation.MemorySize")
        if isinstance(mem, (bytes, bytearray)):
            mem = int.from_bytes(bytes(mem[:8]), "little") if mem else None
        out.append({"name": name.strip(), "provider": str(val("ProviderName") or ""), "driver": val("DriverVersion"),
                    "vram_bytes": mem if isinstance(mem, int) and mem > 0 else None})
    if vendor:
        keys = {"amd": ("advanced micro devices", "amd", "radeon"), "nvidia": ("nvidia",), "intel": ("intel",)}[vendor]
        out = [a for a in out if any(s in (a["provider"] + " " + a["name"]).lower() for s in keys)]
    return out


def windows_amd_cards(adapters=None) -> list:
    """The AMD cards HIP runs on, as windows_display_adapters lists them.  A Ryzen's integrated Radeon is listed too
    (512 MiB on a Ryzen 5 7600): beside a card with 2 GiB or more it is left out (an APU alone keeps it)."""
    cards = windows_display_adapters("amd") if adapters is None else adapters
    return [c for c in cards if (c["vram_bytes"] or 0) >= 2 << 30] or cards


def gpu_count(amd=False) -> int:
    """How many cards there are to list: NVML's count, the AMD cards in the KFD topology, or the fixture's."""
    from serve.gpus import mock_fixture
    fx = mock_fixture()
    if fx is not None:
        return len(fx.get("cards") or [])
    if amd:
        n = 0
        while n < 16 and amd_device_dir(n) is not None:
            n += 1
        return n
    return nvml_count()


def free_vram_mib(index=0, amd=False):
    """Free VRAM of a card in MiB, or None when it cannot be read."""
    g = gpu_reader(index, amd)
    if not g.ok():
        return None
    r = g.read()
    if r.get("mem_total") is None or r.get("mem_used") is None:
        return None
    return int((r["mem_total"] - r["mem_used"]) >> 20)


# ------------------------------------------------------------------------------------------------ CPU / RAM
def _cpu_topology(root: str = "/sys/devices/system"):
    """(sockets, NUMA nodes) of this computer: how many CPU packages it has and how many memory nodes (None where it cannot be read).
    Linux reads /sys, Windows asks GetLogicalProcessorInformationEx / GetNumaHighestNodeNumber, macOS asks sysctl (Apple silicon and every Mac
    sold so far: one package, no NUMA)."""
    try:
        if os.name == "nt":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            needed = ctypes.c_ulong(0)
            kernel32.GetLogicalProcessorInformationEx(3, None, ctypes.byref(needed))          # RelationProcessorPackage
            buf = ctypes.create_string_buffer(needed.value)
            sockets = None
            if needed.value and kernel32.GetLogicalProcessorInformationEx(3, buf, ctypes.byref(needed)):
                sockets, offset = 0, 0
                while offset + 8 <= needed.value:
                    relationship = int.from_bytes(buf.raw[offset:offset + 4], "little")
                    size = int.from_bytes(buf.raw[offset + 4:offset + 8], "little")
                    if size <= 0:
                        break
                    sockets += 1 if relationship == 3 else 0
                    offset += size
            high = ctypes.c_ulong(0)
            numa = high.value + 1 if kernel32.GetNumaHighestNodeNumber(ctypes.byref(high)) else None
            return (sockets or None), numa
        if sys.platform == "darwin":
            import subprocess
            out = subprocess.run(["sysctl", "-n", "hw.packages"], capture_output=True, text=True, timeout=5).stdout.strip()
            return (int(out) if out.isdigit() else 1), None
        import glob
        packages = set()
        for f in glob.glob(os.path.join(root, "cpu", "cpu[0-9]*", "topology", "physical_package_id")):
            try:
                with open(f, encoding="utf-8") as fh:
                    packages.add(fh.read().strip())
            except OSError:
                continue
        nodes = len(glob.glob(os.path.join(root, "node", "node[0-9]*")))
        return (len(packages) or None), (nodes or None)
    except Exception:                                        # noqa: BLE001 - a missing figure is shown as missing, never guessed
        return None, None


def _cpu_name():
    if os.name == "nt":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return sysinfo.processor() or None


class _CpuRamFallback:
    """CPU load and RAM without psutil."""

    def __init__(self):
        self.prev = self._times()

    def _times(self):
        if os.name == "nt":
            idle, kern, user = (ctypes.c_ulonglong() for _ in range(3))
            if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
                return idle.value, kern.value + user.value           # kernel time includes idle
            return None
        try:
            with open("/proc/stat") as fh:
                f = [int(x) for x in fh.readline().split()[1:]]
            return f[3] + f[4], sum(f)
        except (OSError, ValueError):
            return None

    def cpu(self):
        cur = self._times()
        prev, self.prev = self.prev, cur
        if not cur or not prev or cur[1] == prev[1]:
            return None
        return max(0.0, min(100.0, 100.0 * (1 - (cur[0] - prev[0]) / (cur[1] - prev[1]))))

    @staticmethod
    def ram():
        if os.name == "nt":
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return m.ullTotalPhys - m.ullAvailPhys, m.ullTotalPhys
            return None, None
        try:
            with open("/proc/meminfo") as fh:
                info = dict(line.split(":", 1) for line in fh)
            total = int(info["MemTotal"].split()[0]) * 1024
            avail = int(info["MemAvailable"].split()[0]) * 1024
            return total - avail, total
        except (OSError, KeyError, ValueError):
            return None, None


# ------------------------------------------------------------------------------------------------ disks
# The engine's model files, by their argument: the Disk card reads the drives that hold them (a low-RAM start streams
# the experts from the pack's drive), not every drive's total.
MODEL_FILE_ARGS = (("--pack", "model pack"), ("--native", "dense weights"), ("--native-dense-gguf", "dense weights"),
                   ("--native-head-gguf", "head"), ("--ple-gguf", "PLE table"), ("--embd-gguf", "token embedding"),
                   ("--mtp", "MTP drafter"))


def model_files(args, cwd=None) -> list:
    """[(role, absolute path)] of the engine's model files that exist (the args' relative paths are the cwd's)."""
    out, args = [], [str(a) for a in args or []]
    for flag, role in MODEL_FILE_ARGS:
        if flag in args[:-1]:
            v = args[args.index(flag) + 1]
            path = v if os.path.isabs(v) else os.path.join(cwd or ".", v)
            if os.path.exists(path):
                out.append((role, os.path.abspath(path)))
    return out


def _win_disk(path):
    """Windows: the physical disk(s) under a path's volume ("PhysicalDrive1", psutil's per-disk names) and the volume's
    name ("D:"), by IOCTL_VOLUME_GET_VOLUME_DISK_EXTENTS on the volume (no administrator rights needed)."""
    import ctypes.wintypes as w
    import struct
    k = ctypes.WinDLL("kernel32", use_last_error=True)          # its own handle: argtypes stay off ctypes.windll
    k.CreateFileW.restype = w.HANDLE
    k.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, w.LPVOID, w.DWORD, w.DWORD, w.HANDLE]
    k.DeviceIoControl.argtypes = [w.HANDLE, w.DWORD, w.LPVOID, w.DWORD, w.LPVOID, w.DWORD, ctypes.POINTER(w.DWORD),
                                  w.LPVOID]
    k.GetVolumePathNameW.argtypes = [w.LPCWSTR, w.LPWSTR, w.DWORD]
    k.GetVolumeNameForVolumeMountPointW.argtypes = [w.LPCWSTR, w.LPWSTR, w.DWORD]
    k.CloseHandle.argtypes = [w.HANDLE]
    mount, vol = ctypes.create_unicode_buffer(1024), ctypes.create_unicode_buffer(1024)
    if not k.GetVolumePathNameW(path, mount, 1024) or not k.GetVolumeNameForVolumeMountPointW(mount.value, vol, 1024):
        return None
    h = k.CreateFileW(vol.value.rstrip("\\"), 0, 3, None, 3, 0, None)      # no access, shared read/write, existing
    if h is None or h == ctypes.c_void_p(-1).value:
        return None
    try:
        out, n = ctypes.create_string_buffer(8 + 24 * 32), w.DWORD()
        if not k.DeviceIoControl(h, 0x00560000, None, 0, out, len(out), ctypes.byref(n), None):
            return None
    finally:
        k.CloseHandle(h)
    count = struct.unpack_from("<I", out.raw, 0)[0]
    disks = sorted({struct.unpack_from("<I", out.raw, 8 + 24 * j)[0] for j in range(min(count, 32))})
    if not disks:
        return None
    return tuple(f"PhysicalDrive{d}" for d in disks), mount.value.rstrip("\\") or mount.value


def _unescape_mount(s):
    for code, ch in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        s = s.replace(code, ch)
    return s


def _mount_of(path, mountinfo="/proc/self/mountinfo"):
    """Linux: (mount point, source) of the mount that holds `path` (the longest mount point above it)."""
    best = (None, None)
    try:
        with open(mountinfo, encoding="utf-8", errors="replace") as f:
            for line in f:
                head, sep, tail = line.partition(" - ")
                a, b = head.split(), tail.split()
                if not sep or len(a) < 5 or len(b) < 2:
                    continue
                mp = _unescape_mount(a[4])
                if (path == mp or path.startswith(mp.rstrip("/") + "/")) and len(mp) >= len(best[0] or ""):
                    best = (mp, _unescape_mount(b[1]))
    except OSError:
        pass
    return best


def _linux_disk(path, sysfs="/sys", mountinfo="/proc/self/mountinfo"):
    """Linux: the block device that holds `path` - its name in /proc/diskstats (psutil's per-disk names: "nvme0n1p2",
    "dm-0", "md0") - from the file's device number, else (btrfs and other filesystems with an anonymous device
    number) the mount's /dev source; and a label, its mount point.  None when neither names a block device (overlayfs
    in a container, ZFS, a network share): the Disk card then reads every drive."""
    rp = os.path.realpath(path)
    mount, source = _mount_of(rp, mountinfo)
    name = None
    try:
        st = os.stat(rp)
        if os.major(st.st_dev) != 0:
            d = os.path.realpath(os.path.join(sysfs, "dev", "block", f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}"))
            if os.path.isdir(d):
                name = os.path.basename(d)
    except (OSError, AttributeError):
        pass
    if name is None and source and source.startswith("/dev/"):
        name = os.path.basename(os.path.realpath(source))
    if not name:
        return None
    return (name,), (f"{mount} ({name})" if mount else name)


def disk_of(path):
    """(psutil per-disk names, label) of the drive that holds `path`, or None when it cannot be told."""
    try:
        return _win_disk(path) if os.name == "nt" else _linux_disk(path)
    except (OSError, ValueError, AttributeError):
        return None


def model_disks(files) -> list:
    """The drives that hold the model's files: [{"keys": (psutil names), "label": "D:", "roles": [...]}], the drive of
    the first file (the pack) first.  [] when any file's drive cannot be told (the card then reads every drive)."""
    out = []
    for role, path in files or []:
        d = disk_of(path)
        if d is None:
            return []
        keys, label = d
        for e in out:
            if e["keys"] == keys:
                if role not in e["roles"]:
                    e["roles"].append(role)
                break
        else:
            out.append({"keys": keys, "label": label, "roles": [role]})
    return out


def pip_install_command(package="psutil") -> str:
    """The command that installs `package` for the Python this server runs under (pythonw.exe -> its python.exe)."""
    exe = sys.executable or "python"
    import ntpath                                       # (os.path on Windows; also when a test sets os.name)
    if os.name == "nt" and ntpath.basename(exe).lower() == "pythonw.exe":
        alt = ntpath.join(ntpath.dirname(exe), "python.exe")
        exe = alt if os.path.exists(alt) else exe
    return f'{exe if " " not in exe else chr(34) + exe + chr(34)} -m pip install {package}'


def _amd_link_now(r: dict):
    """An AMD card's link generation now: its own negotiated one, never above the slowest hop to the root port."""
    own, path = r.get("pcie_own_gen"), r.get("pcie_gen")
    return min(own, path) if own and path else (own or path)


def _card_info(g) -> dict:
    """A reader's fixed facts; a reader without info() (the Intel server's xe reader) gives its name only."""
    try:
        return g.info() if hasattr(g, "info") else {"name": g.name(), "uuid": None, "pci": None}
    except Exception:  # noqa: BLE001 - a card's facts are extra: the readings go on without them
        return {"name": None, "uuid": None, "pci": None}


# ------------------------------------------------------------------------------------------------ the sampler
class Telemetry:
    def __init__(self, extra=None, gpu_index=0, gpu_indices=None, amd=False, on_tick=None, files=None):
        """`extra()` -> dict of more series to record each second (the server's tok/s).  `gpu_index`: the card the
        engine runs on, numbered as nvidia-smi and NVML number them (by PCI bus); `gpu_indices`: all of them when
        the model is split across several (issue #112) - the gpu_* readings are then their total (memory, power,
        PCIe traffic), mean (load) or hottest (temperature).  "gpus" has every card's own reading, the cards the engine
        does not use too (read every OTHER_CARDS_S), and the static "gpus" each card's name, UUID and PCI bus id.
        `set_engine_cards()` changes which cards the totals are of (the server's layout, serve/gpus.py).
        `on_tick()`: called on this thread before each sample (the server refreshes the GPU layout there from cached
        pieces, so it costs a few dictionary lookups).
        `files`: [(role, path)] of the model's files - the disk reading is of the drives that hold them.  `amd`: the AMD
        backend's cards, numbered as HIP numbers them, read from sysfs (#301)."""
        self.extra = extra
        self.on_tick = on_tick
        self.amd = amd
        self.lock = threading.Lock()
        self.now: dict = {}
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        idx = list(gpu_indices) if gpu_indices and len(gpu_indices) > 1 else [gpu_index]
        self.cards = {i: gpu_reader(i, amd) for i in range(gpu_count(amd))}
        for i in idx:
            if i not in self.cards:
                self.cards[i] = gpu_reader(i, amd)
        self.card_info = {i: _card_info(g) for i, g in sorted(self.cards.items()) if g.ok()}
        try:
            import psutil  # noqa: F401
            self.ps = sys.modules["psutil"]
        except ImportError:
            self.ps = None
        self.fallback = _CpuRamFallback()
        # #1587: no readings for an AMD card on Windows yet, but its name (and VRAM size) are in the registry, so the
        # dashboard and strata_bench do not call the run "CPU only" (read once; set_engine_cards names the cards)
        self._win_amd_cards = windows_amd_cards() if amd and os.name == "nt" else None
        self.static = {
            "cpu_name": _cpu_name(),
            "cores": (self.ps.cpu_count(logical=False) if self.ps else None) or None,
            "threads": os.cpu_count(),
            "sockets": _cpu_topology()[0],
            "numa_nodes": _cpu_topology()[1],
            "psutil": self.ps is not None,
            "psutil_install": None if self.ps else pip_install_command("psutil"),
            "gpus": [{"index": i, **info} for i, info in self.card_info.items()],
        }
        self.gpu_every = gpu_poll_s()
        self._gpu_at, self._gpu_cache = None, {}
        self._others_at, self._others = None, {}
        self.set_engine_cards(idx)
        self.disks = model_disks(files) if self.ps else []
        self.static["disks"] = [{"label": d["label"], "roles": d["roles"], "devices": list(d["keys"])}
                                for d in self.disks]
        self.static["disk_scope"] = "model" if self.disks else "all"
        if self.ps is None:
            print("[strata] the Monitor's disk reading needs psutil, which this Python does not have: "
                  f"{self.static['psutil_install']}", file=sys.stderr, flush=True)
        self._disk_prev = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_engine_cards(self, indices):
        """The cards the gpu_* totals are of (the ones the engine uses).  Unknown or unreadable ones are left out; with
        none left, the first card as before."""
        idx = [i for i in dict.fromkeys(indices or []) if i in self.cards]
        gpus = [(i, self.cards[i]) for i in idx]
        gpus = [(i, g) for i, g in gpus if g.ok()] or gpus[:1] or sorted(self.cards.items())[:1] \
            or [(0, gpu_reader(0, self.amd))]
        gpu_name = " + ".join(g.name() or "?" for _, g in gpus) if gpus[0][1].ok() else None
        win_amd = None
        if gpu_name is None and self._win_amd_cards:
            # #1587: the registry's AMD cards (the class order taken as HIP's order)
            cards = self._win_amd_cards
            win_amd = [cards[i] for i in (idx or [gpus[0][0]]) if 0 <= i < len(cards)] or cards[:1]
            gpu_name = " + ".join(c["name"] for c in win_amd) if win_amd else None
        # the macOS reader knows the device's working set (the VRAM budget of the unified memory); None elsewhere
        vram_total = None
        if win_amd is None and isinstance(gpus[0][1], _Metal) and gpus[0][1].ok():
            ws = metal_working_set()
            vram_total = ws >> 20 if ws else None
        with self.lock:
            self.gpus, self.gpu = gpus, gpus[0][1]
            self._gpu_at = self._others_at = None                # the next sample reads every card at once
            self.static["gpu_name"] = gpu_name
            self.static["gpu_vram_total_mib"] = (sum(c["vram_bytes"] for c in win_amd) >> 20
                                                 if win_amd and all(c["vram_bytes"] for c in win_amd) else vram_total)
            self.static["gpu_count"] = len(gpus)
            # #1380: the AMD readings are the amdgpu driver's Linux sysfs files; a Windows AMD card has none yet, and the
            # dashboard said "not readable (NVML)" or showed empty tiles with no word why
            self.static["gpu_note"] = ("no GPU load or VRAM readings for AMD cards on Windows yet (Linux reads them from "
                                       "the amdgpu driver); the engine's own VRAM figures are in its log"
                                       if self.amd and not self.gpu.ok() else getattr(self.gpu, "note", None))
            self.static["engine_gpus"] = [i for i, _ in gpus]

    def engine_cards(self):
        with self.lock:
            return [i for i, _ in self.gpus]

    def _disk(self):
        """(read MB/s, write MB/s, per drive) of the model's drives, or of every drive when they are not known."""
        if not self.ps:
            return None, None, None
        try:
            c = self.ps.disk_io_counters(perdisk=True) if self.disks else self.ps.disk_io_counters()
        except (OSError, RuntimeError):
            return None, None, None
        if c is None:   # psutil found no disk (a gVisor container, Windows with its disk counters off)
            return None, None, None
        t = time.time()
        if self.disks:
            if not all(k in c for d in self.disks for k in d["keys"]):
                return None, None, None
            cur = [(sum(c[k].read_bytes for k in d["keys"]), sum(c[k].write_bytes for k in d["keys"]))
                   for d in self.disks]
        else:
            cur = [(c.read_bytes, c.write_bytes)]
        prev, self._disk_prev = self._disk_prev, (t, cur)
        if prev is None or t <= prev[0] or len(prev[1]) != len(cur):
            return None, None, None
        dt = t - prev[0]
        rates = [((r - pr) / dt / 2**20, (w - pw) / dt / 2**20) for (r, w), (pr, pw) in zip(cur, prev[1])]
        per = [{"label": d["label"], "read_mb": round(r, 2), "write_mb": round(w, 2)}
               for d, (r, w) in zip(self.disks, rates)] if self.disks else None
        return sum(r for r, _ in rates), sum(w for _, w in rates), per

    def _gpu_sample(self) -> dict:
        """The gpu_* totals of the engine's cards and every card's own reading ("gpus"); the last reading again until
        STRATA_GPU_POLL_S has passed (none at all with 0)."""
        t = time.monotonic()
        if not self.gpu_every:
            return {}
        with self.lock:
            gpus, first, at = list(self.gpus), self.gpu, self._gpu_at
        if at is not None and t - at < self.gpu_every - 0.05:
            return self._gpu_cache
        if not first.ok():
            self._gpu_at, self._gpu_cache = t, {}
            return {}
        reads = [(i, g.read()) for i, g in gpus]
        mine = {i for i, _ in gpus}
        others = [(i, g) for i, g in sorted(self.cards.items()) if i not in mine and g.ok()]
        if others and (self._others_at is None or t - self._others_at >= max(OTHER_CARDS_S, 5 * self.gpu_every)):
            self._others, self._others_at = {i: g.read() for i, g in others}, t
        s = {}
        g = dict(reads[0][1])
        if len(reads) > 1:
            def vals(k):
                return [r[k] for _, r in reads if r.get(k) is not None]
            for k in ("mem_used", "mem_total", "power", "power_limit", "pcie_rx_mb", "pcie_tx_mb"):
                v = vals(k)
                g[k] = sum(v) if v else None
            u = vals("util")
            g["util"] = sum(u) / len(u) if u else None
            hot = vals("temp")
            g["temp"] = max(hot) if hot else None
        s.update({f"gpu_{k}": v for k, v in g.items()})
        every = {i: r for i, r in self._others.items() if i not in mine}
        every.update(dict(reads))
        s["gpus"] = [{"index": i, "name": self.cards[i].name() if i in self.cards else None, "engine": i in mine,
                      "util": r.get("util"), "mem_used": r.get("mem_used"),
                      "mem_total": r.get("mem_total"), "temp": r.get("temp"), "power": r.get("power"),
                      "power_limit": r.get("power_limit"),
                      # the link it runs at now (it drops to Gen1 at idle to save power); the best one is static.
                      # AMD: the card's own negotiated speed, capped by the slowest hop above it (#1733)
                      "pcie_gen": _amd_link_now(r) if self.amd else r.get("pcie_gen"),
                      "pcie_width": r.get("pcie_width")}
                     for i, r in sorted(every.items())]
        self._gpu_at, self._gpu_cache = t, s
        return s

    def sample(self):
        s = dict(self._gpu_sample())
        if self.ps:
            try:
                s["cpu"] = self.ps.cpu_percent(interval=None)
                vm = self.ps.virtual_memory()
                s["ram_used"], s["ram_total"] = vm.total - vm.available, vm.total
            except (OSError, RuntimeError):
                pass
        else:
            s["cpu"] = self.fallback.cpu()
            s["ram_used"], s["ram_total"] = self.fallback.ram()
        s["disk_read_mb"], s["disk_write_mb"], per = self._disk()
        if per is not None:
            s["disks"] = per
        if self.extra:
            try:
                s.update(self.extra())
            except Exception:  # noqa: BLE001 - telemetry must never take the server down
                pass
        return s

    def close(self):
        """Ends the sampler thread (a server that stops, a test's service): it used to run for the life of the process."""
        self._stop.set()

    def _loop(self):
        """One sample a second, for as long as the server lives.

        A failing sample must not end the thread: on a machine where the GPU/psutil source is not readable
        (a CI runner, a container without NVML) the first exception used to kill the sampler outright, so
        `GET /metrics` silently stopped updating and `test_responses.NoLeakedSampler` saw "the server did
        not start its hardware sampler" - the thread had started and died between two readings.
        """
        complained = False
        while not self._stop.is_set():
            if self.on_tick:
                try:
                    self.on_tick()
                except Exception:  # noqa: BLE001 - the GPU layout is extra information, never a reason to stop
                    pass
            try:
                s = self.sample()
            except Exception as e:                    # noqa: BLE001 - a monitor must outlive its source
                if not complained:                    # once, not once a second
                    print(f"[strata] metrics: a sample failed ({e}); the sampler keeps trying", flush=True)
                    complained = True
                self._stop.wait(1.0)
                continue
            with self.lock:
                self.now = s
                for k in ("gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "ram_used",
                          "disk_read_mb", "tok_s", "prefill_tok_s_mean"):
                    v = s.get(k)
                    self.hist[k].append(round(v, 2) if isinstance(v, float) else v)
            self._stop.wait(1.0)

    def snapshot(self):
        with self.lock:
            return {"now": dict(self.now), "history": {k: list(v) for k, v in self.hist.items()},
                    "static": dict(self.static)}
