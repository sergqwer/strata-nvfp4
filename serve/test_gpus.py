"""The Monitor's GPU list (serve/gpus.py) and the disk card's drives (serve/telemetry.py): which card each engine device
is, what the engine does with it, and the readings - without any GPU (fixtures and fake readers)."""
import json
import os
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from serve import gpus, telemetry
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]

# five cards, NVML's numbering = PCI order; GPU 0 is a different model
CARDS = [{"index": 0, "name": "RTX 4060 Ti", "pci": "0000:01:00.0", "uuid": "GPU-aaaa0000-1111"},
         {"index": 1, "name": "RTX 4090", "pci": "0000:21:00.0", "uuid": "GPU-bbbb0000-2222"},
         {"index": 2, "name": "RTX 4090", "pci": "0000:41:00.0", "uuid": "GPU-cccc0000-3333"},
         {"index": 3, "name": "RTX 5090", "pci": "0000:61:00.0", "uuid": "GPU-dddd0000-4444"},
         {"index": 4, "name": "RTX 3060", "pci": "0000:81:00.0", "uuid": "GPU-eeee0000-5555"}]
SAME = [dict(c, name="RTX 4090") for c in CARDS[:4]]

# the engine's own lines (src/program/generate.cpp), as a 2-GPU log has them - with the indent another tool added
SPLIT_LOG = """\
strata generate: expert cache auto: 9.10 GiB free, 600 MiB reserved
  strata generate: layer split across 2 GPUs: CUDA0, then CUDA1 (split auto)
  strata generate: layer split: CUDA1 runs layers 35-47, expert cache 2869 slots (7.39 GiB), 2869 of its 6656 profiled pairs; slot 0 verified
  strata generate: layer split: CUDA0 runs layers 0-34
INFO logit_bias=1 context=262144 expert_slots=6000
READY 262144 stop
""".splitlines()


def never(env):
    raise AssertionError("the CUDA driver was asked although the environment said enough")


class PciKey(unittest.TestCase):
    def test_spellings(self):
        self.assertEqual(gpus.pci_key("00000000:01:00.0"), "0000:01:00.0")      # NVML
        self.assertEqual(gpus.pci_key("0000:01:00.0"), "0000:01:00.0")          # CUDA
        self.assertEqual(gpus.pci_key("01:00.0"), "0000:01:00.0")
        self.assertEqual(gpus.pci_key("0000:C1:00.1"), "0000:c1:00.1")
        self.assertEqual(gpus.pci_key("0001:A2:1F.0"), "0001:a2:1f.0")
        self.assertIsNone(gpus.pci_key("pci0000:00"))
        self.assertIsNone(gpus.pci_key(None))


class CudaCards(unittest.TestCase):
    """CUDA ordinal -> card, read from the engine's environment the way CUDA reads it."""

    def test_pci_bus_id_order_with_a_list(self):
        # the server's own setting for a config's "gpu" (ordered_gpus may put the faster card last)
        env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "2,1,3,4"}
        self.assertEqual(gpus.cuda_cards(env, CARDS, probe=never), ([2, 1, 3, 4], "pci_bus_id"))

    def test_pci_order_is_the_bus_not_the_list_order(self):
        shuffled = [CARDS[3], CARDS[0], CARDS[2], CARDS[1]]
        env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        self.assertEqual(gpus.cuda_cards(env, shuffled, probe=never), ([0, 1, 2, 3], "pci_bus_id"))

    def test_bus_ids_that_sort_as_numbers(self):
        cards = [{"index": 0, "name": "A", "pci": "0000:0a:00.0"}, {"index": 1, "name": "B", "pci": "0000:09:00.0"}]
        self.assertEqual(gpus.cuda_cards({"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}, cards, probe=never)[0], [1, 0])

    def test_fastest_first_over_identical_cards(self):
        # CUDA's default order: identical cards tie and keep the bus order - no driver call needed (the 4x 4090 server)
        self.assertEqual(gpus.cuda_cards({}, SAME, probe=never), ([0, 1, 2, 3], "same_model"))
        self.assertEqual(gpus.cuda_cards({"CUDA_VISIBLE_DEVICES": "3,1"}, SAME, probe=never), ([3, 1], "same_model"))

    def test_fastest_first_over_different_cards_asks_the_driver(self):
        asked = []

        def probe(env):
            asked.append(dict(env))
            return ["0000:61:00.0", "0000:21:00.0"]          # the driver put the 5090 first
        env = {"CUDA_VISIBLE_DEVICES": "0,1"}
        self.assertEqual(gpus.cuda_cards(env, CARDS, probe=probe), ([3, 1], "cuda_driver"))
        self.assertEqual(asked, [env])                       # the engine's environment, as it is

    def test_no_driver_answer_assumes_nvidia_smi_numbers(self):
        self.assertEqual(gpus.cuda_cards({"CUDA_VISIBLE_DEVICES": "0,3"}, CARDS, probe=lambda env: None),
                         ([0, 3], "assumed"))
        self.assertEqual(gpus.cuda_cards({}, CARDS[:2], probe=lambda env: None), ([0, 1], "assumed"))

    def test_uuids_and_their_prefixes(self):
        env = {"CUDA_VISIBLE_DEVICES": "GPU-dddd0000-4444,GPU-bbbb"}
        self.assertEqual(gpus.cuda_cards(env, CARDS, probe=never), ([3, 1], "uuid"))

    def test_the_list_ends_at_an_entry_that_names_no_card(self):
        env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        self.assertEqual(gpus.cuda_cards(dict(env, CUDA_VISIBLE_DEVICES="0,7,1"), CARDS, probe=never)[0], [0])
        self.assertEqual(gpus.cuda_cards(dict(env, CUDA_VISIBLE_DEVICES="1,1"), CARDS, probe=never)[0], [1])
        self.assertEqual(gpus.cuda_cards(dict(env, CUDA_VISIBLE_DEVICES="-1"), CARDS, probe=never)[0], [])
        self.assertEqual(gpus.cuda_cards(dict(env, CUDA_VISIBLE_DEVICES=""), CARDS, probe=never)[0], [])
        self.assertEqual(gpus.cuda_cards(dict(env, CUDA_VISIBLE_DEVICES="GPU-ffff"), CARDS, probe=never)[0], [])

    def test_mig_asks_the_driver(self):
        got = gpus.cuda_cards({"CUDA_VISIBLE_DEVICES": "MIG-1234"}, SAME, probe=lambda env: ["0000:41:00.0"])
        self.assertEqual(got, ([2], "cuda_driver"))

    def test_hip_order(self):
        self.assertEqual(gpus.hip_cards({"HIP_VISIBLE_DEVICES": "1,0"}, 2), ([1, 0], "hip_order"))
        self.assertEqual(gpus.hip_cards({}, 3), ([0, 1, 2], "hip_order"))
        self.assertEqual(gpus.hip_cards({"HIP_VISIBLE_DEVICES": "0,5"}, 2), ([0], "hip_order"))


class Layout(unittest.TestCase):
    def test_two_gpu_split_from_the_log(self):
        lay = gpus.parse_layout(SPLIT_LOG)
        self.assertTrue(lay["split"])
        self.assertEqual(lay["stages"], {0: {"layers": [0, 34]},
                                         1: {"layers": [35, 47], "experts": 2869, "gib": 7.39}})
        r = gpus.roles(lay, 2, {"expert_slots_primary": 3100, "expert_cache_primary_mib": 8192})
        self.assertEqual(r[0], {"role": "stage", "stage": 1, "stages": 2, "layers": [0, 34], "experts": 3100, "gib": 8.0})
        self.assertEqual(r[1]["role"], "stage")
        self.assertEqual((r[1]["stage"], r[1]["layers"], r[1]["experts"]), (2, [35, 47], 2869))

    def test_one_gpu(self):
        lay = gpus.parse_layout(["strata generate: expert cache 8000 slots, 20.00 GiB of VRAM; policy is",
                                 "strata generate: expert cache 8644 slots, 22.26 GiB of VRAM; policy is"])
        self.assertFalse(lay["split"])
        self.assertEqual(lay["cache"], {"experts": 8644, "gib": 22.26})            # the last try is the one it kept
        self.assertEqual(gpus.roles(lay, 1, {}), [{"role": "model", "experts": 8644, "gib": 22.26}])
        # an engine that reports CUDA0's own cache in its INFO line: that number wins
        r = gpus.roles(lay, 2, {"expert_slots_primary": 8600, "expert_cache_primary_mib": 22528})
        self.assertEqual(r, [{"role": "model", "experts": 8600, "gib": 22.0}, {"role": "unused"}])

    def test_split_skipped_because_it_fits(self):
        lay = gpus.parse_layout([SPLIT_LOG[1], "strata generate: layer split skipped (--split-skip-if-fits): CUDA0 "
                                              "(NVIDIA GeForce RTX 5090) holds all 9984 profiled pairs - one GPU"])
        r = gpus.roles(lay, 2, {})
        self.assertEqual(r[0]["role"], "model")
        self.assertIn("split-skip-if-fits", r[0]["note"])
        self.assertEqual(r[1], {"role": "unused"})

    def test_peer_and_helpers(self):
        lay = gpus.parse_layout([
            "strata generate: peer GPU 1: 4200 experts, 10.81 GiB (filled in 3.2 s); with the primary's 8000 that is "
            "12200 of 12288 on the GPUs"])
        self.assertEqual(gpus.roles(lay, 2, {})[1], {"role": "peer", "experts": 4200, "gib": 10.81})
        lay = gpus.parse_layout([
            "strata generate: CUDA1: 3000 additional experts, 7.73 GiB; results return through pinned host rows",
            "strata generate: CUDA2: 2500 additional experts, 6.44 GiB; results return through pinned host rows"])
        r = gpus.roles(lay, 4, {})
        self.assertEqual([x["role"] for x in r], ["model", "helper", "helper", "unused"])
        self.assertEqual(r[2]["experts"], 2500)

    def test_split_onto_one_card(self):
        """`--layer-split 24 --split-device 0`: both stages on CUDA0 - one card running every layer, not stage 2."""
        lay = gpus.parse_layout([
            "strata generate: layer split across 2 GPUs: CUDA0, then CUDA0 (split 24)",
            "strata generate: layer split: CUDA0 runs layers 24-47, expert cache 3000 slots (7.70 GiB), 3000 of its "
            "6000 profiled pairs; slot 0 verified",
            "strata generate: layer split: CUDA0 runs layers 0-23"])
        r = gpus.roles(lay, 1, {"expert_slots_primary": 2500, "expert_cache_primary_mib": 6400})
        self.assertEqual(r[0]["role"], "model")
        self.assertEqual((r[0]["layers"], r[0]["experts"], r[0]["gib"]), ([0, 47], 5500, 13.95))
        self.assertIn("split-device 0", r[0]["note"])
        planned = gpus.roles(gpus.planned_layout(["--layer-split", "24", "--split-device", "0"], 1), 1, {}, planned=True)
        self.assertEqual(planned, [{"role": "model", "layers": [0, None], "planned": True,
                                    "note": "a layer split onto one card (--split-device 0): every stage runs here"}])

    def test_measured_links(self):
        """The engine's PCIe probes: CUDA0's at start, each later stage's on a split (-1.0 = the probe failed)."""
        lay = gpus.parse_layout(SPLIT_LOG + [
            "strata generate: PCIe probe: 24.6 GB/s host->device (best of 24.6 20.4 16.8 16.8) -> pcie_frac 0.25 "
            "(default 0.25)",
            "strata generate: layer split: CUDA1 PCIe probe -1.0 GB/s (best of ) -> pcie_frac 0.55"])
        self.assertEqual(lay["links"], {0: {"link_gbps": 24.6, "pcie_frac": 0.25},
                                        1: {"link_gbps": None, "pcie_frac": 0.55}})
        r = gpus.roles(lay, 2, {})
        self.assertEqual((r[0]["link_gbps"], r[0]["pcie_frac"]), (24.6, 0.25))
        self.assertNotIn("link_gbps", r[1])
        self.assertEqual(r[1]["pcie_frac"], 0.55)
        lay = gpus.parse_layout(["strata generate: PCIe probe: 6.8 GB/s host->device (best of 6.7 6.8); the canonical "
                                 "pack starts at pcie_frac 0.20 and refines it from the CPU pool's speed"])
        self.assertEqual(lay["links"][0], {"link_gbps": 6.8, "pcie_frac": 0.2})

    def test_planned_from_the_args(self):
        lay = gpus.planned_layout(["--layer-split", "24,36,42"], 4)
        self.assertEqual({d: s["layers"] for d, s in lay["stages"].items()},
                         {0: [0, 23], 1: [24, 35], 2: [36, 41], 3: [42, None]})
        r = gpus.roles(lay, 4, {}, planned=True)
        self.assertTrue(all(x["planned"] for x in r))
        lay = gpus.planned_layout(["--layer-split", "auto"], 3)
        self.assertEqual({d: s["layers"] for d, s in lay["stages"].items()},
                         {0: [0, None], 1: [None, None], 2: [None, None]})
        lay = gpus.planned_layout(["--layer-split", "20", "--split-device", "2", "--expert-cache-device1", "auto"], 3)
        self.assertEqual((sorted(lay["stages"]), sorted(lay["helpers"])), ([0, 2], [1]))
        lay = gpus.planned_layout(["--expert-cache-device2", "3000"], 3)
        self.assertEqual((lay["split"], sorted(lay["helpers"])), (False, [2]))
        lay = gpus.planned_layout(["--peer-device", "1"], 2)
        self.assertEqual(gpus.roles(lay, 2, {}, planned=True)[1], {"role": "peer", "planned": True})
        self.assertEqual(gpus.roles(gpus.planned_layout([], 1), 1, {}, planned=True), [{"role": "model", "planned": True}])

    def test_rows(self):
        lay = gpus.parse_layout(SPLIT_LOG)
        engines = [{"replica": None, "cards": [2, 1], "roles": gpus.roles(lay, 2, {}), "state": "ready"}]
        rows = gpus.view(CARDS[:3], {2: {"util": 80, "mem_used": 5 << 30, "mem_total": 24 << 30}}, engines, encoder=0)
        self.assertEqual([(r["index"], r["cuda"], r["role"]) for r in rows], [(0, None, "encoder"), (1, 1, "stage"),
                                                                            (2, 0, "stage")])
        self.assertEqual((rows[2]["util"], rows[2]["layers"], rows[1]["layers"]), (80, [0, 34], [35, 47]))
        self.assertFalse(rows[0]["visible"])
        # replicas: a card shows the replica that runs on it
        engines = [{"replica": 0, "cards": [0], "roles": [{"role": "model"}], "state": "ready"},
                   {"replica": 1, "cards": [1], "roles": [{"role": "model"}], "state": "ready"}]
        rows = gpus.view(CARDS[:2], {}, engines)
        self.assertEqual([(r["replica"], r["role"]) for r in rows], [(0, "model"), (1, "model")])


class LayoutKeeper(unittest.TestCase):
    """serve/gpus.Layout over an engine object: its spawn env/args, its log from this start on, its state."""

    def engine(self, log, start=0, **kw):
        e = SimpleNamespace(spawn=("strata", ["--layer-split", "auto", "--expert-profile", "p.bin"], None, log,
                                   {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "3,1"}, False, None),
                            log_path=log, log_start=start, unloaded=False, ended=False, max_context=262144,
                            info={"expert_slots_primary": 3100, "expert_cache_primary_mib": 8192})
        for k, v in kw.items():
            setattr(e, k, v)
        return e

    def test_ready_engine_reads_its_start_only(self):
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "strata.log")
            old = "strata generate: layer split: CUDA1 runs layers 20-47, expert cache 1 slots (0.01 GiB)\n"
            Path(log).write_text(old + "\n".join(SPLIT_LOG) + "\n", encoding="utf-8")
            svc = SimpleNamespace(engine=self.engine(log, start=len(old)))
            with mock.patch.dict(os.environ, {"STRATA_MOCK_GPUS": ""}):
                lay = gpus.Layout(svc)
                self.assertEqual(lay.refresh(CARDS), [1, 3])
                rows = lay.rows(CARDS[:4], {})["cards"]
        self.assertEqual([(r["index"], r["cuda"], r["role"], r.get("layers")) for r in rows],
                         [(0, None, "unused", None), (1, 1, "stage", [35, 47]), (2, None, "unused", None),
                          (3, 0, "stage", [0, 34])])

    def test_unloaded_engine_shows_the_plan(self):
        svc = SimpleNamespace(engine=self.engine(None, unloaded=True))
        with mock.patch.dict(os.environ, {"STRATA_MOCK_GPUS": ""}):
            lay = gpus.Layout(svc)
            lay.refresh(CARDS)
            rows = lay.rows(CARDS[:4], {})["cards"]
        self.assertEqual([(r["index"], r["role"], r.get("planned"), r.get("engine_state")) for r in rows],
                         [(0, "unused", None, None), (1, "stage", True, "unloaded"), (2, "unused", None, None),
                          (3, "stage", True, "unloaded")])

    def test_replicas(self):
        a, b = self.engine(None), self.engine(None)
        a.spawn = ("strata", [], None, None, {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0"})
        b.spawn = ("strata", [], None, None, {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1"})

        class Replicas:                                  # serve/replicas.ReplicaEngine keeps them in its __dict__
            def __init__(self):
                self.engines = [a, b]
        with mock.patch.dict(os.environ, {"STRATA_MOCK_GPUS": ""}):
            lay = gpus.Layout(SimpleNamespace(engine=Replicas()))
            self.assertEqual(lay.refresh(CARDS), [0, 1])
            rows = lay.rows(CARDS[:3], {})["cards"]
        self.assertEqual([(r["replica"], r["cuda"], r["role"]) for r in rows],
                         [(0, 0, "model"), (1, 0, "model"), (None, None, "unused")])


class MockMode(unittest.TestCase):
    """STRATA_MOCK_GPUS: fixture cards behind the same telemetry and API, for the page without the GPUs."""

    def setUp(self):
        gpus._mock_cache.clear()

    def test_fixture_through_telemetry(self):
        with mock.patch.dict(os.environ, {"STRATA_MOCK_GPUS": "gpus-mixed-roles.json"}):
            t = telemetry.Telemetry()
            try:
                self.assertEqual([g["index"] for g in t.static["gpus"]], [0, 1, 2, 3, 4])
                self.assertEqual(t.static["gpus"][3]["pci"], "0000:61:00.0")
                t.set_engine_cards([2, 1, 3, 4])
                s = t.sample()
                self.assertEqual([(g["index"], g["engine"]) for g in s["gpus"]],
                                 [(0, False), (1, True), (2, True), (3, True), (4, True)])
                self.assertEqual(s["gpu_mem_total"], (24564 * 2 + 32607 + 12288) << 20)
                self.assertEqual(t.static["gpu_name"].count(" + "), 3)
            finally:
                t.close()

    def test_api(self):
        with mock.patch.dict(os.environ, {"STRATA_MOCK_GPUS": "gpus-mixed-roles.json"}):
            tok = ByteTokenizer()
            svc = Service(MockEngine(tok, "x"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
            httpd = serve(svc, port=0)
            try:
                base = f"http://127.0.0.1:{httpd.server_address[1]}"
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    m = json.load(urllib.request.urlopen(base + "/metrics", timeout=10))
                    if any(c.get("util") is not None for c in m["gpus"]["cards"]) and \
                            m["hardware_static"].get("engine_gpus") == [1, 2, 3, 4]:
                        break
                    time.sleep(0.2)
                cards = m["gpus"]["cards"]
                self.assertTrue(m["gpus"]["mock"])
                self.assertEqual(m["gpus"]["mapping"], "pci_bus_id")
                self.assertEqual([(c["index"], c["cuda"], c["role"]) for c in cards],
                                 [(0, None, "encoder"), (1, 1, "stage"), (2, 0, "stage"), (3, 2, "stage"),
                                  (4, 3, "helper")])
                self.assertEqual([c.get("layers") for c in cards[1:4]], [[14, 29], [0, 13], [30, 47]])
                # the PCIe link: the best one the slot gives (static), what the card runs at, the engine's measurement
                self.assertEqual([(c["pcie_gen_max"], c["pcie_width_max"]) for c in cards],
                                 [(4, 16), (4, 16), (4, 16), (4, 16), (3, 16)])
                self.assertEqual((cards[4]["pcie_width"], cards[3]["pcie_gen_gpu_max"]), (4, 5))
                self.assertEqual([c.get("link_gbps") for c in cards], [None, 24.1, 24.6, 25.3, None])
                self.assertEqual(m["hardware_static"]["engine_gpus"], [1, 2, 3, 4])   # the totals follow the layout
                st = json.load(urllib.request.urlopen(base + "/v1/status", timeout=10))
                self.assertEqual([g["role"] for g in st["machine"]["gpus"]],
                                 ["encoder", "stage", "stage", "stage", "helper"])
            finally:
                httpd.shutdown()
                httpd.server_close()
                svc.telemetry.close()

    def test_reader_without_info(self):
        """The Intel server swaps telemetry.gpu_reader for its xe reader, which has no info(): the name is enough."""
        class Xe:
            def ok(self):
                return True

            def name(self):
                return "Intel Arc Pro B70"

            def read(self):
                return {"util": 5, "mem_used": 1 << 30, "mem_total": 32 << 30}
        with mock.patch.object(telemetry, "gpu_reader", lambda index=0, amd=False: Xe()), \
                mock.patch.object(telemetry, "gpu_count", lambda amd=False: 0), \
                mock.patch.dict(os.environ, {"STRATA_MOCK_GPUS": ""}):
            t = telemetry.Telemetry()
            t.close()
            self.assertEqual(t.static["gpus"], [{"index": 0, "name": "Intel Arc Pro B70", "uuid": None, "pci": None}])
            self.assertEqual(t.sample()["gpus"][0]["util"], 5)

    def test_amd_link_is_capped_by_the_slowest_hop(self):
        """#1733: a Gen4 card behind a Gen3 bridge runs Gen3 - the row must not say Gen4 (nor miss the bottleneck)."""
        self.assertEqual(telemetry._amd_link_now({"pcie_own_gen": 4, "pcie_gen": 3}), 3)
        self.assertEqual(telemetry._amd_link_now({"pcie_own_gen": 1, "pcie_gen": 3}), 1)     # idle: its own drop
        self.assertEqual(telemetry._amd_link_now({"pcie_own_gen": None, "pcie_gen": 3}), 3)
        a = telemetry._Amd.__new__(telemetry._Amd)
        a.dev, a.hwmon = "/nonexistent/0000:03:00.0", None
        with mock.patch.object(a, "link", return_value={"pcie_gen": 3, "pcie_width": 8}), \
                mock.patch.object(telemetry._Amd, "_read", staticmethod(lambda d, n: "16.0 GT/s PCIe")), \
                mock.patch.object(a, "name", return_value="AMD Radeon AI PRO R9700"):
            info = a.info()
        self.assertEqual((info["pcie_gen_max"], info["pcie_width_max"], info["pcie_gen_gpu_max"]), (3, 8, 4))

    def test_generated_four_cards(self):
        with mock.patch.dict(os.environ, {"STRATA_MOCK_GPUS": "4"}):
            fx = gpus.mock_fixture()
        lay = gpus.parse_layout(fx["engine"]["log"])
        self.assertEqual({d: s["layers"] for d, s in lay["stages"].items()},
                         {0: [0, 11], 1: [12, 23], 2: [24, 35], 3: [36, 47]})


class Disks(unittest.TestCase):
    """The disk card reads the drives that hold the model's files."""

    def test_model_files(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "models").mkdir()
            (Path(d) / "models" / "ple.gguf").write_bytes(b"x")
            (Path(d) / "models" / "pack").mkdir()
            got = telemetry.model_files(["--pack", "models/pack", "--ple-gguf", "models/ple.gguf", "--mtp", "nope",
                                         "--max-context", "4096"], cwd=d)
            self.assertEqual(got, [("model pack", os.path.abspath(os.path.join(d, "models/pack"))),
                                   ("PLE table", os.path.abspath(os.path.join(d, "models/ple.gguf")))])

    def test_mountinfo(self):
        with tempfile.TemporaryDirectory() as d:
            mi = Path(d) / "mountinfo"
            mi.write_text("22 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n"
                          "40 22 0:45 / /data rw,relatime shared:20 - btrfs /dev/sdb1 rw,space_cache\n"
                          "41 22 0:46 / /mnt/my\\040models rw - ext4 /dev/sdc1 rw\n"
                          "42 22 0:47 / /var/lib/docker/overlay rw - overlay overlay rw\n", encoding="utf-8")
            self.assertEqual(telemetry._mount_of("/data/models/pack", str(mi)), ("/data", "/dev/sdb1"))
            self.assertEqual(telemetry._mount_of("/datax/m", str(mi)), ("/", "/dev/nvme0n1p2"))
            self.assertEqual(telemetry._mount_of("/mnt/my models/x", str(mi)), ("/mnt/my models", "/dev/sdc1"))
            # btrfs: an anonymous device number - the mount's /dev source names the drive
            st = SimpleNamespace(st_dev=0)
            with mock.patch.object(telemetry.os, "stat", return_value=st), \
                    mock.patch.object(telemetry.os, "major", create=True, return_value=0), \
                    mock.patch.object(telemetry.os, "minor", create=True, return_value=45), \
                    mock.patch.object(telemetry.os.path, "realpath", side_effect=lambda p: p):
                self.assertEqual(telemetry._linux_disk("/data/models/pack", mountinfo=str(mi)),
                                 (("sdb1",), "/data (sdb1)"))
                self.assertIsNone(telemetry._linux_disk("/var/lib/docker/overlay/x", mountinfo=str(mi)))

    @unittest.skipUnless(sys.platform.startswith("linux"), "sysfs device numbers")
    def test_linux_device_number(self):
        got = telemetry._linux_disk(__file__)
        if got is not None:                              # a container's overlay names no drive
            self.assertTrue(got[0][0])

    @unittest.skipUnless(os.name == "nt", "Windows volumes")
    def test_windows_volume(self):
        got = telemetry.disk_of(__file__)
        self.assertIsNotNone(got)
        self.assertTrue(got[0][0].startswith("PhysicalDrive"))
        self.assertRegex(got[1], r"^[A-Za-z]:")

    def fake_psutil(self, counters):
        vm = SimpleNamespace(total=64 << 30, available=40 << 30)

        def io(perdisk=False):
            c = counters()
            if perdisk:
                return c
            return SimpleNamespace(read_bytes=sum(x.read_bytes for x in c.values()),
                                   write_bytes=sum(x.write_bytes for x in c.values()))
        return SimpleNamespace(cpu_count=lambda logical=True: 8, cpu_percent=lambda interval=None: 1.0,
                               virtual_memory=lambda: vm, disk_io_counters=io)

    def test_rates_per_model_drive(self):
        n = {"i": 0}

        def counters():
            n["i"] += 1
            k = n["i"]
            return {"PhysicalDrive0": SimpleNamespace(read_bytes=k * (100 << 20), write_bytes=0),
                    "PhysicalDrive1": SimpleNamespace(read_bytes=k * (2000 << 20), write_bytes=k * (1 << 20)),
                    "PhysicalDrive2": SimpleNamespace(read_bytes=k * (999 << 20), write_bytes=0)}
        drives = [{"keys": ("PhysicalDrive1",), "label": "D:", "roles": ["model pack", "dense weights"]},
                  {"keys": ("PhysicalDrive0",), "label": "C:", "roles": ["PLE table"]}]
        with mock.patch.dict(sys.modules, {"psutil": self.fake_psutil(counters)}), \
                mock.patch.object(telemetry, "model_disks", return_value=drives), \
                mock.patch.dict(os.environ, {"STRATA_MOCK_GPUS": "", "STRATA_GPU_POLL_S": "0"}):
            t = telemetry.Telemetry(files=[("model pack", "x")])
            t.close()
            self.assertEqual(t.static["disk_scope"], "model")
            self.assertEqual([d["label"] for d in t.static["disks"]], ["D:", "C:"])
            clock = iter([1000.0, 1001.0])
            with mock.patch.object(telemetry.time, "time", lambda: next(clock)):
                t._disk_prev = None
                t._disk()
                r, w, per = t._disk()
        self.assertAlmostEqual(r, 2100, delta=1)       # the model's drives only: not PhysicalDrive2's 999 MB/s
        self.assertAlmostEqual(w, 1, delta=0.01)
        self.assertEqual([(d["label"], round(d["read_mb"])) for d in per], [("D:", 2000), ("C:", 100)])

    def test_all_drives_when_the_model_drive_is_unknown(self):
        def counters():
            return {"sda": SimpleNamespace(read_bytes=1, write_bytes=2)}
        with mock.patch.dict(sys.modules, {"psutil": self.fake_psutil(counters)}), \
                mock.patch.object(telemetry, "model_disks", return_value=[]), \
                mock.patch.dict(os.environ, {"STRATA_MOCK_GPUS": "", "STRATA_GPU_POLL_S": "0"}):
            t = telemetry.Telemetry(files=[("model pack", "x")])
            t.close()
        self.assertEqual((t.static["disk_scope"], t.static["disks"]), ("all", []))

    def test_psutil_missing_names_the_command(self):
        with mock.patch.object(telemetry.sys, "executable", r"C:\venv\Scripts\pythonw.exe"), \
                mock.patch.object(telemetry.os, "name", "nt"), \
                mock.patch.object(telemetry.os.path, "exists", return_value=True):
            cmd = telemetry.pip_install_command()
        self.assertTrue(cmd.endswith("python.exe -m pip install psutil"), cmd)
        self.assertNotIn("pythonw", cmd)
        with mock.patch.object(telemetry.sys, "executable", "/opt/my venv/bin/python3"):
            self.assertEqual(telemetry.pip_install_command(), '"/opt/my venv/bin/python3" -m pip install psutil')


if __name__ == "__main__":
    unittest.main()
