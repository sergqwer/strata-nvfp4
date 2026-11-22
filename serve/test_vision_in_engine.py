"""The picture reader inside the engine (vision.in_engine, strata --vision-gpu): the engine reads a picture on the GPU in borrowed expert slots (VENC);
a picture it cannot read goes to the section's CPU reader, started the first time it is needed; STRATA_VISION_IN_ENGINE=0 or "in_engine": false keep
the CPU reader as before; the engine's arguments carry --vision-gpu only when the engine reads the pictures.

    python -m serve.test_vision_in_engine
"""
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import serve.server as server

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def vcfg(**extra):
    return {"exe": "/engine/strata-vision", "mmproj": "/m/mmproj.gguf", "model": "/m/model.gguf", "gpu": False, "threads": 16,
            "max_tokens": 1024, "in_engine": True, **extra}


class FakeEngine:
    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.calls = answer, error, []

    def alive(self):
        return True

    def vision_encode(self, image, out):
        self.calls.append((image, out))
        assert os.path.isfile(image), "the picture is written before the engine reads it"
        if self.error:
            raise ValueError(self.error)
        Path(out).write_bytes(struct.pack("<5i", 0x31455653, self.answer, self.answer, 1, 4) + b"\x00" * (16 * self.answer))
        return self.answer


class Switch(unittest.TestCase):
    def test_in_engine_needs_the_key(self):
        self.assertTrue(server.vision_in_engine({"vision": vcfg()}))
        self.assertFalse(server.vision_in_engine({"vision": vcfg(in_engine=False)}))
        self.assertFalse(server.vision_in_engine({"vision": {k: v for k, v in vcfg().items() if k != "in_engine"}}))

    def test_the_environment_switch_keeps_the_cpu_reader(self):
        with mock.patch.dict(os.environ, {"STRATA_VISION_IN_ENGINE": "0"}):
            self.assertFalse(server.vision_in_engine({"vision": vcfg()}))

    def test_not_on_amd_or_several_gpus(self):
        self.assertFalse(server.vision_in_engine({"vision": vcfg(), "backend": "hip"}))
        with mock.patch.object(server, "gpu_list", return_value=[0, 1]):
            self.assertFalse(server.vision_in_engine({"vision": vcfg()}))


class EngineArgs(unittest.TestCase):
    def args(self, cfg):
        with mock.patch.object(server.vram_fit, "fit_start", side_effect=lambda c, a, *_: (a, None, "")):
            return server.engine_args(cfg, announce=False)

    def test_the_engine_gets_the_reader(self):
        a = self.args({"args": ["--serve", "--vision"], "vision": vcfg(min_tokens=64)})
        i = a.index("--vision-gpu")
        self.assertTrue(a[i + 1].endswith("mmproj.gguf"))
        self.assertEqual(a[a.index("--vision-max-tokens") + 1], "1024")
        self.assertEqual(a[a.index("--vision-min-tokens") + 1], "64")
        self.assertNotIn("--vision-gpu-preload", a)
        self.assertIn("--vision-gpu-preload", self.args({"args": ["--serve", "--vision"], "vision": vcfg(preload=True)}))

    def test_the_cpu_reader_config_is_unchanged(self):
        a = self.args({"args": ["--serve", "--vision"], "vision": vcfg(in_engine=False)})
        self.assertNotIn("--vision-gpu", a)


class Reading(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name) / "work"
        d.mkdir()
        with mock.patch.object(server.Vision, "work_dir", return_value=d):
            self.v = server.EngineVision(vcfg())
        self.started = []
        self.v.restart = lambda: self.started.append(1)

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_reader_process_at_the_start(self):
        self.assertIsNone(self.v.proc)
        self.assertTrue(self.v.alive())                 # the service never starts the CPU reader with the engine
        self.assertNotIn("--gpu", self.v.spawn[0])       # its fallback is the CPU reader

    def test_the_engine_reads_and_the_result_is_cached(self):
        eng = FakeEngine(answer=260)
        self.v.engine_of = lambda: eng
        path, n = self.v.encode(PNG)
        self.assertEqual(n, 260)
        self.assertTrue(path.is_file())
        self.assertEqual(self.v.encode(PNG), (path, n))
        self.assertEqual(len(eng.calls), 1)
        self.assertNotIn(" ", eng.calls[0][1])           # the output path is the last word of the line
        self.assertFalse(Path(eng.calls[0][0]).exists())  # the picture file is gone after the read

    def test_a_refusal_goes_to_the_cpu_reader(self):
        eng = FakeEngine(error="VENC: the expert cache has no room for the picture reader (1100 MiB)")
        self.v.engine_of = lambda: eng
        with mock.patch.object(server.Vision, "alive", return_value=False), \
                mock.patch.object(server.Vision, "restart") as rs, \
                mock.patch.object(server.Vision, "encode", return_value=(Path("x.sve"), 7)) as enc:
            self.assertEqual(self.v.encode(PNG), (Path("x.sve"), 7))
        rs.assert_called_once()
        enc.assert_called_once()
        self.assertIsNone(self.v.engine_off)             # the next picture asks the engine again

    def test_an_engine_without_the_reader_is_not_asked_again(self):
        eng = FakeEngine(error="VENC needs an engine started with --vision-gpu <mmproj> (one NVIDIA GPU, --serve)")
        self.v.engine_of = lambda: eng
        with mock.patch.object(server.Vision, "alive", return_value=True), \
                mock.patch.object(server.Vision, "encode", return_value=(Path("x.sve"), 7)):
            self.v.encode(PNG)
            self.v.encode(PNG + b"1")
        self.assertEqual(len(eng.calls), 1)
        self.assertIn("--vision-gpu", self.v.engine_off)

    def test_no_engine_reads_on_the_cpu(self):
        self.v.engine_of = lambda: None
        with mock.patch.object(server.Vision, "alive", return_value=True), \
                mock.patch.object(server.Vision, "encode", return_value=(Path("x.sve"), 7)) as enc:
            self.v.encode(PNG)
        enc.assert_called_once()


if __name__ == "__main__":
    unittest.main()
