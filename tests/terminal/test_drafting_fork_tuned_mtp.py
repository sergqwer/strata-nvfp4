"""strata-nvfp4 0.2.15-nvfp4.4: Basalt's self-distilled draft layer ships for this fork's GPTQ set.  drafting_fetch.ensure_fork_tuned_mtp
puts dense.bin and dense.txt (pinned, checked) into <mtp>/rt-tuned and links the set's experts.bin and draft_vocab.bin beside them; Auto's
MTP then drafts with that folder.  No hard-coded development folder is looked at any more."""

from __future__ import annotations

import hashlib
import http.server
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from strata_control import drafting, drafting_fetch

GPTQ = ("qwen3.8-flash-next-gptq-nvfp4", "qwen3.8-flash-next-nvfp4-gptq", "qwen3.8-flash-next-nvfp4-gptq-uncensored")
NOT_GPTQ = ("qwen3.8-flash-next-nvidia-nvfp4", "qwen3.8-flash-next-mix-nvfp4", "qwen3.8-flash-next-q8_0", "qwen3.8-flash-next-iq3_s")


class _Handler(http.server.BaseHTTPRequestHandler):
    files: dict[str, bytes] = {}
    hits: list[str] = []

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        name = self.path.rsplit("/", 1)[-1]
        type(self).hits.append(name)
        body = type(self).files.get(name)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


def tray_args(data: Path) -> list[str]:
    return ["--pack", str(data / "pack"), "--native", str(data / "dense.gguf"), "--expert-cache", "auto", "--prefill", "auto",
            "--spec", "6", "--spec-min-p", "0.7", "--mtp", str(data / "mtp"), "--max-context", "524288", "--kv", "int8"]


class TunedLayer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name) / "qwen-nvfp4-gptq"
        self.mtp = self.data / "mtp"
        self.mtp.mkdir(parents=True)
        for name, body in (("experts.bin", b"E" * 4096), ("draft_vocab.bin", b"V" * 64), ("dense.bin", b"D" * 512), ("dense.txt", b"t\n")):
            (self.mtp / name).write_bytes(body)
        self.blobs = {"dense.bin": b"T" * 512, "dense.txt": b"t\n"}
        self.files = tuple((n, len(b), hashlib.sha256(b).hexdigest()) for n, b in self.blobs.items())
        _Handler.files, _Handler.hits = dict(self.blobs), []
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}/"
        env = mock.patch.dict("os.environ", {"STRATA_UPDATE_ALLOW_HTTP": "1", "STRATA_MTP_TUNED_DIR": "", "STRATA_AUTO_NEW_MTP": ""})
        env.start()
        self.addCleanup(env.stop)

    def fetch(self, files=None):
        return drafting_fetch.ensure_fork_tuned_mtp(self.mtp, base_url=self.base, files=files or self.files)

    def test_the_pins_are_the_uploaded_files(self):
        self.assertEqual(drafting.FORK_MTP_TUNED_REPO, "Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata")
        self.assertRegex(drafting.FORK_MTP_TUNED_REVISION, r"^[0-9a-f]{40}$")
        self.assertIn(f"/resolve/{drafting.FORK_MTP_TUNED_REVISION}/mtp/rt-tuned/", drafting.FORK_MTP_TUNED_BASE_URL)
        names = [n for n, _s, _h in drafting.FORK_MTP_TUNED_FILES]
        self.assertEqual(names, ["dense.bin", "dense.txt"])
        for _n, size, sha in drafting.FORK_MTP_TUNED_FILES:
            self.assertGreater(size, 0)
            self.assertRegex(sha, r"^[0-9a-f]{64}$")
        self.assertFalse(hasattr(drafting, "FORK_MTP_DEV_TUNED"), "no development folder is looked at")

    def test_only_the_gptq_set_gets_it(self):
        for name in GPTQ:
            self.assertTrue(drafting.fork_tuned_wanted(name), name)
        for name in NOT_GPTQ:
            self.assertFalse(drafting.fork_tuned_wanted(name), name)

    def test_fetch_puts_the_layer_in_place_and_links_the_sets_files(self):
        out = self.fetch()
        self.assertEqual(out, self.mtp / "rt-tuned")
        self.assertEqual((out / "dense.bin").read_bytes(), self.blobs["dense.bin"])
        self.assertEqual((out / "dense.txt").read_bytes(), self.blobs["dense.txt"])
        for name in ("experts.bin", "draft_vocab.bin"):
            self.assertTrue(os.path.samefile(out / name, self.mtp / name), name)       # the same volume: a hard link, no second copy
        self.assertEqual((self.mtp / "dense.bin").read_bytes(), b"D" * 512, "the set's own layer is untouched")
        self.assertTrue(drafting.fork_tuned_mtp_complete(out))
        self.assertFalse(list(out.glob("*.part")))

    def test_a_second_run_downloads_nothing(self):
        self.fetch()
        _Handler.hits = []
        with mock.patch.object(drafting_fetch.transfer, "download", side_effect=AssertionError("downloaded again")):
            self.fetch()
        self.assertEqual(_Handler.hits, [])

    def test_a_copy_where_links_fail_and_the_vocabulary_follows_the_set(self):
        with mock.patch.object(drafting.os, "link", side_effect=OSError("no links on this drive")):
            out = self.fetch()
            self.assertFalse(os.path.samefile(out / "experts.bin", self.mtp / "experts.bin"))
            self.assertEqual((out / "experts.bin").read_bytes(), (self.mtp / "experts.bin").read_bytes())
            (self.mtp / "draft_vocab.bin").write_bytes(b"C" * 80)                     # another subset chosen (setup's refresh_draft_vocab)
            self.assertTrue(drafting.sync_fork_tuned_mtp(self.mtp))
        self.assertEqual((out / "draft_vocab.bin").read_bytes(), b"C" * 80)

    def test_a_wrong_file_is_refused_and_not_used(self):
        bad = (("dense.bin", 512, "0" * 64), self.files[1])
        with self.assertRaises(drafting_fetch.DraftingError):
            self.fetch(files=bad)
        self.assertFalse((self.mtp / "rt-tuned" / "dense.bin").exists())
        self.assertIsNone(drafting.fork_tuned_mtp_dir(tray_args(self.data), self.data))

    def test_without_the_sets_own_layer_nothing_is_fetched(self):
        (self.mtp / "experts.bin").unlink()
        with self.assertRaises(drafting_fetch.DraftingError):
            self.fetch()
        self.assertEqual(_Handler.hits, [])

    def test_auto_drafts_with_the_fetched_layer(self):
        args = tray_args(self.data)
        self.assertIsNone(drafting.fork_tuned_mtp_dir(args, self.data), "nothing here yet: the set's own layer")
        self.fetch()
        self.assertEqual(drafting.fork_tuned_mtp_dir(args, self.data), self.mtp / "rt-tuned")
        cfg = {"args": list(args), "model_name": GPTQ[1]}
        with mock.patch.object(drafting, "computer_language_signals", lambda: {}), \
                mock.patch.object(drafting, "pack_ready", lambda d: False):
            decision = drafting.apply(cfg, data_dir=self.data, backend="cuda", archs=["sm_120"], vram_mib=32607)
        self.assertEqual(decision.drafter, "mtp")
        a = cfg["args"]
        self.assertEqual(a[a.index("--mtp") + 1], str(self.mtp / "rt-tuned"))
        self.assertEqual(a[a.index("--mtp-head") + 1], "proxy")
        with mock.patch.dict("os.environ", {"STRATA_AUTO_NEW_MTP": "0"}), \
                mock.patch.object(drafting, "computer_language_signals", lambda: {}), \
                mock.patch.object(drafting, "pack_ready", lambda d: False):
            cfg = {"args": list(args), "model_name": GPTQ[1]}
            drafting.apply(cfg, data_dir=self.data, backend="cuda", archs=["sm_120"], vram_mib=32607)
        self.assertEqual(cfg["args"][cfg["args"].index("--mtp") + 1], str(self.mtp), "the old Auto keeps the set's own layer")

    def test_an_incomplete_folder_is_not_used(self):
        tuned = self.mtp / "rt-tuned"
        tuned.mkdir()
        (tuned / "dense.bin.part").write_bytes(b"T" * 100)
        (tuned / "experts.bin").write_bytes(b"x")
        self.assertIsNone(drafting.fork_tuned_mtp_dir(tray_args(self.data), self.data))


if __name__ == "__main__":
    unittest.main()
