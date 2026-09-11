#!/usr/bin/env python3
"""Dataset contract. No GPU; builds its own tiny corpus in a temp dir.

The contract exists so bench has exactly one loader and no per-dataset branches.
These tests pin the three rules that make that possible: audio paths are relative,
duration is declared (it is LAAL's T), and the encoding is declared because
headerless PCM cannot be sniffed.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.data import audio as audio_mod
from bench.data import manifest
from bench.errors import BenchDataError

SR = 16000

SPEC = """name: tiny
split: test
languages: [en, fr]
audio: {format: flac, sample_rate: 16000}
provides: {transcript: true, translations: [ko]}
primary_metric: wer
group_rule: "conv"
bench_defaults: {trailing_silence_ms: 1000}
"""

ROWS = [
    {"id": "a", "audio": "audio/a.flac", "duration": 1.5, "group": "conv-1",
     "speaker": "S1", "src_lang": "en",
     "reference": {"transcript": "HELLO WORLD", "translations": {"ko": "안녕 세계"}}},
    {"id": "b", "audio": "audio/b.flac", "duration": 2.25, "group": "conv-1",
     "speaker": "S2", "src_lang": "fr",
     "reference": {"transcript": "BONJOUR LE MONDE", "translations": {"ko": "안녕 세상"}}},
    {"id": "c", "audio": "audio/c.flac", "duration": 0.8, "group": "conv-2",
     "speaker": "S1", "src_lang": "en",
     "reference": {"transcript": "GOOD BYE", "translations": {"ko": "안녕히"}}},
]


def tone(secs: float, hz: float = 220.0) -> np.ndarray:
    t = np.linspace(0, secs, int(SR * secs), endpoint=False)
    return (0.1 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def build(root: Path, *, spec: str = SPEC, rows=None) -> Path:
    (root / "audio").mkdir(parents=True, exist_ok=True)
    (root / "dataset.yml").write_text(spec, encoding="utf-8")
    rows = ROWS if rows is None else rows
    for row in rows:
        name = Path(row["audio"]).name
        if name.endswith(".pcm"):
            (tone(row["duration"], 330) * 32767).astype(np.int16).tofile(root / row["audio"])
        elif not (root / row["audio"]).exists():
            sf.write(str(root / row["audio"]), tone(row["duration"]), SR)
    with open(root / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return root


class DatasetCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = build(Path(self._tmp.name) / "tiny")

    def tearDown(self):
        self._tmp.cleanup()

    def load(self, **kw):
        return manifest.load("tiny", root=self.root, **kw)

    def rewrite(self, rows):
        with open(self.root / "manifest.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


class TestLoad(DatasetCase):
    def test_happy_path(self):
        spec = self.load()
        self.assertEqual([i.id for i in spec.items], ["a", "b", "c"])
        self.assertEqual(spec.n_sessions, 2)
        self.assertEqual(spec.languages, ["en", "fr"])
        self.assertEqual(spec.translation_langs, ["ko"])
        self.assertEqual(spec.items[1].src_lang, "fr")
        self.assertEqual(spec.items[0].reference_translation("ko"), "안녕 세계")
        self.assertEqual(spec.items[0].reference_translation("de"), "")
        self.assertEqual(manifest.verify(spec), [])

    def test_groups_are_contiguous_and_within_group_order_is_preserved(self):
        # conv-2's item first, then conv-1's two in reverse file order.
        self.rewrite([ROWS[2], ROWS[1], ROWS[0]])
        ids = [i.id for i in self.load().items]
        # grouped: conv-1 items together, then conv-2
        self.assertEqual(ids, ["b", "a", "c"])
        groups = [i.group for i in self.load().items]
        self.assertEqual(groups, ["conv-1", "conv-1", "conv-2"])

    def test_within_group_order_is_not_reordered_by_id(self):
        # ACL 60/60 relies on this: speaking order can disagree with id order.
        rows = [{**ROWS[1], "id": "z_first", "group": "g"},
                {**ROWS[0], "id": "a_second", "group": "g"}]
        self.rewrite(rows)
        self.assertEqual([i.id for i in self.load().items], ["z_first", "a_second"])

    def test_limit_applies_after_sorting(self):
        self.assertEqual([i.id for i in self.load(limit=2).items], ["a", "b"])

    def test_group_defaults_to_id(self):
        rows = [{**ROWS[0]}]
        rows[0].pop("group")
        self.rewrite(rows)
        self.assertEqual(self.load().items[0].group, "a")

    def test_manifest_sha_changes_with_contents(self):
        before = self.load().manifest_sha256
        self.rewrite(ROWS[:2])
        self.assertNotEqual(before, self.load().manifest_sha256)


class TestRejections(DatasetCase):
    def test_absolute_audio_path(self):
        rows = [{**ROWS[0], "audio": str(self.root / "audio/a.flac")}]
        self.rewrite(rows)
        with self.assertRaises(BenchDataError) as ctx:
            self.load()
        self.assertIn("relative", str(ctx.exception))

    def test_missing_duration(self):
        rows = [{k: v for k, v in ROWS[0].items() if k != "duration"}]
        self.rewrite(rows)
        with self.assertRaises(BenchDataError) as ctx:
            self.load()
        self.assertIn("duration", str(ctx.exception))

    def test_duplicate_id(self):
        self.rewrite([ROWS[0], ROWS[0]])
        with self.assertRaises(BenchDataError) as ctx:
            self.load()
        self.assertIn("duplicate", str(ctx.exception))

    def test_unknown_audio_format(self):
        (self.root / "dataset.yml").write_text(
            SPEC.replace("format: flac", "format: mp3"), encoding="utf-8")
        with self.assertRaises(BenchDataError) as ctx:
            self.load()
        self.assertIn("audio.format", str(ctx.exception))

    def test_missing_spec_points_at_the_contract(self):
        (self.root / "dataset.yml").unlink()
        with self.assertRaises(BenchDataError) as ctx:
            self.load()
        self.assertIn("datasets/README.md", str(ctx.exception))

    def test_unknown_src_lang(self):
        self.rewrite([{**ROWS[0], "src_lang": "klingon"}])
        with self.assertRaises(BenchDataError):
            self.load()


class TestVerify(DatasetCase):
    def test_missing_audio_is_reported_not_raised(self):
        (self.root / "audio/b.flac").unlink()
        problems = manifest.verify(self.load())
        self.assertEqual(len(problems), 1)
        self.assertIn("audio missing", problems[0])

    def test_duration_mismatch_is_caught(self):
        # the manifest claims 9 seconds; the file on disk is 1.5
        self.rewrite([{**ROWS[0], "duration": 9.0}])
        problems = manifest.verify(self.load())
        self.assertTrue(any("duration" in p for p in problems), problems)

    def test_empty_transcript_is_caught(self):
        self.rewrite([{**ROWS[0], "reference": {"transcript": "  ",
                                                "translations": {"ko": "x"}}}])
        self.assertTrue(any("empty transcript" in p for p in manifest.verify(self.load())))

    def test_missing_reference_translation_is_caught(self):
        self.rewrite([{**ROWS[0], "reference": {"transcript": "HI", "translations": {}}}])
        problems = manifest.verify(self.load())
        self.assertTrue(any("ko reference translation" in p for p in problems), problems)


class TestAudio(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_flac_roundtrip_length(self):
        path = self.dir / "x.flac"
        sf.write(str(path), tone(2.0), SR)
        audio = audio_mod.load_window(path, "flac")
        self.assertAlmostEqual(len(audio) / SR, 2.0, places=3)
        self.assertAlmostEqual(audio_mod.probe_duration(path, "flac"), 2.0, places=3)

    def test_window_slicing_reads_only_the_window(self):
        path = self.dir / "x.flac"
        sf.write(str(path), tone(5.0), SR)
        audio = audio_mod.load_window(path, "flac", offset=1.0, duration=2.0)
        self.assertAlmostEqual(len(audio) / SR, 2.0, places=3)

    def test_raw_pcm_path(self):
        path = self.dir / "x.pcm"
        (tone(1.0) * 32767).astype(np.int16).tofile(path)
        audio = audio_mod.load_window(path, "pcm_s16le")
        self.assertAlmostEqual(len(audio) / SR, 1.0, places=3)
        self.assertAlmostEqual(audio_mod.probe_duration(path, "pcm_s16le"), 1.0, places=3)

    def test_raw_pcm_window(self):
        path = self.dir / "x.pcm"
        (tone(4.0) * 32767).astype(np.int16).tofile(path)
        audio = audio_mod.load_window(path, "pcm_s16le", offset=1.0, duration=1.5)
        self.assertAlmostEqual(len(audio) / SR, 1.5, places=3)

    def test_pcm_bytes_roundtrip(self):
        audio = tone(0.5)
        raw = audio_mod.to_pcm_bytes(audio)
        self.assertEqual(len(raw), len(audio) * 2)
        back = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
        self.assertLess(float(np.max(np.abs(back - audio))), 1e-3)

    def test_silence_bytes_length(self):
        self.assertEqual(len(audio_mod.silence_bytes(1000)), SR * 2)

    def test_missing_file_raises(self):
        with self.assertRaises(BenchDataError):
            audio_mod.load_window(self.dir / "nope.flac", "flac")


if __name__ == "__main__":
    unittest.main(verbosity=2)
