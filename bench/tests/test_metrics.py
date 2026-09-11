#!/usr/bin/env python3
"""Metrics over frozen rows. No GPU, no model, no network.

Two things here are load-bearing. test_empty_hypothesis_is_charged shows the
legacy WER reporting a perfect score on a corpus where an utterance failed
completely, which is why bench does not use it as the primary number. And the
routing tests show a misrouted segment being kept out of BLEU, so a
language-detection failure cannot masquerade as a translation-quality one.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench import config, metrics
from bench.errors import BenchConfigError
from bench.metrics import asr, commit, latency, routing, translation


def seg(**kw):
    base = {"segment_id": 1, "text": "", "translation": "", "commit_reason": "seg",
            "language": "en", "target_lang": "ko", "fsl_sec": None,
            "decision_audio_sec": None, "recv_elapsed_sec": None,
            "direction_refixed": False}
    base.update(kw)
    return base


def row(**kw):
    base = {"id": "x", "status": "ok", "src_lang": "en", "speaker": "S",
            "duration_sec": 5.0, "reference": "", "hypothesis": "",
            "reference_translations": {}, "segments": []}
    base.update(kw)
    return base


def make_cfg(metric_list, languages=None, vad_min_silence_ms=800):
    raw = {
        "name": "t",
        "dataset": {"name": "d"},
        "languages": languages or {"lang": "en", "target": "ko"},
        "stity": {
            "transcription": {"name": "qwen3"},
            "translation": "none",
            "commit": "seg",
            "vad": {"min_silence_ms": vad_min_silence_ms},
            "gpu_memory_utilization": 0.5,
        },
        "metrics": metric_list,
    }
    return config.parse(raw)


class FakeDataset:
    def __init__(self, name="d", has_transcript=True, translation_langs=(),
                 languages=("en",)):
        self.name = name
        self.has_transcript = has_transcript
        self.translation_langs = list(translation_langs)
        # Which source languages the audio actually contains. The reference check
        # needs it: it only demands references for directions this data triggers.
        self.languages = list(languages)


class TestWer(unittest.TestCase):
    def test_known_value(self):
        # 4 reference words, 1 substitution -> 0.25
        rows = [row(reference="the cat sat down", hypothesis="the cat sat up")]
        self.assertAlmostEqual(asr.corpus_wer(rows), 0.25)

    def test_normalization_ignores_case_and_punctuation(self):
        rows = [row(reference="HELLO WORLD", hypothesis="hello, world.")]
        self.assertEqual(asr.corpus_wer(rows), 0.0)

    def test_corpus_not_mean_of_per_item(self):
        # 1/1 errors on a short row, 0/10 on a long one.
        # corpus = 1/11; mean of per-item would be 0.5.
        rows = [row(reference="a", hypothesis="b"),
                row(reference="a b c d e f g h i j", hypothesis="a b c d e f g h i j")]
        self.assertAlmostEqual(asr.corpus_wer(rows), 1 / 11)

    def test_empty_hypothesis_is_charged(self):
        rows = [row(reference="hello world", hypothesis="hello world"),
                row(reference="good bye", hypothesis="")]
        self.assertAlmostEqual(asr.corpus_wer(rows), 0.5)
        self.assertEqual(asr.wer_scored_only(rows), 0.0)
        self.assertEqual(asr.count_empty_hypotheses(rows), 1)

    def test_the_two_agree_when_nothing_is_empty(self):
        rows = [row(reference="the cat sat down", hypothesis="the cat sat up"),
                row(reference="a b c d", hypothesis="a b x d")]
        self.assertAlmostEqual(asr.corpus_wer(rows), asr.wer_scored_only(rows))


class TestCer(unittest.TestCase):
    def test_known_value(self):
        # reference strips to 4 chars, one substitution
        self.assertAlmostEqual(asr.per_item_cer("가나다라", "가나다마"), 0.25)

    def test_whitespace_and_punctuation_ignored(self):
        self.assertEqual(asr.per_item_cer("안녕 하세요.", "안녕하세요"), 0.0)

    def test_empty_hypothesis_is_a_full_deletion(self):
        self.assertEqual(asr.corpus_cer([row(reference="가나다라", hypothesis="")]), 1.0)


class TestLatency(unittest.TestCase):
    def test_fsl_mean(self):
        segs = [seg(fsl_sec=0.10, commit_reason="dot"), seg(fsl_sec=0.30, commit_reason="dot")]
        stats = latency.fsl_stats(segs, min_silence_ms=800)
        self.assertAlmostEqual(stats["avg_fsl_sec"], 0.20)
        self.assertAlmostEqual(stats["avg_fsl_normalized_sec"], 0.20)

    def test_vad_normalization_tracks_configured_silence(self):
        segs = [seg(fsl_sec=0.10, commit_reason="vad")]
        self.assertAlmostEqual(
            latency.fsl_stats(segs, min_silence_ms=800)["avg_fsl_normalized_sec"], 0.90)
        self.assertAlmostEqual(
            latency.fsl_stats(segs, min_silence_ms=400)["avg_fsl_normalized_sec"], 0.50)

    def test_laal_matches_the_formula(self):
        # one segment, 2 target words, d = 2000ms, T = 4000ms, |Y_ref| = 2
        # delays expand to [2000, 2000]; LAAL = mean(2000 - 0*T/2, 2000 - 1*2000) = 1000
        segs = [seg(translation="aa bb", decision_audio_sec=2.0)]
        out = latency.laal_for_item(segs, src_duration_sec=4.0, ref_text="aa bb")
        self.assertAlmostEqual(out["laal_ms"], 1000.0)

    def test_laal_caps_at_source_length(self):
        segs = [seg(translation="aa", decision_audio_sec=99.0)]
        capped = latency.laal_for_item(segs, src_duration_sec=4.0, ref_text="aa",
                                       cap_source=True)["laal_ms"]
        uncapped = latency.laal_for_item(segs, src_duration_sec=4.0, ref_text="aa",
                                         cap_source=True)["laal_uncapped_ms"]
        self.assertAlmostEqual(capped, 4000.0)
        self.assertAlmostEqual(uncapped, 99000.0)

    def test_first_token_latency_skips_empty_translations(self):
        segs = [seg(translation="", recv_elapsed_sec=1.0),
                seg(translation="있다", recv_elapsed_sec=2.5)]
        self.assertEqual(latency.first_token_latency_sec(segs), 2.5)


class TestCommit(unittest.TestCase):
    def test_counts_and_finish_ratio(self):
        segs = [seg(commit_reason="dot"), seg(commit_reason="dot"),
                seg(commit_reason="finish"), seg(commit_reason="vad")]
        stats = commit.commit_stats(segs)
        self.assertEqual(stats["total"], 4)
        self.assertEqual(stats["counts"]["dot"], 2)
        self.assertAlmostEqual(stats["ratios"]["dot"], 0.5)
        self.assertAlmostEqual(stats["finish_ratio"], 0.25)

    def test_unknown_reason_is_not_silently_a_seg(self):
        stats = commit.commit_stats([seg(commit_reason="typo")])
        self.assertEqual(stats["counts"]["seg"], 0)
        self.assertEqual(stats["counts"]["typo"], 1)


class TestRouting(unittest.TestCase):
    def setUp(self):
        self.langs = make_cfg(["routing"],
                              languages={"map": {"en": "ko", "ko": "en", "fr": "ko"}}).languages

    def test_all_correct(self):
        rows = [row(id="a", src_lang="fr",
                    segments=[seg(language="fr", target_lang="ko")]),
                row(id="b", src_lang="ko",
                    segments=[seg(language="ko", target_lang="en")])]
        stats = routing.routing_stats(rows, self.langs)
        self.assertEqual(stats["lang_detect_accuracy"], 1.0)
        self.assertEqual(stats["route_accuracy"], 1.0)
        self.assertEqual(stats["n_misrouted"], 0)

    def test_misdetection_shows_in_both_numbers_and_the_matrix(self):
        # French audio detected as English -> routed to ko, which happens to match,
        # so detection is wrong while routing looks right.
        rows = [row(id="a", src_lang="fr",
                    segments=[seg(language="en", target_lang="ko")])]
        stats = routing.routing_stats(rows, self.langs)
        self.assertEqual(stats["lang_detect_accuracy"], 0.0)
        self.assertEqual(stats["route_accuracy"], 1.0)
        self.assertEqual(stats["confusion"], {"fr->en": 1})

    def test_misroute_is_counted_and_item_listed(self):
        rows = [row(id="a", src_lang="ko",
                    segments=[seg(language="ko", target_lang="ko")])]
        stats = routing.routing_stats(rows, self.langs)
        self.assertEqual(stats["n_misrouted"], 1)
        self.assertEqual(stats["misrouted_items"], ["a"])

    def test_direction_refix_is_counted(self):
        rows = [row(segments=[seg(direction_refixed=True), seg()])]
        self.assertEqual(routing.routing_stats(rows, self.langs)["n_direction_refix"], 1)


class TestRequirementChecks(unittest.TestCase):
    def test_bleu_without_translation_references_dies_early(self):
        cfg = make_cfg(["bleu"])
        with self.assertRaises(BenchConfigError) as ctx:
            metrics.check_requirements(cfg.metrics, dataset=FakeDataset(),
                                       languages=cfg.languages, realtime=True)
        self.assertIn("provides.translations", str(ctx.exception))
        self.assertIn("sacrebleu", str(ctx.exception))

    def test_bleu_with_wrong_language_references_dies_early(self):
        # en audio, en->ko run, but the dataset only carries German references.
        cfg = make_cfg(["bleu"])
        with self.assertRaises(BenchConfigError) as ctx:
            metrics.check_requirements(
                cfg.metrics,
                dataset=FakeDataset(translation_langs=["de"], languages=["en"]),
                languages=cfg.languages, realtime=True)
        self.assertIn("missing", str(ctx.exception))

    def test_only_the_directions_the_data_triggers_are_required(self):
        # en->ko pair on English-only audio: the ko->en leg never fires, so an
        # English reference must not be demanded.
        cfg = make_cfg(["laal"])
        metrics.check_requirements(
            cfg.metrics,
            dataset=FakeDataset(translation_langs=["ko"], languages=["en"]),
            languages=cfg.languages, realtime=True)

    def test_mixed_language_data_requires_every_triggered_direction(self):
        cfg = make_cfg(["laal"], languages={"map": {"en": "ko", "ko": "en", "fr": "ko"}})
        with self.assertRaises(BenchConfigError) as ctx:
            metrics.check_requirements(
                cfg.metrics,
                dataset=FakeDataset(translation_langs=["ko"],
                                    languages=["en", "ko", "fr"]),
                languages=cfg.languages, realtime=True)
        self.assertIn("'en'", str(ctx.exception))

    def test_wer_without_transcript_dies_early(self):
        cfg = make_cfg(["wer"])
        with self.assertRaises(BenchConfigError) as ctx:
            metrics.check_requirements(cfg.metrics,
                                       dataset=FakeDataset(has_transcript=False),
                                       languages=cfg.languages, realtime=True)
        self.assertIn("provides.transcript", str(ctx.exception))

    def test_commit_and_routing_need_nothing(self):
        cfg = make_cfg(["commit", "routing"])
        metrics.check_requirements(cfg.metrics, dataset=FakeDataset(has_transcript=False),
                                   languages=cfg.languages, realtime=False)


class TestAggregate(unittest.TestCase):
    def test_null_metric_always_has_a_reason(self):
        cfg = make_cfg(["wer", "fsl", "commit"])
        rows = [row(reference="", hypothesis="", segments=[seg()])]
        aggregate, diagnostics = metrics.run_metrics(rows, cfg)
        for key, value in aggregate.items():
            if value is not None:
                continue
            owner = metrics.KEY_OWNER.get(key)
            self.assertIsNotNone(owner, f"aggregate key {key!r} is not declared in SPECS")
            self.assertIn(owner, diagnostics["uncomputable"],
                          f"{key} is null with no diagnostics entry: {diagnostics}")

    def test_errored_rows_are_excluded_from_scoring(self):
        cfg = make_cfg(["wer"])
        rows = [row(reference="a b", hypothesis="a b"),
                row(id="bad", status="error", reference="x y z", hypothesis="")]
        aggregate, _ = metrics.run_metrics(rows, cfg)
        self.assertEqual(aggregate["wer"], 0.0)

    def test_empty_hypothesis_surfaces_in_the_aggregate(self):
        cfg = make_cfg(["wer"])
        rows = [row(reference="a b", hypothesis="a b"), row(reference="c d", hypothesis="")]
        aggregate, _ = metrics.run_metrics(rows, cfg)
        self.assertEqual(aggregate["n_empty_hypothesis"], 1)
        self.assertAlmostEqual(aggregate["wer"], 0.5)
        self.assertEqual(aggregate["wer_scored_only"], 0.0)


@unittest.skipUnless(translation.available(), "sacrebleu is not installed")
class TestBleu(unittest.TestCase):
    def test_identical_text_scores_100(self):
        out = translation.bleu_for_pairs([("the cat sat", "the cat sat")], target_lang="en")
        self.assertAlmostEqual(out["bleu"], 100.0, places=4)

    def test_misrouted_segments_are_excluded(self):
        cfg = make_cfg(["bleu"], languages={"map": {"en": "ko", "ko": "en"}})
        rows = [
            row(id="good", src_lang="en", reference_translations={"ko": "안녕하세요"},
                segments=[seg(language="en", target_lang="ko", translation="안녕하세요")]),
            row(id="bad", src_lang="ko", reference_translations={"ko": "안녕하세요"},
                segments=[seg(language="ko", target_lang="ko", translation="전혀 다른 말")]),
        ]
        aggregate, _ = metrics.run_metrics(rows, cfg)
        self.assertIsNotNone(aggregate["bleu"])
        self.assertGreater(aggregate["bleu"], aggregate["bleu_all"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
