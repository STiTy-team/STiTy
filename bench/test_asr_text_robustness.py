import asyncio
import unittest
from unittest.mock import patch

from bench.asr_text_robustness import evaluate_robustness_rows, score_robustness
from core.utils.metrics.text import levenshtein, normalize_words


class _Languages:
    def expected_target(self, source):
        return "en" if source == "ko" else "ko"


class _Translator:
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def start(self, **kwargs):
        pass

    async def translate(self, text, target_lang, source_lang=None, context=None):
        self.calls.append(text)
        if text in self.fail_on:
            raise RuntimeError("backend down")
        return f"{target_lang}:{text}", source_lang


GOLD = "오늘 오후 세 시에 만나요."


def _row(hypothesis=GOLD, reference=GOLD):
    return {
        "id": "a",
        "status": "ok",
        "src_lang": "ko",
        "reference": reference,
        "hypothesis": hypothesis,
        "hypothesis_translation": "old",
        "reference_translations": {"en": "Let's meet at 3 PM today."},
        "metric_inputs": {},
    }


def _similarities(pairs):
    return {row["id"]: .8 for row in pairs}


def _run(rows, translator, levels=(.2, .4)):
    with patch("core.utils.metrics.meaning.chrfpp_sentence",
               side_effect=lambda candidate, reference: 100.0 if GOLD in candidate else 60.0):
        return asyncio.run(evaluate_robustness_rows(
            rows, translator=translator, languages=_Languages(),
            noise_levels=list(levels), seed=2, similarity_scorer=_similarities))


def _wer(reference, hypothesis):
    words = normalize_words(reference)
    return levenshtein(words, normalize_words(hypothesis)) / len(words)


class AsrTextRobustnessTest(unittest.TestCase):

    def test_clean_is_the_gold_transcript_and_noisy_is_the_real_asr_output(self):
        translator = _Translator()
        rows = _run([_row(hypothesis="오늘 오후 세시 만나요")], translator)
        block = rows[0]["metric_inputs"]["asr_robustness"]
        self.assertEqual(translator.calls[0], GOLD)
        self.assertEqual(block["clean_source"], "reference_transcript")
        self.assertEqual(block["noisy_source"], "asr_hypothesis")
        self.assertEqual(block["noisy_transcript"], "오늘 오후 세시 만나요")
        self.assertAlmostEqual(block["noise_level"], _wer(GOLD, "오늘 오후 세시 만나요"))
        self.assertEqual(block["noisy_quality"], {"chrfpp": 60.0})
        self.assertEqual(block["similarity"], .8)
        score = score_robustness(rows).aggregate["asr_robustness"]
        self.assertAlmostEqual(score["quality_drop"]["chrfpp"]["absolute_drop"], 40.0)

    def test_synthetic_curve_is_placed_at_the_measured_word_error_rate(self):
        rows = _run([_row()], _Translator())
        block = rows[0]["metric_inputs"]["asr_robustness"]
        self.assertNotIn("noisy_quality", block)  # no ASR output distinct from the gold
        clean, *noisy = block["quality_by_noise"]
        self.assertEqual(clean["noise_level"], 0.0)
        for point, variant in zip(noisy, block["variants"]):
            self.assertEqual(point["noise_level"], variant["wer_from_clean"])
            self.assertEqual(point["quality"]["similarity"], .8)
        score = score_robustness(rows)
        self.assertIn("quality_noise_degradation", score.aggregate["asr_robustness"])
        self.assertIn("asr_robustness.quality_drop", score.unavailable)

    def test_empty_asr_output_is_a_catastrophic_failure_not_a_skipped_row(self):
        translator = _Translator()
        rows = _run([_row(hypothesis="")], translator)
        block = rows[0]["metric_inputs"]["asr_robustness"]
        self.assertTrue(block["catastrophic"])
        self.assertEqual(block["similarity"], 0.0)
        self.assertNotIn("", translator.calls)
        score = score_robustness(rows).aggregate["asr_robustness"]
        self.assertEqual(score["catastrophic_failures"]["count"], 1)
        self.assertEqual(score["translation_invariance"]["n_pairs"], 1)

    def test_translator_failure_is_left_out_of_both_drop_and_invariance(self):
        rows = _run([_row(hypothesis="오늘 세시")], _Translator(fail_on={"오늘 세시"}))
        block = rows[0]["metric_inputs"]["asr_robustness"]
        self.assertNotIn("noisy_quality", block)
        self.assertNotIn("similarity", block)
        self.assertEqual(rows[0]["robustness_translation_status"], "degraded")


if __name__ == "__main__":
    unittest.main()
