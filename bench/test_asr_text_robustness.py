import asyncio
import unittest
from unittest.mock import patch

from bench.asr_text_robustness import evaluate_robustness_rows, score_robustness
from bench.text_noise import perturb_transcript


class _Languages:
    def expected_target(self, source):
        return "en" if source == "ko" else "ko"


class _Translator:
    def __init__(self):
        self.calls = []

    def start(self, **kwargs):
        pass

    async def translate(self, text, target_lang, source_lang=None, context=None):
        self.calls.append(text)
        return f"{target_lang}:{text}", source_lang


def _row():
    return {
        "id": "a",
        "status": "ok",
        "src_lang": "ko",
        "hypothesis": "오늘 오후 세 시에 만나요.",
        "hypothesis_translation": "old",
        "reference_translations": {"en": "Let's meet at 3 PM today."},
        "metric_inputs": {},
    }


class AsrTextRobustnessTest(unittest.TestCase):

    def test_noise_is_reproducible_and_nonempty_protocol(self):
        first = perturb_transcript("오늘 오후 세 시에 만나요.", lang="ko", level=.3,
                                   seed=7, item_id="a")
        second = perturb_transcript("오늘 오후 세 시에 만나요.", lang="ko", level=.3,
                                    seed=7, item_id="a")
        self.assertEqual(first, second)
        self.assertNotEqual(first[0], "오늘 오후 세 시에 만나요.")
        self.assertTrue(first[1])

    def test_clean_and_noisy_are_translated_and_scored(self):
        translator = _Translator()

        def similarities(pairs):
            return {row["id"]: .8 for row in pairs}

        with patch("bench.asr_text_robustness._chrf_quality",
                   side_effect=lambda candidate, reference: 100.0 if "오늘 오후" in candidate else 60.0):
            rows = asyncio.run(evaluate_robustness_rows(
                [_row()], translator=translator, languages=_Languages(),
                noise_levels=[.2, .4], seed=2, similarity_scorer=similarities))
        block = rows[0]["metric_inputs"]["asr_robustness"]
        self.assertEqual(len(translator.calls), 3)
        self.assertEqual(len(block["variants"]), 2)
        self.assertEqual(block["similarity"], .8)
        self.assertIn("chrfpp", block["clean_quality"])
        score = score_robustness(rows)
        self.assertIn("quality_drop", score.aggregate["asr_robustness"])
        self.assertIn("translation_invariance", score.aggregate["asr_robustness"])
        self.assertIn("quality_noise_degradation", score.aggregate["asr_robustness"])


if __name__ == "__main__":
    unittest.main()

