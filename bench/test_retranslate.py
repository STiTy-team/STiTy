import asyncio
import unittest

from bench.retranslate import retranslate_rows, score_translation_rows


class _Languages:
    def expected_target(self, _source):
        return "de"


class _Translator:
    def __init__(self):
        self.calls = []
        self.started = []

    def start(self, **kwargs):
        self.started.append(kwargs)

    async def translate(self, text, target_lang, source_lang=None, context=None):
        self.calls.append((text, target_lang, source_lang, list(context or [])))
        return f"DE:{text}", source_lang or "en"


def _row(item_id, originals):
    return {
        "id": item_id,
        "status": "ok",
        "group": "dialogue-1",
        "src_lang": "en",
        "reference": " ".join(originals),
        "hypothesis": " ".join(originals),
        "hypothesis_translation": "OLD",
        "reference_translations": {"de": "REF"},
        "segments": [
            {"original": text, "translation": "OLD", "language": "en",
             "target_lang": "de", "commit_reason": "vad",
             "decision_audio_sec": index + 1, "recv_elapsed_sec": index + 1.2}
            for index, text in enumerate(originals)
        ],
        "metric_inputs": {
            "meaning": {"xcomet": .99},
            "critical_information": {
                "reference_spans": [{"type": "time", "canonical_value": "15:00"}],
                "candidate_spans": [{"type": "time", "canonical_value": "14:00"}],
            },
            "fluency": {"judge": {"score": 5}},
            "intent": {
                "polarity": {"reference": "positive", "candidate": "negative"}
            },
        },
        "wer": 0.1,
        "laal_ms": 123,
    }


class RetranslateTest(unittest.TestCase):

    def test_item_scope_replays_segments_and_resets_context(self):
        translator = _Translator()
        rows = asyncio.run(retranslate_rows(
            [_row("a", ["one", "two"]), _row("b", ["three"])],
            translator=translator, languages=_Languages(), context_scope="item"))

        self.assertEqual([call[3] for call in translator.calls], [[], ["one"], []])
        self.assertEqual(rows[0]["hypothesis_translation"], "DE:one DE:two")
        self.assertNotIn("wer", rows[0])
        self.assertNotIn("laal_ms", rows[0])
        self.assertNotIn("decision_audio_sec", rows[0]["segments"][0])
        self.assertEqual(rows[0]["segments"][0]["source_timing"]["decision_audio_sec"], 1)

    def test_group_scope_carries_context_and_invalidates_old_candidate_annotations(self):
        translator = _Translator()
        rows = asyncio.run(retranslate_rows(
            [_row("a", ["one"]), _row("b", ["two"])],
            translator=translator, languages=_Languages(), context_scope="group"))

        self.assertEqual([call[3] for call in translator.calls], [[], ["one"]])
        inputs = rows[0]["metric_inputs"]
        self.assertNotIn("meaning", inputs)
        self.assertNotIn("fluency", inputs)
        self.assertNotIn("candidate_spans", inputs["critical_information"])
        self.assertEqual(inputs["intent"]["polarity"], {"reference": "positive"})

    def test_translation_score_excludes_inherited_asr_metrics(self):
        translator = _Translator()
        rows = asyncio.run(retranslate_rows(
            [_row("a", ["one"])], translator=translator,
            languages=_Languages(), context_scope="item"))
        score = score_translation_rows(rows, languages=_Languages())
        self.assertFalse({"wer", "cer", "laal_ms", "commit_stats"} & set(score.aggregate))
        self.assertNotIn("critical_information", score.aggregate)
        self.assertEqual(
            score.unavailable["critical_information.critical_span_f1"],
            "missing metric_inputs.critical_information.candidate_spans")
        self.assertTrue(all(key.startswith("bleu") or "." in key
                            for key in score.unavailable))


if __name__ == "__main__":
    unittest.main()
