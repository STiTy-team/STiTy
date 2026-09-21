import unittest
from unittest.mock import patch

from core.utils import metrics
from core.utils.metrics import (asr_robustness, context, critical_information,
                                fluency, intent, meaning)


class _Languages:
    def expected_target(self, _src_lang):
        return "en"


def _span(kind, value):
    return {"type": kind, "canonical_value": value, "text": str(value)}


class TranslationMetricFunctionsTest(unittest.TestCase):

    def test_xcomet_official_adapter_keeps_scores_and_spans(self):
        class _Metadata:
            error_spans = [[{"start": 0, "end": 3, "severity": "minor"}]]

        class _Output:
            scores = [.75]
            system_score = .75
            metadata = _Metadata()

        class _Model:
            def predict(self, rows, **kwargs):
                self.rows = rows
                self.kwargs = kwargs
                return _Output()

        model = _Model()
        with patch.object(meaning, "_xcomet_model", return_value=model):
            result = meaning.xcomet_score(
                [{"id": "a", "src": "x", "mt": "y", "ref": "z"}], gpus=0)
        self.assertEqual(result["score"], .75)
        self.assertEqual(result["per_item"], {"a": .75})
        self.assertEqual(result["error_spans"]["a"][0]["severity"], "minor")
        self.assertEqual(model.kwargs["gpus"], 0)

    def test_existing_metric_functions(self):
        self.assertEqual(
            critical_information.normalized_value_accuracy(
                [_span("time", "15:00")], [_span("time", "15:00")])["accuracy"],
            1.0,
        )
        self.assertEqual(
            critical_information.critical_span_f1(
                [_span("person", "kim")], [_span("person", "kim")])["f1"],
            1.0,
        )
        self.assertEqual(
            critical_information.critical_fact_error_rate([
                {"id": "a", "reference_spans": [_span("time", "15:00")],
                 "candidate_spans": []}
            ])["error_rate"],
            1.0,
        )
        self.assertEqual(fluency.spoken_fluency_judge_score([{"score": 4}])["score"], 4)
        self.assertEqual(fluency.mqm_fluency_error_rate([
            {"target_token_count": 5, "errors": [{"severity": "minor"}]}
        ])["error_rate"], 0.2)
        self.assertEqual(context.context_mqm_score([{"score": 80}])["score"], 80)
        self.assertEqual(context.dialogue_inconsistency_rate([
            {"inconsistencies": []}
        ])["error_rate"], 0)
        self.assertEqual(context.context_contrastive_accuracy([
            {"selected_correct": True}
        ])["accuracy"], 1)
        self.assertAlmostEqual(asr_robustness.asr_induced_quality_drop([
            {"clean_quality": {"xcomet": .9}, "noisy_quality": {"xcomet": .7}}
        ])["xcomet"]["absolute_drop"], .2)
        self.assertEqual(asr_robustness.translation_invariance_score([
            {"similarity": .8}
        ])["score"], .8)
        self.assertGreater(asr_robustness.quality_noise_degradation([
            {"noise_level": 0, "quality": {"xcomet": 1}},
            {"noise_level": 1, "quality": {"xcomet": 0}},
        ])["xcomet"]["degradation_slope"], 0)
        self.assertEqual(intent.polarity_preservation_accuracy([
            {"reference": "negative", "candidate": "positive"}
        ])["polarity_reversal_rate"], 1)
        self.assertEqual(intent.speech_act_macro_f1([
            {"reference": ["question"], "candidate": ["question"]}
        ])["macro_f1"], 1)
        self.assertEqual(intent.modality_stance_preservation_accuracy([
            {"reference": "certain", "candidate": "possible"}
        ])["mean_ordinal_distance"], 2)

    def test_pseudo_perplexity_is_not_averaged_across_target_languages(self):
        items = [metrics.Utterance.from_row({
            "id": "en", "metric_inputs": {"fluency": {
                "target_lm_pseudo_perplexity": 2.0, "target_lang": "en",
                "target_lm_model": "en-lm"}},
        }), metrics.Utterance.from_row({
            "id": "ko", "metric_inputs": {"fluency": {
                "target_lm_pseudo_perplexity": 8.0, "target_lang": "ko",
                "target_lm_model": "ko-lm"}},
        })]
        values, _ = fluency.corpus(items)
        result = values["fluency"]["target_lm_pseudo_perplexity"]
        self.assertNotIn("pseudo_perplexity", result)
        self.assertEqual(result["by_target"]["en"]["pseudo_perplexity"], 2.0)
        self.assertEqual(result["by_target"]["ko"]["pseudo_perplexity"], 8.0)

    def test_all_six_axes_are_wired_into_score_run(self):
        common = {
            "status": "ok", "src_lang": "ko", "reference": "오후 세 시야",
            "hypothesis": "오후 세 시야", "hypothesis_translation": "It is 3 PM.",
            "reference_translations": {"en": "It is 3 PM."},
            "segments": [{"original": "오후 세 시야", "translation": "It is 3 PM.",
                          "language": "ko", "target_lang": "en", "commit_reason": "vad"}],
        }
        rows = []
        for index, quality in enumerate((.9, .7)):
            rows.append({"id": str(index), **common, "metric_inputs": {
                "meaning": {"xcomet": quality, "metricx_24": 1 - quality},
                "critical_information": {
                    "reference_spans": [_span("time", "15:00")],
                    "candidate_spans": [_span("time", "15:00")],
                },
                "fluency": {"judge": {"score": 4}, "mqm_errors": [],
                            "target_token_count": 4,
                            "target_lm_pseudo_perplexity": 2.0},
                "context": {"mqm": {"score": 4}, "inconsistencies": [],
                            "contrastive": {"selected_correct": True}},
                "asr_robustness": {
                    "clean_quality": {"xcomet": .9}, "noisy_quality": {"xcomet": quality},
                    "similarity": quality, "noise_level": index * .2,
                    "quality_by_noise": [
                        {"noise_level": 0, "quality": {"xcomet": .9}},
                        {"noise_level": .2, "quality": {"xcomet": .7}},
                    ],
                },
                "intent": {
                    "polarity": {"reference": "positive", "candidate": "positive"},
                    "speech_act": {"reference": ["statement"],
                                   "candidate": ["statement"]},
                    "modality": {"reference": "certain", "candidate": "certain"},
                },
            }})

        fake_chrf = {"score": 100.0, "signature": "test", "n_scored": 2,
                     "per_item_scores": [100.0, 100.0]}
        with patch.object(meaning, "chrfpp_score", return_value=fake_chrf):
            score = metrics.score_run(rows, languages=_Languages())
        expected = {
            "meaning": {"xcomet", "metricx_24", "chrfpp"},
            "critical_information": {"normalized_value_accuracy", "critical_span_f1",
                                     "critical_fact_error_rate"},
            "fluency": {"spoken_fluency_judge", "mqm_fluency_error_rate",
                        "target_lm_pseudo_perplexity"},
            "context": {"context_mqm", "dialogue_inconsistency_rate",
                        "context_contrastive_accuracy"},
            "asr_robustness": {"quality_drop", "translation_invariance",
                               "quality_noise_degradation"},
            "intent": {"polarity_preservation", "speech_act_macro_f1",
                       "modality_stance_preservation"},
        }
        for axis, names in expected.items():
            self.assertEqual(set(score.aggregate[axis]), names)


if __name__ == "__main__":
    unittest.main()
