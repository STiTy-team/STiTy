import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from core.utils import metrics
from core.utils.metrics import (asr_robustness, context, critical_information,
                                fluency, intent, meaning)


class _Languages:
    def expected_target(self, _src_lang):
        return "en"


@contextmanager
def _modules(**fakes):
    names = {name.replace("__", "."): module for name, module in fakes.items()}
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules.update(names)
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _span(kind, value):
    return {"type": kind, "canonical_value": value, "text": str(value)}


class TranslationMetricFunctionsTest(unittest.TestCase):

    def test_comet_adapter_keeps_scores_and_optional_spans(self):
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
        with patch.object(meaning, "_comet_model", return_value=model):
            result = meaning.comet_score(
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
            {"clean_quality": {"comet": .9}, "noisy_quality": {"comet": .7}}
        ])["comet"]["absolute_drop"], .2)
        self.assertEqual(asr_robustness.translation_invariance_score([
            {"similarity": .8}
        ])["score"], .8)
        self.assertGreater(asr_robustness.quality_noise_degradation([
            {"noise_level": 0, "quality": {"comet": 1}},
            {"noise_level": 1, "quality": {"comet": 0}},
        ])["comet"]["degradation_slope"], 0)
        self.assertEqual(intent.polarity_preservation_accuracy([
            {"reference": "negative", "candidate": "positive"}
        ])["polarity_reversal_rate"], 1)
        self.assertEqual(intent.speech_act_macro_f1([
            {"reference": ["question"], "candidate": ["question"]}
        ])["macro_f1"], 1)
        self.assertEqual(intent.modality_stance_preservation_accuracy([
            {"reference": "certain", "candidate": "possible"}
        ])["mean_ordinal_distance"], 2)

    def test_relative_quality_drop_keeps_the_sign_of_the_absolute_drop(self):
        result = asr_robustness.asr_induced_quality_drop([
            {"id": "echo", "clean_quality": {"chrfpp": 5}, "noisy_quality": {"chrfpp": 15}},
            {"id": "b", "clean_quality": {"chrfpp": 60}, "noisy_quality": {"chrfpp": 40}},
            {"id": "c", "clean_quality": {"chrfpp": 60}, "noisy_quality": {"chrfpp": 45}},
        ])["chrfpp"]
        self.assertAlmostEqual(result["absolute_drop"], 25 / 3)
        self.assertAlmostEqual(result["relative_drop"], 25 / 125)
        self.assertAlmostEqual(result["per_item"]["echo"]["relative_drop"], -2.0)

    def test_metricx_runtime_disables_the_decoder_cache(self):
        class _Config:
            use_cache = True

        class _Model:
            config = _Config()

            def to(self, _device):
                return self

            def eval(self):
                return self

        class _MT5ForRegression:
            @classmethod
            def from_pretrained(cls, _name, **_kwargs):
                return _Model()

        package = types.ModuleType("metricx24")
        models = types.ModuleType("metricx24.models")
        models.MT5ForRegression = _MT5ForRegression
        with _modules(metricx24=package, metricx24__models=models), \
                patch("transformers.AutoTokenizer.from_pretrained", return_value=object()):
            _tokenizer, model = meaning._metricx_runtime("model", "tokenizer", "cpu")
        self.assertFalse(model.config.use_cache)

    def test_comet_runs_the_ungated_da_checkpoint_named_by_its_env_var(self):
        seen = []

        def scorer(samples, *, model_name):
            seen.append(model_name)
            return {"score": .8, "model": model_name, "n_scored": len(samples)}

        item = metrics.Utterance.from_row({
            "id": "a", "src_lang": "ko", "hypothesis": "x", "hypothesis_translation": "y",
            "reference_translations": {"en": "z"}})
        self.assertEqual(meaning.DEFAULT_COMET_MODEL, "Unbabel/wmt22-comet-da")
        with patch.object(meaning, "comet_score", scorer), \
                patch.dict(os.environ, {"STITY_COMET_MODEL": meaning.DEFAULT_COMET_MODEL}):
            axis, _unavailable = meaning.corpus([item], languages=_Languages())
        self.assertEqual(seen, ["Unbabel/wmt22-comet-da"])
        self.assertEqual(axis["meaning"]["comet"]["score"], .8)

    def test_comet_download_failure_reports_the_hub_error(self):
        class GatedRepoError(Exception):
            pass

        def download_model(name):
            raise KeyError(f"Model '{name}' not supported by COMET.")

        def hf_hub_download(_repo, _filename):
            raise GatedRepoError(
                "403 Client Error.\n\nCannot access gated repo.\n"
                "Access to model Unbabel/wmt22-cometkiwi-da is restricted.")

        comet = types.ModuleType("comet")
        comet.download_model = download_model
        comet.load_from_checkpoint = lambda path: path
        hub = types.ModuleType("huggingface_hub")
        hub.hf_hub_download = hf_hub_download
        item = metrics.Utterance.from_row({
            "id": "a", "src_lang": "ko", "hypothesis": "x", "hypothesis_translation": "y",
            "reference_translations": {"en": "z"}})
        with _modules(comet=comet, huggingface_hub=hub), \
                patch.dict(os.environ, {"STITY_COMET_MODEL": "Unbabel/wmt22-cometkiwi-da"}):
            _axis, unavailable = meaning.corpus([item], languages=_Languages())
        reason = unavailable["meaning.comet"]
        self.assertIn("GatedRepoError", reason)
        self.assertIn("Access to model Unbabel/wmt22-cometkiwi-da is restricted.", reason)
        self.assertNotIn("not supported by COMET", reason)

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
                "meaning": {"comet": quality, "metricx_24": 1 - quality},
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
                    "clean_quality": {"comet": .9}, "noisy_quality": {"comet": quality},
                    "similarity": quality, "noise_level": index * .2,
                    "quality_by_noise": [
                        {"noise_level": 0, "quality": {"comet": .9}},
                        {"noise_level": .2, "quality": {"comet": .7}},
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
            "meaning": {"comet", "metricx_24", "chrfpp"},
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
