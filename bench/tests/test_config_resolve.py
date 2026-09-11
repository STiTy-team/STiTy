#!/usr/bin/env python3
"""Config resolution. No GPU, no torch, no model.

The point of these tests is the commit-policy landmine: the production server
derives enable_dot_commit from the weights path (_infer_dot_commit_default), so
the same command yields a different policy per checkpoint. Bench must resolve it
from the config name alone, and test_commit_is_independent_of_model_path is what
holds that line.
"""
import ast
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench import config
from bench.errors import BenchConfigError

SERVER = (Path(__file__).resolve().parents[2] / "Qwen3-ASR" / "examples"
          / "streaming_websocket_server.py")

BASE = {
    "name": "t",
    "dataset": {"name": "librispeech"},
    "languages": {"lang": "en", "target": "ko"},
    "stity": {
        "transcription": {"name": "qwen3", "model": "Qwen/Qwen3-ASR-1.7B"},
        "translation": "none",
        "commit": "seg",
        "gpu_memory_utilization": 0.5,
    },
    "metrics": ["wer"],
}


# Only 'stity' is merged, so a patch can set one field and inherit the rest.
# Every other key is replaced outright -- merging 'languages' would silently
# combine BASE's lang/target pair with a patched map and trip the guard that
# forbids exactly that combination.
_MERGED_KEYS = {"stity"}


def cfg(**patch):
    raw = copy.deepcopy(BASE)
    for key, value in patch.items():
        if key in _MERGED_KEYS and isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key] = {**raw[key], **value}
        else:
            raw[key] = value
    return config.parse(raw)


def streaming_config_fields() -> set[str]:
    """StreamingConfig's annotated field names, read from source.

    Parsed rather than imported: importing the server pulls in torch and vLLM,
    which would make this test slow and GPU-adjacent for no benefit.
    """
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "StreamingConfig":
            return {s.target.id for s in node.body if isinstance(s, ast.AnnAssign)}
    raise AssertionError("StreamingConfig not found in the server source")


class TestCommitResolution(unittest.TestCase):
    def test_table(self):
        expected = {
            "seg": (False, False, False, False),
            "punct": (False, True, True, True),
            "static": (True, False, False, True),
        }
        for mode, want in expected.items():
            c = cfg(stity={"commit": mode}).stity.commit
            got = (c.always_commit, c.enable_dot_commit, c.dot_commit_confirm, c.hide_seg)
            self.assertEqual(got, want, f"commit {mode}")

    def test_commit_is_independent_of_model_path(self):
        baseline = "Qwen/Qwen3-ASR-1.7B"
        finetuned = "models/Qwen3-ASR-1.7B-en-dailytalk-seg"
        for mode in ("seg", "punct", "static"):
            a = cfg(stity={"commit": mode,
                           "transcription": {"name": "qwen3", "model": baseline}})
            b = cfg(stity={"commit": mode,
                           "transcription": {"name": "qwen3", "model": finetuned}})
            self.assertEqual(a.stity.commit, b.stity.commit, f"commit drifted for {mode}")
            for key in ("enable_dot_commit", "always_commit", "dot_commit_confirm"):
                self.assertEqual(a.streaming_config_kwargs()[key],
                                 b.streaming_config_kwargs()[key], key)

    def test_unknown_mode_lists_available(self):
        with self.assertRaises(BenchConfigError) as ctx:
            cfg(stity={"commit": "dot"})
        self.assertIn("punct", str(ctx.exception))
        self.assertIn("seg", str(ctx.exception))


class TestLanguages(unittest.TestCase):
    def test_map_and_pair_together_is_rejected(self):
        with self.assertRaises(BenchConfigError) as ctx:
            cfg(languages={"map": {"en": "ko"}, "lang": "en", "target": "ko"})
        self.assertIn("mutually exclusive", str(ctx.exception))

    def test_map_routes_three_languages(self):
        c = cfg(languages={"map": {"en": "ko", "ko": "en", "fr": "ko"}})
        self.assertEqual(c.languages.expected_target("fr"), "ko")
        self.assertEqual(c.languages.expected_target("ko"), "en")
        self.assertEqual(c.languages.expected_target("en"), "ko")
        self.assertEqual(c.languages.source_langs, ["en", "fr", "ko"])

    def test_pair_rule_cannot_route_a_third_language(self):
        c = cfg(languages={"lang": "en", "target": "ko"})
        self.assertEqual(c.languages.expected_target("fr"), "en")
        self.assertEqual(c.languages.covers(["en", "ko", "fr"]), ["fr"])

    def test_identity_entry_is_rejected(self):
        with self.assertRaises(BenchConfigError) as ctx:
            cfg(languages={"map": {"en": "en"}})
        self.assertIn("identity", str(ctx.exception))

    def test_language_names_are_accepted(self):
        c = cfg(languages={"lang": "English", "target": "Korean"})
        self.assertEqual((c.languages.lang, c.languages.target), ("en", "ko"))

    def test_unknown_language_dies(self):
        with self.assertRaises(BenchConfigError):
            cfg(languages={"lang": "en", "target": "klingon"})


class TestGuards(unittest.TestCase):
    def test_gpu_memory_utilization_has_no_default(self):
        raw = copy.deepcopy(BASE)
        del raw["stity"]["gpu_memory_utilization"]
        with self.assertRaises(BenchConfigError) as ctx:
            config.parse(raw)
        self.assertIn("gpu_memory_utilization", str(ctx.exception))

    def test_local_translation_caps_gpu(self):
        with self.assertRaises(BenchConfigError) as ctx:
            cfg(stity={"translation": "local", "gpu_memory_utilization": 0.8})
        self.assertIn("7.2", str(ctx.exception))
        self.assertEqual(
            cfg(stity={"translation": "local", "gpu_memory_utilization": 0.5})
            .stity.translation.name, "local")

    def test_latency_metrics_require_realtime_pacing(self):
        with self.assertRaises(BenchConfigError) as ctx:
            cfg(metrics=["wer", "laal"], pacing={"send_interval_ms": 0})
        self.assertIn("real-time", str(ctx.exception))

    def test_unknown_metric_lists_available(self):
        with self.assertRaises(BenchConfigError) as ctx:
            cfg(metrics=["wer", "mer"])
        self.assertIn("routing", str(ctx.exception))

    def test_unknown_key_is_rejected(self):
        with self.assertRaises(BenchConfigError) as ctx:
            cfg(stity={"chunk_size": 2.0})
        self.assertIn("chunk_size", str(ctx.exception))

    def test_rank_by_must_be_a_requested_metric(self):
        with self.assertRaises(BenchConfigError):
            cfg(metrics=["wer"], logs={"rank_by": "bleu"})


class TestStreamingConfigContract(unittest.TestCase):
    def test_kwargs_match_the_dataclass_exactly(self):
        fields = streaming_config_fields()
        kwargs = set(cfg().streaming_config_kwargs())
        self.assertEqual(kwargs - fields, set(), "bench passes fields StreamingConfig lacks")
        self.assertEqual(fields - kwargs, set(), "bench leaves StreamingConfig fields to default")

    def test_translation_backends_map_to_flags(self):
        none_kw = cfg(stity={"translation": "none"}).streaming_config_kwargs()
        self.assertTrue(none_kw["no_translation"])
        self.assertFalse(none_kw["enable_gpt_translation"])
        gpt_kw = cfg(stity={"translation": {"name": "gpt", "model": "gpt-5.4-mini"}})\
            .streaming_config_kwargs()
        self.assertTrue(gpt_kw["enable_gpt_translation"])
        self.assertFalse(gpt_kw["no_translation"])
        self.assertEqual(gpt_kw["translation_model"], "gpt-5.4-mini")

    def test_vad_disabled_sets_no_vad(self):
        self.assertTrue(cfg(stity={"vad": {"enabled": False}})
                        .streaming_config_kwargs()["no_vad"])
        self.assertFalse(cfg().streaming_config_kwargs()["no_vad"])


class TestPerModelOptions(unittest.TestCase):
    """Per-model settings belong to the model and must reach StreamingConfig.

    The failure being guarded against is the silent one: an option that is
    accepted, dropped, and never applied still yields a plausible number.
    """

    def test_transcription_options_reach_streaming_config(self):
        c = cfg(stity={"transcription": {"name": "qwen3", "model": "baseline",
                                         "chunk_size_sec": 1.0, "max_new_tokens": 256,
                                         "beam_size": 2, "enforce_eager": True}})
        kw = c.streaming_config_kwargs()
        self.assertEqual(kw["chunk_size_sec"], 1.0)
        self.assertEqual(kw["max_new_tokens"], 256)
        self.assertEqual(kw["beam_size"], 2)
        self.assertTrue(kw["enforce_eager"])

    def test_translation_options_reach_streaming_config(self):
        c = cfg(stity={"translation": {"name": "gpt", "model": "gpt-5.4-mini",
                                       "context_window": 7}})
        kw = c.streaming_config_kwargs()
        self.assertTrue(kw["enable_gpt_translation"])
        self.assertEqual(kw["translation_model"], "gpt-5.4-mini")
        self.assertEqual(kw["context_window"], 7)

    def test_unknown_transcription_option_is_rejected(self):
        with self.assertRaises(BenchConfigError) as ctx:
            cfg(stity={"transcription": {"name": "qwen3", "chunk_size": 1.0}})
        self.assertIn("chunk_size", str(ctx.exception))
        self.assertIn("chunk_size_sec", str(ctx.exception))

    def test_unknown_translation_option_is_rejected(self):
        with self.assertRaises(BenchConfigError) as ctx:
            cfg(stity={"translation": {"name": "gpt", "ctx": 7}})
        self.assertIn("ctx", str(ctx.exception))

    def test_options_do_not_leak_between_components(self):
        with self.assertRaises(BenchConfigError):
            # context_window belongs to gpt, not to the ASR model
            cfg(stity={"transcription": {"name": "qwen3", "context_window": 5}})

    def test_defaults_apply_when_the_model_says_nothing(self):
        kw = cfg().streaming_config_kwargs()
        self.assertEqual(kw["chunk_size_sec"], 2.0)
        self.assertEqual(kw["max_new_tokens"], 128)
        self.assertEqual(kw["beam_size"], 1)

    def test_model_may_override_gpu_memory(self):
        c = cfg(stity={"transcription": {"name": "qwen3",
                                         "gpu_memory_utilization": 0.3}})
        self.assertEqual(c.streaming_config_kwargs()["gpu_memory_utilization"], 0.3)

    def test_per_model_option_changes_the_fingerprint(self):
        a = cfg(stity={"transcription": {"name": "qwen3", "chunk_size_sec": 1.0}})
        b = cfg(stity={"transcription": {"name": "qwen3", "chunk_size_sec": 2.0}})
        self.assertNotEqual(a.fingerprint(), b.fingerprint())


class TestModelAliases(unittest.TestCase):
    def test_finetuned_resolves_per_language(self):
        en = cfg(languages={"lang": "en", "target": "ko"},
                 stity={"transcription": {"name": "qwen3", "model": "finetuned"}})
        ko = cfg(languages={"lang": "ko", "target": "en"},
                 stity={"transcription": {"name": "qwen3", "model": "finetuned"}})
        self.assertIn("-en-", en.streaming_config_kwargs()["model_path"])
        self.assertIn("-ko-", ko.streaming_config_kwargs()["model_path"])

    def test_alias_without_weights_for_the_language_raises(self):
        with self.assertRaises(BenchConfigError) as ctx:
            cfg(languages={"lang": "fr", "target": "ko"},
                stity={"transcription": {"name": "qwen3", "model": "finetuned"}})
        self.assertIn("no weights for language", str(ctx.exception))

    def test_unknown_alias_does_not_fall_back_to_baseline(self):
        with self.assertRaises(BenchConfigError):
            cfg(stity={"transcription": {"name": "qwen3", "model": "finetund"}})


class TestFingerprint(unittest.TestCase):
    def test_commit_mode_changes_it(self):
        self.assertNotEqual(cfg(stity={"commit": "seg"}).fingerprint(),
                            cfg(stity={"commit": "punct"}).fingerprint())

    def test_language_mode_changes_it(self):
        self.assertNotEqual(cfg(languages={"lang": "en", "target": "ko"}).fingerprint(),
                            cfg(languages={"map": {"en": "ko"}}).fingerprint())

    def test_same_config_is_stable(self):
        self.assertEqual(cfg().fingerprint(), cfg().fingerprint())


class TestShippedConfigs(unittest.TestCase):
    def test_they_load(self):
        for path in sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yml")):
            with self.subTest(path.name):
                c = config.load(path)
                self.assertTrue(c.name)
                self.assertTrue(c.fingerprint())


if __name__ == "__main__":
    unittest.main(verbosity=2)
