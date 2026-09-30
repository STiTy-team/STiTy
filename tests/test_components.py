"""The new components on CPU: script labels, commit filter, enhancer, VAD tuners, qwen-la,
qwen3.5:v2 prompts, cascade:v2 wiring, bench diagnostics."""
import asyncio
import unittest
from types import SimpleNamespace

import numpy as np

from core.components.enhancement.spectral import FRAME, SpectralGate
from core.components.filter.commit_rules import CommitRules
from core.components.labeler.script import ScriptLabeler
from core.components.registry import Partial, Speech, Transcribed, Translated
from core.components.transcription.qwen3_la import Qwen3LocalAgreement, common_prefix
from core.components.translation.local_qwen_3_5_4b.translator_v2 import Qwen35TranslationV2
from core.components.vad.tuners import Adaptive, Fixed, Ramp, parse_tuner
from core.utils import scripts
from core.utils.audio import from_pcm_bytes, to_pcm_bytes
from core.utils.glossary import Glossary

from tests.fake_asr import commits, fake_cfg, feed, make, timeline

EN = "language English<asr_text>"


def committed(text, language, at=1.0):
    return Transcribed(original=text, language=language, commit_reason="seg",
                       decision_audio_sec=at, committed_elapsed_sec=at)


def part(cls, **options):
    return cls(cls.validate(options, kind=cls.__module__.split(".")[2]), cfg=fake_cfg())


class Scripts(unittest.TestCase):
    def test_dominant_script(self):
        self.assertEqual(scripts.dominant_script("그건 블루투스 페어링 문제입니다."), "hangul")
        self.assertEqual(scripts.dominant_script("Another day, change."), "latin")
        self.assertEqual(scripts.dominant_script("これは日本語です"), "kana")
        self.assertIsNone(scripts.dominant_script("... 123"))

    def test_language_for_script_needs_one_candidate(self):
        self.assertEqual(scripts.language_for_script("latin", candidates=["ko", "en"]), "en")
        self.assertEqual(scripts.language_for_script("latin", candidates=["en", "es"],
                                                     fallback="x"), "x")
        self.assertEqual(scripts.language_for_script("hangul", candidates=["en"],
                                                     fallback="en"), "en")

    def test_foreign_share_ignores_latin_names_in_korean(self):
        self.assertEqual(scripts.foreign_share("김 부장님, Mr. Kim 오셨어요", "ko"), 0.0)
        self.assertGreater(scripts.foreign_share("우리这边数据不好", "ko"), 0.4)


class Labeler(unittest.TestCase):
    def test_relabels_by_script_among_conversation_languages(self):
        labeler = part(ScriptLabeler)
        labeler.start(languages=["ko", "en"])
        out = labeler.label([committed("Another day, change.", "ko"),
                             committed("다섯 명이요, 김태호예요.", "en"),
                             committed("Okay.", "en"),
                             Partial(text="And it's the Bluetooth", language="ko", seq=1)])
        self.assertEqual([r.language for r in out], ["en", "ko", "en", "en"])

    def test_empty_label_gets_filled(self):
        labeler = part(ScriptLabeler)
        labeler.start(languages=["ko", "en"])
        self.assertEqual(labeler.label([committed("Because I saw a video", "")])[0].language, "en")


class Filter(unittest.TestCase):
    def rules(self, **options):
        keeper = part(CommitRules, **options)
        keeper.start(languages=["ko", "en"])
        return keeper

    def test_fillers_only(self):
        keeper = self.rules(fillers=True)
        out = keeper.filter([committed("음.", "ko"), committed("어, 음", "ko"),
                             committed("아니, 아니.", "ko"), committed("Um, okay.", "en")])
        self.assertEqual([r.original for r in out], ["아니, 아니.", "Um, okay."])

    def test_extra_fillers(self):
        keeper = self.rules(fillers=True, extra_fillers={"ko": ["아니"]})
        self.assertEqual(keeper.filter([committed("아니, 아니.", "ko")]), [])

    def test_foreign_script(self):
        keeper = self.rules(foreign_script=True)
        out = keeper.filter([committed("啊。", "ko"), committed("세크 좋아요", "ko")])
        self.assertEqual([r.original for r in out], ["세크 좋아요"])

    def test_repeats_within_window(self):
        keeper = self.rules(repeats=2, repeat_window_sec=10.0)
        out = keeper.filter([committed("안녕하세요.", "ko", 1.0), committed("안녕하세요.", "ko", 2.0),
                             committed("안녕하세요.", "ko", 3.0), committed("안녕하세요.", "ko", 20.0)])
        self.assertEqual([r.decision_audio_sec for r in out], [1.0, 20.0])

    def test_canned_phrases(self):
        keeper = self.rules(canned_phrases=["I want to buy a new car."])
        out = keeper.filter([committed("I want to buy a new car.", "en"),
                             committed("I want to buy a new car soon.", "en")])
        self.assertEqual([r.original for r in out], ["I want to buy a new car soon."])

    def test_partials_pass_through(self):
        keeper = self.rules(fillers=True)
        partial = Partial(text="음", language="ko", seq=1)
        self.assertEqual(keeper.filter([partial]), [partial])


class Enhancer(unittest.TestCase):
    def test_streaming_keeps_length_and_reduces_noise(self):
        rng = np.random.default_rng(0)
        t = np.arange(16000 * 4) / 16000
        clean = 0.3 * np.sin(2 * np.pi * 220 * t) * (t > 1.5)
        noisy = (clean + 0.05 * rng.standard_normal(t.size)).astype(np.float32)
        gate = part(SpectralGate, vad_input="enhanced", asr_input="mix", mix_weight=0.2)
        gate.start()
        vad_out, asr_out = [], []
        for start in range(0, noisy.size, 3200):
            routed = gate.enhance(to_pcm_bytes(noisy[start:start + 3200]))
            vad_out.append(from_pcm_bytes(routed.vad))
            asr_out.append(from_pcm_bytes(routed.asr))
        enhanced = np.concatenate(vad_out)
        self.assertEqual(enhanced.size, noisy.size)
        self.assertEqual(np.concatenate(asr_out).size, noisy.size)
        quiet = slice(FRAME + 3200, 16000)
        self.assertLess(np.std(enhanced[quiet]), 0.5 * np.std(noisy[quiet]))

    def test_raw_route_is_the_input_delayed(self):
        gate = part(SpectralGate, vad_input="raw", asr_input="raw")
        gate.start()
        signal = np.linspace(-0.5, 0.5, 6400).astype(np.float32)
        out = np.concatenate([from_pcm_bytes(gate.enhance(to_pcm_bytes(signal[i:i + 3200])).asr)
                              for i in (0, 3200)])
        np.testing.assert_allclose(out[FRAME:], from_pcm_bytes(to_pcm_bytes(signal))[:-FRAME],
                                   atol=1e-4)


class Tuners(unittest.TestCase):
    def test_fixed_matches_the_silero_iterator_rule(self):
        tuner = Fixed({}, threshold=0.5, min_silence_ms=800)
        self.assertIsNone(tuner.should_end(799.9, 3.0, None))
        self.assertEqual(tuner.should_end(800.0, 3.0, None), 800)

    def test_ramp_shortens_with_segment_length(self):
        tuner = Ramp({"end_ms": 300, "ramp_sec": 8, "max_segment_sec": 15},
                     threshold=0.5, min_silence_ms=800)
        self.assertEqual(tuner.min_silence_ms(0.0), 800)
        self.assertEqual(tuner.min_silence_ms(4.0), 550)
        self.assertEqual(tuner.min_silence_ms(9.0), 300)
        self.assertEqual(tuner.min_silence_ms(16.0), 100)

    def test_adaptive_pauses_and_noise_floor(self):
        tuner = Adaptive({"pauses": True, "noise_floor": True, "noise_window_sec": 1.0},
                         threshold=0.5, min_silence_ms=800)
        tuner.start()
        for pause in (200, 250, 300, 220, 260, 240):
            tuner.on_pause(pause)
        self.assertLess(tuner.min_silence_ms(3.0), 400)
        for _ in range(40):
            tuner.observe(0.45, triggered=False)
        self.assertAlmostEqual(tuner.threshold(), 0.7)

    def test_parse_tuner_rejects_unknown_settings(self):
        with self.assertRaises(ValueError):
            parse_tuner({"name": "ramp", "speed": 3})
        self.assertEqual(parse_tuner("fixed"), {"name": "fixed"})


class LocalAgreement(unittest.TestCase):
    def test_common_prefix(self):
        self.assertEqual(common_prefix([["a", "b", "c"], ["a", "b", "d"]]), ["a", "b"])

    def test_commits_agreed_words_at_sentence_ends(self):
        script = timeline([(2.0, f"{EN}Hello there. How are"),
                           (4.0, f"{EN}Hello there. How are you doing"),
                           (6.0, f"{EN}Hello there. How are you doing today? I")])
        out = asyncio.run(feed(make(Qwen3LocalAgreement, script), 6.2))
        reasons = [(r.original, r.commit_reason) for r in out if isinstance(r, Transcribed)]
        self.assertEqual(reasons[0], ("Hello there.", "agree"))
        self.assertEqual(commits(out)[-1], "How are you doing today? I")

    def test_max_pending_words_forces_a_commit(self):
        words = " ".join(f"w{i}" for i in range(14))
        script = timeline([(2.0, f"{EN}{words} x"), (4.0, f"{EN}{words} y")])
        part = make(Qwen3LocalAgreement, script, max_pending_words=12)
        out = asyncio.run(feed(part, 4.2))
        self.assertEqual([r.commit_reason for r in out if isinstance(r, Transcribed)][0],
                         "agree-max")


class TranslatorV2(unittest.TestCase):
    def translator(self, replies, **options):
        t = Qwen35TranslationV2(Qwen35TranslationV2.validate(options, kind="translation"),
                                cfg=fake_cfg())
        t.glossary = Glossary(name="t", terms=({"ko": "을지로", "en": "Euljiro"},))
        t.context_turns, t.sentence, t.sent = 1, {}, []
        replies = iter(replies)

        async def chat(messages):
            t.sent.append(messages[-1]["content"])
            return next(replies)

        t._chat = chat
        return t

    def test_glossary_pairs_go_into_the_prompt(self):
        t = self.translator(["This is Euljiro."])
        asyncio.run(t.translate("여기가 을지로예요.", "en", "ko"))
        self.assertIn("을지로 -> Euljiro", t.sent[0])

    def test_leak_retry_keeps_the_cleaner_reply(self):
        t = self.translator(["우리这边数据不好", "우리 쪽 데이터가 안 좋아요"], leak_retry=True)
        reply, _ = asyncio.run(t.translate("Our data is bad.", "ko", "en"))
        self.assertEqual(reply, "우리 쪽 데이터가 안 좋아요")

    def test_meta_answer_is_retried(self):
        t = self.translator(["Please translate this sentence into Korean", "아니면"],
                            meta_guard=True)
        reply, _ = asyncio.run(t.translate("Or even", "ko", "en"))
        self.assertEqual(reply, "아니면")

    def test_continuation_shows_the_sentence_so_far(self):
        t = self.translator(["다섯 명이요,", "김태호로 예약했어요."], continuation=True)
        asyncio.run(t.translate("다섯 명이요,", "en", "ko"))
        asyncio.run(t.translate("김태호로 예약했습니다.", "en", "ko"))
        self.assertIn("already on screen", t.sent[1])
        self.assertIn("다섯 명이요,", t.sent[1])
        self.assertEqual(t.sentence[("ko", "en")], [])

    def test_same_language_passes_through(self):
        t = self.translator([])
        self.assertEqual(asyncio.run(t.translate("안녕", "ko", "ko")), ("안녕", "ko"))


class CascadeV2(unittest.TestCase):
    def test_split_at_vad_cuts_the_chunk_at_the_end_sample(self):
        from core.pipeline.cascade_v2 import Room

        calls = []

        class Transcriber:
            async def transcribe(self, pcm):
                calls.append(("transcribe", len(pcm) // 2))
                return []

            async def flush(self, reason, speech):
                calls.append(("flush", reason))
                return []

        class Detector:
            def detect(self, pcm):
                return Speech(started_at=0.0, ended_at=0.05, silence_waited_out_sec=0.8)

        room = Room({"transcription": Transcriber(), "vad": Detector()}, split_at_vad=True)
        asyncio.run(room.hear(b"\x00\x00" * 3200))
        self.assertEqual(calls, [("transcribe", 800), ("flush", "vad"), ("transcribe", 2400)])

    def test_mock_pipeline_end_to_end(self):
        from core import config
        from core.pipeline import build

        spec = {"pipeline": {"name": "cascade:v2", "transcription": "mock",
                             "vad": {"name": "mock", "speech_chunks": 3},
                             "translation": "mock", "labeler": "script",
                             "filter": {"name": "commit-rules", "fillers": True},
                             "split_at_vad": True},
                "commit": "seg", "gpu_memory_utilization": 0.1}
        cfg = SimpleNamespace(stity=config.parse_pipeline(spec, name="t"))
        pipe = build(cfg)
        pipe.start(languages=["en"], target_lang="ko")
        out = []
        for _ in range(7):
            out += asyncio.run(pipe.listen(b"\x01\x00" * 3200))
        out += asyncio.run(pipe.finish())
        translated = [r for r in out if isinstance(r, Translated)]
        self.assertTrue(translated)
        self.assertTrue(all(r.translation.startswith("[ko]") for r in translated))


class Diagnostics(unittest.TestCase):
    def test_passthrough_and_gaps(self):
        from bench.metrics.diagnostics import diagnostics

        records = [
            {"type": "transcribed", "decision_audio_sec": 1.0, "original": "안녕하세요"},
            {"type": "translated", "original": "안녕하세요", "translation": "안녕하세요",
             "language": "en", "target_lang": "en"},
            {"type": "transcribed", "decision_audio_sec": 5.0, "original": "Hi"},
            {"type": "translated", "original": "Hi", "translation": "Hi", "language": "en",
             "target_lang": "en"},
            {"type": "speech", "started_at": 0.0, "ended_at": 4.0},
        ]
        row = {"status": "ok", "records": records, "audio_sec": 60.0, "src_lang": "ko",
               "reference": "", "transcription_output": "안녕하세요 Hi"}
        values = diagnostics([row], "en")
        self.assertEqual(values["passthrough_share"], 1.0)
        self.assertEqual(values["commit_gap_max_sec"], 4.0)
        self.assertEqual(values["speech_segment_max_sec"], 4.0)
        self.assertEqual(values["commits_per_min"], 2.0)


if __name__ == "__main__":
    unittest.main()
