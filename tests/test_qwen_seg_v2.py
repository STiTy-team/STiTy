"""qwen-seg:v2 against qwen-seg on scripted decodes.

Each scenario replays a decode sequence that reproduces one WORKLOG issue. For each:
  - qwen-seg (v1) shows the bug,
  - qwen-seg:v2 with only that flag on does not,
  - qwen-seg:v2 with every flag off gives exactly v1's output (parity).
"""
import asyncio
import unittest

from core.components.registry import Speech
from core.components.transcription.qwen3_seg import Qwen3SegTranscription
from core.components.transcription.qwen3_seg_v2 import (
    FLAGS,
    Qwen3SegTranscriptionV2,
    clean_partial,
    covered_prefix,
    find_loop,
    repeats_tail,
)

from tests.fake_asr import FakeVad, commits, feed, make, partials, timeline

EN = "language English<asr_text>"
KO = "language Korean<asr_text>"


def run(coro):
    return asyncio.run(coro)


SCENARIOS = {
    "A2 held fragment": dict(
        script=timeline([(2.0, f"{EN}Needless to say, <SEG> the")],
                        final={0.0: f"{EN}Needless to say, <SEG> the rest is history."}),
        seconds=3.0, flush_at=3.0,
        speech=Speech(started_at=0.2, ended_at=3.0, silence_waited_out_sec=0.8)),
    "A3 boundary": dict(
        script=timeline([(2.0, f"{EN}It is in La Plata. <SEG>"),
                         (4.0, f"{EN}A city fifty kilometers away. <SEG>")]),
        seconds=4.2),
    "A1 no speech": dict(
        script=timeline([(2.0, f"{EN}Real speech here. <SEG>"),
                         (4.0, f"{EN}Real speech here. <SEG>"),
                         (6.0, f"{EN}I want to buy a new car. <SEG>")]),
        seconds=6.4, flush_at=4.6,
        speech=Speech(started_at=0.2, ended_at=4.6, silence_waited_out_sec=0.8),
        vad=[(0.2, 3.8)]),
    "A4 carry": dict(
        script=timeline([(2.0, f"{EN}Hello there. <SEG> General"),
                         (4.0, f"{EN}Hello there. <SEG> General Kenobi you"),
                         (6.0, f"{EN}Hello there. <SEG> General Kenobi you are"),
                         (8.0, f"{EN}Hello there. <SEG> General Kenobi you are a bold"),
                         (10.0, f"{EN}Hello there. <SEG>language English")],
                        final={0.0: f"{EN}General Kenobi, you are a bold one."}),
        seconds=10.0, flush_at=10.0,
        speech=Speech(started_at=0.2, ended_at=10.0, silence_waited_out_sec=0.8)),
    "N5 resync": dict(
        script=timeline([(2.0, f"{EN}A one. <SEG> B two. <SEG> C"),
                         (4.0, f"{EN}A one B two. <SEG> C three. <SEG> D")]),
        seconds=4.2),
    "A5 loop": dict(
        script=timeline([(2.0, f"{KO}네 알겠습니다. <SEG> 아니, 아니, 아니. <SEG> 아니, 아니.")],
                        final={0.0: "language None<asr_text>"}),
        seconds=2.2, languages=("ko",)),
    "A8 language list": dict(
        script=timeline([(2.0, f"{KO}English, Chinese, Japanese, Korean. <SEG>")]),
        seconds=2.2, languages=("ko",)),
    "A9 partial header": dict(
        script=timeline([(2.0, f"{EN}Hello. <SEG>language English General")]),
        seconds=2.2),
}

FIX = {
    "A2 held fragment": "keep_held_fragment",
    "A3 boundary": "strict_boundary_dedup",
    "A1 no speech": "no_speech_since_vad_reset",
    "A4 carry": "carry_uncommitted_audio",
    "N5 resync": "resync_cursor",
    "A5 loop": "loop_guard",
    "A8 language list": "drop_language_lists",
    "A9 partial header": "clean_partials",
}


def replay(cls, scenario: dict, **flags) -> list:
    part = make(cls, scenario["script"], **flags)
    vad = FakeVad(scenario["vad"]) if "vad" in scenario else None
    return run(feed(part, scenario["seconds"], languages=scenario.get("languages", ("en",)),
                    vad=vad, flush_at=scenario.get("flush_at"),
                    speech=scenario.get("speech")))


def records_key(records) -> list:
    return [(type(r).__name__, getattr(r, "original", getattr(r, "text", None)),
             getattr(r, "language", None), getattr(r, "commit_reason", None)) for r in records]


class Parity(unittest.TestCase):
    def test_all_flags_off_matches_v1(self):
        for name, scenario in SCENARIOS.items():
            with self.subTest(name):
                self.assertEqual(records_key(replay(Qwen3SegTranscription, scenario)),
                                 records_key(replay(Qwen3SegTranscriptionV2, scenario)))


class Fixes(unittest.TestCase):
    def v1(self, name):
        return replay(Qwen3SegTranscription, SCENARIOS[name])

    def v2(self, name):
        return replay(Qwen3SegTranscriptionV2, SCENARIOS[name], **{FIX[name]: True})

    def test_a2_held_fragment_survives_the_flush(self):
        self.assertEqual(commits(self.v1("A2 held fragment")), ["the rest is history."])
        self.assertEqual(commits(self.v2("A2 held fragment")),
                         ["Needless to say, the rest is history."])

    def test_a3_boundary_dedup_keeps_a_new_sentence(self):
        self.assertEqual(commits(self.v1("A3 boundary")), ["It is in La Plata."])
        self.assertEqual(commits(self.v2("A3 boundary")),
                         ["It is in La Plata.", "A city fifty kilometers away."])

    def test_a1_commit_over_silence_after_a_vad_reset_is_dropped(self):
        self.assertIn("I want to buy a new car.", commits(self.v1("A1 no speech")))
        self.assertNotIn("I want to buy a new car.", commits(self.v2("A1 no speech")))

    def test_a4_carried_audio_is_decoded_at_the_flush(self):
        v1, v2 = commits(self.v1("A4 carry")), commits(self.v2("A4 carry"))
        self.assertNotIn("General Kenobi, you are a bold one.", v1)
        self.assertIn("General Kenobi, you are a bold one.", v2)

    def test_n5_merged_units_do_not_swallow_the_next_one(self):
        self.assertNotIn("C three.", commits(self.v1("N5 resync")))
        self.assertIn("C three.", commits(self.v2("N5 resync")))

    def test_a5_alternating_filler_loop_is_cut(self):
        loops = lambda records: [c for c in commits(records) if c.count("아니") >= 2]
        self.assertTrue(loops(self.v1("A5 loop")))
        self.assertFalse(loops(self.v2("A5 loop")))
        self.assertIn("네 알겠습니다.", commits(self.v2("A5 loop")))

    def test_a8_language_name_list_is_not_committed(self):
        self.assertTrue(commits(self.v1("A8 language list")))
        self.assertFalse(commits(self.v2("A8 language list")))

    def test_a9_partials_carry_no_language_header(self):
        self.assertTrue(any("language" in p for p in partials(self.v1("A9 partial header"))))
        self.assertFalse(any("language" in p for p in partials(self.v2("A9 partial header"))))


class Helpers(unittest.TestCase):
    def test_flags_are_all_settable(self):
        settings = Qwen3SegTranscriptionV2.validate(
            {"model": "Org/fake", **{flag: True for flag in FLAGS}}, kind="transcription")
        self.assertTrue(all(settings[flag] for flag in FLAGS))

    def test_repeats_tail_needs_the_whole_unit(self):
        self.assertFalse(repeats_tail("It is in La Plata.", "A city fifty kilometers away."))
        self.assertTrue(repeats_tail("It is in La Plata.", "La Plata."))
        self.assertFalse(repeats_tail("I told you that", "At the other end"))

    def test_find_loop_ignores_punctuation_and_seg(self):
        text = "네 알겠습니다. <SEG> 아니, 아니, 아니. <SEG> 아니, 아니."
        cut = find_loop(text)
        self.assertEqual(text[:cut], "네 알겠습니다. <SEG> 아니,")
        self.assertIsNone(find_loop("one two three four five six"))

    def test_covered_prefix_is_in_order(self):
        text = "A one B two. <SEG> C three. <SEG> D"
        self.assertEqual(text[:covered_prefix(text, "A one. B two.")], "A one B two.")
        self.assertEqual(covered_prefix("네. <SEG> 새로운 말", "다른 말. 네."), 0)

    def test_clean_partial(self):
        self.assertEqual(clean_partial("Hello. language English General"), "Hello. General")
        self.assertEqual(clean_partial("Hello, lang"), "Hello,")
        self.assertEqual(clean_partial("l"), "")

    def test_n7_dot_carry_dedup_applies_to_seg_triggers(self):
        part = make(Qwen3SegTranscriptionV2, timeline([]), dedup_dot_carry=True)
        part.start(languages=["ko"])
        part.slot["dot_switch_prev_committed"] = "아, 아니에요, 괜찮습니다."
        found = part._find_duplicate("괜찮습니다.", stage="extract", trigger="seg")
        self.assertEqual(found[0], "dot-suffix-dedup")
        v1 = make(Qwen3SegTranscription, timeline([]))
        v1.start(languages=["ko"])
        v1.slot["dot_switch_prev_committed"] = "아, 아니에요, 괜찮습니다."
        self.assertIsNone(v1._find_duplicate("괜찮습니다.", stage="extract", trigger="seg"))

    def test_bias_context_goes_into_every_new_state(self):
        from core.utils.glossary import Glossary

        part = make(Qwen3SegTranscriptionV2, timeline([]), bias_recent_commits=2)
        part.glossary = Glossary(name="t", terms=({"ko": "을지로", "en": "Euljiro"},))
        part.start(languages=["ko", "en"])
        self.assertIn("을지로", part.model.contexts[-1])
        self.assertIn("Euljiro", part.model.contexts[-1])


if __name__ == "__main__":
    unittest.main()
