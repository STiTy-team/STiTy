import asyncio
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from core.utils.metrics.critical_information import (
    ReferenceSpanCache,
    annotate_candidate_critical_information,
    annotate_critical_information,
    annotate_reference_critical_information,
    gold_reference_spans,
)
from core.utils.metrics.critical_values import extract_critical_values
from core.utils.metrics.fluency import annotate_fluency
from core.utils.metrics.judge import OpenAIJsonJudge
from core.utils.metrics.text import levenshtein, normalize_words
from core.utils.metrics.text_noise import perturb_transcript


class _Judge:
    model = "fake-judge"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def ask(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class _Message:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)


class _Response:
    def __init__(self, content, prompt_tokens, completion_tokens):
        self.choices = [_Choice(content)]
        self._usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}

    def model_dump(self):
        return {"choices": [{"finish_reason": "stop"}], "usage": dict(self._usage)}


class _Completions:
    def __init__(self, responses):
        self.responses = list(responses)

    async def create(self, **_kwargs):
        return self.responses.pop(0)


class _Chat:
    def __init__(self, responses):
        self.completions = _Completions(responses)


class _Client:
    def __init__(self, responses):
        self.chat = _Chat(responses)


class AnnotationTest(unittest.TestCase):

    def test_number_span_excludes_trailing_comma(self):
        text = "In 1990, it cost 12,000 won."
        number = next(v for v in extract_critical_values(text, "en") if v["type"] == "number")
        self.assertEqual(number["text"], "1990")
        self.assertEqual(text[number["start"]:number["end"]], "1990")
        self.assertEqual(number["canonical_value"], "1990")

    def _values(self, text, lang):
        return {(v["type"], v["text"], v["canonical_value"])
                for v in extract_critical_values(text, lang)}

    def test_dotted_meridiem_after_clock_time_is_read(self):
        # FLEURS ko_1748: four translators wrote "11:35 p.m." and were scored wrong.
        self.assertEqual(self._values("They put it out by 11:35 p.m.", "en"),
                         {("time", "11:35 p.m.", "23:35")})
        self.assertEqual(self._values("at 11:35 today", "en"), {("time", "11:35", "11:35")})
        # The sentence's own full stop is not part of an undotted "PM".
        self.assertEqual(self._values("by 11:35 PM.", "en"), {("time", "11:35 PM", "23:35")})

    def test_english_number_words_compose(self):
        # FLEURS ko_1980: "one-hundred percent" was read as 1, failing every system.
        self.assertEqual(self._values("with one-hundred percent certainty", "en"),
                         {("unit", "one-hundred percent", "100 %")})
        self.assertEqual(self._values("with 100% certainty", "en"), {("unit", "100%", "100 %")})
        self.assertEqual(self._values("twenty-five people", "en"),
                         {("number", "twenty-five", "25")})
        self.assertEqual(self._values("three hundred and five seats", "en"),
                         {("number", "three hundred and five", "305")})
        self.assertEqual(self._values("about 2 million users", "en"),
                         {("number", "2 million", "2000000")})

    def test_hyphenated_name_and_bare_pronoun_one_are_not_numbers(self):
        # FLEURS ko_1946 ("COVID-19") and ko_1677 ("One day").
        self.assertEqual(self._values("COVID-19 policies", "en"), set())
        self.assertEqual(self._values("One day, the one I liked left.", "en"), set())
        self.assertEqual(self._values("It costs one dollar.", "en"),
                         {("money", "one dollar", "USD:1")})
        self.assertEqual(self._values("pages 3-4", "en"),
                         {("number", "3", "3"), ("number", "4", "4")})

    def test_korean_answer_and_demonstrative_are_not_numbers(self):
        self.assertEqual(self._values("네, 이 시간에 이 분이 오셨어요.", "ko"), set())
        self.assertEqual(self._values("세 시간 걸려요.", "ko"), {("number", "세", "3")})

    def test_korean_numerals_need_a_counter(self):
        self.assertEqual(self._values("사과 두 개랑 한 사람", "ko"),
                         {("number", "두", "2"), ("number", "한", "1")})
        self.assertEqual(self._values("한국 세상", "ko"), set())
        self.assertEqual(self._values("두 번 말했어", "ko"), {("number", "두", "2")})
        self.assertEqual(self._values("두 번째 줄", "ko"), {("ordinal", "두 번째", "2")})

    def test_korean_scale_numerals_compose(self):
        self.assertEqual(self._values("3만 원", "ko"), {("money", "3만 원", "KRW:30000")})
        self.assertEqual(self._values("1억 2천만 원이", "ko"),
                         {("money", "1억 2천만 원", "KRW:120000000")})
        self.assertEqual(self._values("삼백 명", "ko"), {("number", "삼백", "300")})
        self.assertEqual(self._values("이천에 가요", "ko"), set())
        self.assertEqual(self._values("각각 6 만 원씩", "ko"),
                         {("money", "6 만 원", "KRW:60000")})
        self.assertEqual(self._values("각각 6 만원씩", "ko"), {("money", "6 만원", "KRW:60000")})
        self.assertEqual(self._values("3 백화점", "ko"), {("number", "3", "3")})

    def test_korean_clock_times(self):
        self.assertEqual(self._values("두 시 반에 봐", "ko"), {("time", "두 시 반", "02:30")})
        self.assertEqual(self._values("오후 세 시쯤", "ko"), {("time", "오후 세 시", "15:00")})
        self.assertEqual(self._values("밤 열 시", "ko"), {("time", "밤 열 시", "22:00")})
        self.assertEqual(self._values("오전 다섯 시인가요?", "ko"),
                         {("time", "오전 다섯 시", "05:00")})
        self.assertEqual(self._values("7시는 차 있고 7시 30분은 돼요.", "ko"),
                         {("time", "7시", "07:00"), ("time", "7시 30분", "07:30")})

    def test_judge_appends_usage_for_every_billed_call(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "judge_usage.jsonl"
            judge = OpenAIJsonJudge(
                model="gpt-5.4-mini", usage_log=log,
                client=_Client([_Response("not json", 1000, 100),
                                _Response('{"score": 5}', 2000, 200)]))
            judge.retry_delay_sec = 0
            result = asyncio.run(judge.ask(purpose="fluency", system="s", payload={}))
            lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(result, {"score": 5})
        self.assertEqual([line["prompt_tokens"] for line in lines], [1000, 2000])
        self.assertEqual([line["completion_tokens"] for line in lines], [100, 200])
        self.assertEqual({line["model"] for line in lines}, {"gpt-5.4-mini"})
        self.assertEqual({line["purpose"] for line in lines}, {"fluency"})
        self.assertAlmostEqual(lines[0]["cost"], (1000 * .75 + 100 * 4.5) / 1_000_000)
        self.assertAlmostEqual(lines[1]["cumulative_cost"],
                               (3000 * .75 + 300 * 4.5) / 1_000_000)
        self.assertEqual(judge.usage.snapshot()["calls"], 2)

    def test_ko_en_values_share_canonical_forms(self):
        korean = extract_critical_values("오후 세 시에 12,000원 보내 줘", "ko")
        english = extract_critical_values("Send 12,000 won at 3 PM", "en")
        self.assertIn(("time", "15:00"), {(v["type"], v["canonical_value"]) for v in korean})
        self.assertIn(("time", "15:00"), {(v["type"], v["canonical_value"]) for v in english})
        self.assertIn(("money", "KRW:12000"),
                      {(v["type"], v["canonical_value"]) for v in korean})
        self.assertIn(("money", "KRW:12000"),
                      {(v["type"], v["canonical_value"]) for v in english})

    def test_critical_annotation_merges_entities_and_values(self):
        reference = "Meet Kim at Gangnam Station at 3 PM."
        candidate = "Meet Kim at Gangnam Station at 4 PM."
        ref_place = "Gangnam Station"
        ref_start = reference.index(ref_place)
        cand_start = candidate.index(ref_place)
        judge = _Judge([{
            "reference_spans": [{"type": "location", "text": ref_place,
                                 "start": ref_start, "end": ref_start + len(ref_place),
                                 "canonical_value": "gangnam_station"}],
        }, {
            "candidate_spans": [{"type": "location", "text": ref_place,
                                 "start": cand_start, "end": cand_start + len(ref_place),
                                 "canonical_value": "gangnam_station"}],
            "alignments": [{"reference_index": 1, "candidate_index": 0,
                            "relation": "equivalent"}],
        }])
        result = asyncio.run(annotate_critical_information(
            judge=judge, source="오후 세 시에 강남역에서 김을 만나요",
            reference=reference, candidate=candidate,
            source_lang="ko", target_lang="en"))
        refs = {(v["type"], v["canonical_value"]) for v in result["reference_spans"]}
        hyps = {(v["type"], v["canonical_value"]) for v in result["candidate_spans"]}
        self.assertIn(("location", "gangnam_station"), refs)
        self.assertIn(("time", "15:00"), refs)
        self.assertIn(("time", "16:00"), hyps)
        self.assertEqual(result["annotation_source"], "automatic")
        self.assertEqual(result["alignment"][0]["candidate_text"], ref_place)

    def test_reference_spans_are_extracted_without_the_candidate(self):
        # The reference set is the denominator every system is scored against, so it
        # must not depend on which candidate happened to be in the same call.
        judge = _Judge([{"reference_spans": [
            {"type": "location", "text": "Paris", "start": 9, "end": 14,
             "canonical_value": "paris"}]}])
        result = asyncio.run(annotate_reference_critical_information(
            judge=judge, source="3시에 파리에서", reference="At 3 PM, Paris.",
            source_lang="ko", target_lang="en"))
        self.assertNotIn("candidate", judge.calls[0]["payload"])
        self.assertEqual({(v["type"], v["canonical_value"], v["origin"])
                          for v in result["reference_spans"]},
                         {("time", "15:00", "rule"), ("location", "paris", "llm")})

    def test_candidate_annotation_rescues_only_paraphrased_reference_values(self):
        reference_spans = [{"type": "number", "text": "12", "start": 4, "end": 6,
                            "canonical_value": "12", "origin": "rule"}]
        candidate = "Buy a dozen eggs, not 13 apples."
        judge = _Judge([{"candidate_spans": [
            # the same value in other words: accepted
            {"type": "number", "text": "a dozen", "canonical_value": "12"},
            # on top of the rule's own 13: rejected, the rule's value stands
            {"type": "number", "text": "13", "canonical_value": "12"},
            # a value the reference does not have: rejected
            {"type": "number", "text": "apples", "canonical_value": "99"},
            # "domain term" is the prompt's own wording for the term type
            {"type": "domain term", "text": "eggs", "canonical_value": "egg"},
        ], "alignments": []}])
        result = asyncio.run(annotate_candidate_critical_information(
            judge=judge, source="", reference="Buy 12 eggs.", candidate=candidate,
            reference_spans=reference_spans, source_lang="ko", target_lang="en"))
        self.assertEqual({(v["type"], v["text"], v["canonical_value"], v["origin"])
                          for v in result["candidate_spans"]},
                         {("number", "a dozen", "12", "llm_value"),
                          ("number", "13", "13", "rule"),
                          ("term", "eggs", "egg", "llm")})

    def test_gold_span_without_offsets_is_preserved(self):
        judge = _Judge([{"reference_spans": []}, {"candidate_spans": [], "alignments": []}])
        result = asyncio.run(annotate_critical_information(
            judge=judge, source="김민수에게 전화해", reference="Call Minsoo Kim",
            candidate="Call Minsoo Kim", source_lang="ko", target_lang="en",
            gold_reference_spans=[{"type": "person", "text": "Minsoo Kim",
                                   "canonical_value": "kim_minsoo",
                                   "accepted_values": ["김민수"]}]))
        self.assertEqual(result["reference_spans"][0]["canonical_value"], "kim_minsoo")
        self.assertEqual(result["reference_spans"][0]["origin"], "gold")
        self.assertEqual(result["annotation_source"], "human_gold+automatic")

    def test_only_human_reference_spans_count_as_gold(self):
        manifest = {"reference_spans": [{"type": "person", "canonical_value": "kim"}]}
        legacy_automatic = {"annotation_source": "automatic", "reference_spans": [
            {"type": "person", "canonical_value": "kim"}]}
        mixed = {"annotation_source": "human_gold+automatic", "reference_spans": [
            {"type": "person", "canonical_value": "kim", "origin": "gold"},
            {"type": "time", "canonical_value": "15:00", "origin": "rule"}]}
        self.assertEqual(len(gold_reference_spans(manifest)), 1)
        self.assertEqual(gold_reference_spans(legacy_automatic), [])
        self.assertEqual([v["origin"] for v in gold_reference_spans(mixed)], ["gold"])

    def test_reference_cache_keeps_the_first_written_spans(self):
        async def run(directory):
            path = Path(directory) / "cache.jsonl"
            first, second = ReferenceSpanCache(path), ReferenceSpanCache(path)

            async def racing_factory():
                # Another process wrote this key while we were computing ours.
                await first.get("k", lambda: _value("theirs"))
                return {"reference_spans": ["ours"]}

            ours = await second.get("k", racing_factory)
            again = await ReferenceSpanCache(path).get("k", lambda: _value("never"))
            return ours, again

        async def _value(name):
            return {"reference_spans": [name]}

        with tempfile.TemporaryDirectory() as directory:
            ours, again = asyncio.run(run(directory))
        self.assertEqual(ours, {"reference_spans": ["theirs"]})
        self.assertEqual(again, {"reference_spans": ["theirs"]})

    def test_fluency_annotation_repairs_unique_span_offset(self):
        candidate = "I has a ticket."
        judge = _Judge([{
            "score": 3,
            "reason": "Agreement error",
            "errors": [{"category": "grammar", "severity": "major",
                        "start": 99, "end": 102, "text": "has",
                        "explanation": "Use have"}],
        }])
        result = asyncio.run(annotate_fluency(
            judge=judge, candidate=candidate, target_lang="en",
            previous_turns=["Earlier turn"])).copy()
        self.assertEqual(result["judge"]["score"], 3)
        self.assertEqual(result["mqm_errors"][0]["start"], 2)
        self.assertEqual(result["target_token_count"], 4)
        self.assertEqual(judge.calls[0]["payload"]["preceding_target_turns"], ["Earlier turn"])

    def test_fluency_error_offsets_resolve_to_the_nearest_occurrence(self):
        # A repeated word is exactly the error whose text appears more than once.
        candidate = "I saw the the cat."
        judge = _Judge([{"score": 3, "reason": "", "errors": [
            {"category": "repetition", "severity": "minor", "start": 11, "end": 14,
             "text": "the"},
            {"category": "grammar", "severity": "minor", "start": 0, "end": 3,
             "text": "not in the text"},
        ]}])
        result = asyncio.run(annotate_fluency(judge=judge, candidate=candidate,
                                              target_lang="en"))
        self.assertEqual([(e["start"], e["end"]) for e in result["mqm_errors"]], [(10, 13)])
        self.assertEqual(result["mqm_unlocated_errors"], 1)


_EN_TEXT = (
    "We should meet at the station before the train leaves because the tickets are "
    "already paid and the manager wants everyone there early. My sister called "
    "yesterday about the dinner reservation, and she asked whether the restaurant "
    "still serves the seafood pasta that we ordered last time. I think the weather "
    "will be fine this weekend, so we could walk to the museum after lunch and then "
    "visit the market near the river where they sell fresh bread and flowers. "
) * 4
_KO_TEXT = (
    "내일 아침에 회의가 있어서 일찍 출발해야 할 것 같아요. 어제 동생이 저녁 예약 때문에 "
    "전화했는데 식당에서 해산물 파스타를 아직 파는지 물어봤어요. 이번 주말에는 날씨가 "
    "괜찮을 것 같으니까 점심 먹고 박물관까지 걸어가서 강 근처 시장도 구경해요. "
) * 4


def _wer(reference, hypothesis):
    words = normalize_words(reference)
    return levenshtein(words, normalize_words(hypothesis)) / len(words)


class TextNoiseTest(unittest.TestCase):

    def test_noise_is_reproducible_and_nonempty_protocol(self):
        first = perturb_transcript("오늘 오후 세 시에 만나요.", lang="ko", level=.3,
                                   seed=7, item_id="a")
        second = perturb_transcript("오늘 오후 세 시에 만나요.", lang="ko", level=.3,
                                    seed=7, item_id="a")
        self.assertEqual(first, second)
        self.assertNotEqual(first[0], "오늘 오후 세 시에 만나요.")
        self.assertTrue(first[1])

    def test_substitutions_are_the_most_common_word_error(self):
        # ASR errors are mostly substitutions. The earlier generator only ever
        # deleted, repeated or dropped punctuation: its dictionary substitutions
        # fired on 0 of 90 Korean variants.
        for lang, text in (("en", _EN_TEXT), ("ko", _KO_TEXT)):
            with self.subTest(lang=lang):
                _, operations = perturb_transcript(text, lang=lang, level=.3, seed=1,
                                                   item_id="x")
                kinds = Counter(op["type"] for op in operations)
                self.assertGreater(kinds["substitution"], kinds["word_deletion"])
                self.assertGreater(kinds["word_deletion"], 0)
                self.assertGreater(kinds["word_repetition"], 0)

    def test_requested_level_approximates_the_word_error_rate(self):
        for level in (.1, .3):
            noisy, _ = perturb_transcript(_EN_TEXT, lang="en", level=level, seed=3,
                                          item_id="x")
            self.assertAlmostEqual(_wer(_EN_TEXT, noisy), level, delta=.08)


if __name__ == "__main__":
    unittest.main()
