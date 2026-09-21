import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from bench.quality_annotators import (
    OpenAIJsonJudge,
    annotate_critical_information,
    annotate_fluency,
    extract_critical_values,
)


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


class QualityAnnotatorsTest(unittest.TestCase):

    def test_number_span_excludes_trailing_comma(self):
        text = "In 1990, it cost 12,000 won."
        number = next(v for v in extract_critical_values(text, "en") if v["type"] == "number")
        self.assertEqual(number["text"], "1990")
        self.assertEqual(text[number["start"]:number["end"]], "1990")
        self.assertEqual(number["canonical_value"], "1990")

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
            "candidate_spans": [{"type": "location", "text": ref_place,
                                 "start": cand_start, "end": cand_start + len(ref_place),
                                 "canonical_value": "gangnam_station"}],
            "alignments": [{"reference_index": 0, "candidate_index": 0,
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

    def test_gold_span_without_offsets_is_preserved(self):
        judge = _Judge([{"reference_spans": [], "candidate_spans": [], "alignments": []}])
        result = asyncio.run(annotate_critical_information(
            judge=judge, source="김민수에게 전화해", reference="Call Minsoo Kim",
            candidate="Call Minsoo Kim", source_lang="ko", target_lang="en",
            gold_reference_spans=[{"type": "person", "text": "Minsoo Kim",
                                   "canonical_value": "kim_minsoo",
                                   "accepted_values": ["김민수"]}]))
        self.assertEqual(result["reference_spans"][0]["canonical_value"], "kim_minsoo")
        self.assertEqual(result["annotation_source"], "human_gold+automatic")

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


if __name__ == "__main__":
    unittest.main()
