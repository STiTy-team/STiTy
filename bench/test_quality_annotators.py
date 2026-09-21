import asyncio
import unittest

from bench.quality_annotators import (
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


class QualityAnnotatorsTest(unittest.TestCase):

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
