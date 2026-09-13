"""`PYTHONPATH=. python -m unittest core.meaning_segmentator.autoseg.runtime.test_agents_judge`"""
import unittest

from . import agents_judge as aj

BASE = """[Role]
r

[Core Principles]
- (1) ask this
- (2) ask that

[Scoring Rules]
- combine as (1 - contradiction) x cohesion

[Decision Procedure]
d

[Output Rules]
frozen text

[Examples]
Input: a <SEG:?> b
Output: a <SEG:50> b
"""


class Findings(unittest.TestCase):
    def test_허용된_섹션만_통과(self):
        got = aj.clean_findings({"findings": [
            {"error_type": "boundary", "diagnosis": "d",
             "edit": {"where": "scoring_rules", "action": "add", "text": "x"}},
            {"error_type": "boundary", "diagnosis": "d",
             "edit": {"where": "core_principles", "action": "add", "text": "y"}}]})
        self.assertEqual([f["edit"]["where"] for f in got], ["core_principles"])

    def test_replace_인데_target_없으면_버린다(self):
        got = aj.clean_findings({"findings": [
            {"edit": {"where": "core_principles", "action": "replace", "text": "y"}}]})
        self.assertEqual(got, [])

    def test_최대_3개(self):
        many = [{"edit": {"where": "examples", "action": "add", "text": f"t{i}"}} for i in range(9)]
        self.assertEqual(len(aj.clean_findings({"findings": many})), 3)

    def test_빈_입력도_견딘다(self):
        self.assertEqual(aj.clean_findings({}), [])
        self.assertEqual(aj.clean_findings({"findings": None}), [])


class ParsePrompt(unittest.TestCase):
    def test_정상(self):
        new = BASE.replace("- (2) ask that", "- (2) ask that, and check the seam")
        pr, log = aj.parse_prompt({"prompt": new, "changelog": ["seam"]}, BASE, budget=10_000)
        self.assertEqual(pr, new)
        self.assertEqual(log, ["seam"])

    def test_동결_섹션을_고치면_거부(self):
        """[Output Rules] 는 출력 형식, [Scoring Rules] 는 측정 절차라 둘 다 라벨이 바뀔 때만 바뀐다."""
        new = BASE.replace("frozen text", "frozen text plus")
        pr, errs = aj.parse_prompt({"prompt": new}, BASE, budget=10_000)
        self.assertIsNone(pr)
        self.assertTrue(any("[Output Rules]" in e for e in errs))

    def test_점수_규칙에_조건을_넣으면_거부(self):
        new = BASE.replace("- combine as (1 - contradiction) x cohesion",
                           "- combine as (1 - contradiction) x cohesion\n- if comma, lower")
        pr, errs = aj.parse_prompt({"prompt": new}, BASE, budget=10_000)
        self.assertIsNone(pr)
        self.assertTrue(any("[Scoring Rules]" in e for e in errs))

    def test_섹션이_빠지면_거부(self):
        pr, errs = aj.parse_prompt({"prompt": "[Role]\nonly"}, BASE, budget=10_000)
        self.assertIsNone(pr)
        self.assertTrue(errs)

    def test_길이_초과_거부(self):
        pr, errs = aj.parse_prompt({"prompt": BASE}, BASE, budget=10)
        self.assertIsNone(pr)
        self.assertTrue(any("길이 초과" in e for e in errs))

    def test_빈_출력_거부(self):
        self.assertEqual(aj.parse_prompt({}, BASE, 100)[0], None)


class Size(unittest.TestCase):
    def test_brief_의_editable_cap_은_상한에서_고정분을_뺀_값(self):
        b = aj.size_brief(BASE, budget=len(BASE))
        editable = sum(b["sections"][h] for h in aj.EDITABLE)
        self.assertEqual(b["editable_cap"], editable)
        self.assertEqual(aj.size_brief(BASE, budget=len(BASE) + 50)["editable_cap"], editable + 50)

    def test_feedback_은_초과량과_늘어난_섹션을_먼저(self):
        draft = BASE.replace("- (2) ask that", "- (2) ask that" + "x" * 40)
        fb = aj.size_feedback(draft, BASE, budget=len(BASE))
        self.assertEqual(fb["over_by"], 40)
        self.assertEqual(fb["cut_from"][0], "[Core Principles]")
        self.assertEqual(fb["sections"]["[Core Principles]"]["change"], 40)

    def test_길이만_넘었을_때만_되돌린다(self):
        self.assertTrue(aj.only_too_long(["길이 초과: 10 > 5"]))
        self.assertFalse(aj.only_too_long(["길이 초과: 10 > 5", "동결 섹션이 바뀌었다: [Output Rules]"]))
        self.assertFalse(aj.only_too_long([]))


class MeasurementInOnePlace(unittest.TestCase):
    """측정 절차는 [Scoring Rules] 에만 쓴다 — [Output Rules] 에 옛 라벨 서술이 남으면
    한 프롬프트 안에 목표 설명이 둘이 된다 (judge01 v0 에서 실제로 났다)."""

    def test_output_rules_는_측정을_서술하지_않는다(self):
        from . import agents_distill as ad
        s = ad.output_rules(True, "source", meaning_in_scoring_rules=True)
        self.assertIn("Scoring Rules section", s)
        for old in ("quality_before", "entailment", "MEASURED"):
            self.assertNotIn(old, s)

    def test_output_rules_본문에_섹션_헤더가_없다(self):
        """본문에 "[Scoring Rules]" 가 있으면 섹션 경계로 잡혀 replace_section 이 꼬리를 복제한다."""
        from . import agents_distill as ad
        s = ad.output_rules(True, "source", meaning_in_scoring_rules=True)
        body = s.split("\n", 1)[1]
        for h in ad.SECTIONS:
            self.assertNotIn(h, body)
        once = aj.replace_section(BASE, "[Output Rules]", s)
        self.assertEqual(aj.replace_section(once, "[Output Rules]", s), once)
        self.assertEqual(aj.section_of(once, "[Output Rules]"), s.strip())

    def test_writer_와_pe_가_scoring_rules_를_측정_자리로_안다(self):
        self.assertIn("ONLY place that states how the target was measured",
                      aj.writer_system(True, ["German"]))
        self.assertIn("[Scoring Rules] states how the target was measured",
                      aj.engineer_system(100, 100))


if __name__ == "__main__":
    unittest.main()
