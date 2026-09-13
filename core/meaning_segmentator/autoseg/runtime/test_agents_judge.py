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
        """[Output Rules] 는 측정 절차 서술이라 라벨이 바뀔 때만 바뀐다."""
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


if __name__ == "__main__":
    unittest.main()
