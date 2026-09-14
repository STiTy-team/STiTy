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

    def test_길이만_넘었을_때만_되돌린다(self):
        self.assertTrue(aj.only_too_long(["길이 초과: 10 > 5"]))
        self.assertFalse(aj.only_too_long(["길이 초과: 10 > 5", "동결 섹션이 바뀌었다: [Output Rules]"]))
        self.assertFalse(aj.only_too_long([]))


class Edits(unittest.TestCase):
    def test_단위는_원칙_한_줄과_예시_한_쌍(self):
        u = {x["id"]: x for x in aj.edit_units(BASE)}
        self.assertEqual(sorted(u), ["C1", "C2", "E1"])
        self.assertEqual(u["C1"]["text"], "- (1) ask this")
        self.assertEqual(u["E1"]["text"], "Input: a <SEG:?> b\nOutput: a <SEG:50> b")
        self.assertEqual(u["C2"]["chars"], len("- (2) ask that"))

    def test_바꾸기_지우기_끼우기(self):
        pr, note, _draft, deltas, _skipped = aj.parse_edits({"edits": [
            {"op": "replace", "id": "C2", "text": "- (2) check the seam"},
            {"op": "delete", "id": "C1"},
            {"op": "insert_after", "id": "E1", "text": "Input: c <SEG:?> d\nOutput: c <SEG:10> d"}],
            "changelog": ["seam"]}, BASE, budget=10_000)
        self.assertIsNotNone(pr, note)
        self.assertNotIn("ask this", pr)
        self.assertEqual([x["id"] for x in aj.edit_units(pr)], ["C1", "E1", "E2"])
        self.assertEqual(aj.edit_units(pr)[0]["text"], "- (2) check the seam")
        for h in ("[Role]", "[Scoring Rules]", "[Decision Procedure]", "[Output Rules]"):
            self.assertEqual(aj.section_of(pr, h), aj.section_of(BASE, h))
        self.assertEqual(deltas[1]["chars_change"], -(len("- (1) ask this") + 1))
        self.assertEqual(note, ["seam"])

    def test_같은_본문으로_바꾸면_프롬프트가_그대로다(self):
        """재조립이 공백을 바꾸면 아무것도 안 고친 편집도 길이 상한에 걸린다."""
        for base in (BASE, BASE.rstrip("\n")):
            same = [{"op": "replace", "id": u["id"], "text": u["text"]} for u in aj.edit_units(base)]
            pr, note, _d, _deltas, _sk = aj.parse_edits({"edits": same}, base, budget=len(base))
            self.assertEqual(pr, base, note)

    def test_본문_머리의_단위_id_는_뗀다(self):
        pr, *_ = aj.parse_edits({"edits": [
            {"op": "insert_after", "id": "C2", "text": "C11: - new rule"},
            {"op": "insert_after", "id": "E1", "text": "E2) Input: c <SEG:?> d\nOutput: c <SEG:9> d"}]},
            BASE, 10_000)
        texts = [u["text"] for u in aj.edit_units(pr)]
        self.assertIn("- new rule", texts)
        self.assertIn("Input: c <SEG:?> d\nOutput: c <SEG:9> d", texts)

    def test_맨_끝에_붙이기(self):
        """끝에 붙이는 자리가 없어 PE 가 없는 번호를 만들어 썼다 (judge07 에서 네 이터가 날아갔다)."""
        for uid in ("C_end", "C3"):          # C3 = 마지막(C2) 바로 다음 번호
            pr, note, _d, deltas, skipped = aj.parse_edits(
                {"edits": [{"op": "insert_after", "id": uid, "text": "- (3) ask last"}]},
                BASE, 10_000)
            self.assertIsNotNone(pr, note)
            self.assertEqual(skipped, [])
            self.assertEqual([u["text"] for u in aj.edit_units(pr)][:3],
                             ["- (1) ask this", "- (2) ask that", "- (3) ask last"])
        pr, _n, _d, _de, _sk = aj.parse_edits(
            {"edits": [{"op": "insert_after", "id": "E_end",
                        "text": "Input: c <SEG:?> d\nOutput: c <SEG:9> d"}]}, BASE, 10_000)
        self.assertEqual([u["id"] for u in aj.edit_units(pr)], ["C1", "C2", "E1", "E2"])

    def test_한참_뒤_번호는_건너뛴다(self):
        pr, errs, _d, _de, skipped = aj.parse_edits(
            {"edits": [{"op": "insert_after", "id": "C9", "text": "- (9) ask"}]}, BASE, 10_000)
        self.assertIsNone(pr)
        self.assertTrue(any("C9" in s["reason"] for s in skipped), skipped)

    def test_맨_앞에_끼우기(self):
        pr, *_ = aj.parse_edits({"edits": [{"op": "insert_after", "id": "C0",
                                             "text": "- (0) first"}]}, BASE, 10_000)
        self.assertEqual(aj.edit_units(pr)[0]["text"], "- (0) first")

    def test_잘못된_편집만_있으면_반려(self):
        bad = [
            [{"op": "replace", "id": "C9", "text": "x"}],
            [{"op": "replace", "id": "C1", "text": "[Output Rules]\nhijack"}],
            [{"op": "replace", "id": "E1", "text": "just words"}],
            [{"op": "rewrite", "id": "C1", "text": "y"}],
            [],
        ]
        for edits in bad:
            pr, errs, _draft, _deltas, skipped = aj.parse_edits({"edits": edits}, BASE, 10_000)
            self.assertIsNone(pr, edits)
            self.assertTrue(errs, edits)
            self.assertTrue(skipped, edits)

    def test_잘못된_편집만_빼고_나머지는_적용한다(self):
        """묶음을 통째로 반려하면 멀쩡한 편집까지 버려진다 (judge07 iter 1)."""
        pr, note, _draft, deltas, skipped = aj.parse_edits({"edits": [
            {"op": "replace", "id": "C1", "text": "- (1) ask this better"},
            {"op": "replace", "id": "C9", "text": "없는 단위"},
            {"op": "delete", "id": "C2"},
            {"op": "replace", "id": "C2", "text": "두 번째"}]}, BASE, 10_000)
        self.assertIsNotNone(pr, note)
        self.assertIn("- (1) ask this better", pr)
        self.assertNotIn("- (2) ask that", pr)
        self.assertEqual([d["id"] for d in deltas], ["C1", "C2"])
        self.assertEqual([(s["edit"], s["id"]) for s in skipped], [(1, "C9"), (3, "C2")])

    def test_길이만_넘으면_적용_결과와_실측_초과량을_돌려준다(self):
        edits = [{"op": "insert_after", "id": "C2", "text": "- " + "x" * 40}]
        pr, errs, draft, deltas, _sk = aj.parse_edits({"edits": edits}, BASE, budget=len(BASE))
        self.assertIsNone(pr)
        self.assertTrue(aj.only_too_long(errs))
        fb = aj.edit_feedback(draft, BASE, len(BASE), deltas)
        self.assertEqual(fb["over_by"], len(draft) - len(BASE))
        self.assertGreater(fb["over_by"], 0)
        self.assertEqual(fb["your_edits"][0]["id"], "C2")


class Provenance(unittest.TestCase):
    GAIN = {"mean": 0.0166, "lo": 0.0025, "hi": 0.0311}

    def test_시작은_전부_v0(self):
        prov = aj.init_provenance(BASE)
        self.assertEqual(len(prov), 3)
        self.assertTrue(all(r["origin"] == "v0" and r["adopted_delta"] is None
                            for r in prov.values()))

    def test_채택하면_남은_단위는_기록을_잇고_새_단위는_Δ_를_받는다(self):
        prov = aj.init_provenance(BASE)
        prov["- (1) ask this"]["critic_hits"] = 3
        new = BASE.replace("- (2) ask that", "- (2) check the seam")
        got = aj.adopt_provenance(prov, new, 4, self.GAIN)
        self.assertEqual(got["- (1) ask this"]["critic_hits"], 3)
        self.assertEqual(got["- (2) check the seam"],
                         {"origin": "iter 4", "adopted_delta": 0.0166, "adopted_ci_lo": 0.0025,
                          "critic_hits": 0})
        self.assertNotIn("- (2) ask that", got)

    def test_critic_target_앞부분으로_지적_횟수를_센다(self):
        base = BASE.replace("- (1) ask this", "1) Does what follows the cut overturn the claim?")
        prov = aj.init_provenance(base)
        aj.count_critic_hits(prov, [
            {"edit": {"target": "Does what follows the cut overturn"}},
            {"edit": {"target": "ask"}},                       # 너무 짧아 세지 않는다
            {"edit": {}}])
        self.assertEqual(prov["1) Does what follows the cut overturn the claim?"]["critic_hits"], 1)
        self.assertEqual(prov["- (2) ask that"]["critic_hits"], 0)

    def test_PE_입력_단위에_출처가_붙는다(self):
        prov = aj.init_provenance(BASE)
        u = aj.units_with_provenance(BASE, prov)
        self.assertEqual({k for k in u[0]} >= {"id", "chars", "text", "origin", "critic_hits"},
                         True)

    def test_이력용_편집_요약은_원문_앞부분을_싣는다(self):
        got = aj.edit_summary(BASE, [{"op": "replace", "id": "C2", "text": "- (2) new"},
                                     {"op": "delete", "id": "E1"}, "junk"])
        self.assertEqual(got[0], {"op": "replace", "id": "C2", "kind": "change",
                                  "was": "- (2) ask that", "now": "- (2) new"})
        self.assertEqual((got[1]["kind"], got[1]["was"][:6]), ("delete", "Input:"))
        self.assertEqual(len(got), 2)

    def test_편집별_증감에_압축_표시가_남는다(self):
        _pr, _n, _d, deltas, _sk = aj.parse_edits({"edits": [
            {"op": "replace", "id": "C1", "kind": "paraphrase", "text": "- (1) ask"},
            {"op": "delete", "id": "C2"}]}, BASE, 10_000)
        self.assertEqual([d["kind"] for d in deltas], ["paraphrase", "delete"])


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
