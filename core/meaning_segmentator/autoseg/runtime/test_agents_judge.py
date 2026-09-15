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

[Output Rules]
frozen text

[Examples]
Input: a <SEG:?> b
Output: a <SEG:50> b
"""


class Findings(unittest.TestCase):
    def test_editable_sections_only(self):
        got = aj.clean_findings({"findings": [
            {"diagnosis": "d",
             "edit": {"where": "scoring_rules", "action": "add", "text": "x"}},
            {"diagnosis": "d",
             "edit": {"where": "core_principles", "action": "add", "text": "y"}}]})
        self.assertEqual([f["edit"]["where"] for f in got], ["core_principles"])

    def test_replace_needs_target(self):
        got = aj.clean_findings({"findings": [
            {"edit": {"where": "core_principles", "action": "replace", "text": "y"}}]})
        self.assertEqual(got, [])

    def test_max_three(self):
        many = [{"edit": {"where": "examples", "action": "add", "text": f"t{i}"}} for i in range(9)]
        self.assertEqual(len(aj.clean_findings({"findings": many})), 3)

    def test_empty_input(self):
        self.assertEqual(aj.clean_findings({}), [])
        self.assertEqual(aj.clean_findings({"findings": None}), [])


class ParsePrompt(unittest.TestCase):
    def test_valid(self):
        new = BASE.replace("- (2) ask that", "- (2) ask that, and check the seam")
        pr, log = aj.parse_prompt({"prompt": new, "changelog": ["seam"]}, BASE, budget=10_000)
        self.assertEqual(pr, new)
        self.assertEqual(log, ["seam"])

    def test_frozen_changed(self):
        """[Output Rules] 는 출력 형식, [Scoring Rules] 는 측정 절차라 둘 다 라벨이 바뀔 때만 바뀐다."""
        new = BASE.replace("frozen text", "frozen text plus")
        pr, errs = aj.parse_prompt({"prompt": new}, BASE, budget=10_000)
        self.assertIsNone(pr)
        self.assertTrue(any("[Output Rules]" in e for e in errs))

    def test_scoring_rules_condition(self):
        new = BASE.replace("- combine as (1 - contradiction) x cohesion",
                           "- combine as (1 - contradiction) x cohesion\n- if comma, lower")
        pr, errs = aj.parse_prompt({"prompt": new}, BASE, budget=10_000)
        self.assertIsNone(pr)
        self.assertTrue(any("[Scoring Rules]" in e for e in errs))

    def test_missing_section(self):
        pr, errs = aj.parse_prompt({"prompt": "[Role]\nonly"}, BASE, budget=10_000)
        self.assertIsNone(pr)
        self.assertTrue(errs)

    def test_over_length(self):
        pr, errs = aj.parse_prompt({"prompt": BASE}, BASE, budget=10)
        self.assertIsNone(pr)
        self.assertTrue(any("길이 초과" in e for e in errs))

    def test_empty_output(self):
        self.assertEqual(aj.parse_prompt({}, BASE, 100)[0], None)


class Size(unittest.TestCase):
    def test_editable_cap(self):
        b = aj.size_brief(BASE, budget=len(BASE))
        editable = sum(b["sections"][h] for h in aj.EDITABLE)
        self.assertEqual(b["editable_cap"], editable)
        self.assertEqual(aj.size_brief(BASE, budget=len(BASE) + 50)["editable_cap"], editable + 50)

    def test_headroom_words(self):
        h = aj.size_brief(BASE, budget=len(BASE) + 100)["headroom"]
        self.assertEqual(h["chars"], 100)
        cpw = len(BASE) / len(BASE.split())
        self.assertEqual(h["words"], int(100 / cpw))
        units = [u["chars"] for u in aj.edit_units(BASE) if u["id"].startswith("C")]
        typical = int(sum(units) / len(units))
        self.assertEqual(h["typical_principle_words"], int(typical / cpw))
        self.assertEqual(h["principles"], round(100 / typical, 1))
        self.assertEqual(aj.size_brief(BASE, budget=len(BASE))["headroom"]["words"], 0)

    def test_only_too_long(self):
        self.assertTrue(aj.only_too_long(["길이 초과: 10 > 5"]))
        self.assertFalse(aj.only_too_long(["길이 초과: 10 > 5", "동결 섹션이 바뀌었다: [Output Rules]"]))
        self.assertFalse(aj.only_too_long([]))


class Edits(unittest.TestCase):
    def test_units(self):
        u = {x["id"]: x for x in aj.edit_units(BASE)}
        self.assertEqual(sorted(u), ["C1", "C2", "E1"])
        self.assertEqual(u["C1"]["text"], "- (1) ask this")
        self.assertEqual(u["E1"]["text"], "Input: a <SEG:?> b\nOutput: a <SEG:50> b")
        self.assertEqual(u["C2"]["chars"], len("- (2) ask that"))

    def test_replace_delete_insert(self):
        pr, note, _draft, deltas, _skipped = aj.parse_edits({"edits": [
            {"op": "replace", "id": "C2", "text": "- (2) check the seam"},
            {"op": "delete", "id": "C1"},
            {"op": "insert_after", "id": "E1", "text": "Input: c <SEG:?> d\nOutput: c <SEG:10> d"}],
            "changelog": ["seam"]}, BASE, budget=10_000)
        self.assertIsNotNone(pr, note)
        self.assertNotIn("ask this", pr)
        self.assertEqual([x["id"] for x in aj.edit_units(pr)], ["C1", "E1", "E2"])
        self.assertEqual(aj.edit_units(pr)[0]["text"], "- (2) check the seam")
        for h in ("[Role]", "[Scoring Rules]", "[Output Rules]"):
            self.assertEqual(aj.section_of(pr, h), aj.section_of(BASE, h))
        self.assertEqual(deltas[1]["chars_change"], -(len("- (1) ask this") + 1))
        self.assertEqual(note, ["seam"])

    def test_same_text_noop(self):
        """재조립이 공백을 바꾸면 아무것도 안 고친 편집도 길이 상한에 걸린다."""
        for base in (BASE, BASE.rstrip("\n")):
            same = [{"op": "replace", "id": u["id"], "text": u["text"]} for u in aj.edit_units(base)]
            pr, note, _d, _deltas, _sk = aj.parse_edits({"edits": same}, base, budget=len(base))
            self.assertEqual(pr, base, note)

    def test_strip_id_prefix(self):
        pr, *_ = aj.parse_edits({"edits": [
            {"op": "insert_after", "id": "C2", "text": "C11: - new rule"},
            {"op": "insert_after", "id": "E1", "text": "E2) Input: c <SEG:?> d\nOutput: c <SEG:9> d"}]},
            BASE, 10_000)
        texts = [u["text"] for u in aj.edit_units(pr)]
        self.assertIn("- new rule", texts)
        self.assertIn("Input: c <SEG:?> d\nOutput: c <SEG:9> d", texts)

    def test_insert_end(self):
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

    def test_id_past_last(self):
        pr, errs, _d, _de, skipped = aj.parse_edits(
            {"edits": [{"op": "insert_after", "id": "C9", "text": "- (9) ask"}]}, BASE, 10_000)
        self.assertIsNone(pr)
        self.assertTrue(any("C9" in s["reason"] for s in skipped), skipped)

    def test_insert_top(self):
        pr, *_ = aj.parse_edits({"edits": [{"op": "insert_after", "id": "C0",
                                             "text": "- (0) first"}]}, BASE, 10_000)
        self.assertEqual(aj.edit_units(pr)[0]["text"], "- (0) first")

    def test_all_invalid(self):
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

    def test_skip_invalid_only(self):
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

    def test_length_feedback(self):
        edits = [{"op": "insert_after", "id": "C2", "text": "- " + "x" * 40}]
        pr, errs, draft, deltas, _sk = aj.parse_edits({"edits": edits}, BASE, budget=len(BASE))
        self.assertIsNone(pr)
        self.assertTrue(aj.only_too_long(errs))
        fb = aj.edit_feedback(draft, BASE, len(BASE), deltas)
        self.assertEqual(fb["over_by"], len(draft) - len(BASE))
        self.assertGreater(fb["over_by"], 0)
        self.assertGreaterEqual(fb["over_by_words"] * int(len(BASE) / len(BASE.split())), fb["over_by"])
        self.assertEqual(fb["your_edits"][0]["id"], "C2")


class Provenance(unittest.TestCase):
    GAIN = {"mean": 0.0166, "lo": 0.0025, "hi": 0.0311}

    def test_init_v0(self):
        prov = aj.init_provenance(BASE)
        self.assertEqual(len(prov), 3)
        self.assertTrue(all(r["origin"] == "v0" and r["adopted_delta"] is None
                            for r in prov.values()))

    def test_adopt(self):
        prov = aj.init_provenance(BASE)
        prov["- (1) ask this"]["origin"] = "iter 2"
        new = BASE.replace("- (2) ask that", "- (2) check the seam")
        got = aj.adopt_provenance(prov, new, 4, self.GAIN)
        self.assertEqual(got["- (1) ask this"]["origin"], "iter 2")
        self.assertEqual(got["- (2) check the seam"],
                         {"origin": "iter 4", "adopted_delta": 0.0166, "adopted_ci_lo": 0.0025})
        self.assertNotIn("- (2) ask that", got)

    def test_units_with_provenance(self):
        prov = aj.init_provenance(BASE)
        u = aj.units_with_provenance(BASE, prov)
        self.assertEqual({k for k in u[0]} >= {"id", "chars", "text", "origin", "adopted_ci_lo"},
                         True)

    def test_history_brief(self):
        hist = [{"iter": 1, "candidate": 1, "adopted": False, "screened_out": True,
                 "delta": {"mean": -0.01, "lo": -0.03, "hi": 0.01, "n": 410},
                 "edits": [{"op": "replace", "id": "C2"}], "changelog": ["x"]},
                {"iter": 1, "adopted": False, "delta": {"mean": 0.01, "lo": -0.004, "hi": 0.02},
                 "gain": {"mean": 0.008, "lo": 0.001, "hi": 0.016, "pooled": True},
                 "findings": ["long"], "edits": [], "diagnosis": {"why": "w", "lesson": "L"}},
                {"iter": 2, "adopted": False, "reason": ["길이 초과"], "edits": []}]
        b = aj.history_brief(hist)
        self.assertEqual(b[0], {"iter": 1, "adopted": False, "delta_mean": -0.01, "delta_lo": -0.03,
                                "edits": [{"op": "replace", "id": "C2"}], "candidate": 1,
                                "measured_on": "screen subset, lost to a sibling"})
        self.assertEqual((b[1]["delta_mean"], b[1]["lesson"]), (0.008, "L"))
        self.assertNotIn("findings", b[1])
        self.assertEqual(b[2]["rejected_before_measuring"], ["길이 초과"])

    def test_edit_summary(self):
        got = aj.edit_summary(BASE, [{"op": "replace", "id": "C2", "text": "- (2) new"},
                                     {"op": "delete", "id": "E1"}, "junk"])
        self.assertEqual(got[0], {"op": "replace", "id": "C2", "kind": "change",
                                  "was": "- (2) ask that", "now": "- (2) new"})
        self.assertEqual((got[1]["kind"], got[1]["was"][:6]), ("delete", "Input:"))
        self.assertEqual(len(got), 2)

    def test_paraphrase_kind(self):
        _pr, _n, _d, deltas, _sk = aj.parse_edits({"edits": [
            {"op": "replace", "id": "C1", "kind": "paraphrase", "text": "- (1) ask"},
            {"op": "delete", "id": "C2"}]}, BASE, 10_000)
        self.assertEqual([d["kind"] for d in deltas], ["paraphrase", "delete"])


class MeasurementInOnePlace(unittest.TestCase):
    """측정 절차는 [Scoring Rules] 에만 쓴다 — [Output Rules] 에 옛 라벨 서술이 남으면
    한 프롬프트 안에 목표 설명이 둘이 된다 (judge01 v0 에서 실제로 났다)."""

    def test_output_rules_no_measurement(self):
        from . import agents_distill as ad
        s = ad.output_rules(True, "source", meaning_in_scoring_rules=True)
        self.assertIn("Scoring Rules section", s)
        for old in ("quality_before", "entailment", "MEASURED"):
            self.assertNotIn(old, s)

    def test_output_rules_no_header(self):
        """본문에 "[Scoring Rules]" 가 있으면 섹션 경계로 잡혀 replace_section 이 꼬리를 복제한다."""
        from . import agents_distill as ad
        s = ad.output_rules(True, "source", meaning_in_scoring_rules=True)
        body = s.split("\n", 1)[1]
        for h in ad.SECTIONS:
            self.assertNotIn(h, body)
        once = aj.replace_section(BASE, "[Output Rules]", s)
        self.assertEqual(aj.replace_section(once, "[Output Rules]", s), once)
        self.assertEqual(aj.section_of(once, "[Output Rules]"), s.strip())

    def test_scoring_rules_is_the_place(self):
        self.assertIn("ONLY place that states how the target was measured",
                      aj.writer_system(True, ["German"]))
        self.assertIn("[Scoring Rules] states how the target was measured",
                      aj.engineer_system(100, 100))


if __name__ == "__main__":
    unittest.main()


class LabeledExampleRefs(unittest.TestCase):
    def test_resolve(self):
        ex = {"s1": "Input: c <SEG:?> d\nOutput: c <SEG:10> d"}
        edits, bad = aj.resolve_labeled_examples(
            [{"op": "replace", "id": "E1", "labeled_example": "s1"},
             {"op": "insert_after", "id": "E_end", "labeled_example": "zz"},
             {"op": "replace", "id": "C1", "text": "- kept"}], ex)
        self.assertEqual([e["id"] for e in edits], ["E1", "C1"])
        self.assertEqual(edits[0]["text"], ex["s1"])
        self.assertEqual(edits[0]["kind"], "example")
        self.assertEqual(bad[0]["reason"], "없는 labeled_example 'zz'")
        pr, _n, _d, _dl, _s = aj.parse_edits({"edits": edits}, BASE, budget=len(BASE) + 100)
        self.assertIn(ex["s1"], pr)
        self.assertNotIn("a <SEG:50> b", pr)

    def test_text_wins(self):
        edits, bad = aj.resolve_labeled_examples(
            [{"op": "replace", "id": "E1", "text": "Input: x\nOutput: x", "labeled_example": "s1"}],
            {})
        self.assertEqual(edits[0]["text"], "Input: x\nOutput: x")
        self.assertEqual(bad, [])


class CandidateRoles(unittest.TestCase):
    def test_examples_only(self):
        keep, bad = aj.enforce_role([{"op": "replace", "id": "C1", "text": "x"},
                                     {"op": "replace", "id": "E1", "labeled_example": "s1"}],
                                    "examples_only")
        self.assertEqual([e["id"] for e in keep], ["E1"])
        self.assertEqual(bad[0]["id"], "C1")

    def test_single_small(self):
        keep, bad = aj.enforce_role([{"op": "replace", "id": "C1", "text": "- short"},
                                     {"op": "delete", "id": "C2"}], "single_small")
        self.assertEqual(len(keep), 1)
        self.assertEqual(bad[0]["reason"], "single_small 인데 둘째 이후 편집")
        keep, bad = aj.enforce_role([{"op": "replace", "id": "C1", "text": "w " * 61}], "single_small")
        self.assertEqual(keep, [])
        self.assertIn("61단어", bad[0]["reason"])
        keep, _ = aj.enforce_role([{"op": "replace", "id": "E1", "labeled_example": "s1"}],
                                  "single_small")
        self.assertEqual(len(keep), 1)

    def test_free(self):
        e = [{"op": "delete", "id": "C1"}, {"op": "delete", "id": "C2"}]
        self.assertEqual(aj.enforce_role(e, "free"), (e, []))


class Skeleton(unittest.TestCase):
    def test_five_sections(self):
        self.assertEqual(aj.check_skeleton(BASE), [])
        with_dp = BASE.replace("[Output Rules]", "[Decision Procedure]\nd\n\n[Output Rules]")
        self.assertEqual(aj.check_skeleton(with_dp), ["허용하지 않는 섹션: [Decision Procedure]"])
        self.assertIn("섹션 없음: [Examples]", aj.check_skeleton(BASE.split("[Examples]")[0]))
        self.assertNotIn("[Decision Procedure]", aj.size_brief(BASE, len(BASE))["sections"])
