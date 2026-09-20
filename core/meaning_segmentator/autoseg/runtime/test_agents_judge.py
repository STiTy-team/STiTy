"""`PYTHONPATH=. python -m unittest core.meaning_segmentator.autoseg.runtime.test_agents_judge`"""
import unittest

from . import agents_judge as aj

BASE = """[Role]
r

[Scoring Rules]
- combine as (1 - contradiction) x cohesion

[Core Principles]
- (1) ask this
- (2) ask that

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
            self.assertEqual([u["text"] for u in aj.edit_units(pr) if u["id"][0] == "C"],
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
        self.assertEqual([u["text"] for u in aj.edit_units(pr)
                          if u["id"][0] == "C"][0], "- (0) first")

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
        flat = " ".join(aj.writer_system(True, ["German"]).split())
        self.assertIn("ONLY place that states how the target was measured", flat)
        # PE 에게도 측정이 어디에 적혀 있는지, 그리고 그 절은 못 고친다는 것을 말한다.
        pe = " ".join(aj.engineer_system(100, 100).split())
        self.assertIn("[Scoring Rules] says what was MEASURED (cohesion, contra, target)", pe)
        self.assertIn("neither is yours to change", pe)


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
        # 상한은 80단어다. 60 이었을 때는 원칙 한 줄(질문 + 정도 축, 실측 40~57단어)이 경계에
        # 닿아 정상 편집이 걸릴 수 있었다.
        ok, bad_ok = aj.enforce_role([{"op": "replace", "id": "C1", "text": "w " * 57}], "single_small")
        self.assertEqual((len(ok), bad_ok), (1, []))
        keep, bad = aj.enforce_role([{"op": "replace", "id": "C1", "text": "w " * 81}], "single_small")
        self.assertEqual(keep, [])
        self.assertIn("81단어", bad[0]["reason"])
        keep, _ = aj.enforce_role([{"op": "replace", "id": "E1", "labeled_example": "s1"}],
                                  "single_small")
        self.assertEqual(len(keep), 1)

    def test_free(self):
        e = [{"op": "delete", "id": "C1"}, {"op": "delete", "id": "C2"}]
        self.assertEqual(aj.enforce_role(e, "free"), (e, []))


class Skeleton(unittest.TestCase):
    def test_section_order(self):
        self.assertEqual(aj.check_skeleton(BASE), [])
        with_dp = BASE.replace("[Output Rules]", "[Decision Procedure]\nd\n\n[Output Rules]")
        self.assertEqual(aj.check_skeleton(with_dp), ["허용하지 않는 섹션: [Decision Procedure]"])
        self.assertIn("섹션 없음: [Examples]", aj.check_skeleton(BASE.split("[Examples]")[0]))
        self.assertNotIn("[Decision Procedure]", aj.size_brief(BASE, len(BASE))["sections"])


class LengthTarget(unittest.TestCase):
    """모델에게 알리는 길이 목표는 검사 상한보다 낮다 — 모델은 상한을 1% 안팎 넘긴다."""

    def test_margin_below_cap(self):
        self.assertEqual(aj.soft_target(8054, 7322), int(8054 * 0.95))

    def test_never_below_current(self):
        # 천장에 닿아 상한 == 현재 길이일 때 줄이라고 요구하지 않는다
        self.assertEqual(aj.soft_target(7322, 7322), 7322)
        self.assertEqual(aj.soft_target(7400, 7322), 7322)


class RewriteShorten(unittest.TestCase):
    """Writer 가 다시 쓴 초안이 길이만 넘으면 초안 자체를 편집 단위로 쪼개 PE 에게 준다."""

    def test_shorten_user(self):
        draft = BASE.replace("- (2) ask that", "- (2) ask that " + "y" * 40)
        findings = [{"diagnosis": "d", "evidence": "e", "edit": {}}]
        u = aj.shorten_user(draft, len(BASE), findings)
        self.assertEqual([x["id"] for x in u["units"]], ["C1", "C2", "E1"])
        self.assertTrue(all(x["origin"] == "rewrite" for x in u["units"]))
        self.assertIn("[Role]", u["fixed_sections"])
        self.assertEqual(u["findings"], findings)
        self.assertEqual(u["size"]["budget"], len(BASE))
        fb = u["size_feedback"]
        self.assertEqual(fb["over_by"], len(draft) - len(BASE))
        self.assertGreater(fb["over_by"], 0)
        self.assertEqual(fb["your_edits"], [])
        self.assertEqual(fb["largest_units"][0]["id"], "C2")

    def test_pe_edits_apply_to_draft(self):
        # PE 가 초안의 id 로 낸 편집이 초안에 적용되고 상한 검사를 통과한다
        draft = BASE.replace("- (2) ask that", "- (2) ask that " + "y" * 40)
        pe = {"changelog": ["shorter"],
              "edits": [{"op": "replace", "id": "C2", "kind": "paraphrase", "text": "- (2) ask that"}]}
        pr, note, _d, deltas, skipped = aj.parse_edits(pe, draft, budget=len(BASE))
        self.assertEqual(pr, BASE)
        self.assertEqual(skipped, [])
        self.assertEqual(deltas[0]["chars_change"], -41)

    def test_shorten_role_drops_inserts(self):
        # 줄이라는 패스에서 PE 가 예시를 더 넣은 것(judge10 iter 2: E_end +262자)은 코드가 뺀다
        edits = [{"op": "delete", "id": "E2"},
                 {"op": "insert_after", "id": "E_end", "labeled_example": "x"},
                 {"op": "replace", "id": "C1", "kind": "paraphrase", "text": "- (1) ask"}]
        keep, bad = aj.enforce_role(edits, "shorten")
        self.assertEqual([e["op"] for e in keep], ["delete", "replace"])
        self.assertEqual([(b["edit"], b["id"]) for b in bad], [(1, "E_end")])
        self.assertEqual(aj.SHORTEN_ROLE, "shorten")
        self.assertNotIn(aj.SHORTEN_ROLE, aj.ROLES)   # 후보 역할이 아니라 축소 패스 전용


class PostmortemReachesNextIter(unittest.TestCase):
    """부검이 지목한 것이 다음 이터의 후보에 실제로 닿는가 — judge10 iter 2 는 세 후보 전부
    iter 1 부검이 지목한 실측 예시(en_us_733, en_us_1294)를 다시 넣었다."""

    def test_history_brief_carries_blamed(self):
        hist = [{"iter": 1, "adopted": False, "delta": {"mean": -0.01, "lo": -0.03},
                 "edits": [], "diagnosis": {"why": "w", "lesson": "L", "blamed": ["E2 (x)"]}}]
        b = aj.history_brief(hist)[0]
        self.assertEqual(b["blamed"], ["E2 (x)"])
        self.assertEqual(b["why"], "w")

    def test_edit_summary_keeps_labeled_id(self):
        edits = [{"op": "replace", "id": "E1", "labeled_example": "en_us_733", "text": "Input: W"}]
        self.assertEqual(aj.edit_summary(BASE, edits)[0]["labeled_example"], "en_us_733")
        self.assertNotIn("labeled_example", aj.edit_summary(BASE, [{"op": "delete", "id": "E1"}])[0])

    def test_exclude_examples_used_by_rejected(self):
        examples = {"a": "Input: x <SEG:?> y\nOutput: x <SEG:10> y",
                    "b": "Input: p <SEG:?> q\nOutput: p <SEG:20> q"}
        rejected = ["[Examples]\nInput: x <SEG:?> y\nOutput: x <SEG:10> y\n"]
        kept, dropped = aj.exclude_used_examples(examples, rejected)
        self.assertEqual(list(kept), ["b"])
        self.assertEqual(dropped, ["a"])
        self.assertEqual(aj.exclude_used_examples(examples, [])[0], examples)

    def test_critic_replace_target_must_exist(self):
        # Critic 이 last_revision 본문을 현재 프롬프트에 있는 줄 알고 겨눈 것(judge10 iter 2 Whistler)
        got = aj.clean_findings({"findings": [
            {"edit": {"where": "examples", "action": "replace", "target": "Input: Whistler", "text": "y"}},
            {"edit": {"where": "core_principles", "action": "replace", "target": "- (1) ask this", "text": "z"}},
            {"edit": {"where": "core_principles", "action": "add", "text": "w"}}]}, prompt=BASE)
        self.assertEqual([f["edit"]["action"] for f in got], ["replace", "add"])
        self.assertEqual(got[0]["edit"]["target"], "- (1) ask this")

    def test_blamed_ids_become_text(self):
        # 부검의 "E2 (…)" 는 그 이터 후보의 id — 다음 이터엔 다른 단위가 E2 다. 본문을 붙인다
        cand = BASE.replace("- (2) ask that", "- (2) ask that about negation scope")
        got = aj.blamed_with_text(["C2 (encouraged early cuts)", "E9 (missing)", "nonsense"], cand)
        self.assertEqual(got[0], "C2 «- (2) ask that about negation scope» (encouraged early cuts)")
        self.assertEqual(got[1], "E9 (missing)")
        self.assertEqual(got[2], "nonsense")

    def test_history_brief_keeps_changelog_for_rewrite(self):
        hist = [{"iter": 2, "candidate": 1, "adopted": False, "screened_only": True,
                 "delta": {"mean": -0.01, "lo": -0.03}, "edits": [],
                 "changelog": ["rewrite: [Core Principles]/[Examples] 를 새로 씀"]}]
        self.assertEqual(aj.history_brief(hist)[0]["changelog"], hist[0]["changelog"])
        hist[0]["edits"] = [{"op": "delete", "id": "E1"}]
        self.assertNotIn("changelog", aj.history_brief(hist)[0])

    def test_exclude_examples_already_in_prompt(self):
        examples = {"a": "Input: a <SEG:?> b\nOutput: a <SEG:50> b", "b": "Input: p\nOutput: p"}
        kept, dropped = aj.exclude_used_examples(examples, [BASE])
        self.assertEqual(dropped, ["a"])


class DedupeCandidates(unittest.TestCase):
    def test_identical_text_dropped(self):
        # judge12 iter 1: examples_only 와 single_small 이 같은 편집(E4 → en_us_738)을 냈다 — 선별 두 번은 낭비
        keep, dropped = aj.dedupe_candidates([(0, "A"), (1, "B"), (2, "A"), (3, "B ")])
        self.assertEqual(keep, [0, 1])
        self.assertEqual(dropped, [(2, 0), (3, 1)])


class NearMiss(unittest.TestCase):
    def test_near_miss_flagged_in_history_brief(self):
        h = {"iter": 1, "adopted": False, "edits": [],
             "gain": {"mean": 0.0074, "lo": -0.0003, "hi": 0.0153, "pooled": True},
             "diagnosis": {"by_bin": {"≤3": 0.0129}, "lesson": "keep"}}
        self.assertTrue(aj.is_near_miss(h))
        b = aj.history_brief([h])[0]
        self.assertTrue(b["near_miss"])
        self.assertEqual(b["measured_on"], "test-A 200 + test-B 200 pooled")
        self.assertEqual(b["by_bin"], {"≤3": 0.0129})

    def test_near_miss_not_for_screened_or_negative(self):
        base = {"iter": 1, "adopted": False, "edits": []}
        self.assertFalse(aj.is_near_miss({**base, "screened_only": True,
                                          "delta": {"mean": 0.02, "lo": -0.001}}))
        self.assertFalse(aj.is_near_miss({**base, "delta": {"mean": -0.006, "lo": -0.018}}))
        self.assertFalse(aj.is_near_miss({**base, "delta": {"mean": 0.006, "lo": -0.02}}))
        self.assertNotIn("near_miss", aj.history_brief([{**base, "delta": {"mean": -0.006, "lo": -0.018}}])[0])

    def test_history_brief_carries_vs_base(self):
        h = {"iter": 2, "adopted": False, "edits": [], "built_on": "iter 1 near miss (+0.0074)",
             "delta": {"mean": 0.0026, "lo": -0.0085, "hi": 0.014},
             "diagnosis": {"by_bin": {"≤3": 0.0047},
                           "vs_base": {"delta": {"mean": -0.0038, "lo": -0.01, "hi": 0.003},
                                       "by_bin": {"≤3": -0.0083}, "n_worse": 1, "n_better": 0}}}
        b = aj.history_brief([h])[0]
        self.assertEqual(b["built_on"], "iter 1 near miss (+0.0074)")
        self.assertEqual(b["vs_base"], {"delta_mean": -0.0038, "delta_lo": -0.01, "by_bin": {"≤3": -0.0083}})
        self.assertNotIn("near_miss", b)


class ProhibitionForm(unittest.TestCase):
    """narrow_rule 은 결속문이어야 한다 — 금지문만으로 쓰면 코드가 버린다."""

    def test_pure_prohibition_rejected(self):
        self.assertTrue(aj.is_prohibition("Do not cut immediately after a main verb."))
        self.assertTrue(aj.is_prohibition("- Avoid cutting between a determiner and its noun."))

    def test_prohibition_with_binding_clause_allowed(self):
        # judge16 채택본의 실제 문면
        self.assertFalse(aj.is_prohibition(
            "- Narrow check: do not cut between a numeral and an immediately following unit "
            "phrase that completes it; keep the number and its unit together."))

    def test_binding_allowed(self):
        self.assertFalse(aj.is_prohibition(
            "Keep a head noun together with the post-nominal modifier that identifies it."))

    def test_enforce_role_drops_prohibition(self):
        edits = [{"op": "insert_after", "id": "C3", "text": "Do not cut after a main verb."}]
        keep, bad = aj.enforce_role(edits, "narrow_rule")
        self.assertEqual(keep, [])
        self.assertIn("금지문", bad[0]["reason"])

    def test_enforce_role_keeps_binding(self):
        edits = [{"op": "insert_after", "id": "C3", "text": "Keep a numeral with its unit."}]
        keep, bad = aj.enforce_role(edits, "narrow_rule")
        self.assertEqual(len(keep), 1)
        self.assertEqual(bad, [])

    def test_findings_cap_is_configurable(self):
        blob = {"findings": [{"diagnosis": f"d{i}", "evidence": "e",
                              "type_predicate": f"p{i}",
                              "edit": {"where": "core_principles", "action": "add",
                                       "text": "t"}} for i in range(6)]}
        self.assertEqual(len(aj.clean_findings(blob, cap=4)), 4)
        self.assertEqual(aj.clean_findings(blob, cap=4)[0]["type_predicate"], "p0")


class PruneRole(unittest.TestCase):
    """빼기 후보 — 원칙 하나를 지우는 것만 허용한다. 더하기는 스물넷 연속 음수였다."""

    def test_single_delete_of_principle_passes(self):
        keep, bad = aj.enforce_role([{"op": "delete", "id": "C5"}], "prune")
        self.assertEqual(len(keep), 1)
        self.assertEqual(bad, [])

    def test_insert_is_dropped(self):
        keep, bad = aj.enforce_role(
            [{"op": "insert_after", "id": "C3", "text": "Keep a numeral with its unit."}], "prune")
        self.assertEqual(keep, [])
        self.assertIn("delete 만 된다", bad[0]["reason"])

    def test_example_unit_is_dropped(self):
        keep, bad = aj.enforce_role([{"op": "delete", "id": "E2"}], "prune")
        self.assertEqual(keep, [])
        self.assertIn("[Order Principles] 단위가 아니다", bad[0]["reason"])

    def test_second_edit_is_dropped(self):
        keep, bad = aj.enforce_role([{"op": "delete", "id": "C5"},
                                     {"op": "delete", "id": "C6"}], "prune")
        self.assertEqual([e["id"] for e in keep], ["C5"])
        self.assertIn("둘째 이후 편집", bad[0]["reason"])

    def test_prune_is_a_known_role(self):
        self.assertIn("prune", aj.ROLES)
