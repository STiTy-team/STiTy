"""`PYTHONPATH=. python -m unittest core.meaning_segmentator.autoseg.test_loop_judge`"""
import unittest

from . import loop_judge as lj
from .runtime import data


def sent(i, text):
    return data.Sentence(id=f"s{i}", text=text)


def labels_for(texts, contra, coh):
    """타깃 하나짜리 라벨 — `adq_l`/`adq_r` 자리에 cohesion 이 들어간다 (run25 형식)."""
    per = []
    for i, t in enumerate(texts):
        n = len(t.split()) - 1
        per.append({"id": f"s{i}", "contra": [contra(i, j) for j in range(1, n + 1)],
                    "ent": [0.0] * n, "contra_floor": [0.0] * n,
                    "adq_l": [coh(i, j) for j in range(1, n + 1)],
                    "adq_r": [coh(i, j) for j in range(1, n + 1)],
                    "hyp_units": list(range(1, n + 1))})
    return {"X": per}


class Promising(unittest.TestCase):
    def test_promising(self):
        self.assertTrue(lj.promising({"mean": 0.0105, "lo": -0.0045}))
        self.assertFalse(lj.promising({"mean": 0.003, "lo": -0.004}))
        self.assertFalse(lj.promising({"mean": 0.02, "lo": -0.02}))


class Decide(unittest.TestCase):
    def test_accept(self):
        self.assertEqual(lj.decide({"lo": 0.02}), "accept")

    def test_confirm(self):
        self.assertEqual(lj.decide({"lo": 0.002}), "confirm")

    def test_reject(self):
        self.assertEqual(lj.decide({"lo": 0.0}), "reject")
        self.assertEqual(lj.decide({"lo": -0.01}), "reject")


class Sets(unittest.TestCase):
    def setUp(self):
        self.texts = ["a b c d e f g h i j k l"]
        self.sents = [sent(0, self.texts[0])]

    def test_oracle_top_k(self):
        lab = labels_for(self.texts, lambda i, j: 0.0, lambda i, j: 1.0 if j == 6 else 0.1)
        got = lj.oracle_sets(lab, self.sents, spaced=True, min_gap=1)
        self.assertEqual(got[(0, 1)], (6,))

    def test_k_from_one(self):
        """T 격자와 달리 문장마다 k 가 1..kmax 로 전부 들어온다."""
        lab = labels_for(self.texts, lambda i, j: 0.0, lambda i, j: 0.1 * j)
        got = lj.oracle_sets(lab, self.sents, spaced=True, min_gap=1)
        ks = sorted(k for _i, k in got)
        self.assertEqual(ks, [1, 2, 3, 4, 5])          # 12어절 / 평균 조각 2 이상
        self.assertEqual(len(got[(0, 3)]), 3)

    def test_contra_pushes_down(self):
        lab = labels_for(self.texts, lambda i, j: 0.99 if j == 6 else 0.0,
                         lambda i, j: 1.0 if j in (6, 9) else 0.1)
        got = lj.oracle_sets(lab, self.sents, spaced=True, min_gap=1)
        self.assertEqual(got[(0, 1)], (9,))

    def test_policy_from_rows(self):
        rows = [{"id": "s0", "positions": [3, 6, 9], "scores": [10, 90, 20]}]
        got = lj.policy_sets(rows, self.sents, spaced=True, min_gap=1)
        self.assertEqual(got[(0, 1)], (6,))
        self.assertEqual(got[(0, 2)], (6, 9))

    def test_skip_unscored(self):
        rows = [{"id": "s0", "positions": [3], "scores": None}]
        self.assertEqual(lj.policy_sets(rows, self.sents, True, 1), {})


class SearchSets(unittest.TestCase):
    def test_best_per_t(self):
        import json, tempfile
        from pathlib import Path
        blob = {"0": {"6": {"greedy": [3], "sets": {"3": 0.70, "6": 0.81, "3,9": 0.66}},
                      "4": {"greedy": [3, 9], "sets": {"3,9": 0.55}}},
                "9": {"6": {"greedy": [3], "sets": {"3": 0.90}}}}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "set_scores_dev.json"
            p.write_text(json.dumps(blob), encoding="utf-8")
            sets, vals = lj.search_sets(p, [4, 6], limit=5)
        self.assertEqual(sets[(0, 6)], (6,))
        self.assertAlmostEqual(vals[(0, 6)], 0.81)
        self.assertEqual(sets[(0, 4)], (3, 9))
        self.assertNotIn((9, 6), sets)          # limit 밖 문장은 뺀다

    def test_t_outside_grid(self):
        import json, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.json"
            p.write_text(json.dumps({"0": {"12": {"greedy": [3], "sets": {"3": 0.5}}}}),
                         encoding="utf-8")
            sets, _ = lj.search_sets(p, [4, 6], limit=5)
        self.assertEqual(sets, {})


class Latency(unittest.TestCase):
    def test_bins_by_chunk(self):
        """12어절 문장: k=1 이면 조각 6.0 (≤7 칸), k=5 면 2.0 (≤3 칸)."""
        sents = [sent(0, " ".join(["w"] * 12))]
        got = lj.by_latency(sents, {(0, 1): 0.8, (0, 5): 0.4}, spaced=True)
        self.assertEqual(got, {"≤3": 0.4, "≤7": 0.8})


class ContraReader(unittest.TestCase):
    """`H_set` 의 모순 항을 어느 타깃에서 읽는가 — 소스 NLI 와 번역 NLI 에서 규칙이 다르다."""

    def lab(self, per_target, source_mode):
        """타깃별 contra 벡터 하나씩. 위치는 j=1,2 둘뿐이다."""
        extra = {"contra_source": "source"} if source_mode else {}
        return {tgt: [{"id": "s0", "contra": vec, "ent": [0.0] * len(vec),
                       "contra_floor": [0.0] * len(vec), "adq_l": [0.5] * len(vec),
                       "adq_r": [0.5] * len(vec), "hyp_units": [1] * len(vec), **extra}]
                for tgt, vec in per_target.items()}

    def test_source_mode_reads_one_target(self):
        """소스 모드는 네 타깃에 같은 값이 복사돼 있다 — 첫 타깃만 읽는다.

        값이 갈린 라벨을 일부러 넣어 규칙 자체를 고정한다. 평균이면 0.4 가 나온다.
        """
        lab = self.lab({"Chinese": [0.2, 0.9], "German": [0.6, 0.1]}, source_mode=True)
        f = lj.contra_reader(lab, ["Chinese", "German"])
        self.assertAlmostEqual(f(0, 1), 0.2)
        self.assertAlmostEqual(f(0, 2), 0.9)

    def test_source_mode_follows_targets_order_not_dict_order(self):
        lab = self.lab({"Chinese": [0.2, 0.9], "German": [0.6, 0.1]}, source_mode=True)
        f = lj.contra_reader(lab, ["German", "Chinese"])
        self.assertAlmostEqual(f(0, 1), 0.6)

    def test_translation_mode_means_over_targets(self):
        """번역 NLI 는 타깃마다 값이 갈린다 — cohesion·라벨과 같은 규칙(타깃 평균)으로 모은다."""
        lab = self.lab({"Chinese": [0.2, 0.9], "German": [0.6, 0.1]}, source_mode=False)
        f = lj.contra_reader(lab, ["Chinese", "German"])
        self.assertAlmostEqual(f(0, 1), 0.4)
        self.assertAlmostEqual(f(0, 2), 0.5)

    def test_label_files_without_the_field_are_translation_mode(self):
        """`contra_source` 가 붙기 전 라벨 파일은 번역 NLI 다 — `labels.contra_source_of` 기본값."""
        lab = self.lab({"Chinese": [0.2], "German": [0.6]}, source_mode=False)
        for per in lab.values():
            per[0].pop("contra_source", None)
        self.assertAlmostEqual(lj.contra_reader(lab, ["Chinese", "German"])(0, 1), 0.4)

    def test_single_target_is_the_same_either_way(self):
        for mode in (True, False):
            f = lj.contra_reader(self.lab({"Chinese": [0.3]}, source_mode=mode), ["Chinese"])
            self.assertAlmostEqual(f(0, 1), 0.3)


class Cases(unittest.TestCase):
    def setUp(self):
        self.texts = ["a b c d e f g h i j k l"]
        self.sents = [sent(0, self.texts[0])]
        self.lab = labels_for(self.texts, lambda i, j: 0.0,
                              lambda i, j: {3: 0.9, 6: 0.2, 9: 0.8}.get(j, 0.1))

    def test_no_loss_no_cases(self):
        pol = {(0, 1): (3,)}
        got = lj.build_cases(self.sents, self.lab, pol, pol, {(0, 1): 0.7}, {(0, 1): 0.7},
                             True, 1, 5)[0]
        self.assertEqual(got, [])

    def test_dropped_and_added(self):
        pol, ora = {(0, 1): (6,)}, {(0, 1): (3,)}
        got = lj.build_cases(self.sents, self.lab, pol, ora, {(0, 1): 0.60}, {(0, 1): 0.75},
                             True, 1, 5)[0]
        self.assertEqual(len(got), 1)
        c = got[0]
        self.assertEqual(c["cuts"], 1)
        self.assertAlmostEqual(c["avg_chunk"], 6.0)
        self.assertEqual([d["pos"] for d in c["diff"]["dropped"]], [6])
        self.assertEqual([d["pos"] for d in c["diff"]["added"]], [3])
        self.assertAlmostEqual(c["gap"], 0.15, places=4)
        self.assertIn("‖", c["policy"]["text"])

    def test_cut_by_loss(self):
        texts = ["a b c d e f g h i j k l"] * 3
        sents = [sent(i, t) for i, t in enumerate(texts)]
        lab = labels_for(texts, lambda i, j: 0.0, lambda i, j: 0.5)
        pol = {(i, 1): (6,) for i in range(3)}
        ora = {(i, 1): (3,) for i in range(3)}
        pol_h = {(0, 1): 0.5, (1, 1): 0.1, (2, 1): 0.3}
        ora_h = {(0, 1): 0.6, (1, 1): 0.9, (2, 1): 0.5}
        got = lj.build_cases(sents, lab, pol, ora, pol_h, ora_h, True, 1, 2)[0]
        self.assertEqual([c["id"] for c in got], ["s1", "s2"])


class PickCases(unittest.TestCase):
    def test_round_robin_bins(self):
        gaps = [(0.9, (0, 1)), (0.8, (1, 1)), (0.7, (0, 2)), (0.2, (2, 5)), (0.1, (3, 5)),
                (-0.1, (4, 5))]
        bin_of = lambda key: "≤3" if key[1] >= 5 else "≤10"
        got = lj.pick_cases(gaps, bin_of, 10)
        self.assertEqual([k for _g, k in got], [(2, 5), (0, 1), (3, 5), (1, 1)])
        self.assertEqual([k for _g, k in lj.pick_cases(gaps, bin_of, 3)],
                         [(2, 5), (0, 1), (3, 5)])

    def test_contra_kill_flag(self):
        texts = ["a b c d e f g h i j k l"]
        lab = labels_for(texts, lambda i, j: 0.9 if j == 6 else 0.0, lambda i, j: 0.8)
        got = lj.build_cases([sent(0, texts[0])], lab, {(0, 1): (6,)}, {(0, 1): (3,)},
                             {(0, 1): 0.05}, {(0, 1): 0.8}, True, 1, 5)[0]
        self.assertTrue(got[0]["contra_kill"])
        self.assertAlmostEqual(got[0]["policy_worst_contra"], 0.9)
        self.assertEqual(got[0]["latency_bin"], "≤7")


class RevisionDiagnosis(unittest.TestCase):
    def test_bins_and_movers(self):
        sents = [sent(i, " ".join(["w"] * 12)) for i in range(3)]
        old_h = {(0, 1): 0.8, (1, 2): 0.6, (2, 5): 0.4}
        new_h = {(0, 1): 0.75, (1, 2): 0.66, (2, 5): 0.5}
        sets = {(0, 1): (6,), (1, 2): (4, 8), (2, 5): (2, 4, 6, 8, 10)}
        d = lj.revision_diagnosis(sents, old_h, new_h, sets, sets, True)
        self.assertEqual(d["n_worse"], 1)
        self.assertEqual(d["n_better"], 2)
        self.assertEqual(list(d["by_bin"]), ["≤3", "≤5", "≤7"])      # 구간은 짧은 쪽부터
        self.assertEqual((d["worst"][0]["id"], d["worst"][0]["delta"]), ("s0", -0.05))
        self.assertEqual(d["best"][0]["id"], "s2")
        self.assertIn("‖", d["worst"][0]["after"])


class Violations(unittest.TestCase):
    def test_by_rule(self):
        rows = [{"id": "a", "valid": True, "first_pass": True, "violations": []},
                {"id": "b", "valid": False, "first_pass": False,
                 "violations": ["too_few_tags", "text_modified"]},
                {"id": "c", "valid": True, "first_pass": False, "violations": ["too_few_tags"]}]
        got = lj.violation_summary(rows)
        self.assertEqual((got["n_rows"], got["valid"], got["first_pass"]), (3, 2, 1))
        self.assertEqual(list(got["by_rule"]), ["too_few_tags", "text_modified"])
        self.assertEqual(got["by_rule"]["too_few_tags"], {"n": 2, "ids": ["b", "c"]})


class FinalReport(unittest.TestCase):
    def test_report(self):
        md = lj.final_report(
            "judge08",
            {"prompt": 0.61, "v0": 0.59, "oracle": 0.67},
            {"prompt": {"≤3": 0.45, "≤10": 0.77}, "v0": {"≤3": 0.44, "≤10": 0.76},
             "oracle": {"≤3": 0.5, "≤10": 0.8}},
            {"mean": 0.02, "lo": 0.005, "hi": 0.035, "n": 1236, "n_clusters": 100},
            [{"iter": 1, "adopted": True, "gain": {"mean": 0.02, "lo": 0.005, "hi": 0.035},
              "diagnosis": {"why": "짧은 조각에서 올랐다"}},
             {"iter": 2, "adopted": False, "reason": ["길이 초과: 9 > 8"]},
             {"iter": 3, "candidate": 1, "adopted": False, "screened_out": True,
              "delta": {"mean": -0.01, "lo": -0.03, "hi": 0.01}}],
            12.34, 1.0)
        self.assertIn("judge08", md)
        self.assertIn("| ≤3 |", md)
        self.assertIn("+0.0200", md)
        self.assertIn("반려", md)
        self.assertIn("선별 탈락 (후보 1)", md)
        self.assertIn("$12.34", md)


class CheckpointVerdict(unittest.TestCase):
    H = {(i, k): 0.5 + 0.01 * ((i * 7 + k) % 5) for i in range(30) for k in (1, 2, 3)}

    def prev(self, h):
        return {"iter": 2, "value": sum(h.values()) / len(h), "prompt": "P",
                "h": [[i, k, v] for (i, k), v in h.items()]}

    def test_first_saves(self):
        self.assertEqual(lj.checkpoint_verdict(None, self.H), ("save", None))

    def test_noise_no_rollback(self):
        noisy = {key: v - 0.0024 + 0.01 * ((key[0] % 3) - 1) for key, v in self.H.items()}
        verdict, boot = lj.checkpoint_verdict(self.prev(self.H), noisy)
        self.assertEqual(verdict, "save")
        self.assertLess(boot["mean"], 0)

    def test_clear_drop_rollback(self):
        worse = {key: v - 0.05 for key, v in self.H.items()}
        self.assertEqual(lj.checkpoint_verdict(self.prev(self.H), worse)[0], "rollback")

    def test_no_pairs_point_compare(self):
        old = {"iter": 2, "value": 0.6, "prompt": "P"}
        self.assertEqual(lj.checkpoint_verdict(old, self.H), ("rollback", None))


class LengthCap(unittest.TestCase):
    def test_growth(self):
        self.assertEqual(lj.length_cap(8652, 8664, 0.05, 1.3), int(8652 * 1.05))

    def test_headroom_after_shrink(self):
        """시작 길이 고정이면 채택 후 여유가 0 이 됐다 — 증가율은 줄어든 길이에서도 여유를 준다."""
        cap = lj.length_cap(8318, 8664, 0.05, 1.3)
        self.assertGreater(cap - 8318, 400)

    def test_ceiling(self):
        self.assertEqual(lj.length_cap(11000, 8664, 0.05, 1.3), int(8664 * 1.3))


class State(unittest.TestCase):
    def test_roundtrip(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / lj.STATE_FILE
            ck = {"iter": 2, "value": 0.61, "prompt": "P2"}
            prov = {"- rule": {"origin": "v0", "adopted_delta": None, "adopted_ci_lo": None}}
            lj.save_state(p, 2, "P3", [{"iter": 1, "adopted": False}], ck, 9000, 12.5, prov)
            got = lj.load_state(p)
            self.assertEqual((got["done"], got["prompt"], got["checkpoint"], got["v0_len"],
                              got["provenance"]), (2, "P3", ck, 9000, prov))
            self.assertEqual([x.name for x in Path(d).iterdir()], [lj.STATE_FILE])

    def test_missing(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(lj.load_state(Path(d) / lj.STATE_FILE))

    def test_prior_spend(self):
        """실행마다 게이트웨이 누적이 0 에서 다시 시작한다 — 합이 아니라 런 누적의 최댓값."""
        import json, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            (r / "iter_00").mkdir()
            (r / "iter_00" / "metrics.json").write_text(json.dumps({"usage": {"cost": 7.4}}))
            (r / "iter_02").mkdir()
            (r / "iter_02" / "metrics.json").write_text(
                json.dumps({"usage": {"cost": 3.0}, "run_total_cost": 10.4}))
            self.assertAlmostEqual(lj.prior_spend(r), 10.4)
            lj.save_state(r / lj.STATE_FILE, 2, "P", [], None, 1, 11.0, {})
            self.assertAlmostEqual(lj.prior_spend(r), 11.0)


if __name__ == "__main__":
    unittest.main()


class LabeledExamples(unittest.TestCase):
    def test_pairs_from_labels(self):
        sents = [sent(0, "a b c d e f"), sent(1, "p q r s t u")]
        # s0: 위치 3 이 가장 좋고(contra 0, coh 1) 위치 1 이 가장 나쁘다
        lab = labels_for([s.text for s in sents],
                         lambda i, j: 0.9 if j == 1 else 0.0,
                         lambda i, j: 1.0 if j == 3 else 0.5)
        ex = lj.labeled_examples(sents, lab, [{"id": "s0"}, {"id": "없음"}], True, 1)
        self.assertEqual(list(ex), ["s0"])
        inp, out = ex["s0"].split("\n")
        self.assertTrue(inp.startswith("Input: a <SEG:?> b <SEG:?> c <SEG:?> d <SEG:?> e <SEG:?> f"))
        self.assertTrue(out.startswith("Output: a <SEG:0> b"))          # 최악 자리 0
        self.assertIn("c <SEG:100> d", out)                              # 최선 자리 100
        self.assertEqual(out.count("<SEG:"), 5)

    def test_min_gap(self):
        sents = [sent(0, "a b c d e f")]
        lab = labels_for([sents[0].text], lambda i, j: 0.0, lambda i, j: 0.5)
        inp, out = lj.labeled_examples(sents, lab, [{"id": "s0"}], True, 2)["s0"].split("\n")
        self.assertEqual(inp.count("<SEG:?>"), 3)
        self.assertEqual(out.count("<SEG:"), 3)
        self.assertTrue(inp.startswith("Input: a b <SEG:?> c"))


class ExampleSentences(unittest.TestCase):
    def test_find_in_prompt(self):
        sents = [sent(0, "a b c d e f"), sent(1, "p q r s t u"), sent(2, "x y z")]
        pr = ("[Role]\nr\n\n[Scoring Rules]\ns\n\n[Core Principles]\n- one\n\n[Examples]\n"
              "Input: a <SEG:?> b <SEG:?> c <SEG:?> d <SEG:?> e <SEG:?> f\n"
              "Output: a <SEG:1> b <SEG:2> c <SEG:3> d <SEG:4> e <SEG:5> f\n\n"
              "Input: hand <SEG:?> written\nOutput: hand <SEG:50> written\n\n"
              "[Output Rules]\no\n")
        self.assertEqual(lj.example_sentences(pr, sents), {0})
        self.assertEqual(lj.example_sentences(pr, sents[1:]), set())


class ScreenIndices(unittest.TestCase):
    def test_rotates_with_fixed_seed(self):
        a = lj.screen_indices(150, 50, 1)
        b = lj.screen_indices(150, 50, 2)
        self.assertEqual(len(a), 50)
        self.assertEqual(a, sorted(a))
        self.assertNotEqual(a, b)
        self.assertEqual(a, lj.screen_indices(150, 50, 1))
        self.assertEqual(lj.screen_indices(30, 50, 1), list(range(30)))


class CaseRotation(unittest.TestCase):
    def test_exclude_previous_sentences(self):
        # 채택이 없으면 손해 순위가 같아 같은 12문장이 또 뽑힌다(judge10 iter 1·2 동일) — 직전 이터 문장은 뺀다
        gaps = [(0.9, (0, 1)), (0.8, (1, 1)), (0.7, (2, 2)), (0.2, (3, 5))]
        bin_of = lambda key: "≤3" if key[1] >= 5 else "≤10"
        got = [k for _g, k in lj.pick_cases(gaps, bin_of, 3, exclude={0, 3})]
        self.assertEqual(got, [(1, 1), (2, 2)])
        self.assertEqual(len(lj.pick_cases(gaps, bin_of, 3)), 3)


class V0Examples(unittest.TestCase):
    def test_spread_by_length(self):
        class S:
            def __init__(self, i, n): self.id, self.text = f"s{i}", " ".join(["w"] * n)
        sents = [S(i, n) for i, n in enumerate([5, 30, 12, 8, 20, 15, 25, 10])]
        ids = lj.pick_example_ids(sents, spaced=True, n=4)
        self.assertEqual(len(ids), 4)
        by_id = {s.id: s for s in sents}
        lens = [len(by_id[i].text.split()) for i in ids]
        self.assertEqual(lens, sorted(lens))
        self.assertLess(min(lens), 12)
        self.assertGreater(max(lens), 20)
        self.assertEqual(ids, lj.pick_example_ids(sents, spaced=True, n=4))

    def test_examples_section_from_measured(self):
        body = lj.examples_section({"a": "Input: x <SEG:?> y\nOutput: x <SEG:50> y",
                                    "b": "Input: p <SEG:?> q\nOutput: p <SEG:10> q"})
        self.assertTrue(body.startswith("[Examples]\n"))
        self.assertEqual(body.count("Input:"), 2)
        self.assertIn("\n\n", body)


class LossByBin(unittest.TestCase):
    def test_share_and_quota(self):
        # test-A 실측: 손실의 61% 가 ≤3, 26% 가 ≤5 — 사례 균등 배분(3/3/2/2/2)은 그 몫을 못 본다
        gaps = [(0.2, (0, 5)), (0.2, (1, 5)), (0.2, (2, 5)), (0.1, (3, 2)), (0.02, (4, 1)), (-0.1, (5, 1))]
        bin_of = lambda key: {5: "≤3", 2: "≤5", 1: "≤99"}[key[1]]
        lb = lj.loss_by_bin(gaps, bin_of)
        self.assertEqual(lb["≤3"]["pairs"], 3)
        self.assertAlmostEqual(lb["≤3"]["loss_share"], 0.6 / 0.72, places=3)
        self.assertAlmostEqual(lb["≤99"]["gap_mean"], (0.02 - 0.1) / 2, places=6)
        q = lj.case_quota(lb, 12)
        self.assertEqual(sum(q.values()), 12)
        self.assertGreaterEqual(min(q.values()), 1)
        self.assertGreater(q["≤3"], q["≤5"])

    def test_pick_with_quota(self):
        gaps = [(0.9, (0, 5)), (0.8, (1, 5)), (0.7, (2, 5)), (0.6, (3, 2)), (0.5, (4, 2)), (0.1, (5, 1))]
        bin_of = lambda key: {5: "≤3", 2: "≤5", 1: "≤99"}[key[1]]
        got = [k for _g, k in lj.pick_cases(gaps, bin_of, 4, quota={"≤3": 2, "≤5": 1, "≤99": 1})]
        self.assertEqual(sorted(got), [(0, 5), (1, 5), (3, 2), (5, 1)])


class NearMissBase(unittest.TestCase):
    def test_picks_highest_lo_since_last_adoption(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            rd = Path(d)
            for it, files in ((1, {"prompt.txt": "P1", "candidate_3.txt": "C13"}),
                              (2, {"prompt.txt": "P2"})):
                (rd / f"iter_{it:02d}").mkdir()
                for fn, body in files.items():
                    (rd / f"iter_{it:02d}" / fn).write_text(body, encoding="utf-8")
            hist = [
                {"iter": 1, "candidate": 3, "adopted": False, "full_scored": True,
                 "delta": {"mean": 0.0059, "lo": -0.0064, "hi": 0.0188}},
                {"iter": 1, "adopted": False, "delta": {"mean": 0.0064, "lo": -0.006, "hi": 0.018},
                 "gain": {"mean": 0.0074, "lo": -0.0003, "hi": 0.0153, "pooled": True},
                 "diagnosis": {"by_bin": {}}},
                {"iter": 2, "adopted": False, "delta": {"mean": -0.01, "lo": -0.02, "hi": 0.0},
                 "diagnosis": {"by_bin": {}}},
            ]
            text, h = lj.near_miss_base(hist, rd)
            self.assertEqual(text, "P1")
            self.assertEqual(h["iter"], 1)
            self.assertNotIn("candidate", h)
            # 채택 뒤의 것만 본다
            hist2 = hist + [{"iter": 3, "adopted": True, "gain": {"mean": 0.02, "lo": 0.01, "hi": 0.03}}]
            self.assertIsNone(lj.near_miss_base(hist2, rd))
            # 근소 기각이 없으면 None
            self.assertIsNone(lj.near_miss_base([hist[2]], rd))


class DecideMean(unittest.TestCase):
    """--adopt-rule mean — 평균 > 0 + 퇴행 가드."""

    def test_mean_positive_accepts(self):
        bins = {"≤3": {"mean": 0.02, "pairs": 700}, "≤99": {"mean": 0.001, "pairs": 400}}
        self.assertEqual(lj.decide({"mean": 0.004, "lo": -0.01}, rule="mean",
                                   bins=bins, guard=0.01), "accept")

    def test_mean_zero_rejects(self):
        self.assertEqual(lj.decide({"mean": 0.0, "lo": 0.0}, rule="mean", bins={}, guard=0.01),
                         "reject")

    def test_guard_blocks_collapsed_bin(self):
        # judge14 iter 1 실측 꼴 — 평균은 양수인데 ≤99 가 무너졌다
        bins = {"≤3": {"mean": 0.02, "pairs": 700}, "≤99": {"mean": -0.0226, "pairs": 400}}
        self.assertEqual(lj.decide({"mean": 0.005, "lo": -0.01}, rule="mean",
                                   bins=bins, guard=0.01), "reject_guard")

    def test_thin_bin_gets_double_slack(self):
        bins = {"≤99": {"mean": -0.015, "pairs": 100}}      # 짝 300 미만 → −0.02 까지 봐준다
        self.assertEqual(lj.decide({"mean": 0.005}, rule="mean", bins=bins, guard=0.01), "accept")

    def test_lo_rule_unchanged(self):
        self.assertEqual(lj.decide({"lo": 0.02, "mean": 0.0}), "accept")


class CaseExclusion(unittest.TestCase):
    """이터 간 제외 — (문장, 구간) 키와 절단집합 겹침."""

    def setUp(self):
        self.texts = ["a b c d e f g h i j k l"]
        self.sents = [sent(0, self.texts[0])]
        self.lab = labels_for(self.texts, lambda i, j: 0.0,
                              lambda i, j: {3: 0.9, 6: 0.2, 9: 0.8}.get(j, 0.1))
        self.pol = {(0, 1): (6,), (0, 3): (2, 6, 10)}
        self.ora = {(0, 1): (3,), (0, 3): (3, 6, 9)}
        self.pol_h = {(0, 1): 0.60, (0, 3): 0.50}
        self.ora_h = {(0, 1): 0.75, (0, 3): 0.70}

    def build(self, **kw):
        return lj.build_cases(self.sents, self.lab, self.pol, self.ora, self.pol_h, self.ora_h,
                              True, 1, 5, **kw)[0]

    def test_sentence_exclusion_drops_every_bin(self):
        self.assertEqual(self.build(exclude_ids={"s0"}), [])

    def test_bin_key_keeps_other_bins(self):
        shown = self.build()[0]
        got = self.build(exclude_keys={(shown["id"], shown["latency_bin"])})
        self.assertTrue(got)
        self.assertNotIn(shown["latency_bin"], [c["latency_bin"] for c in got])

    def test_overlapping_cut_set_is_skipped(self):
        # (0,1) 의 절단 (6,) 은 (0,3) 의 (2,6,10) 과 Jaccard 1/3 — 기본 0.5 에서는 안 걸린다
        self.assertEqual(len(self.build(shown_cuts={"s0": [(6,)]}, jaccard_max=0.3)), 0)
        self.assertTrue(self.build(shown_cuts={"s0": [(6,)]}, jaccard_max=0.9))


class CandidatePlan(unittest.TestCase):
    """--candidates-cross — (역할 × finding) 한 번씩."""

    R4 = ["free", "narrow_rule", "examples_only", "single_small"]

    def test_cross_covers_every_pair_once(self):
        got = lj.candidate_plan(self.R4, 3, True, 6)
        self.assertEqual(len(got), 12)
        self.assertEqual(len(set(got)), 12)
        self.assertEqual({r for r, _f in got}, set(self.R4))
        self.assertEqual({f for _r, f in got}, {0, 1, 2})

    def test_cross_adapts_to_finding_count(self):
        self.assertEqual(len(lj.candidate_plan(self.R4, 4, True, 6)), 16)
        self.assertEqual(len(lj.candidate_plan(self.R4, 1, True, 6)), 4)

    def test_cap_limits_gate_cost(self):
        got = lj.candidate_plan(self.R4, 5, True, 6, cap=16)
        self.assertEqual(len(got), 16)

    def test_default_is_independent_modulo(self):
        # 종전 배정 — 역할 4 · finding 4 면 주기가 4 라 같은 쌍이 반복된다
        got = lj.candidate_plan(self.R4, 4, False, 12)
        self.assertEqual(len(got), 12)
        self.assertEqual(len(set(got)), 4)

    def test_no_roles_falls_back_to_free(self):
        self.assertEqual(lj.candidate_plan([], 3, True, 2), [("free", 0), ("free", 1)])


class MergeWinners(unittest.TestCase):
    """유형별 1등 합치기 — 같은 단위를 두 번 고치는 편집은 하나만 남긴다."""

    BASE = ("[Role]\nr\n\n[Scoring Rules]\ns\n\n[Core Principles]\n- C one\n- C two\n\n"
            "[Output Rules]\no\n\n[Examples]\nInput: a <SEG:?> b\nOutput: a <SEG:5> b\n")

    def win(self, lo, edits):
        return {"delta": {"lo": lo, "mean": lo + 0.01}, "pe": {"changelog": ["c"], "edits": edits}}

    def test_two_inserts_merge(self):
        w1 = self.win(0.01, [{"op": "insert_after", "id": "C1", "kind": "change",
                              "text": "- keep a numeral with its unit"}])
        w2 = self.win(0.02, [{"op": "insert_after", "id": "C2", "kind": "change",
                              "text": "- keep a head noun with the modifier that identifies it"}])
        cand, _note, _d, dropped = lj.merge_winners(self.BASE, [w1, w2], 9999)
        self.assertIsNotNone(cand)
        self.assertIn("keep a numeral", cand)
        self.assertIn("keep a head noun", cand)
        self.assertEqual(dropped, [])

    def test_same_unit_keeps_higher_lo(self):
        lowlo = self.win(0.005, [{"op": "replace", "id": "C1", "kind": "change", "text": "- low"}])
        highlo = self.win(0.03, [{"op": "replace", "id": "C1", "kind": "change", "text": "- high"}])
        cand, _note, _d, dropped = lj.merge_winners(self.BASE, [lowlo, highlo], 9999)
        self.assertIn("- high", cand)
        self.assertNotIn("- low", cand)
        self.assertEqual(dropped, ["C1"])

    def test_no_edits_returns_none(self):
        cand, note, _d, _dr = lj.merge_winners(self.BASE, [self.win(0.01, [])], 9999)
        self.assertIsNone(cand)
        self.assertEqual(note, ["합칠 편집이 없다"])


class PickKey(unittest.TestCase):
    """선별과 채택은 같은 자로 본다 — 갈리면 평균이 높고 분산이 큰 후보가 조용히 떨어진다."""

    BOLD = {"mean": 0.0120, "lo": -0.0076, "hi": 0.0316}   # 합본 — 크게 움직여 se 가 크다
    MEEK = {"mean": 0.0040, "lo": 0.0001, "hi": 0.0079}    # 단일 — 작게 움직여 se 가 작다

    def test_mean_rule_prefers_higher_mean(self):
        key = lj.pick_key("mean")
        self.assertIs(max([self.BOLD, self.MEEK], key=key), self.BOLD)

    def test_lo_rule_prefers_higher_lower_bound(self):
        key = lj.pick_key("lo")
        self.assertIs(max([self.BOLD, self.MEEK], key=key), self.MEEK)

    def test_mean_rule_screens_on_mean(self):
        key = lj.pick_key("mean")
        self.assertGreater(key(self.BOLD), 0)      # 하한은 음수지만 평균으로는 선별을 통과한다
        self.assertLess(key({"mean": -0.001, "lo": -0.02, "hi": 0.018}), 0)


class GuardBeforeSelection(unittest.TestCase):
    """퇴행 가드는 뽑기 전에 후보마다 매긴다 — 뒤에 매기면 본채점 두 번이 통째로 버려진다."""

    GUARD = 0.01

    def fulls(self):
        ok = {"j": 0, "boot": {"mean": 0.004, "lo": 0.0001},
              "bins": {"≤5": {"mean": 0.002, "pairs": 400}, "≤99": {"mean": 0.005, "pairs": 400}}}
        breach = {"j": 900, "boot": {"mean": 0.012, "lo": -0.008},
                  "bins": {"≤5": {"mean": -0.025, "pairs": 400}, "≤99": {"mean": 0.02, "pairs": 400}}}
        return [ok, breach]

    def clean(self, fulls):
        return [f for f in fulls if not lj.guard_breaches(f["bins"], self.GUARD)]

    def test_breaching_candidate_dropped_before_pick(self):
        fulls = self.fulls()
        clean = self.clean(fulls)
        self.assertEqual([f["j"] for f in clean], [0])
        chosen = max(clean or fulls, key=lambda f: lj.pick_key("mean")(f["boot"]))
        self.assertEqual(chosen["j"], 0)           # 평균은 합본이 높지만 구간을 무너뜨린다

    def test_all_breaching_falls_back_to_full_list(self):
        fulls = self.fulls()
        fulls[0]["bins"]["≤5"] = {"mean": -0.03, "pairs": 400}
        clean = self.clean(fulls)
        self.assertEqual(clean, [])
        chosen = max(clean or fulls, key=lambda f: lj.pick_key("mean")(f["boot"]))
        self.assertEqual(chosen["j"], 900)         # 전부 걸리면 종전대로 decide 가 기각한다
        self.assertEqual(lj.decide(chosen["boot"], rule="mean", bins=chosen["bins"],
                                   guard=self.GUARD), "reject_guard")

    def test_no_guard_keeps_every_candidate(self):
        fulls = self.fulls()
        self.assertEqual(len([f for f in fulls if not lj.guard_breaches(f["bins"], 0.0)]), 2)


class ClassifyType(unittest.TestCase):
    """분류는 덩이로 끊어 부르고, 한 덩이가 깨져도 런은 죽지 않는다."""

    class Sent:
        def __init__(self, i):
            self.id, self.text = f"s{i}", f"sentence {i}"

    class Cache:
        def __init__(self):
            self.d = {}

        def get(self, k):
            return self.d.get(k)

        def put(self, k, v):
            self.d[k] = v

    class GW:
        """짝수 덩이는 앞 두 개를 맞다고 하고, 홀수 덩이는 빈 응답으로 파싱 실패를 낸다."""

        def __init__(self, fail_every=0):
            self.calls, self.fail_every, self.max_tokens = [], fail_every, []

        def chat_json(self, _system, user, **kw):
            import json as _j
            nums = [int(x.split(".")[0]) for x in _j.loads(user)["sentences"]]
            self.calls.append(nums)
            self.max_tokens.append(kw.get("max_tokens"))
            if self.fail_every and len(self.calls) % self.fail_every == 0:
                raise ValueError("JSON 파싱 실패: ")
            return {"matching": nums[:2]}

    def setUp(self):
        self.sents = [self.Sent(i) for i in range(100)]
        self.ids = list(range(100))

    def test_chunks_and_merges(self):
        gw = self.GW()
        got = lj.classify_type(gw, "a numeral precedes a unit", self.sents, self.ids,
                               self.Cache(), chunk=40)
        self.assertEqual([len(c) for c in gw.calls], [40, 40, 20])
        self.assertEqual(got, [0, 1, 40, 41, 80, 81])

    def test_failed_chunk_is_skipped_not_raised(self):
        gw = self.GW(fail_every=2)          # 두 번째 덩이가 깨진다
        lines = []
        got = lj.classify_type(gw, "pred", self.sents, self.ids, self.Cache(),
                               log_fn=lines.append, chunk=40)
        self.assertEqual(got, [0, 1, 80, 81])
        self.assertIn("분류 실패 1덩이", lines[0])

    def test_cache_key_is_per_chunk(self):
        cache, gw = self.Cache(), self.GW()
        lj.classify_type(gw, "pred", self.sents, self.ids, cache, chunk=40)
        again = self.GW()
        got = lj.classify_type(again, "pred", self.sents, self.ids, cache, chunk=40)
        self.assertEqual(again.calls, [])           # 세 덩이 모두 캐시 적중
        self.assertEqual(got, [0, 1, 40, 41, 80, 81])

    def test_budget_is_raised_above_reasoning(self):
        gw = self.GW()
        lj.classify_type(gw, "pred", self.sents, self.ids[:10], self.Cache(), chunk=40)
        self.assertEqual(gw.max_tokens, [8000])     # 4000 은 사고 토큰이 다 먹었다


class FailureBrief(unittest.TestCase):
    """Writer 가 다시 쓸 때 Critic 요약이 아니라 원본 실패(현재 절단 vs 오라클 절단)를 본다."""

    def case(self, cid, gap, bin_="≤3"):
        return {"id": cid, "gap": gap, "latency_bin": bin_,
                "policy": {"text": f"{cid} now ‖ text", "cut": [1]},
                "target": {"text": f"{cid} best ‖ text", "cut": [2]}}

    def test_sorted_by_gap_and_capped(self):
        cases = [self.case(f"s{i}", i / 100) for i in range(12)]
        out = lj.failure_brief(cases, n=3)
        self.assertEqual(out.count("now :"), 3)
        self.assertIn("s11", out.splitlines()[0] + out)      # 가장 큰 gap 이 먼저
        self.assertNotIn("s0 now", out)

    def test_shows_both_cut_sets(self):
        out = lj.failure_brief([self.case("s1", 0.5)])
        self.assertIn("now : s1 now ‖ text", out)
        self.assertIn("best: s1 best ‖ text", out)
        self.assertIn("gap 0.500", out)

    def test_empty_is_empty_string(self):
        self.assertEqual(lj.failure_brief([]), "")

    def test_missing_fields_do_not_raise(self):
        self.assertIn("gap 0.000", lj.failure_brief([{"id": "x"}]))


class AdoptKeysTest(unittest.TestCase):
    """`--adopt-bin` 은 채택 판정에 쓸 짝만 남긴다 — 채점 자체는 건드리지 않는다."""

    def setUp(self):
        self.sents = [sent(0, " ".join(f"w{i}" for i in range(12))),
                      sent(1, " ".join(f"w{i}" for i in range(40)))]

    def test_none_keeps_all(self):
        keys = [(0, 1), (0, 3), (1, 1), (1, 9)]
        self.assertEqual(lj.adopt_keys(self.sents, keys, True, None), keys)

    def test_bin_filters(self):
        keys = [(0, k) for k in (1, 2, 3, 5, 99)] + [(1, k) for k in (1, 5, 99)]
        got = lj.adopt_keys(self.sents, keys, True, "≤3")
        self.assertTrue(got, "≤3 짝이 하나도 없다 — latency_bin 규약이 바뀌었는지 본다")
        for i, k in got:
            n = len(lj.L.units_of(self.sents[i].text, True))
            self.assertEqual(lj.latency_bin(n, k), "≤3")

    def test_bin_unknown_gives_empty(self):
        keys = [(0, 1), (1, 5)]
        self.assertEqual(lj.adopt_keys(self.sents, keys, True, "없는구간"), [])


class InducePicksTest(unittest.TestCase):
    """`induce` 사례 선정 — 구간 비중을 사례 배분과 맞추고, 예시와 극단을 제한한다."""

    def cases(self):
        """실측 분포를 흉내 낸다 — ≤3 이 57%, gap 은 구간과 무관하게 퍼져 있다.

        gap 을 한 구간에 몰아 두면(≤5 만 0.7대) 상위 절반을 잘랐을 때 그 구간만 남아 층화가
        의미를 잃는다. judge24 실측은 gap 중앙 0.17~0.52 에 구간이 고르게 섞여 있었다."""
        out = []
        for i in range(57):
            out.append({"id": f"a{i}", "latency_bin": "≤3", "gap": 0.80 - i * 0.01,
                        "policy": {"H_set": 0.01 if i % 2 == 0 else 0.30}})
        for i in range(28):
            out.append({"id": f"b{i}", "latency_bin": "≤5", "gap": 0.78 - i * 0.01,
                        "policy": {"H_set": 0.01 if i % 2 == 0 else 0.30}})
        for i in range(9):
            out.append({"id": f"c{i}", "latency_bin": "≤7", "gap": 0.75 - i * 0.02,
                        "policy": {"H_set": 0.20}})
        for i in range(6):
            out.append({"id": f"d{i}", "latency_bin": "≤10", "gap": 0.70 - i * 0.03,
                        "policy": {"H_set": 0.20}})
        return out

    def test_follows_bin_share(self):
        got = lj.induce_picks(self.cases(), set(), 8)
        self.assertEqual(len(got), 8)
        n3 = sum(1 for c in got if c["latency_bin"] == "≤3")
        # ≤3 이 전체의 60% 이므로 gap 이 작아도 가장 많이 들어가야 한다
        self.assertGreaterEqual(n3, 4, f"≤3 이 {n3} 개뿐이다 — gap 순 정렬로 되돌아갔다")

    def test_excludes_example_sentences(self):
        got = lj.induce_picks(self.cases(), {"a0", "b0", "b1"}, 8)
        self.assertFalse({c["id"] for c in got} & {"a0", "b0", "b1"})

    def test_caps_contra_killed(self):
        """모순으로 죽은 사례는 절반까지 — **채울 수 있을 때만** 지킨다.

        상위 절반으로 좁히면 그 안이 대부분 죽은 사례라(gap 큰 자리가 곧 그런 자리다) 상한을
        끝까지 지키면 묶음이 8개에서 6개로 줄어든다. 개수를 채우는 쪽이 낫다고 판단했다."""
        got = lj.induce_picks(self.cases(), set(), 8)
        self.assertEqual(len(got), 8, "개수를 못 채웠다")
        killed = sum(1 for c in got if c["policy"]["H_set"] < 0.05)
        self.assertLessEqual(killed, 4, "살아 있는 사례가 충분한데 상한을 넘었다")

    def test_uses_only_top_half(self):
        """하위 절반(거의 맞힌 자리)은 쓰지 않는다 — judge24 에서 네 번 다 해로웠다."""
        cs = self.cases()
        median = sorted(c["gap"] for c in cs)[len(cs) // 2]
        for part in (0, 1):
            got = lj.induce_picks(cs, set(), 8, part=part, parts=2)
            self.assertEqual(len(got), 8)
            worst = min(c["gap"] for c in got)
            self.assertGreaterEqual(worst, median - 1e-9,
                                    f"part {part} 가 중앙값 {median:.3f} 아래를 썼다({worst:.3f})")

    def test_empty_pool(self):
        self.assertEqual(lj.induce_picks([], set(), 8), [])
        self.assertEqual(lj.induce_picks(self.cases(), set(), 0), [])


class MisorderCost(unittest.TestCase):
    """깊이별 "한 번 잘못 고르는 손해". Critic 은 이 값이 깊이에 따라 줄어드는지로
    "어느 깊이까지 순서를 신경 써야 하나" 를 판단한다 — 그러니 모양이 살아야 한다."""

    def test_stays_positive_while_labels_keep_falling(self):
        """고원이 없는 분포 — 어느 깊이에서도 "남은 것보다 나은 자리" 가 있으므로 손해가 0 이
        되지 않는다. 값이 완만히 줄어드는 것은 꼬리가 짧아지는 산술 효과다(선형 분포에서 0.475
        → 0.175). 실측 라벨은 꼬리가 길어 그 감소가 훨씬 작았다(0.1405 → 0.1185)."""
        rows = [{"labels": [1.0 - 0.05 * i for i in range(20)]} for _ in range(30)]
        got = lj.misorder_cost(rows)
        self.assertEqual(len(got), 13)
        self.assertGreater(got["13"], 0, f"고원이 없는데 손해가 0 이 됐다: {got}")
        vals = [got[str(d)] for d in range(1, 14)]
        self.assertEqual(vals, sorted(vals, reverse=True), f"단조 감소가 아니다: {got}")

    def test_falls_when_only_the_top_few_matter(self):
        """상위 셋만 좋고 나머지가 동일한 분포 — 그때는 손해가 깊이에서 0 으로 간다."""
        rows = [{"labels": [0.9, 0.8, 0.7] + [0.3] * 17} for _ in range(30)]
        got = lj.misorder_cost(rows)
        self.assertGreater(got["1"], 0.1)
        self.assertAlmostEqual(got["8"], 0.0, places=6,
                               msg=f"남은 것이 다 같으면 손해가 0 이어야 한다: {got}")

    def test_skips_rows_without_labels(self):
        self.assertEqual(lj.misorder_cost([{"labels": []}, {"scores": [1, 2]}]), {})
        self.assertEqual(lj.misorder_cost([]), {})


class RejectedByBin(unittest.TestCase):
    """기각 이력은 **이 런 안에서만** 나온다 — 다른 런의 기각은 데이터도 프롬프트도 달라
    근거가 못 된다. 그래서 입력이 그 런의 history 하나뿐이다."""

    def hist(self):
        return [{"iter": 1, "adopted": False, "edits": [{"op": "insert_after"}],
                 "diagnosis": {"by_bin": {"≤3": {"mean": -0.004}}}},
                {"iter": 2, "adopted": True, "edits": [],
                 "diagnosis": {"by_bin": {"≤3": {"mean": 0.01}}}},
                {"iter": 3, "adopted": False, "edits": [{"op": "delete"}],
                 "diagnosis": {"by_bin": {"≤3": {"mean": -0.009}}}},
                {"iter": 4, "adopted": False, "edits": []}]

    def test_keeps_only_rejected_with_a_postmortem(self):
        got = lj.rejected_by_bin(self.hist())
        self.assertEqual([g["iter"] for g in got], [1, 3],
                         "채택본이나 부검 없는 항목이 섞였다")
        self.assertIn("by_bin", got[0])

    def test_keeps_the_most_recent(self):
        h = [{"iter": i, "adopted": False, "edits": [],
              "diagnosis": {"by_bin": {"≤3": {"mean": -0.001 * i}}}} for i in range(1, 11)]
        got = lj.rejected_by_bin(h, n_max=3)
        self.assertEqual([g["iter"] for g in got], [8, 9, 10])

    def test_empty(self):
        self.assertEqual(lj.rejected_by_bin([]), [])


class FindingKind(unittest.TestCase):
    """`kind` 는 **문면**이 정한다 — Critic 이 붙인 라벨을 믿지 않는다. 그 값으로 편집 칸과 역할
    배분이 갈리는데, judge31 에서 PE 가 뒤에서 형태를 바꿔 써 라벨과 실제가 어긋났다."""

    ORDER_SEC = "[Order Principles]\n- keep order\n"

    def blob(self, *pairs):
        """(라벨, 문면) 쌍들."""
        return {"findings": [
            {**({"kind": k} if k is not None else {}), "diagnosis": f"d{i}",
             "evidence": "e", "type_predicate": "t",
             "edit": {"where": "core_principles", "action": "add", "text": t}}
            for i, (k, t) in enumerate(pairs)]}

    def test_text_decides_not_label(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        # 라벨이 뒤집혀 있어도 문면대로 간다.
        got = aj.clean_findings(self.blob(("check", "- prefer a cut at A over one at B"),
                                         ("order", "- keep a head with its modifier")), cap=9)
        self.assertEqual([f["kind"] for f in got], ["order", "check"])
        self.assertTrue(all(f.get("kind_relabeled") for f in got))

    def test_prohibition_with_before_stays_unary(self):
        """`is_ordering` 의 넓은 표지("before ")로 판정하면 단항 금지문이 이항으로 승격된다 —
        judge31 이터 1 의 finding 이 그 꼴이었다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        got = aj.clean_findings(self.blob(
            (None, "- Do not cut immediately before a clausal attachment; keep it with its head.")),
            cap=9)
        self.assertEqual([f["kind"] for f in got], ["check"])

    def test_order_routes_to_order_principles(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        b = self.blob(("order", "- prefer a cut at A over one at B"))
        self.assertEqual(aj.clean_findings(b, self.ORDER_SEC, cap=9)[0]["edit"]["where"],
                         "order_principles")
        # 그 섹션이 없는 프롬프트로 도는 런에서는 원칙 칸에 그대로 둔다 — `kind` 는 살린다.
        got = aj.clean_findings(b, "[Core Principles]\n- y\n", cap=9)[0]
        self.assertEqual((got["kind"], got["edit"]["where"]), ("order", "core_principles"))

    def test_two_lists_and_cap_keeps_one_of_each(self):
        """cap 에 앞에서 잘리면 `order` 가 사라지고 이항 편집을 쓰는 역할이 안 돈다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        mk = lambda t: {"diagnosis": "d", "evidence": "e", "type_predicate": "t",
                        "edit": {"where": "core_principles", "action": "add", "text": t}}
        b = {"checks": [mk("- keep A with B"), mk("- keep C with D"), mk("- keep E with F")],
             "orders": [mk("- prefer A over B")]}
        got = aj.clean_findings(b, self.ORDER_SEC, cap=3)
        self.assertEqual([f["kind"] for f in got], ["order", "check", "check"])

    def test_empty_list_is_allowed(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        mk = {"diagnosis": "d", "evidence": "e", "type_predicate": "t",
              "edit": {"where": "core_principles", "action": "add", "text": "- keep A with B"}}
        got = aj.clean_findings({"checks": [mk], "orders": []}, self.ORDER_SEC, cap=3)
        self.assertEqual([f["kind"] for f in got], ["check"])


class KindEnforced(unittest.TestCase):
    """칸이 형태를 정한다 — PE 가 형태를 바꿔 쓸 여지를 없앤다."""

    def edit(self, uid, text):
        return [{"op": "insert_after", "id": uid, "text": text}]

    def test_order_must_use_order_section(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        keep, bad = aj.enforce_kind(self.edit("C_end", "prefer A over B"), "order")
        self.assertEqual((keep, len(bad)), ([], 1))
        keep, bad = aj.enforce_kind(self.edit("O_end", "prefer A over B"), "order")
        self.assertEqual((len(keep), bad), (1, []))

    def test_check_may_not_compare(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        keep, bad = aj.enforce_kind(self.edit("C_end", "rank A higher than B"), "check")
        self.assertEqual((keep, len(bad)), ([], 1))
        keep, bad = aj.enforce_kind(self.edit("O_end", "keep A with B"), "check")
        self.assertEqual((keep, len(bad)), ([], 1))
        keep, bad = aj.enforce_kind(self.edit("C_end", "keep A with B"), "check")
        self.assertEqual((len(keep), bad), (1, []))

    def test_skipped_without_order_section(self):
        """그 섹션이 없는 프롬프트로 도는 런에서는 칸 검사를 건너뛴다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        keep, bad = aj.enforce_kind(self.edit("C_end", "prefer A over B"), "order",
                                    has_order_section=False)
        self.assertEqual((len(keep), bad), (1, []))

    def test_fallback_role_accepts_order_unit(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        keep, bad = aj.enforce_role(self.edit("O_end", "prefer a cut at A over one at B"),
                                    "fallback")
        self.assertEqual((len(keep), bad), (1, []))

    def test_fallback_instruction_names_the_section_that_survives(self):
        """역할 지시문이 가리키는 칸과 `enforce_kind` 가 허용하는 칸이 같아야 한다.

        같지 않으면 그 역할의 후보는 **매 이터 조용히 사라진다.** 지시문이 이항 편집을
        `[Core Principles]` 에 넣으라고 적혀 있던 동안, PE 는 적힌 대로 C 단위에 넣었고
        `enforce_kind` 는 적힌 대로 버렸다. 두 번 시도하고 후보가 반려되는데, 로그만 보면
        PE 가 말을 안 들은 것처럼 보인다. 점수에 안 잡히는 종류의 손실이라 지시문 쪽을
        문자열로 못 박는다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        spec = aj.ENGINEER_SYSTEM[aj.ENGINEER_SYSTEM.index('"fallback"'):]
        spec = spec[:spec.index('"induce"')]          # 다음 역할 설명이 시작되는 자리까지
        self.assertIn('"insert_after" into [Order Principles]', spec)
        self.assertNotIn('"insert_after" into [Core Principles]', spec)


class PromptCarriesNoEnvironmentConstant(unittest.TestCase):
    """**코드가 주입하는 프롬프트에는 이 평가 환경에서만 참인 값을 넣지 않는다.**

    프롬프트는 배포되는 산출물이다. 평가 환경의 상수를 품고 있으면 다른 코퍼스·언어쌍·지연 설정에서
    모델에게 **틀린 말**을 하게 된다. 실측은 문서와 커밋 메시지가 담당한다.

    실제로 둘이 걸렸다. `[Output Rules]` 를 고친 judge35 의 변형이 "약 여덟 자리가 채택된다" 고 썼는데
    그건 dev 500 과 이 latency bin 구성에서 잰 값이고(평균 k 8.42), 배포 예산이 다르면 거짓이 된다 —
    dev 에서 +0.0017 로 재였지만 그래서 안 넣었다. 그리고 `[Scoring Rules]` 에 "아래 등급이 보통 가장
    많은 자리를 담는다" 가 있었는데 그것도 이 데이터의 분포다.

    구조를 말하는 것은 된다("지연 예산이 몇 개를 남길지 정한다"). 그 환경의 값은 안 된다("약 여덟")."""

    def injected(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        from core.meaning_segmentator.autoseg.runtime import agents_distill as ad
        return {"[Scoring Rules]": aj.scoring_rules(["Chinese", "German"]),
                "[Output Rules]": ad.output_rules(True, "source", meaning_in_scoring_rules=True)}

    def test_no_distribution_claim_or_measured_quantity(self):
        import re
        pat = re.compile(r"\busually\b|\bmost of\b|\babout \d|\d+\s*%")
        for name, txt in self.injected().items():
            hits = [ln.strip()[:90] for ln in txt.split("\n") if pat.search(ln)]
            self.assertEqual(hits, [], f"{name} 에 환경 의존 주장: {hits}")

    def test_no_run_name(self):
        """런 이름은 프롬프트에도 에이전트 지시문에도 넣지 않는다 — 읽는 쪽이 그 런을 모른다."""
        import re
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        pat = re.compile(r"\bjudge\d|\brun\d")
        for name, txt in self.injected().items():
            self.assertIsNone(pat.search(txt), f"{name} 에 런 이름")
        for name, txt in (("WRITER_SYSTEM", aj.WRITER_SYSTEM),
                          ("ENGINEER_SYSTEM", aj.ENGINEER_SYSTEM),
                          ("CRITIC_SYSTEM", aj.CRITIC_SYSTEM)):
            self.assertIsNone(pat.search(txt), f"{name} 에 런 이름")


class GradedPrinciplesAreTheDefault(unittest.TestCase):
    """새 v0 는 **질문 + 정도 절** 구조로 생성돼야 한다. 정도 절이 빠지면 이득이 조용히 사라진다."""

    def test_skeleton_orders_by_severity(self):
        """골격의 등급 안 기본 서열이 정도를 가리켜야 한다 — 안 그러면 정도 절을 써도 읽히지 않는다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        sr = " ".join(aj.scoring_rules(["German"]).split())
        self.assertIn("order positions by how MILDLY the Core Principles' concerns apply", sr)
        # 옛 표면형 기본값은 사라졌다
        self.assertNotIn("prefer the one whose preceding stretch can stand alone", sr)

    def test_writer_is_told_to_write_both_parts(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        w = " ".join(aj.WRITER_SYSTEM.split())
        self.assertIn("Each line has two parts", w)
        self.assertIn("what makes a cut at such a position worse, and what makes it milder", w)
        self.assertIn("The question decides the band; the severity decides the order INSIDE a band", w)

    def test_detects_a_question_with_no_severity(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        base = ("[Role]\nr\n\n" + aj.scoring_rules(["German"]) + "\n\n"
                "[Core Principles]\n"
                "- Does the continuation reverse what came before?\n"
                "- Does the cut strand a required complement? A cut here is worse the shorter that"
                " remainder is, and mildest when it is a detachable afterthought.\n\n"
                "[Order Principles]\n- prefer A over B\n\n[Output Rules]\n- x\n\n"
                "[Examples]\nInput: a <SEG:?> b\nOutput: a <SEG:5> b\n")
        self.assertEqual(aj.ungraded_principles(base), ["C1"])
        self.assertEqual(aj.check_skeleton(base), [])        # 골격 검사로는 안 잡힌다

    def test_direction_alone_is_not_a_severity_axis(self):
        """combo 의 원칙들이 여기 걸린다 — 물음표 뒤에 문장이 있지만 **방향만** 말한다.

        "such outcomes increase contradiction risk" 는 어느 쪽으로 미는지는 알려주지만 얼마나인지,
        무엇이 더 심하고 무엇이 더 가벼운지는 말하지 않는다. 그러면 등급 안 두 자리를 여전히 못
        가른다. 물음표 뒤에 글이 있는지만 보면 이것을 통과시킨다 — 실제로 한 번 통과시켰다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        head = ("[Role]\nr\n\n" + aj.scoring_rules(["German"]) + "\n\n[Core Principles]\n")
        tail = ("\n\n[Order Principles]\n- prefer A over B\n\n[Output Rules]\n- x\n\n"
                "[Examples]\nInput: a <SEG:?> b\nOutput: a <SEG:5> b\n")
        direction = "- Does the continuation negate what came before? Such outcomes increase contradiction risk."
        self.assertEqual(aj.ungraded_principles(head + direction + tail), ["C1"])
        # 양쪽 끝이 있으면 통과한다. 가벼운 쪽을 "better" 로 써도 된다.
        for axis in ("- Does it negate? A cut here is worse the more complete the reversal, mildest for a qualifier.",
                     "- Does it negate? A cut here is better when the reversal is partial, worse when total."):
            self.assertEqual(aj.ungraded_principles(head + axis + tail), [], axis[:40])

    def test_uses_the_same_boundary_as_the_severity_role(self):
        """검사와 역할이 경계를 다르게 보면, 통과한 편집이 뒤에서 죽거나 빈 절이 통과한다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        q = "- Does the cut strand a required complement?"
        full = ("[Role]\nr\n\n" + aj.scoring_rules(["German"]) + "\n\n[Core Principles]\n" + q
                + "\n\n[Order Principles]\n- prefer A over B\n\n[Output Rules]\n- x\n\n"
                "[Examples]\nInput: a <SEG:?> b\nOutput: a <SEG:5> b\n")
        self.assertEqual(aj.ungraded_principles(full), ["C1"])       # 검사: 비었다
        keep, bad = aj.enforce_role([{"op": "replace", "id": "C1", "text": q, "_was": q}], "severity")
        self.assertEqual(keep, [])                                   # 역할: 같은 이유로 거부
        self.assertIn("정도 축이 아니다", bad[0]["reason"])


class WriterBudgetIsComputed(unittest.TestCase):
    """길이 예산은 **코드가 고정분을 세어서** 넘긴다 — 지시문에 숫자를 박으면 낡는다.

    종전 지시문은 "9500자 이하, 주어진 두 블록이 약 3,000" 이었다. 그 숫자는 원칙에 정도 축을
    요구하기 **전**에 쓴 것이다. 실측으로 고정분이 7,164자가 됐고(골격 3,253 + 출력 규약 1,456 +
    예시 2,455) 9,500 중 2,336자만 남는데, 거기에 [Role] 과 원칙 여덟(질문 + 정도 축)과 예외 서넛을
    넣으라는 것은 **불가능한 요구**다.

    Writer 후보 여덟이 전부 10,433~11,037자를 냈고 그중 셋은 뒤쪽 원칙의 정도 축을 빠뜨렸다 —
    길이가 차니까 끝을 줄인 것이다. 사람이 쓴 판도 같은 구조에서 10,331자다. **지시가 틀렸지
    Writer 가 못 쓴 것이 아니다.**"""

    def test_cap_follows_the_fixed_content(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        import re
        for fixed in (7164, 5000):
            w = aj.writer_system(True, ["German"], fixed)
            m = re.search(r"under (\d+) characters. Of that, (\d+)", w)
            self.assertIsNotNone(m, w[-400:])
            self.assertEqual(int(m.group(2)), fixed)
            self.assertEqual(int(m.group(1)) - fixed, 4200)

    def test_no_stale_number_is_left_in_the_instruction(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        w = aj.writer_system(True, ["German"], 7164)
        self.assertNotIn("9500", w)
        self.assertNotIn("already take about", w)

    def test_writer_is_told_to_budget_severity_first(self):
        """뒤쪽 원칙의 정도 축이 빠지는 것이 실측된 실패 모습이다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        w = " ".join(aj.writer_system(True, ["German"], 7164).split())
        self.assertIn("budget the severity in before you write the questions", w)
        self.assertIn("no room left to grade the last of them", w)

    def test_the_real_fixed_content_leaves_room_for_what_is_asked(self):
        """고정분이 실제로 얼마인지 세어, Writer 몫이 요구량보다 큰지 확인한다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        from core.meaning_segmentator.autoseg.runtime import agents_distill as ad
        fixed = (len(aj.scoring_rules(["Chinese", "Japanese", "German", "Spanish"]))
                 + len(ad.output_rules(True, "source", meaning_in_scoring_rules=True)))
        # 원칙 8 × 300자 + 예외 3 × 250자 + Role 450자 ≈ 3,600
        self.assertGreater(4200, 3600)
        self.assertGreater(fixed, 4000)     # 주입분만으로 이미 크다 — 상한을 고정할 수 없는 이유


class OrderPrinciplesAreExceptions(unittest.TestCase):
    """`[Core Principles]` 가 주 척도이고 `[Order Principles]` 는 그 척도가 틀리는 **사례**다.

    종전 계약은 골격 문장 그대로 O 줄이 등급 안 기본 서열을 *"overrides that default"* 하는 것이었다.
    그런데 이득을 내는 것이 정도 기반 서열이므로, 덮어쓰면 이득을 깎는다 — 같은 에이전트 편집이
    v0 위 −0.0015, 정도 절을 붙인 판 위 −0.0053 이었다. 칸이 문제가 아니라 **계약**이 문제였다.

    이름과도 맞는다: Core 가 중심 척도를 들고, Order 는 그 척도로 잘못 서는 짝을 하나씩 집는다."""

    def rules(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        return " ".join(aj.scoring_rules(["German"]).split())

    def writer(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        return " ".join(aj.WRITER_SYSTEM.split())

    def test_severity_is_the_rule_not_a_fallback(self):
        r = self.rules()
        self.assertIn("Ordering by severity is the rule, not a fallback", r)
        self.assertIn("apply it to every pair in the band", r)

    def test_order_principles_are_exceptions_that_settle_only_their_pair(self):
        r = self.rules()
        self.assertIn("The Order Principles are its EXCEPTIONS", r)
        self.assertIn("settles only the pair it names", r)
        self.assertIn("Outside the configurations they name, order by severity", r)

    def test_the_override_contract_is_gone(self):
        """이 문구가 돌아오면 정도 절의 이득이 다시 밀려난다."""
        r = self.rules()
        self.assertNotIn("overrides that default", r)
        self.assertNotIn("read only to settle the order of positions that already share a band", r)

    def test_definition_and_contract_tell_one_story(self):
        """정의문과 계약문이 어긋나면 모델이 둘 중 하나를 고르게 된다."""
        r = self.rules()
        self.assertIn("the severity in the same line", r)       # Core 가 두 일을 한다
        self.assertIn("are exceptions only", r)                 # O 는 예외뿐
        self.assertIn("a general preference is either what severity already gives", r)

    def test_writer_is_told_to_write_few_and_narrow(self):
        w = self.writer()
        self.assertIn("holds the EXCEPTIONS to ordering by severity", w)
        self.assertIn("do not restate that", w)
        self.assertIn("Write few and write them narrow", w)
        # 일반 선호는 어느 칸에도 안 된다 — 종전 O1·O3 이 정확히 그것이었다
        self.assertIn("belongs nowhere", w)

    def test_the_duplicated_default_is_not_reintroduced(self):
        """옛 표면형 기본값(앞이 홀로 설 수 있는 쪽 / 늦은 쪽)은 Writer 가 O1·O3 으로 되풀이하던 것이다."""
        r, w = self.rules(), self.writer()
        self.assertNotIn("prefer the one whose preceding stretch can stand alone", r)
        self.assertIn("prefer the one whose preceding stretch stands alone", w)   # 금지 예시로만 등장


class SeverityVocabularyIsCalibrated(unittest.TestCase):
    """`has_severity` 의 표지는 **실제 생성물로 맞춘 것**이다.

    처음에 `worse`/`severe` 만 두었더니 Writer 가 낸 정상 절 넷을 거부했다 —
    "is most damaging; separating optional adjuncts is least", "Most harmful when …; milder when …".
    Writer 는 damaging·harmful 을 쓰고 나는 worse·severe 만 찾고 있었다. **차단 검사에서 오탐은
    후보를 조용히 잃는 것**이므로(같은 종류 버그를 두 번 겪었다) 목록이 넉넉해야 한다.

    반대쪽도 지켜야 한다: 방향만 말하는 절("such outcomes increase contradiction risk")은 통과하면
    안 된다. 그것이 통과하면 등급 안 서열 재료가 없는 원칙이 조용히 들어온다."""

    AXES = [   # Writer 와 사람이 실제로 쓴 형태들
        "Is it? An explicit sentential negation is the most severe; a narrow modifier is the mildest.",
        "Is it? Splitting a verb from its primary object is most damaging; separating optional adjuncts is least.",
        "Is it? Most harmful when a relative clause supplies the main predicate; milder when it is an aside.",
        "Is it? It is worst where later material reattaches the adjunct; it is mild when detachable.",
        "Is it? A cut here is better when the reversal is partial, worse when total.",
    ]
    DIRECTIONS = [   # 방향만 — 통과하면 안 된다
        "Is it? Such outcomes increase contradiction risk.",
        "Is it? If an essential complement would be lost, cohesion is reduced.",
        "Is it? Separating head and modifier lowers cohesion.",
        "Is it?",
        "Is it? If yes, expect damage.",
    ]

    def test_accepts_every_shape_seen_in_real_output(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        for c in self.AXES:
            self.assertTrue(aj.has_severity(c), c[:70])

    def test_rejects_direction_only(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        for c in self.DIRECTIONS:
            self.assertFalse(aj.has_severity(c), c[:70])

    def test_both_ends_are_required_not_just_one(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        self.assertFalse(aj.has_severity("Is it? It is most damaging when the object is stranded."))
        self.assertFalse(aj.has_severity("Is it? It is mildest when the tail is detachable."))

    def test_the_reason_names_the_vocabulary(self):
        """거부만 하고 무엇을 찾는지 안 알려주면 PE 가 맞출 수 없다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        _k, bad = aj.check_core_unit_shape(
            [{"op": "insert_after", "id": "C_end", "text": self.DIRECTIONS[0]}], "narrow_rule")
        r = bad[0]["reason"]
        self.assertIn("양쪽 끝", r)
        self.assertIn(aj.WORSE_WORDS[0], r)
        self.assertIn(aj.MILDER_WORDS[0], r)


class EveryRoleWritesBothParts(unittest.TestCase):
    """`[Core Principles]` 에 줄을 넣거나 바꾸는 편집은 **역할과 무관하게** 두 부분이어야 한다.

    지시문만으로는 새어 나간다 — `narrow_rule` 은 "금지문 말고 결속문" 을 세 이터 연속 어겼다.
    그리고 질문만 있는 원칙이 들어가면 프롬프트는 멀쩡해 보이고 골격 검사도 통과하지만, 그 원칙은
    등급 배정만 하고 등급 안 서열에는 아무것도 주지 못한다. **점수에 안 잡히는 손실**이다."""

    Q = "- Does the continuation reverse what came before?"
    AXIS = " A cut here is worse the more complete the reversal, and mildest for a bare qualifier."
    DIRECTION = " Such outcomes increase contradiction risk."

    def test_every_role_that_touches_c_is_checked(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        for role in ("free", "single_small", "narrow_rule", "induce", "replace", "fallback"):
            for op in ("insert_after", "replace"):
                keep, bad = aj.check_core_unit_shape(
                    [{"op": op, "id": "C2", "text": self.Q + self.DIRECTION}], role)
                self.assertEqual(keep, [], f"{role}/{op}")
                self.assertIn("두 부분이 아니다", bad[0]["reason"])
                ok, bad2 = aj.check_core_unit_shape(
                    [{"op": op, "id": "C2", "text": self.Q + self.AXIS}], role)
                self.assertEqual((len(ok), bad2), (1, []), f"{role}/{op}")

    def test_other_sections_and_deletes_are_untouched(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        for e in ({"op": "insert_after", "id": "O_end", "text": "- prefer A over B"},
                  {"op": "insert_after", "id": "E_end", "text": "Input: a\nOutput: b"},
                  {"op": "delete", "id": "C3"},
                  {"op": "replace", "id": "E1", "labeled_example": "en_us_1"}):
            keep, bad = aj.check_core_unit_shape([e], "free")
            self.assertEqual((len(keep), bad), (1, []), str(e))

    def test_one_judgement_in_one_place(self):
        """세 군데가 같은 함수를 쓴다 — 따로 판정하면 한쪽을 통과한 편집이 다른 쪽에서 죽는다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        axis, direction = self.Q + self.AXIS, self.Q + self.DIRECTION
        self.assertTrue(aj.has_severity(axis))
        self.assertFalse(aj.has_severity(direction))
        # 역할 검사
        self.assertEqual(aj.enforce_role(
            [{"op": "replace", "id": "C1", "text": direction, "_was": axis}], "severity")[0], [])
        # 모양 검사
        self.assertEqual(aj.check_core_unit_shape(
            [{"op": "replace", "id": "C1", "text": direction}], "free")[0], [])
        # v0 검사
        head = ("[Role]\nr\n\n" + aj.scoring_rules(["German"]) + "\n\n[Core Principles]\n")
        tail = ("\n\n[Order Principles]\n- prefer A over B\n\n[Output Rules]\n- x\n\n"
                "[Examples]\nInput: a <SEG:?> b\nOutput: a <SEG:5> b\n")
        self.assertEqual(aj.ungraded_principles(head + direction + tail), ["C1"])
        self.assertEqual(aj.ungraded_principles(head + axis + tail), [])

    def test_pe_is_told_which_job_each_role_serves(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        e = " ".join(aj.ENGINEER_SYSTEM.split())
        self.assertIn("The prompt has two jobs and every role serves exactly one of them", e)
        self.assertIn("BAND ASSIGNMENT", e)
        self.assertIn("ORDER INSIDE A BAND", e)
        self.assertIn("EXCEPTIONS", e)
        self.assertIn("The role that serves it is \"severity\"", e)
        self.assertIn("Code checks it and drops the edit otherwise, whatever your role is", e)


class BulletsSurviveAnEdit(unittest.TestCase):
    """교체한 줄이 글머리 기호를 잃으면 그 칸만 들쭉날쭉해진다 — 실측으로 O1 이 그렇게 들어갔다."""

    def prompt(self) -> str:
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        return ("[Role]\nr\n\n" + aj.scoring_rules(["German"]) + "\n\n[Core Principles]\n"
                "- Does the cut strand a complement? It is worse when short, milder otherwise.\n\n"
                "[Order Principles]\n- prefer A over B\n- prefer C over D\n\n"
                "[Output Rules]\n- x\n\n[Examples]\nInput: a <SEG:?> b\nOutput: a <SEG:5> b\n")

    def test_a_replacement_without_a_bullet_gets_one(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        out, errs, _ = aj.apply_edits(self.prompt(),
                                      [{"op": "replace", "id": "O1", "text": "prefer E over F"}])
        self.assertEqual(errs, [])
        lines = [l for l in out.split("[Order Principles]")[1].split("[Output Rules]")[0].strip().split("\n") if l.strip()]
        self.assertEqual(lines, ["- prefer E over F", "- prefer C over D"])

    def test_an_existing_bullet_is_not_doubled(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        out, _, _ = aj.apply_edits(self.prompt(),
                                   [{"op": "replace", "id": "O1", "text": "- prefer E over F"}])
        self.assertIn("- prefer E over F", out)
        self.assertNotIn("- - ", out)

    def test_a_section_without_bullets_stays_without_them(self):
        """없던 기호를 만들지 않는다 — 같은 칸의 다른 줄이 전부 글머리일 때만 붙인다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        pr = self.prompt().replace("- prefer A over B\n- prefer C over D",
                                   "prefer A over B\nprefer C over D")
        out, _, _ = aj.apply_edits(pr, [{"op": "replace", "id": "O1", "text": "prefer E over F"}])
        self.assertIn("prefer E over F", out)
        self.assertNotIn("- prefer E over F", out)


class BothGatesUseTheSameRuler(unittest.TestCase):
    """1차와 2차 문턱은 **한 플래그**가 정한다 — 2차에 하한이 박혀 있으면 플래그가 거짓말이 된다.

    실측으로 `--adopt-rule mean` 을 주고 돌렸는데 후보가 여럿이라 2차가 하한 > 0 으로 갈랐다.
    로그는 "하한 최고 후보 채택" 이라고 정직하게 찍었지만, 설정한 규칙과 작동한 규칙이 달랐다."""

    def cand(self, mean: float, lo: float) -> dict:
        return {"j": 0, "boot_avg": {"mean": mean, "lo": lo, "hi": mean + 0.01}}

    def test_mean_rule_passes_a_candidate_whose_lower_bound_is_negative(self):
        c = [self.cand(+0.0015, -0.0048)]
        self.assertEqual(len(lj.confirmed(c, "mean")), 1)
        self.assertEqual(lj.confirmed(c, "lo"), [])

    def test_lo_rule_is_unchanged(self):
        c = [self.cand(+0.0093, +0.0030)]
        self.assertEqual(len(lj.confirmed(c, "lo")), 1)
        self.assertEqual(len(lj.confirmed(c, "mean")), 1)

    def test_a_negative_mean_never_passes(self):
        c = [self.cand(-0.0002, -0.0070)]
        self.assertEqual(lj.confirmed(c, "mean"), [])
        self.assertEqual(lj.confirmed(c, "lo"), [])

    def test_the_ruler_comes_from_pick_key(self):
        """줄 세우는 자와 가르는 자가 같은 함수에서 나와야 한다 — 갈라지면 평균이 높고 분산이 큰
        후보가 선택 단계에서 조용히 떨어진다."""
        cs = [self.cand(+0.0015, -0.0048), self.cand(+0.0009, +0.0002)]
        ok = lj.confirmed(cs, "mean")
        key = lj.pick_key("mean")
        self.assertEqual(max(ok, key=lambda f: key(f["boot_avg"])), cs[0])
        self.assertEqual(lj.confirmed(cs, "lo"), [cs[1]])


class TheCapIsToldBeforeWriting(unittest.TestCase):
    """길이 상한은 **쓰기 전에** 전달된다 — 거부 사유로만 알려 주면 후보가 한 번 쓰고 한 번 죽는다.

    `replace` 가 네 런 연속 첫 시도에서 그렇게 죽었다(+171 / +133 / +216 / +400 대 상한
    80 / 105 / 101 / 248). 재시도는 정확한 숫자를 받으면 통과했으니 PE 가 글자를 못 세는 것이 아니라
    셀 기준을 안 받고 있었다. 검사와 알려 주는 값이 **한 식에서 나와야** 한다."""

    UNIT = ("- Does the cut strand a required complement? A cut there is worse the shorter the"
            " remainder is, and mildest when what follows is a detachable afterthought.")

    def prompt(self) -> str:
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        return ("[Role]\nr\n\n" + aj.scoring_rules(["German"]) + "\n\n[Core Principles]\n"
                + self.UNIT + "\n\n[Order Principles]\n- prefer A over B\n\n[Output Rules]\n- x"
                "\n\n[Examples]\nInput: a <SEG:?> b\nOutput: a <SEG:5> b\n")

    def test_budget_matches_what_the_check_allows(self):
        """알려 준 상한에 딱 맞는 문면은 통과하고, 한 자 넘으면 거부된다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        cap = aj.unit_budgets(self.prompt(), "replace")["C1"]
        self.assertEqual(cap, len(self.UNIT) + aj.growth_cap(self.UNIT, aj.REPLACE_GROWTH,
                                                             aj.unit_ceiling("C1")))
        head = "- Does the cut split a name from its appositive? It is worse when "
        fits = head + "x" * (cap - len(head) - len(" and milder otherwise.")) + " and milder otherwise."
        self.assertEqual(len(fits), cap)
        edit = {"op": "replace", "id": "C1", "text": fits, "_was": self.UNIT}
        self.assertEqual(len(aj.enforce_role([edit], "replace")[0]), 1)
        over = dict(edit, text=fits + "x")
        keep, bad = aj.enforce_role([over], "replace")
        self.assertEqual(keep, [])
        self.assertIn("늘었다", bad[0]["reason"])

    def test_a_short_core_line_may_reach_the_ceiling(self):
        """비례식만 두면 거꾸로 된다 — 정도 축이 가장 얇은 줄이 가장 짧아서 상한이 가장 좁다.
        실측으로 `severity` 가 그 줄을 골라 두 번 연속 걸렸다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        short = "- Does the cut strand a complement? Worse when short, milder otherwise."
        self.assertLess(len(short) * 2, aj.LINE_CEILING)
        cap = len(short) + aj.growth_cap(short, aj.SEVERITY_GROWTH, aj.unit_ceiling("C1"))
        self.assertEqual(cap, aj.LINE_CEILING)

    def test_a_long_core_line_keeps_the_proportional_cap(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        long = "- " + "x" * 400
        cap = len(long) + aj.growth_cap(long, aj.SEVERITY_GROWTH, aj.unit_ceiling("C1"))
        self.assertGreater(cap, aj.LINE_CEILING)

    def test_order_lines_do_not_get_the_ceiling(self):
        """그 칸의 계약이 "예외만, 적게, 좁게" 라 길어지는 것 자체가 계약 위반이다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        short = "- prefer A over B"
        self.assertEqual(aj.unit_ceiling("O2"), 0)
        cap = len(short) + aj.growth_cap(short, aj.REPLACE_GROWTH, aj.unit_ceiling("O2"))
        self.assertLess(cap, aj.LINE_CEILING)

    def test_severity_has_a_wider_budget_than_replace(self):
        """천장 위로 올라간 긴 줄에서만 갈린다 — 짧은 줄은 둘 다 천장에서 만난다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        pr = self.prompt().replace(self.UNIT, "- " + "x" * 400 + "? Worse when short, milder when long.")
        self.assertGreater(aj.unit_budgets(pr, "severity")["C1"],
                           aj.unit_budgets(pr, "replace")["C1"])

    def test_roles_without_a_cap_get_no_table(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        self.assertEqual(aj.unit_budgets(self.prompt(), "narrow_rule"), {})
        self.assertEqual(aj.unit_budgets(self.prompt(), "fallback"), {})

    def test_taken_units_are_named_by_id(self):
        """이미 손댄 단위는 본문으로 판정하고 PE 에게는 id 로 알려 준다 — `prune` 이 세 이터 연속
        앞 후보가 손댄 단위를 골라 첫 시도에서 죽었다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        texts = aj.unit_texts(self.prompt())
        self.assertEqual(texts["C1"], self.UNIT)
        self.assertIn("O1", texts)

    def test_both_roles_are_told_to_read_the_table(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        e = " ".join(aj.ENGINEER_SYSTEM.split())
        self.assertEqual(e.count("max_chars_per_unit"), 2)      # replace 와 severity


class PruneOnlyRemovesDeadWeight(unittest.TestCase):
    """정도 축이 들어온 뒤 `prune` 의 대상이 좁아졌다 — 등급 안 서열을 든 줄은 못 지운다.

    한 줄이 두 일을 한다(질문 = 등급, 정도 절 = 등급 안 순서). 그래서 지울 값어치가 있는 C 줄은
    정도 절이 없는 줄뿐이다. 그리고 **검사만 두면 역할이 죽는다** — 모든 C 줄이 등급을 가진 판에서는
    매번 전부 기각되므로, 먼저 세어 지울 것이 없으면 역할을 아예 주지 않는다."""

    def prompt(self, core: str, order: str) -> str:
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        return ("[Role]\nr\n\n" + aj.scoring_rules(["German"]) + "\n\n[Core Principles]\n"
                + core + "\n\n[Order Principles]\n" + order
                + "\n\n[Output Rules]\n- x\n\n[Examples]\nInput: a <SEG:?> b\nOutput: a <SEG:5> b\n")

    GRADED = ("- Does the cut strand a required complement? A cut there is worse the shorter the"
              " remainder is, and mildest when what follows is a detachable afterthought.")
    UNGRADED = "- Does the continuation reverse what came before?"

    def test_nothing_to_prune_when_every_line_is_graded(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        pr = self.prompt(self.GRADED, "- prefer A over B")
        self.assertEqual(aj.prune_targets(pr), [])
        self.assertEqual(lj.usable_roles(["severity", "prune"], pr)[0], ["severity"])
        self.assertIn("지울 수 있는 단위가 없다", lj.usable_roles(["prune"], pr)[1][0][1])

    def test_an_ungraded_line_is_the_target(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        pr = self.prompt(self.UNGRADED + "\n" + self.GRADED, "- prefer A over B")
        self.assertEqual(aj.prune_targets(pr), ["C1"])
        self.assertEqual(lj.usable_roles(["prune"], pr)[0], ["prune"])

    def test_a_spare_order_line_is_also_prunable(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        pr = self.prompt(self.GRADED, "- prefer A over B\n- prefer C over D")
        self.assertEqual(aj.prune_targets(pr), ["O1", "O2"])

    def test_deleting_a_graded_line_is_refused(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        keep, bad = aj.enforce_role(
            [{"op": "delete", "id": "C2", "_was": self.GRADED}], "prune")
        self.assertEqual(keep, [])
        self.assertIn("정도 축을 가진 원칙을 지운다", bad[0]["reason"])

    def test_deleting_an_ungraded_line_passes(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        keep, bad = aj.enforce_role(
            [{"op": "delete", "id": "C1", "_was": self.UNGRADED}], "prune")
        self.assertEqual(len(keep), 1)
        self.assertEqual(bad, [])

    def test_pe_is_told_which_units_it_may_delete(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        e = " ".join(aj.ENGINEER_SYSTEM.split())
        self.assertIn("a [Core Principles] line with NO severity clause", e)
        self.assertIn("Deleting a line that has both throws away ordering information", e)

    def test_examples_role_is_warned_about_holdout(self):
        """`examples_only` 는 실측에서 사례에 맞추고 홀드아웃에서 잃었다 — 지시문이 그것을 말해야 한다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        e = " ".join(aj.ENGINEER_SYSTEM.split())
        self.assertIn("examples are the easiest way to FIT the sentences you were shown", e)
        self.assertIn("lost on held-out ones", e)


class SeverityRole(unittest.TestCase):
    """`severity` 는 **등급 안 서열**을 움직이는 역할이다 — 질문이 아니라 정도 절을 고친다.

    왜 이 표면인가. `[Core Principles]` 의 질문 일곱은 전부 예/아니오이고 전부 한 방향이라 등급
    **배정**은 하지만 등급 **안** 서열은 못 만든다. 그런데 골격은 맨 아래 등급까지 서로 다른 번호를
    요구하고 모델은 그것을 지킨다 — 근거 없이 지킨다(순위 깊이 배수 1위 6.16배, 8위 1.27배).
    질문에 "무엇이 더 심하고 무엇이 더 가벼운가" 를 손으로 붙여 재니 dev 500 3벌에서
    **+0.0076 [+0.0028, +0.0127]** 이고 이득이 ≤3 구간에 +0.0143 으로 몰렸다.

    왜 `[Order Principles]` 가 아닌가. 그 칸의 계약은 골격의 등급 안 기본 서열을 **덮어쓰는** 것이라,
    정도 기반 서열이 이득을 내는 위에서는 이득을 깎는다 — 같은 편집이 v0 위 −0.0015, graded 위
    −0.0053 이었다."""

    Q = ("- Would placing a cut here leave trailing sentence-final material attached to the "
         "preceding word, producing an impossible or misleading fragment?")
    SEV = (" A cut here is worse the shorter that trailing remainder is, and mildest when the"
           " remainder is a detachable afterthought.")

    def edit(self, text, was=None, uid="C7", op="replace"):
        return [{"op": op, "id": uid, "text": text, "_was": was if was is not None else self.Q + self.SEV}]

    def test_rewrites_the_severity_clause(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        new = self.Q + (" A cut here is worse the shorter the remainder and the more it is required "
                        "to complete what precedes it; it is mildest when the remainder is a "
                        "detachable afterthought.")
        keep, bad = aj.enforce_role(self.edit(new), "severity")
        self.assertEqual((len(keep), bad), (1, []))

    def test_question_must_come_back_byte_identical(self):
        """질문은 등급 배정을 정하고 그쪽은 이미 작동한다 — 열면 무엇이 움직였는지 갈라지지 않는다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        for tweak in (self.Q.replace("Would", "Does"),          # 낱말 하나
                      self.Q.replace("  ", " ").replace("cut here", "cut  here"),   # 공백
                      self.Q[:-1] + " or an awkward one?"):     # 질문 확장
            keep, bad = aj.enforce_role(self.edit(tweak + self.SEV), "severity")
            self.assertEqual(keep, [], tweak[:40])
            self.assertIn("질문이 바뀌었다", bad[0]["reason"])

    def test_severity_may_not_be_empty(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        keep, bad = aj.enforce_role(self.edit(self.Q), "severity")
        self.assertEqual(keep, [])
        self.assertIn("정도 축이 아니다", bad[0]["reason"])

    def test_one_end_is_direction_not_an_axis(self):
        """한쪽만 쓰면 방향이지 축이 아니다 — 등급 안 두 자리를 여전히 못 가른다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        keep, bad = aj.enforce_role(
            self.edit(self.Q + " A cut here is worse the shorter that remainder is."), "severity")
        self.assertEqual(keep, [])
        self.assertIn("정도 축이 아니다", bad[0]["reason"])

    def test_only_replace_on_a_c_unit(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        k1, b1 = aj.enforce_role(self.edit(self.Q + self.SEV, op="insert_after"), "severity")
        self.assertEqual(k1, []); self.assertIn("replace 만 된다", b1[0]["reason"])
        k2, b2 = aj.enforce_role(self.edit(self.Q + self.SEV, uid="O1"), "severity")
        self.assertEqual(k2, []); self.assertIn("[Core Principles] 단위가 아니다", b2[0]["reason"])

    def test_caps_growth_in_proportion_to_the_line(self):
        """상한은 **대상 줄 크기에 비례**한다 — 절대값으로 두면 구조가 바뀔 때 조용히 병목이 된다.

        실측으로 원칙 한 줄이 질문만 있던 시절 187자에서 질문 + 정도 축 324자로 커졌고, 그러자 옛
        상수 200 이 막기 시작했다: PE 가 +215자를 쓰려는데 15자 차이로 반려돼 후보를 잃었다.
        Writer 길이 지시(9,500자)에서 겪은 것과 같은 종류다 — 정도 축을 요구하기 전에 쓴 숫자였다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        was = self.Q + self.SEV
        cap = aj.growth_cap(was, aj.SEVERITY_GROWTH, aj.unit_ceiling("C1"))
        self.assertGreater(cap, 200)                      # 옛 절대 상한보다 넉넉하다
        ok = aj.enforce_role(self.edit(was + "y" * (cap - 10)), "severity")
        self.assertEqual((len(ok[0]), ok[1]), (1, []))
        keep, bad = aj.enforce_role(self.edit(was + "y" * (cap + 50)), "severity")
        self.assertEqual(keep, [])
        self.assertIn("늘었다", bad[0]["reason"])
        self.assertIn(str(cap), bad[0]["reason"])         # 얼마까지 되는지 알려준다

    def test_short_lines_still_get_a_floor(self):
        """짧은 줄에서도 최소 폭은 준다 — 비율만 쓰면 짧은 단위를 고칠 수 없게 된다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        self.assertEqual(aj.growth_cap("- Is it? Worse when X, milder when Y.", 0.01), aj.MIN_GROWTH)

    def test_pairs_with_binary_findings_and_keeps_its_section(self):
        """이항 발견을 받되 **칸 검사에 걸리지 않는다** — 역할이 칸을 스스로 고정한다.

        이 둘을 따로 정했다가 `enforce_role` 을 통과한 편집이 뒤에서 죽는 일을 두 번 겪었다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        import core.meaning_segmentator.autoseg.loop_judge as lj
        self.assertEqual(lj.ROLE_KINDS["severity"], {"order"})
        e = [{"op": "replace", "id": "C7", "text": self.Q + self.SEV}]
        keep, bad = aj.enforce_kind([dict(x) for x in e], "order", True, "severity")
        self.assertEqual((len(keep), bad), (1, []))
        # 역할을 안 주면 종전대로 칸 검사가 걸린다
        keep2, bad2 = aj.enforce_kind([dict(x) for x in e], "order", True)
        self.assertEqual(keep2, [])

    def test_verbatim_pinning_is_exempt(self):
        """`--pe-verbatim` 이 문면을 Critic 문면으로 갈아 끼우면 질문이 사라져 역할 검사가 거부한다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        finding = {"edit": {"text": "prefer a cut at A over one at B"}}
        e = [{"op": "replace", "id": "C7", "text": self.Q + self.SEV}]
        out, changed = aj.pin_finding_text([dict(x) for x in e], finding, "severity")
        self.assertEqual(changed, [])
        self.assertEqual(out[0]["text"], self.Q + self.SEV)
        # 면제는 **역할이 아니라 칸** 으로 정해진다 — 다른 역할이 C 를 겨눠도 보존된다.
        out2, changed2 = aj.pin_finding_text([dict(x) for x in e], finding, "replace")
        self.assertEqual(changed2, [])
        self.assertEqual(out2[0]["text"], self.Q + self.SEV)
        # O 단위는 되돌린다
        eo = [{"op": "insert_after", "id": "O_end", "text": "prefer the later one"}]
        _o3, changed3 = aj.pin_finding_text(eo, finding, "fallback")
        self.assertEqual(len(changed3), 1)

    def test_whole_edit_path_survives(self):
        """역할 검사를 통과한 편집이 **프롬프트가 되는 데까지** 가는지. 이 확인을 빼먹어 두 번 물렸다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        base = ("[Role]\nr\n\n" + aj.scoring_rules(["German"]) + "\n\n"
                "[Core Principles]\n" + self.Q + self.SEV + "\n\n[Order Principles]\n- prefer A over B\n\n"
                "[Output Rules]\n- x\n\n[Examples]\nInput: a <SEG:?> b\nOutput: a <SEG:5> b\n")
        was = {u["id"]: u["text"] for u in aj.edit_units(base)}["C1"]
        new = self.Q + " A cut here is worse the shorter the remainder; mildest when detachable."
        keep, bad = aj.enforce_role([{"op": "replace", "id": "C1", "text": new, "_was": was}], "severity")
        self.assertEqual((len(keep), bad), (1, []))
        cand, note, *_ = aj.parse_edits({"edits": keep}, base, 20_000)
        self.assertIsNotNone(cand, note)
        self.assertIn("mildest when detachable", cand)
        self.assertEqual(aj.check_skeleton(cand, base=base), [])


class ScoringRulesFrozen(unittest.TestCase):
    """`[Scoring Rules]` 는 **루프가 건드리지 않는다.** judge34 에서 `procedure` 역할로 열어 실측했고
    이득이 없었다(절차 줄 −0.0028 / −0.0037). 같은 발견을 판단 줄에 넣은 것(−0.0080 / −0.0157)보다는
    나아서 자리에 뜻은 있지만 부호가 안 바뀐다. 그래서 다시 닫았다.

    닫는 길이 둘이고 **둘 다 있어야 한다** — `UNIT_TAGS` 에 없어 편집이 단위를 못 만들고,
    `FROZEN` 이 통째 재작성 경로를 되돌린다. 하나만 있으면 다른 쪽으로 샌다."""

    BASE = None

    def base(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        return ("[Role]\nr\n\n" + aj.scoring_rules(["German"]) + "\n\n"
                "[Core Principles]\n- a\n\n[Order Principles]\n- prefer A over B\n\n"
                "[Output Rules]\n- x\n\n[Examples]\nInput: a <SEG:?> b\nOutput: a <SEG:5> b\n")

    def test_no_editable_unit_in_it(self):
        """단위가 없으면 PE 가 가리킬 id 자체가 없다 — 이것이 첫째 자물쇠다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        ids = [u["id"] for u in aj.edit_units(self.base())]
        self.assertEqual([i for i in ids if i.startswith("S")], [])
        self.assertNotIn("[Scoring Rules]", aj.UNIT_TAGS)

    def test_edit_naming_an_s_id_is_dropped(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        base = self.base()
        cand, note, *_ = aj.parse_edits(
            {"edits": [{"op": "replace", "id": "S7", "text": "- rewritten procedure"}]},
            base, 20_000)
        self.assertIsNone(cand)

    def test_whole_prompt_path_restores_it(self):
        """둘째 자물쇠 — Writer 가 통째로 다시 써도 `FROZEN` 이 되돌린다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        base = self.base()
        tampered = base.replace("Work in bands", "Just pick the best one")
        self.assertIn("[Scoring Rules]", aj.frozen_intact(base, tampered))
        cand, note = aj.parse_prompt({"prompt": tampered}, base, 20_000)
        self.assertIsNone(cand)
        self.assertTrue(any("[Scoring Rules]" in n for n in note), note)

    def test_nothing_is_unfrozen_for_any_role(self):
        """열려 있는 역할이 하나도 없다. 다시 열려면 `ROLE_UNFREEZES` 한 군데만 고친다 —
        어느 역할에 무엇이 열려 있는지를 두 군데에서 정했다가 `enforce_role` 을 통과한 편집이
        `frozen_intact` 에서 다시 죽는 일을 겪었다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        self.assertEqual(aj.ROLE_UNFREEZES, {})
        for r in aj.ROLES:
            self.assertEqual(aj.unfreezes(r), (), r)

    def test_procedure_role_is_gone(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        import core.meaning_segmentator.autoseg.loop_judge as lj
        self.assertNotIn("procedure", aj.ROLES)
        self.assertNotIn("procedure", lj.UNPAIRED_ROLES)
        self.assertNotIn('"procedure"', aj.ENGINEER_SYSTEM)


class FixedSkeleton(unittest.TestCase):
    """`[Scoring Rules]` 는 **사람이 정하고 코드가 주입하는** 골격이다. Writer 에게 맡기면 선언형
    ("결과가 순위여야 한다")으로 돌아가는데, 절차형과의 차이가 test 560 에서 +0.0084 였다."""

    def test_carries_the_band_procedure(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        sr = aj.scoring_rules(["Chinese", "German"])
        self.assertTrue(sr.startswith("[Scoring Rules]"))
        for must in ("five bands", "one position at a time", "choose again among the rest",
                     "Core Principles decide", "Order Principles are its EXCEPTIONS"):
            self.assertIn(must, sr)
        self.assertIn("Chinese, German", sr)
        self.assertNotIn("__TARGETS__", sr)

    def test_overwrites_a_writer_declarative_section(self):
        """Writer 가 쓴 선언형 절을 주입본이 덮는다 — v0 생성 런에서 이것이 골격을 지킨다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        wrote = ("[Role]\nr\n\n[Scoring Rules]\n- the integer is a RANK; spread distinct integers\n\n"
                 "[Core Principles]\n- a\n\n[Order Principles]\n- prefer A over B\n\n"
                 "[Output Rules]\n- x\n\n[Examples]\nInput: a <SEG:?> b\nOutput: a <SEG:5> b\n")
        fixed = aj.replace_section(wrote, "[Scoring Rules]", aj.scoring_rules(["German"]))
        self.assertIn("five bands", fixed)
        self.assertNotIn("spread distinct integers\n", fixed)
        self.assertEqual(aj.check_skeleton(fixed, base=fixed), [])
        # 판단 두 칸과 예시는 그대로 남고, 주입한 골격은 **편집 단위를 하나도 만들지 않는다.**
        self.assertEqual([u["id"] for u in aj.edit_units(fixed)], ["C1", "O1", "E1"])

    def test_writer_is_told_what_each_section_decides(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        ws = " ".join(aj.writer_system(True, ["German"]).split())
        self.assertIn("[Core Principles] decides WHICH BAND", ws)
        self.assertIn("Every line is UNARY", ws)
        self.assertIn("[Order Principles] holds the EXCEPTIONS to ordering by severity", ws)
        self.assertIn("The question decides the band; the severity decides the order INSIDE a band", ws)
        self.assertIn("Every line is BINARY", ws)
        self.assertIn("[Scoring Rules], [Core Principles], [Order Principles], [Output Rules], "
                      "[Examples]", ws)
        # 두 골격 절은 받아서 그대로 쓴다.
        self.assertIn("[Output Rules] and [Scoring Rules] MUST be copied verbatim", ws)


class InlineHeaders(unittest.TestCase):
    """본문 중간에 쓰인 섹션 헤더는 **경계로 잡혀 그 자리에서 섹션을 자른다.** judge33 의 v0 후보
    둘이 [Role] 안에서 골격을 대괄호째 참조해("must feed the procedure in [Scoring Rules]"),
    골격 주입이 그 문장 중간부터 다음 경계까지를 갈아 [Role] 뒷부분이 유실됐다. 골격 검사는
    그것을 못 잡는다 — 문자열이 있으니 "섹션 없음" 이 아니다."""

    BAD = ("[Role]\nYou must feed the procedure in [Scoring Rules]. Score every marker.\n\n"
           "[Core Principles]\n- a\n\n[Output Rules]\n- x\n\n"
           "[Examples]\nInput: a <SEG:?> b\nOutput: a <SEG:5> b\n")

    def test_strips_only_inline(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        fixed = aj.strip_inline_headers(self.BAD)
        self.assertNotIn("[Scoring Rules]", fixed)
        self.assertIn("Scoring Rules section", fixed)      # 참조 의도는 남는다
        for h in ("[Role]", "[Core Principles]", "[Output Rules]", "[Examples]"):
            self.assertEqual(fixed.count(h), 1, h)         # 줄 맨 앞 헤더는 건드리지 않는다
        self.assertIn("Score every marker.", fixed)        # 뒤 문장이 남아 있다

    def test_role_survives_injection_after_stripping(self):
        """치환을 주입 **전에** 해야 한다 — 순서가 바뀌면 그 자리가 경계가 된 뒤다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        bad_injected = aj.replace_section(self.BAD, "[Scoring Rules]",
                                          aj.scoring_rules(["German"]))
        self.assertNotIn("Score every marker.", bad_injected)   # 유실된다
        good = aj.replace_section(aj.strip_inline_headers(self.BAD), "[Scoring Rules]",
                                  aj.scoring_rules(["German"]))
        self.assertIn("Score every marker.", good)              # 살아남는다
        self.assertIn("five bands", good)

    def test_writer_is_forbidden_to_write_them(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        ws = " ".join(aj.writer_system(True, ["German"]).split())
        self.assertIn("NEVER write a bracketed section header inside the body", ws)


class ReplaceRole(unittest.TestCase):
    """길이 중립 편집 — 원칙 하나를 지우고 그 자리에 발견을 넣는다. 문장을 더한 스물세 번 중
    스물두 번이 음수였고, 지운 편집(C4)은 −0.0001 이었다. 손해가 길이에서 온다면 여기서 0 에서
    출발한다."""

    WAS = ("- Does a following relative clause restrict a preceding noun phrase so that separating "
           "them changes reference? Separating head and modifier lowers cohesion.")

    def pair(self, text, del_id="C4", ins_id="O_end"):
        return [{"op": "delete", "id": del_id, "_was": self.WAS},
                {"op": "insert_after", "id": ins_id, "text": text}]

    def test_accepts_length_neutral_pair(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        keep, bad = aj.enforce_role(
            self.pair("- prefer a cut at A over one at B when both sit in the same band."),
            "replace")
        self.assertEqual((len(keep), bad), (2, []))

    def test_keeps_the_principle_count_not_the_length(self):
        """**문장 수 중립이지 길이 중립이 아니다.**

        처음에는 길이 중립으로 잡았다(대상의 30%). 그런데 원칙 줄이 질문 + 정도 축 두 부분이 되면서
        그 조건이 만족 불가능해졌다 — 발견을 그 형태로 쓰면 자연히 300~400자이고 338자 단위를
        교체하면 순증가가 +200 쯤 난다. 실측으로 다섯 번 연속 걸렸다(+133/+166/+171/+208/+216 대
        상한 80~113). 조건을 두 군데서 다르게 건 다섯 번째 사례다.

        그래서 지킬 것을 문장 수로 옮겼다: 원칙 개수는 그대로(한 줄 지우고 한 줄 넣기), 새 줄은
        한 줄, 글자 수는 두 배까지. 전체 증가는 이터 천장이 따로 막는다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        # 정도 축을 갖춘 실제 크기의 교체는 통과해야 한다
        ok, bad0 = aj.enforce_role(self.pair("- " + "word " * 60), "replace")
        self.assertEqual(bad0, [], bad0)
        self.assertTrue(ok)
        # 두 배를 넘으면 막는다
        keep, bad = aj.enforce_role(self.pair("- " + "word " * 300), "replace")
        self.assertEqual(keep, [])
        self.assertIn("늘었다", bad[0]["reason"])

    def test_rejects_extra_lines(self):
        """한 줄이 한 단위다 — 줄을 늘리면 원칙 개수가 늘어난다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        keep, bad = aj.enforce_role(self.pair("- first line\n- second line"), "replace")
        self.assertEqual(keep, [])
        self.assertIn("여러 줄", bad[0]["reason"])

    def test_requires_both_ops(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        for edits in ([{"op": "insert_after", "id": "C_end", "text": "- keep X with Y"}],
                      [{"op": "delete", "id": "C4", "_was": self.WAS}]):
            keep, bad = aj.enforce_role(edits, "replace")
            self.assertEqual(keep, [], edits)
            self.assertTrue(bad)

    def test_order_principles_line_is_replaceable(self):
        """사람이 박아 둔 [Order Principles] 시작 문장을 **루프가 실측으로 교체할 수 있어야 한다.**
        손댈 수 없게 두면 검증되지 않은 한 줄이 런 끝까지 남는다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        one = [{"op": "replace", "id": "O1", "_was": "- " + "word " * 70,
                "text": "- prefer the cut that does not split a verb from its object."}]
        keep, bad = aj.enforce_role(one, "replace", {"order_units": 1})
        self.assertEqual((len(keep), bad), (1, []))
        self.assertEqual(aj.enforce_kind(one, "order")[1], [])

    def test_last_order_line_may_not_be_deleted(self):
        """칸이 비면 골격이 참조할 것이 없다 — 교체는 되고 삭제는 안 된다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        pair = [{"op": "delete", "id": "O1", "_was": "- " + "word " * 70},
                {"op": "insert_after", "id": "C_end", "text": "- keep X with Y"}]
        keep, bad = aj.enforce_role(pair, "replace", {"order_units": 1})
        self.assertEqual(keep, [])
        self.assertIn("마지막 한 줄", bad[0]["reason"])
        # 줄이 둘 이상이면 지울 수 있다.
        self.assertEqual(len(aj.enforce_role(pair, "replace", {"order_units": 2})[0]), 2)
        # `prune` 도 같은 규칙을 지킨다.
        solo = [{"op": "delete", "id": "O1", "_was": "- prefer A over B"}]
        self.assertEqual(aj.enforce_role(solo, "prune", {"order_units": 1})[0], [])
        self.assertEqual(len(aj.enforce_role(solo, "prune", {"order_units": 2})[0]), 1)

    def test_rejects_non_unit_target(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        keep, bad = aj.enforce_role(
            [{"op": "delete", "id": "E2", "_was": "Input: a\nOutput: b"},
             {"op": "insert_after", "id": "C_end", "text": "- keep X with Y"}], "replace")
        self.assertEqual(keep, [])
        self.assertIn("[Order Principles] 단위가 아니다", bad[0]["reason"])

    def test_refuses_replacing_the_same_unit_twice(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        keep, bad = aj.enforce_role(self.pair("- prefer a cut at A over one at B."), "replace",
                                    {"replaced": {self.WAS}})
        self.assertEqual(keep, [])
        self.assertIn("이미 교체한", bad[0]["reason"])

    def test_a_unit_a_sibling_deleted_is_still_replaceable(self):
        """형제가 지운 줄을 교체하는 것은 **다른 편집**이다 — 한 집합에 섞으면 대상이 몇 줄뿐인
        역할의 후보 자리가 통째로 빈다(실측으로 `prune` 이 세 이터 연속)."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        keep, bad = aj.enforce_role(self.pair("- prefer a cut at A over one at B."), "replace",
                                    {"deleted": {self.WAS}})
        self.assertEqual(len(keep), 2)
        self.assertEqual(bad, [])

    def test_a_unit_a_sibling_replaced_is_still_prunable(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        edit = [{"op": "delete", "id": "O2", "_was": "- prefer A over B"}]
        self.assertEqual(len(aj.enforce_role(edit, "prune",
                                             {"replaced": {"- prefer A over B"},
                                              "order_units": 3})[0]), 1)
        self.assertEqual(aj.enforce_role(edit, "prune",
                                         {"deleted": {"- prefer A over B"},
                                          "order_units": 3})[0], [])

    def test_kind_check_still_bars_binary_text(self):
        """이 역할도 `enforce_kind` 를 지난다 — 단항 발견에 비교문을 쓸 수는 없다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        keep, _bad = aj.enforce_role(
            self.pair("- prefer a cut at A over one at B.", ins_id="C_end"), "replace")
        kept, bad = aj.enforce_kind(keep, "check")
        self.assertTrue(bad)
        # 이항 발견이면 추가는 O 칸으로 가고, 삭제는 C 여도 통과한다.
        kept2, bad2 = aj.enforce_kind(
            aj.enforce_role(self.pair("- prefer a cut at A over one at B."), "replace")[0], "order")
        self.assertEqual((len(kept2), bad2), (2, []))

    def test_pairs_with_both_kinds(self):
        """`ROLE_KINDS` 에 없으므로 check·order 양쪽과 짝지어진다 — 어느 칸에든 넣을 수 있다."""
        plan = lj.candidate_plan(["fallback", "single_small", "replace", "prune"], 3, True, 4, 16,
                                 kinds=["order", "check", "check"])
        self.assertEqual([r for r, _i in plan].count("replace"), 3)
        self.assertEqual(sorted(i for r, i in plan if r == "replace"), [0, 1, 2])


class PinFindingText(unittest.TestCase):
    """PE 가 쓴 문면을 Critic 문면으로 되돌린다."""

    FIND = {"kind": "check", "edit": {"text": "Keep a head with its modifier."}}

    def test_replaces_text_in_the_order_section(self):
        """`[Order Principles]` 단위에는 걸린다 — 그 칸의 형태가 이항 비교이고 발견 문면이 그것이다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        out, ch = aj.pin_finding_text(
            [{"op": "insert_after", "id": "O_end", "text": "rank the later cut higher"}], self.FIND)
        self.assertEqual(out[0]["text"], "Keep a head with its modifier.")
        self.assertEqual(len(ch), 1)

    def test_never_replaces_text_in_the_core_section(self):
        """**PE 가 옳게 쓴 줄을 되돌림이 망치던 자리다.**

        `[Core Principles]` 단위는 질문 + 정도 축 두 부분이어야 하는데 Critic 의 발견 문면은 그 형태가
        아니다 — 평서문이거나 금지문이다. 되돌리면 모양 검사나 금지문 검사가 **반드시** 거부한다.
        실측으로 한 이터에서 후보 다섯 중 둘을 그렇게 잃었고, 로그만 보면 PE 탓처럼 보였다.

        되돌림이 막으려던 것(단항 발견을 서열문으로 고쳐 쓰기)은 모양 검사가 대신 막는다 —
        질문 + 정도 축은 그 자체로 단항이라 이항으로 새어 나갈 수 없다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        good = ("- Does cutting here strand a head that needs a complement? Worst when the complement"
                " is required, mildest when it is optional.")
        out, ch = aj.pin_finding_text(
            [{"op": "insert_after", "id": "C_end", "text": good}], self.FIND)
        self.assertEqual(out[0]["text"], good)
        self.assertEqual(ch, [])
        # 그리고 그 편집은 모양 검사를 통과한다 — 되돌리면 통과하지 못한다
        self.assertEqual(aj.check_core_unit_shape(out, "narrow_rule")[1], [])
        pinned = [{"op": "insert_after", "id": "C_end", "text": self.FIND["edit"]["text"]}]
        self.assertEqual(aj.check_core_unit_shape(pinned, "narrow_rule")[0], [])

    def test_leaves_delete_and_examples(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        edits = [{"op": "delete", "id": "C2", "_was": "old"},
                 {"op": "insert_after", "id": "E_end", "labeled_example": "en_us_1"}]
        out, ch = aj.pin_finding_text(edits, self.FIND)
        self.assertEqual((out, ch), (edits, []))

    def test_noop_when_already_same(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        out, ch = aj.pin_finding_text(
            [{"op": "insert_after", "id": "C_end",
              "text": "Keep  a head   with its modifier."}], self.FIND)
        self.assertEqual(ch, [])


class OrderRulesSection(unittest.TestCase):
    """`[Order Principles]` 는 편집 가능한 칸이어야 하고, 없는 프롬프트도 골격 검사를 통과해야 한다."""

    def test_unit_ids(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        pr = ("[Role]\nr\n\n[Scoring Rules]\ns\n\n[Core Principles]\n- a\n\n[Order Principles]\n- o1\n- o2\n\n"
              "[Output Rules]\n- x\n\n[Examples]\nInput: a <SEG:?> b\nOutput: a <SEG:5> b\n")
        ids = [u["id"] for u in aj.edit_units(pr)]
        self.assertIn("O1", ids)
        self.assertIn("O2", ids)
        self.assertEqual(aj.check_skeleton(pr, base=pr), [])

    def test_section_name_in_body_is_caught(self):
        """본문에 섹션 헤더 문자열을 쓰면 그것이 경계로 잡혀 **그 자리에서 섹션이 잘린다.**
        프롬프트를 손으로 쓸 때 실제로 밟은 함정이다 — 골격 검사가 잡아야 한다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        pr = ("[Role]\nr\n\n[Scoring Rules]\ns\n\n[Core Principles]\n- a\n\n"
              "[Order Principles]\n- these refine what [Core Principles] decided\n\n"
              "[Output Rules]\n- x\n\n[Examples]\nInput: a <SEG:?> b\nOutput: a <SEG:5> b\n")
        self.assertTrue(aj.check_skeleton(pr, base=pr))
        # 편집으로도 들어갈 수 없다.
        _pr, errs, _d = aj.apply_edits(
            pr, [{"op": "insert_after", "id": "O_end", "text": "- see [Order Principles]"}])
        self.assertTrue(errs)

    def test_editing_principles_keeps_order_principles(self):
        """경계 목록에 이 칸이 없으면 `[Core Principles]` 를 고치는 순간 뒤 칸이 통째로 사라진다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        pr = ("[Role]\nr\n\n[Scoring Rules]\ns\n\n[Core Principles]\n- a\n\n[Order Principles]\n- prefer A over B\n\n"
              "[Output Rules]\n- x\n\n[Examples]\nInput: a <SEG:?> b\nOutput: a <SEG:5> b\n")
        new, errs, _d = aj.apply_edits(
            pr, [{"op": "insert_after", "id": "C_end", "text": "- keep X with Y"}])
        self.assertEqual(errs, [])
        self.assertIn("[Order Principles]", new)
        self.assertIn("prefer A over B", new)
        self.assertEqual(aj.check_skeleton(new, base=pr), [])

    def test_order_end_anchor(self):
        """`O_end` 가 앵커로 풀려야 한다 — 정규식에 머리글자를 놓치면 "없는 id" 로 반려된다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        pr = ("[Role]\nr\n\n[Scoring Rules]\ns\n\n[Core Principles]\n- a\n\n[Order Principles]\n- prefer A over B\n\n"
              "[Output Rules]\n- x\n\n[Examples]\nInput: a <SEG:?> b\nOutput: a <SEG:5> b\n")
        for uid in ("O_end", "O2", "O0"):
            new, errs, _d = aj.apply_edits(
                pr, [{"op": "insert_after", "id": uid, "text": "- prefer C over D"}])
            self.assertEqual(errs, [], uid)
            self.assertEqual(len([u for u in aj.edit_units(new) if u["id"].startswith("O")]), 2, uid)

    def test_units_reach_pe_payload(self):
        """PE 입력의 단위 목록과 출처표에 새 칸이 실려야 한다 — 실리지 않으면 PE 는 그 칸을
        모르고, 골격이 참조하는 자리가 비어 있는 채로 이터가 돈다."""
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        pr = ("[Role]\nr\n\n[Scoring Rules]\ns\n\n[Core Principles]\n- a\n\n[Order Principles]\n- prefer A over B\n\n"
              "[Output Rules]\n- x\n\n[Examples]\nInput: a <SEG:?> b\nOutput: a <SEG:5> b\n")
        prov = aj.init_provenance(pr)
        ids = [u["id"] for u in aj.units_with_provenance(pr, prov)]
        self.assertIn("O1", ids)
        self.assertIn("[Order Principles]", aj.section_sizes(pr))

    def test_optional_when_absent_from_base(self):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        pr = ("[Role]\nr\n\n[Scoring Rules]\ns\n\n[Core Principles]\n- a\n\n[Output Rules]\n- x\n\n"
              "[Examples]\nInput: a <SEG:?> b\nOutput: a <SEG:5> b\n")
        self.assertEqual(aj.check_skeleton(pr, base=pr), [])


class CandidatePlanKinds(unittest.TestCase):
    """형식이 내용을 담을 수 있는 짝만 곱한다. `order`(두 자리 비교)는 선호문 역할만,
    `check`(단항 조건)는 결속문·귀납 역할만. `prune` 은 발견과 무관하게 하나."""

    ROLES = ["fallback", "single_small", "induce", "prune"]

    def test_pairs_by_kind(self):
        plan = lj.candidate_plan(self.ROLES, 3, True, 4, 16,
                                 kinds=["check", "order", "check"])
        self.assertEqual(plan, [("fallback", 1),
                                ("single_small", 0), ("single_small", 2),
                                ("induce", 0), ("induce", 2),
                                ("prune", 0)])

    def test_prune_appears_once_even_with_many_findings(self):
        plan = lj.candidate_plan(["prune"], 3, True, 4, 16, kinds=["check"] * 3)
        self.assertEqual(plan, [("prune", 0)])

    def test_unknown_role_takes_both(self):
        plan = lj.candidate_plan(["free"], 2, True, 4, 16, kinds=["check", "order"])
        self.assertEqual(plan, [("free", 0), ("free", 1)])

    def test_falls_back_when_no_pair_survives(self):
        """`order` 만 나왔는데 역할이 결속문뿐이면 이터를 버리는 대신 종전 곱으로 돈다."""
        plan = lj.candidate_plan(["single_small"], 1, True, 4, 16, kinds=["order"])
        self.assertEqual(plan, [("single_small", 0)])

    def test_without_kinds_is_the_old_product(self):
        old = lj.candidate_plan(self.ROLES, 2, True, 4, 16)
        self.assertEqual(len(old), 8)
        self.assertEqual(old[0], ("fallback", 0))
        self.assertEqual(old[-1], ("prune", 1))

    def test_cap(self):
        plan = lj.candidate_plan(self.ROLES, 3, True, 4, 3, kinds=["check", "order", "check"])
        self.assertEqual(len(plan), 3)


class HistoryDepth(unittest.TestCase):
    """PE 는 이력으로 직전 개정을 읽는다. `by_bin` 만으로는 앞쪽 순위만 고친 편집과 깊은 쪽을
    고친 편집이 구별되지 않으므로 깊이 변화도 같이 넘긴다."""

    def brief(self, diag):
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        return aj.history_brief([{"iter": 1, "adopted": False, "edits": [],
                                  "delta": {"mean": -0.002, "lo": -0.01},
                                  "diagnosis": diag}])[0]

    def test_passes_depth_delta(self):
        b = self.brief({"by_bin": {"≤3": {"mean": -0.001}},
                        "rank_depth": {"base": {"1": 6.8}, "revision": {"1": 7.0},
                                       "delta": {"1": 0.21, "8": -0.01}}})
        self.assertEqual(b["rank_depth_delta"], {"1": 0.21, "8": -0.01})

    def test_absent_when_not_measured(self):
        b = self.brief({"by_bin": {"≤3": {"mean": -0.001}}})
        self.assertNotIn("rank_depth_delta", b)
        b = self.brief({"rank_depth": {"base": {}, "revision": {}}})
        self.assertNotIn("rank_depth_delta", b)


class PromptsAreEnvironmentFree(unittest.TestCase):
    """**에이전트 지시문에 이 저장소의 런 이름이나 그 런에서 잰 값을 넣지 않는다.** 다른 코퍼스나
    언어로 옮기면 검증할 수 없는 주장이 되고, 모델은 그것을 사실로 읽는다. 런에서 나오는 수치는
    페이로드(`loss_by_bin`·`rank_depth`·`misorder_cost`·`rejected_by_bin`)로 매 런 다시 계산해
    넘긴다. 사람이 읽는 자리(코드 주석·argparse 도움말)는 이 규칙의 대상이 아니다."""

    def test_no_run_names_or_measurements(self):
        import re
        import core.meaning_segmentator.autoseg.runtime.agents_judge as aj
        bad = re.compile(r'judge\d+|\brun\d\d\b|\d+\.\d+x|\d+ of \d+ measured'
                         r'|five times out of five')
        for name in dir(aj):
            v = getattr(aj, name)
            if name.isupper() and isinstance(v, str) and len(v) > 150:
                self.assertIsNone(bad.search(v), f"{name} 에 런 고유 정보가 있다: "
                                                 f"{(bad.search(v) or '').group(0) if bad.search(v) else ''}")


class ScoreThresholdGrid(unittest.TestCase):
    """점수 임계값을 지연 노브로 쓴다 — T 격자와 나란히 놓을 조건을 만든다.

    T 는 문장 길이만 보고 `k = 어절/T` 를 정하므로 그 문장에 좋은 자리가 있는지와 무관하게 같은
    수를 자른다. 임계값은 점수가 정한다 — 자리가 좋은 문장은 더 잘리고 없는 문장은 덜 잘린다.
    임계값을 올리면 무분절로 수렴하므로 최소 절단을 억지로 보장하지 않는다(그게 곡선의 끝점이다).

    쓸 수 있는 근거는 모델이 문장마다 0~100 을 거의 꽉 쓴다는 실측이다 — covost2 15,525문장에서
    문장 최고점 중앙 99, 최저점 중앙 0, 임계 10/30/50/70/90 에서 문장당 절단
    6.13/4.48/3.42/2.23/1.16 으로 매끄럽게 줄어든다.
    """

    SEG = "Do <SEG:0> you <SEG:50> mean <SEG:100> it?"
    SRC = "Do you mean it?"

    def cut(self, th, spaced=True, seg=None, src=None):
        from core.meaning_segmentator.autoseg.scoring.bleu_eval import cut_at_threshold
        return cut_at_threshold(seg or self.SEG, src or self.SRC, th, spaced)

    def test_raising_the_threshold_merges_pieces(self):
        self.assertEqual(self.cut(0), ["Do", "you", "mean", "it?"])
        self.assertEqual(self.cut(40), ["Do you", "mean", "it?"])
        self.assertEqual(self.cut(60), ["Do you mean", "it?"])

    def test_the_top_of_the_grid_is_unsegmented(self):
        """임계값이 최고점을 넘으면 문장 전체 한 조각 — 곡선의 끝점이다."""
        self.assertEqual(self.cut(101), [self.SRC])

    def test_pieces_rejoin_into_the_source(self):
        """**조각을 이어 붙이면 원문이어야 한다.** 마킹 형식은 태그 양옆에 공백을 두므로 태그를
        버리고 이으면 영어는 공백이 둘이 되고 일본어는 없던 공백이 생긴다 — 그러면 번역 결과가
        원문과 다른 문장의 것이 되어 BLEU 가 흔들린다."""
        for th in (0, 40, 60, 101):
            self.assertEqual(" ".join(self.cut(th)), self.SRC, f"임계 {th}")
        ja_seg, ja_src = "私は <SEG:80> 学校に <SEG:20> 行く", "私は学校に行く"
        for th in (0, 50, 90):
            self.assertEqual("".join(self.cut(th, False, ja_seg, ja_src)), ja_src, f"임계 {th}")

    def test_unscored_tags_and_no_tags_still_work(self):
        self.assertEqual(self.cut(50, True, "a <SEG> b", "a b"), ["a", "b"])
        self.assertEqual(self.cut(50, True, "a b c", "a b c"), ["a b c"])

    def test_grid_shows_up_as_its_own_conditions(self):
        from core.meaning_segmentator.autoseg.scoring.bleu_eval import build_conditions
        rows = [{"text": self.SRC, "seg_text": self.SEG, "by_T": {}}]
        conds = build_conditions(rows, [], True, 8, has_auto=False, score_grid=[40, 90])
        self.assertIn("auto_S40", conds)
        self.assertIn("auto_S90", conds)
        self.assertEqual(conds["auto_S40"][0]["pieces"], ["Do you", "mean", "it?"])
        self.assertEqual(conds["unsegmented"][0]["pieces"], [self.SRC])
