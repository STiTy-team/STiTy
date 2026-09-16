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
        pr = ("[Role]\nr\n\n[Core Principles]\n- one\n\n[Examples]\n"
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
