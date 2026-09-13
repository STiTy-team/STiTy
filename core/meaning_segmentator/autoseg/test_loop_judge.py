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


class Decide(unittest.TestCase):
    def test_하한이_충분히_크면_바로_채택(self):
        self.assertEqual(lj.decide({"lo": 0.02}), "accept")

    def test_경계선은_재채점(self):
        self.assertEqual(lj.decide({"lo": 0.002}), "confirm")

    def test_하한이_0_이하면_기각(self):
        self.assertEqual(lj.decide({"lo": 0.0}), "reject")
        self.assertEqual(lj.decide({"lo": -0.01}), "reject")


class ErrorType(unittest.TestCase):
    def test_뺀_자리가_더_낮으면_경계오류(self):
        self.assertEqual(lj.error_type([{"H": 0.2}], [{"H": 0.8}]), "boundary")

    def test_뺀_자리가_더_높으면_상호작용오류(self):
        """혼자서는 좋은 자리인데 조합에서 틀린 것 — 목표가 탐색 최적일 때만 나온다."""
        self.assertEqual(lj.error_type([{"H": 0.9}], [{"H": 0.5}]), "interaction")

    def test_한쪽이_비면_경계오류로_둔다(self):
        self.assertEqual(lj.error_type([], [{"H": 0.5}]), "boundary")


class Sets(unittest.TestCase):
    def setUp(self):
        self.texts = ["a b c d e f g h i j k l"]
        self.sents = [sent(0, self.texts[0])]

    def test_오라클은_라벨_상위_k개(self):
        lab = labels_for(self.texts, lambda i, j: 0.0, lambda i, j: 1.0 if j == 6 else 0.1)
        got = lj.oracle_sets(lab, self.sents, spaced=True, min_gap=1)
        self.assertEqual(got[(0, 1)], (6,))

    def test_k_를_1부터_훑는다(self):
        """T 격자와 달리 문장마다 k 가 1..kmax 로 전부 들어온다."""
        lab = labels_for(self.texts, lambda i, j: 0.0, lambda i, j: 0.1 * j)
        got = lj.oracle_sets(lab, self.sents, spaced=True, min_gap=1)
        ks = sorted(k for _i, k in got)
        self.assertEqual(ks, [1, 2, 3, 4, 5])          # 12어절 / 평균 조각 2 이상
        self.assertEqual(len(got[(0, 3)]), 3)

    def test_contra_가_높은_자리는_밀린다(self):
        lab = labels_for(self.texts, lambda i, j: 0.99 if j == 6 else 0.0,
                         lambda i, j: 1.0 if j in (6, 9) else 0.1)
        got = lj.oracle_sets(lab, self.sents, spaced=True, min_gap=1)
        self.assertEqual(got[(0, 1)], (9,))

    def test_정책은_채점_행에서_나온다(self):
        rows = [{"id": "s0", "positions": [3, 6, 9], "scores": [10, 90, 20]}]
        got = lj.policy_sets(rows, self.sents, spaced=True, min_gap=1)
        self.assertEqual(got[(0, 1)], (6,))
        self.assertEqual(got[(0, 2)], (6, 9))

    def test_점수가_없는_문장은_건너뛴다(self):
        rows = [{"id": "s0", "positions": [3], "scores": None}]
        self.assertEqual(lj.policy_sets(rows, self.sents, True, 1), {})


class SearchSets(unittest.TestCase):
    def test_T별_최고_집합을_고르고_값을_그대로_쓴다(self):
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

    def test_격자에_없는_T는_뺀다(self):
        import json, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.json"
            p.write_text(json.dumps({"0": {"12": {"greedy": [3], "sets": {"3": 0.5}}}}),
                         encoding="utf-8")
            sets, _ = lj.search_sets(p, [4, 6], limit=5)
        self.assertEqual(sets, {})


class Latency(unittest.TestCase):
    def test_평균_조각으로_묶는다(self):
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

    def test_손해가_없으면_사례가_없다(self):
        pol = {(0, 1): (3,)}
        got = lj.build_cases(self.sents, self.lab, pol, pol, {(0, 1): 0.7}, {(0, 1): 0.7},
                             True, 1, 5)
        self.assertEqual(got, [])

    def test_뺀_자리와_넣은_자리를_모두_싣는다(self):
        pol, ora = {(0, 1): (6,)}, {(0, 1): (3,)}
        got = lj.build_cases(self.sents, self.lab, pol, ora, {(0, 1): 0.60}, {(0, 1): 0.75},
                             True, 1, 5)
        self.assertEqual(len(got), 1)
        c = got[0]
        self.assertEqual(c["cuts"], 1)
        self.assertAlmostEqual(c["avg_chunk"], 6.0)
        self.assertEqual([d["pos"] for d in c["diff"]["dropped"]], [6])
        self.assertEqual([d["pos"] for d in c["diff"]["added"]], [3])
        self.assertAlmostEqual(c["gap"], 0.15, places=4)
        self.assertIn("‖", c["policy"]["text"])

    def test_경계별_H_가_낮은_자리를_골랐으면_경계오류로_분류(self):
        pol, ora = {(0, 1): (6,)}, {(0, 1): (3,)}    # 뺀 자리 H 0.2 < 넣은 자리 0.9
        got = lj.build_cases(self.sents, self.lab, pol, ora, {(0, 1): 0.6}, {(0, 1): 0.75},
                             True, 1, 5)
        self.assertEqual(got[0]["error_type"], "boundary")

    def test_경계별로는_좋은_자리를_골랐으면_상호작용오류(self):
        pol, ora = {(0, 1): (3,)}, {(0, 1): (6,)}    # 뺀 자리 H 0.9 > 넣은 자리 0.2
        got = lj.build_cases(self.sents, self.lab, pol, ora, {(0, 1): 0.6}, {(0, 1): 0.75},
                             True, 1, 5)
        self.assertEqual(got[0]["error_type"], "interaction")

    def test_손해_큰_순으로_자른다(self):
        texts = ["a b c d e f g h i j k l"] * 3
        sents = [sent(i, t) for i, t in enumerate(texts)]
        lab = labels_for(texts, lambda i, j: 0.0, lambda i, j: 0.5)
        pol = {(i, 1): (6,) for i in range(3)}
        ora = {(i, 1): (3,) for i in range(3)}
        pol_h = {(0, 1): 0.5, (1, 1): 0.1, (2, 1): 0.3}
        ora_h = {(0, 1): 0.6, (1, 1): 0.9, (2, 1): 0.5}
        got = lj.build_cases(sents, lab, pol, ora, pol_h, ora_h, True, 1, 2)
        self.assertEqual([c["id"] for c in got], ["s1", "s2"])


if __name__ == "__main__":
    unittest.main()
