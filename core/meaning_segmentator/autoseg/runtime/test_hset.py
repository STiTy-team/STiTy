"""`PYTHONPATH=. python -m unittest core.meaning_segmentator.autoseg.runtime.test_hset`"""
import unittest

from . import hset as H


class FakeTranslator:
    """조각을 대문자로 바꿔 돌려준다 — 이어붙임이 조각 순서를 지키는지 볼 수 있다."""
    def __init__(self):
        self.calls = 0

    def full(self, texts):
        self.calls += len(texts)
        return [t.upper() for t in texts]


class FakeQE:
    """조각 수가 적을수록 높은 점수 — 절단이 많을수록 손해라는 흔한 성질을 흉내낸다."""
    def score(self, srcs, hyps):
        return [1.0 / (1 + h.count(" | ")) for h in hyps]


class CutSet(unittest.TestCase):
    def test_top_k(self):
        u = "a b c d e f g h".split()
        sc = {j: 0.1 * j for j in range(1, 8)}      # 뒤쪽 자리일수록 높다
        got = H.cut_set(u, sc, T=4, spaced=True, min_gap=1)
        self.assertEqual(len(got), 1)               # 8어절 / T4 → 경계 1개
        self.assertEqual(got, (7,))

    def test_min_gap(self):
        u = "a b c d e f g h i j k l".split()
        sc = {j: (1.0 if j in (5, 6) else 0.1) for j in range(1, 12)}
        got = H.cut_set(u, sc, T=4, spaced=True, min_gap=3)
        self.assertNotIn((5, 6), [got])             # 붙어 있는 둘을 같이 못 고른다
        for x, y in zip(got, got[1:]):
            self.assertGreaterEqual(y - x, 3)

    def test_no_scores(self):
        self.assertEqual(H.cut_set("a b c".split(), {}, 2, True, 1), ())


class TopK(unittest.TestCase):
    def test_top_k_positions(self):
        u = "a b c d e f".split()
        sc = {1: 0.1, 2: 0.9, 3: 0.5, 4: 0.7, 5: 0.2}
        self.assertEqual(H.top_k_cuts(u, sc, 2, min_gap=1), (2, 4))

    def test_tie_earlier_wins(self):
        u = "a b c d e".split()
        sc = {1: 0.5, 2: 0.5, 3: 0.1, 4: 0.1}
        self.assertEqual(H.top_k_cuts(u, sc, 1, min_gap=1), (1,))

    def test_min_gap_at_edges(self):
        """자리 1 은 앞 조각이 1어절이라 min_gap 3 에서 못 쓴다."""
        u = "a b c d e f g h".split()
        sc = {1: 0.9, 2: 0.8, 5: 0.7}
        self.assertEqual(H.top_k_cuts(u, sc, 2, min_gap=3), (5,))

    def test_k_zero(self):
        self.assertEqual(H.top_k_cuts(list("abc"), {1: 0.5}, 0, 1), ())


class KRange(unittest.TestCase):
    def test_min_chunk(self):
        self.assertEqual(H.k_range(8, min_chunk=2), [1, 2, 3])
        self.assertEqual(H.k_range(5, min_chunk=2), [1])

    def test_max_k(self):
        self.assertEqual(H.k_range(100, min_chunk=2, max_k=4), [1, 2, 3, 4])

    def test_too_short(self):
        self.assertEqual(H.k_range(4, min_chunk=2), [1])
        self.assertEqual(H.k_range(3, min_chunk=2), [])   # 3어절을 둘로 쪼개면 평균 1.5

    def test_chunk_len(self):
        self.assertAlmostEqual(H.chunk_len(20, 3), 5.0)


class Pieces(unittest.TestCase):
    def test_no_cut(self):
        self.assertEqual(H.pieces_of(list("abcd"), (), True), [(0, 4)])

    def test_one_piece_per_cut(self):
        self.assertEqual(H.pieces_of(list("abcdef"), (2, 4), True), [(0, 2), (2, 4), (4, 6)])


class Scorer(unittest.TestCase):
    def setUp(self):
        self.tr = FakeTranslator()
        self.s = H.HsetScorer(translators={"X": self.tr}, qe=FakeQE(), spaced=True,
                              target_spaced={"X": True})

    def test_contra_penalty(self):
        texts = ["a b c d"]
        no = self.s.score(texts, [(0, (2,))], lambda i, j: 0.0)[0]
        half = self.s.score(texts, [(0, (2,))], lambda i, j: 0.5)[0]
        self.assertAlmostEqual(half, no * 0.5, places=6)

    def test_worst_contra(self):
        """여러 자리를 자르면 contra 는 평균이 아니라 **최댓값**이 걸린다."""
        texts = ["a b c d e f"]
        got = self.s.score(texts, [(0, (2, 4))], lambda i, j: 0.9 if j == 4 else 0.0)[0]
        base = self.s.score(texts, [(0, (2, 4))], lambda i, j: 0.0)[0]
        self.assertAlmostEqual(got, base * 0.1, places=6)

    def test_reuse_translations(self):
        """같은 구간은 한 번만 번역한다 — 집합끼리 조각이 크게 겹치는 것이 이 설계의 전제."""
        texts = ["a b c d e f"]
        self.s.score(texts, [(0, (2,))], lambda i, j: 0.0)
        first = self.tr.calls
        self.s.score(texts, [(0, (2,))], lambda i, j: 0.0)
        self.assertEqual(self.tr.calls, first)


class Bootstrap(unittest.TestCase):
    def test_no_diff(self):
        a = [0.5] * 50
        r = H.paired_bootstrap(a, list(a))
        self.assertEqual(r["mean"], 0.0)
        self.assertLessEqual(r["lo"], 0.0)
        self.assertGreaterEqual(r["hi"], 0.0)

    def test_cluster_by_sentence(self):
        """클러스터를 주면 CI 가 넓어진다 — 문장 안 상관을 무시하지 않는다."""
        a, b, cl = [], [], []
        for s in range(20):                       # 문장 20개, 각 5개 k
            good = s < 11                         # 11문장만 개선 — 문장 단위로 뭉쳐 있다
            for _k in range(5):
                a.append(0.6 if good else 0.4)
                b.append(0.5)
                cl.append(s)
        naive = H.paired_bootstrap(a, b)
        clustered = H.paired_bootstrap(a, b, clusters=cl)
        self.assertEqual(naive["mean"], clustered["mean"])
        self.assertGreater(clustered["hi"] - clustered["lo"], naive["hi"] - naive["lo"])
        self.assertEqual(clustered["n_clusters"], 20)

    def test_consistent_gain(self):
        a = [0.6] * 50
        b = [0.5] * 50
        r = H.paired_bootstrap(a, b)
        self.assertGreater(r["lo"], 0)

    def test_unpaired(self):
        with self.assertRaises(ValueError):
            H.paired_bootstrap([1.0, 2.0], [1.0])


if __name__ == "__main__":
    unittest.main()


class ScorerCacheKey(unittest.TestCase):
    def test_key_by_text(self):
        """dev-A 다음에 dev-B 를 재면 (문장 index, 구간) 이 겹친다. 조각 원문으로 키를 잡아야
        dev-B 문장이 dev-A 조각의 번역을 받지 않는다 (judge08 test 0.4919 오염의 원인)."""
        calls = []

        class Tr:
            def full(self, xs):
                calls.append(list(xs))
                return [f"T({x})" for x in xs]

        s = H.HsetScorer(translators={"X": Tr()}, qe=FakeQE(), spaced=True,
                         target_spaced={"X": True})
        s.score(["a b c d"], [(0, (2,))], lambda i, j: 0.0)
        s.score(["p q r s"], [(0, (2,))], lambda i, j: 0.0)
        self.assertEqual(s.piece("X", ["p", "q", "r", "s"], 0, 2), "T(p q)")
        self.assertEqual(s.piece("X", ["a", "b", "c", "d"], 0, 2), "T(a b)")
        self.assertEqual(sorted(x for c in calls for x in c), ["a b", "c d", "p q", "r s"])
        s.score(["a b c d"], [(0, (2,))], lambda i, j: 0.0)      # 같은 조각은 다시 번역 안 한다
        self.assertEqual(len([x for c in calls for x in c]), 4)


class ContraAggregation(unittest.TestCase):
    """`contra_agg` 는 지표를 바꾸는 손잡이가 아니라 진단 도구다 — 기본은 항상 `max` 여야 한다."""

    class _QE:
        def score(self, srcs, hyps):
            return [1.0] * len(srcs)

    class _TR:
        def full(self, xs):
            return list(xs)

    def scorer(self, agg="max"):
        from core.meaning_segmentator.autoseg.runtime import hset
        kw = {} if agg is None else {"contra_agg": agg}
        return hset.HsetScorer(translators={"de": self._TR()}, qe=self._QE(),
                               spaced=True, target_spaced={"de": True}, **kw)

    def run_one(self, agg, contra):
        sc = self.scorer(agg) if agg else self.scorer(None)
        texts = ["a b c d e f"]
        jobs = [(0, (1, 2, 3))]
        return sc.score(texts, jobs, lambda i, j: contra[j])[0]

    def test_default_is_max(self):
        """기본값이 바뀌면 지금까지 잰 모든 값과 비교 가능성이 끊긴다."""
        from core.meaning_segmentator.autoseg.runtime import hset
        self.assertEqual(hset.HsetScorer.contra_agg, "max")
        c = {1: 0.0, 2: 0.9, 3: 0.0}
        self.assertAlmostEqual(self.run_one(None, c), 1.0 - 0.9, places=6)

    def test_max_is_dominated_by_one_bad_cut(self):
        """이것이 max 의 성질이자 기울기가 희소한 이유다 — 나머지를 아무리 고쳐도 안 움직인다."""
        worst_only = {1: 0.0, 2: 0.9, 3: 0.0}
        all_mid = {1: 0.4, 2: 0.9, 3: 0.4}
        self.assertAlmostEqual(self.run_one("max", worst_only),
                               self.run_one("max", all_mid), places=6)

    def test_mean_sees_what_max_cannot(self):
        """같은 두 집합이 mean 에서는 갈린다 — 진단이 물을 수 있는 것이 이것이다."""
        worst_only = {1: 0.0, 2: 0.9, 3: 0.0}
        all_mid = {1: 0.4, 2: 0.9, 3: 0.4}
        a = self.run_one("mean", worst_only)
        b = self.run_one("mean", all_mid)
        self.assertGreater(a, b)
        self.assertAlmostEqual(a, 1.0 - 0.9 / 3, places=6)

    def test_no_cuts_is_unpenalised_either_way(self):
        for agg in ("max", "mean"):
            sc = self.scorer(agg)
            self.assertAlmostEqual(sc.score(["a b c"], [(0, ())], lambda i, j: 1.0)[0],
                                   1.0, places=6)
