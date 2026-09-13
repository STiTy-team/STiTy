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
    def test_상위_k개를_고른다(self):
        u = "a b c d e f g h".split()
        sc = {j: 0.1 * j for j in range(1, 8)}      # 뒤쪽 자리일수록 높다
        got = H.cut_set(u, sc, T=4, spaced=True, min_gap=1)
        self.assertEqual(len(got), 1)               # 8어절 / T4 → 경계 1개
        self.assertEqual(got, (7,))

    def test_min_gap_을_지킨다(self):
        u = "a b c d e f g h i j k l".split()
        sc = {j: (1.0 if j in (5, 6) else 0.1) for j in range(1, 12)}
        got = H.cut_set(u, sc, T=4, spaced=True, min_gap=3)
        self.assertNotIn((5, 6), [got])             # 붙어 있는 둘을 같이 못 고른다
        for x, y in zip(got, got[1:]):
            self.assertGreaterEqual(y - x, 3)

    def test_점수가_없으면_빈_집합(self):
        self.assertEqual(H.cut_set("a b c".split(), {}, 2, True, 1), ())


class TopK(unittest.TestCase):
    def test_상위_k개(self):
        u = "a b c d e f".split()
        sc = {1: 0.1, 2: 0.9, 3: 0.5, 4: 0.7, 5: 0.2}
        self.assertEqual(H.top_k_cuts(u, sc, 2, min_gap=1), (2, 4))

    def test_동점은_앞쪽이_이긴다(self):
        u = "a b c d e".split()
        sc = {1: 0.5, 2: 0.5, 3: 0.1, 4: 0.1}
        self.assertEqual(H.top_k_cuts(u, sc, 1, min_gap=1), (1,))

    def test_min_gap_은_양끝도_본다(self):
        """자리 1 은 앞 조각이 1어절이라 min_gap 3 에서 못 쓴다."""
        u = "a b c d e f g h".split()
        sc = {1: 0.9, 2: 0.8, 5: 0.7}
        self.assertEqual(H.top_k_cuts(u, sc, 2, min_gap=3), (5,))

    def test_k가_0이면_빈_집합(self):
        self.assertEqual(H.top_k_cuts(list("abc"), {1: 0.5}, 0, 1), ())


class KRange(unittest.TestCase):
    def test_평균_조각이_2어절_아래로_안_간다(self):
        self.assertEqual(H.k_range(8, min_chunk=2), [1, 2, 3])
        self.assertEqual(H.k_range(5, min_chunk=2), [1])

    def test_상한을_지킨다(self):
        self.assertEqual(H.k_range(100, min_chunk=2, max_k=4), [1, 2, 3, 4])

    def test_너무_짧으면_없다(self):
        self.assertEqual(H.k_range(4, min_chunk=2), [1])
        self.assertEqual(H.k_range(3, min_chunk=2), [])   # 3어절을 둘로 쪼개면 평균 1.5

    def test_지연축_변환(self):
        self.assertAlmostEqual(H.chunk_len(20, 3), 5.0)


class Pieces(unittest.TestCase):
    def test_절단이_없으면_통짜(self):
        self.assertEqual(H.pieces_of(list("abcd"), (), True), [(0, 4)])

    def test_절단마다_조각이_하나씩_는다(self):
        self.assertEqual(H.pieces_of(list("abcdef"), (2, 4), True), [(0, 2), (2, 4), (4, 6)])


class Scorer(unittest.TestCase):
    def setUp(self):
        self.tr = FakeTranslator()
        self.s = H.HsetScorer(translators={"X": self.tr}, qe=FakeQE(), spaced=True,
                              target_spaced={"X": True})

    def test_contra_가_점수를_깎는다(self):
        texts = ["a b c d"]
        no = self.s.score(texts, [(0, (2,))], lambda i, j: 0.0)[0]
        half = self.s.score(texts, [(0, (2,))], lambda i, j: 0.5)[0]
        self.assertAlmostEqual(half, no * 0.5, places=6)

    def test_최악의_자리가_깎는다(self):
        """여러 자리를 자르면 contra 는 평균이 아니라 **최댓값**이 걸린다."""
        texts = ["a b c d e f"]
        got = self.s.score(texts, [(0, (2, 4))], lambda i, j: 0.9 if j == 4 else 0.0)[0]
        base = self.s.score(texts, [(0, (2, 4))], lambda i, j: 0.0)[0]
        self.assertAlmostEqual(got, base * 0.1, places=6)

    def test_조각_번역을_재사용한다(self):
        """같은 구간은 한 번만 번역한다 — 집합끼리 조각이 크게 겹치는 것이 이 설계의 전제."""
        texts = ["a b c d e f"]
        self.s.score(texts, [(0, (2,))], lambda i, j: 0.0)
        first = self.tr.calls
        self.s.score(texts, [(0, (2,))], lambda i, j: 0.0)
        self.assertEqual(self.tr.calls, first)


class Bootstrap(unittest.TestCase):
    def test_차이가_없으면_CI_가_0_을_품는다(self):
        a = [0.5] * 50
        r = H.paired_bootstrap(a, list(a))
        self.assertEqual(r["mean"], 0.0)
        self.assertLessEqual(r["lo"], 0.0)
        self.assertGreaterEqual(r["hi"], 0.0)

    def test_같은_문장의_k들은_한_덩이로_뽑힌다(self):
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

    def test_한결같은_개선이면_하한이_양수(self):
        a = [0.6] * 50
        b = [0.5] * 50
        r = H.paired_bootstrap(a, b)
        self.assertGreater(r["lo"], 0)

    def test_짝이_안_맞으면_거부(self):
        with self.assertRaises(ValueError):
            H.paired_bootstrap([1.0, 2.0], [1.0])


if __name__ == "__main__":
    unittest.main()
