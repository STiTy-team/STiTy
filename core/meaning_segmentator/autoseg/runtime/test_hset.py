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
