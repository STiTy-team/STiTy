"""`python -m unittest core.meaning_segmentator.autoseg.runtime.test_agents_distill` (PYTHONPATH=.)"""
import unittest

from . import agents_distill as ad


def r1(xs):
    """백분위는 소수점 둘째 자리에 동점 보조키(≤0.01)가 얹히므로 첫째 자리까지만 본다."""
    return [round(x, 1) for x in xs]


class MeanRankScores(unittest.TestCase):
    def test_single_sample_keeps_order_as_percentile(self):
        # 점수 [10, 30, 20] → 순위 3,1,2 → 백분위 0, 100, 50
        self.assertEqual(r1(ad.mean_rank_scores([[10, 30, 20]])), [0.0, 100.0, 50.0])

    def test_two_samples_average_the_rank(self):
        # 위치 0: 순위 1 과 2 → 평균 1.5 → 백분위 75 ; 위치 1: 2 와 1 → 75 ; 위치 2: 3,3 → 0
        self.assertEqual(r1(ad.mean_rank_scores([[90, 50, 10], [50, 90, 10]])), [75.0, 75.0, 0.0])

    def test_ties_share_the_average_rank(self):
        # 동점 둘은 순위 (1+2)/2 = 1.5 씩 → 75 ; 나머지 3위 → 0. 원점수도 같으니 보조키도 같다.
        s = ad.mean_rank_scores([[40, 40, 5]])
        self.assertEqual(r1(s), [75.0, 75.0, 0.0])
        self.assertEqual(s[0], s[1])

    def test_scale_does_not_matter_across_samples(self):
        # 한 샘플이 전부 높게 찍어도 순위만 본다
        self.assertEqual(r1(ad.mean_rank_scores([[1, 2, 3], [70, 80, 90]])), [0.0, 50.0, 100.0])

    def test_equal_mean_rank_is_broken_by_mean_raw_score(self):
        # 세 위치 모두 평균 순위 2 — 그대로 두면 절단기가 위치(앞쪽)로 정한다.
        # 가운데는 두 샘플 다 80 이라 원점수 평균이 높다 → 그쪽이 조금 앞서야 한다.
        s = ad.mean_rank_scores([[90, 80, 10], [10, 80, 90]])
        self.assertEqual([round(x) for x in s], [50, 50, 50])
        self.assertGreater(s[1], s[0])
        self.assertEqual(s[0], s[2])

    def test_tiebreak_never_overturns_a_rank_difference(self):
        # 순위 한 칸(여기선 50점)은 보조키 최대치 0.01 보다 언제나 크다
        s = ad.mean_rank_scores([[100, 99, 0], [100, 99, 0]])
        self.assertGreater(s[0], s[1])

    def test_single_position_is_midpoint(self):
        self.assertEqual(ad.mean_rank_scores([[42], [7]]), [50.0])



class ScoreMeaningBySource(unittest.TestCase):
    def test_translation_mode_is_the_default_text(self):
        self.assertEqual(ad.score_meaning("translation"), ad.SCORE_MEANING)
        self.assertIn("contradicts that whole-sentence translation", ad.score_meaning("translation"))

    def test_source_mode_describes_source_nli_without_translation(self):
        s = ad.score_meaning("source")
        self.assertIn("whole source sentence", s)
        self.assertNotIn("whole-sentence translation", s)
        self.assertIn("(1 - contradiction) x (quality_before + quality_after) / 2", s)

    def test_output_rules_and_critic_carry_the_mode(self):
        self.assertIn("whole source sentence", ad.output_rules(True, contra_source="source"))
        self.assertIn("whole source sentence", ad.critic_system(contra_source="source"))
        self.assertIn("whole source sentence", ad.writer_system(True, 3, contra_source="source"))
        self.assertNotIn("whole source sentence", ad.output_rules(True))
if __name__ == "__main__":
    unittest.main()
