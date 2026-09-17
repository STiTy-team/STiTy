"""상한에 닿은 행 가려내기 — `hit_cap` 단위 검사.

왜 필요한가. `max_new_tokens` 상한은 폭주 행을 끊어 배치 전체가 128스텝까지 끌려가는 것을
막지만, AlignAtt 은 잘린 토큰을 그대로 `forced` 로 커밋하므로 **자르는 위치가 이후 회차를
전부 바꾼다** (ja 긴문장 300개 중 2개에서 조각수 10 vs 12, 5 vs 13). 그래서 상한에 닿은
행만 골라 상한 없이 다시 돌린다. 그 "닿았다" 판정이 여기서 검사하는 것이다.
"""

import unittest

from core.meaning_segmentator.autoseg.baselines.nmt import hit_cap

EOS, PAD, PREFIX = 1, 0, 3
STOP = {EOS, PAD}


class HitCapTest(unittest.TestCase):
    def test_상한까지_가고_종료토큰이_없으면_닿은_것이다(self):
        seq = list(range(PREFIX)) + [7] * 20
        self.assertTrue(hit_cap(seq, PREFIX, 20, STOP))

    def test_상한보다_짧게_끝나면_안_닿았다(self):
        seq = list(range(PREFIX)) + [7, 7, EOS]
        self.assertFalse(hit_cap(seq, PREFIX, 20, STOP))

    def test_상한_길이라도_종료토큰으로_끝났으면_안_닿았다(self):
        """제 길이로 끝난 행이 마침 상한과 같은 길이일 수 있다."""
        seq = list(range(PREFIX)) + [7] * 19 + [EOS]
        self.assertFalse(hit_cap(seq, PREFIX, 20, STOP))

    def test_다른_행_때문에_패딩이_붙어도_안_닿았다(self):
        """배치는 가장 긴 행에 맞춰 패딩된다 — 먼저 끝난 행에 pad 가 딸려온다."""
        seq = list(range(PREFIX)) + [7, 7, EOS] + [PAD] * 17
        self.assertFalse(hit_cap(seq, PREFIX, 20, STOP))

    def test_상한이_예산_전체면_다시_돌릴_것이_없다(self):
        """상한이 `max_new_tokens` 와 같으면 그게 곧 상한 없는 경우다."""
        seq = list(range(PREFIX)) + [7] * 128
        self.assertFalse(hit_cap(seq, PREFIX, 128, STOP, budget=128))


if __name__ == "__main__":
    unittest.main()
