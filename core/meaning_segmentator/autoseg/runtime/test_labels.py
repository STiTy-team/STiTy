"""`PYTHONPATH=. python -m unittest core.meaning_segmentator.autoseg.runtime.test_labels`"""
import unittest

from . import labels as L


class FakeNLI:
    """(premise, hypothesis) 쌍마다 hypothesis 어절 수 / 10 을 모순 확률로 돌려준다."""
    def score_dual(self, premises, hypotheses):
        c = [len(h.split()) / 10 for h in hypotheses]
        return c, [1 - x for x in c]


def two_target_labels():
    # 문장 "a b c d" → 경계 3개. 타깃마다 번역 contra 가 다르다.
    return {"X": [{"id": "s", "contra": [0.9, 0.8, 0.7], "ent": [0.5, 0.5, 0.5],
                   "contra_floor": [0, 0, 0], "adq_l": [0.1, 0.2, 0.3], "adq_r": [0.4, 0.5, 0.6], "hyp_units": [1, 2, 3]}],
            "Y": [{"id": "s", "contra": [0.1, 0.2, 0.3], "ent": [0.5, 0.5, 0.5],
                   "contra_floor": [0, 0, 0], "adq_l": [0.9, 0.8, 0.7], "adq_r": [0.6, 0.5, 0.4], "hyp_units": [1, 2, 3]}]}


class ApplySourceContra(unittest.TestCase):
    def test_overwrites_contra_with_source_nli_same_for_every_target(self):
        lab = L.apply_source_contra(two_target_labels(), ["a b c d"], True, FakeNLI())
        # prefix 어절 수 1,2,3 → 0.1, 0.2, 0.3 — 두 타깃 모두
        self.assertEqual(lab["X"][0]["contra"], [0.1, 0.2, 0.3])
        self.assertEqual(lab["Y"][0]["contra"], [0.1, 0.2, 0.3])
        self.assertEqual(lab["X"][0]["ent"], [0.9, 0.8, 0.7])

    def test_keeps_translation_contra_and_marks_mode(self):
        lab = L.apply_source_contra(two_target_labels(), ["a b c d"], True, FakeNLI())
        self.assertEqual(lab["X"][0]["contra_mt"], [0.9, 0.8, 0.7])
        self.assertEqual(lab["X"][0]["contra_source"], "source")
        self.assertEqual(L.contra_source_of(lab), "source")
        self.assertEqual(L.contra_source_of(two_target_labels()), "translation")

    def test_adq_untouched_so_label_changes_only_through_contra(self):
        lab = L.apply_source_contra(two_target_labels(), ["a b c d"], True, FakeNLI())
        self.assertEqual(lab["X"][0]["adq_l"], [0.1, 0.2, 0.3])
        # (1-0.1)*(0.1+0.4)/2 = 0.225 ; (1-0.1)*(0.9+0.6)/2 = 0.675 → 평균 0.45
        self.assertAlmostEqual(L.label_value(lab, 0, 1), 0.45)

    def test_is_idempotent(self):
        once = L.apply_source_contra(two_target_labels(), ["a b c d"], True, FakeNLI())
        twice = L.apply_source_contra(once, ["a b c d"], True, FakeNLI())
        self.assertEqual(once, twice)
        self.assertEqual(twice["X"][0]["contra_mt"], [0.9, 0.8, 0.7])


if __name__ == "__main__":
    unittest.main()
