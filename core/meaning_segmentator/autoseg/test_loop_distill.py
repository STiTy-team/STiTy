"""`PYTHONPATH=. python -m unittest core.meaning_segmentator.autoseg.test_loop_distill`

`evaluate` 의 k회 채점 평균. LLM 은 `segment_batch` 를 바꿔치기해 흉내낸다 — 샘플 번호는
넘어온 캐시 파일 이름으로 가른다 (샘플 i 는 `segment_s{i}.json`, 0 은 `segment.json`).
"""
import tempfile
import unittest
from pathlib import Path

from . import loop_distill as ld
from .runtime.pipeline import JsonCache

TEXT = "a b c d e f g"          # 7 어절, min_gap 2 → 후보 2,3,4,5
SAMPLE_SCORES = {0: [90, 50, 10, 30], 1: [50, 90, 10, 30]}


def fake_segment_batch(gw, prompt, texts, cache=None, **kw):
    name = cache.path.name
    i = 0 if name == "segment.json" else int(name[len("segment_s"):-len(".json")])
    outs = [ld._seg_with(TEXT.split(), dict(zip([2, 3, 4, 5], SAMPLE_SCORES[i])), True)
            for _ in texts]
    return outs, [True] * len(texts)


class Sent:
    def __init__(self, id_, text):
        self.id, self.text = id_, text


def labels_for_one_sentence():
    n_b = len(TEXT.split()) - 1
    # 라벨이 전부 같으면 순위상관이 정의되지 않아 evaluate 가 None 을 반올림하려 든다 — 변화를 준다.
    return {"X": [{"contra": [0.0] * n_b, "adq_l": [0.1 * (j + 1) for j in range(n_b)],
                   "adq_r": [0.5] * n_b}]}


class EvaluateKSamples(unittest.TestCase):
    def setUp(self):
        self._orig = ld.segment_batch
        ld.segment_batch = fake_segment_batch
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = JsonCache(Path(self.tmp.name) / "segment.json")

    def tearDown(self):
        ld.segment_batch = self._orig
        self.tmp.cleanup()

    def run_eval(self, k):
        rows, m = ld.evaluate(None, "p", [Sent("s1", TEXT)], labels_for_one_sentence(),
                              True, 2, [3], self.cache, 1, 1, None, k_samples=k)
        return rows[0], m

    def test_k1_keeps_raw_integer_scores(self):
        row, _ = self.run_eval(1)
        self.assertEqual(row["scores"], [90, 50, 10, 30])
        self.assertEqual(row["tag_scale"], 1)
        self.assertIsInstance(row["out"], str)

    def test_k2_averages_ranks_into_percentiles(self):
        row, _ = self.run_eval(2)
        # 순위: 위치2 (1,2)→1.5 ; 위치3 (2,1)→1.5 ; 위치4 4 ; 위치5 3  → 백분위 83.3/83.3/0/33.3
        self.assertEqual([round(s, 1) for s in row["scores"]], [83.3, 83.3, 0.0, 33.3])
        self.assertEqual(row["tag_scale"], 10000)
        self.assertEqual(len(row["out"]), 2)
        self.assertEqual(row["n_valid_samples"], 2)

    def test_k2_cut_uses_averaged_order(self):
        row, _ = self.run_eval(2)
        # T=3 → 경계 1개. 평균 순위 동점(위치 2·3)은 앞쪽 우선이므로 위치 2 를 남긴다.
        self.assertEqual(row["by_T"]["3"]["kept_model"], [2])


if __name__ == "__main__":
    unittest.main()
