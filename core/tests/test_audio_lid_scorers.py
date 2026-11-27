import json
from pathlib import Path

import librosa
import pytest
import torch

from core.components.langid.audio.scorers import SCORERS, build_scorer

FLEURS = Path("/home/mobility/STiTy-team/datasets/fleurs")


def _clip(lang_dir):
    item = json.loads((FLEURS / lang_dir / "manifest.jsonl").open().readline())
    wav, _ = librosa.load(FLEURS / lang_dir / item["audio"], sr=16000, duration=3.0)
    return wav


@pytest.mark.skipif(
    not torch.cuda.is_available() or not FLEURS.exists(), reason="needs GPU + FLEURS"
)
@pytest.mark.parametrize("name", sorted(SCORERS))
def test_scorer_ranks_korean_and_english(name):
    scorer = build_scorer(name)
    ko = scorer.scores(_clip("ko_kr"), ("ko", "en"))
    en = scorer.scores(_clip("en_us"), ("ko", "en"))
    assert ko["ko"] > ko["en"]
    assert en["en"] > en["ko"]
    assert scorer.scores(_clip("ko_kr"), ("ko", "xx"))["xx"] == float("-inf")


@pytest.mark.skipif(
    not torch.cuda.is_available() or not FLEURS.exists(), reason="needs GPU + FLEURS"
)
@pytest.mark.parametrize("name", ["whisper-token", "ecapa"])
def test_batch_matches_one_by_one(name):
    scorer = build_scorer(name)
    clips = [_clip("ko_kr")[: 16000 * s] for s in (1, 2, 3)] + [_clip("en_us")]
    one = [scorer.scores(c, ("ko", "en")) for c in clips]
    many = scorer.scores_batch(clips, ("ko", "en"))
    # Padding shifts ECAPA's feature normalisation a little; the verdict must not change.
    for a, b in zip(one, many):
        assert max(a, key=a.get) == max(b, key=b.get)
        for lang in ("ko", "en"):
            assert abs(a[lang] - b[lang]) < 0.3
