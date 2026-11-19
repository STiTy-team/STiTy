import os

import jiwer

from core.utils import langs

from .common import basic, english, src_codes
from . import r2norm as r2norm_mod
from .numbers import normalize_numbers


def wer(rows: list[dict], r2norm: str | None = None) -> float | None:
    transcripts = _transcripts(rows, r2norm)
    if not transcripts:
        return None
    return jiwer.wer([ref for ref, _ in transcripts], [hyp for _, hyp in transcripts])


def cer(rows: list[dict], r2norm: str | None = None) -> float | None:
    transcripts = _transcripts(rows, r2norm)
    if not transcripts:
        return None
    return jiwer.cer(
        ["".join(ref.split()) for ref, _ in transcripts],
        ["".join(hyp.split()) for _, hyp in transcripts],
    )


def _transcripts(rows: list[dict], r2norm: str | None = None) -> list[tuple[str, str]]:
    # The dataset config's `dataset.r2norm` turns it on; STITY_R2NORM overrides it for ad-hoc rescoring.
    r2norm = os.environ.get("STITY_R2NORM") or r2norm
    transcripts = []
    for row in rows:
        normalize = english if langs.norm_code(row["src_lang"]) == "en" else basic
        if "ko" in src_codes(row):
            # The English normalizer already writes numbers as digits; Korean needs its own pass.
            normalize = lambda text, base=normalize: base(normalize_numbers(text))
        if r2norm:
            # r2 scoring: align fillers, spelling variants, dialect endings and number forms on both sides (r2norm.py).
            normalize = lambda text, base=normalize, use=r2norm_mod.parts(r2norm): base(r2norm_mod.apply(text, use))
        reference = normalize(row["reference"] or "").strip()
        if reference:
            transcripts.append((reference, normalize(row["transcription_output"] or "").strip()))
    return transcripts
