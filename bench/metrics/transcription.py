import jiwer

from core.utils import langs

from .common import basic, english


def wer(rows: list[dict]) -> float | None:
    transcripts = _transcripts(rows)
    if not transcripts:
        return None
    return jiwer.wer([ref for ref, _ in transcripts], [hyp for _, hyp in transcripts])


def cer(rows: list[dict]) -> float | None:
    transcripts = _transcripts(rows)
    if not transcripts:
        return None
    return jiwer.cer(
        ["".join(ref.split()) for ref, _ in transcripts],
        ["".join(hyp.split()) for _, hyp in transcripts],
    )


def _transcripts(rows: list[dict]) -> list[tuple[str, str]]:
    transcripts = []
    for row in rows:
        normalize = english if langs.norm_code(row["src_lang"]) == "en" else basic
        reference = normalize(row["reference"] or "").strip()
        if reference:
            transcripts.append((reference, normalize(row["transcription_output"] or "").strip()))
    return transcripts
