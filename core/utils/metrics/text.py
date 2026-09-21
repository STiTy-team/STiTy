"""Text normalization and edit distance -- the primitives WER, CER and LAAL share.

Own implementations rather than `jiwer`, so the normalization is identical across
datasets: lowercase, punctuation to spaces, collapse whitespace. Korean drops
spaces entirely (CER) because its spacing varies too much between references to
score as content.
"""
import re

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_CER_STRIP = re.compile(r"[\s\.,\?!]")
_WS = re.compile(r"\s+")


def normalize_words(text: str) -> list[str]:
    """Lowercase, punctuation to spaces, split. The unit WER counts."""
    return _PUNCT.sub(" ", (text or "").lower()).split()


def strip_for_cer(text: str) -> str:
    """Drop spaces and sentence punctuation. The unit CER counts."""
    return _CER_STRIP.sub("", text or "")


def levenshtein(ref, hyp) -> int:
    """Edit distance over any two sequences. `ref` empty means every hyp unit is an
    insertion; `hyp` empty means every ref unit is a deletion."""
    if not ref:
        return len(hyp)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def count_units(text: str, unit: str = "word") -> int:
    """LAAL's |Y|.

    word -- whitespace tokens, for languages written with spaces.
    char -- non-space characters, for zh/ja/th where there is no word boundary.

    The choice changes the LAAL value, so it has to be recorded alongside it for
    two numbers to be comparable.
    """
    if not text:
        return 0
    if unit == "word":
        return len(text.split())
    if unit == "char":
        return len(_WS.sub("", text))
    raise ValueError(f"unknown unit: {unit!r} (word|char)")


def mean_or_none(values) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
