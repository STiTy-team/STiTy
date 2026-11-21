"""The language a commit's text is written in, when its script singles out one candidate.

Used only for short spans (`short_text_sec`), where the acoustic scorers are weakest and the
transcript's script was right on every matched short northstar commit. Candidates that share a
script (es/en, fr/de/en) give no verdict, so those sessions stay on the acoustic judgment."""

import re

SCRIPT = {
    "ko": "hangul",
    "ja": "kana",
    "zh": "han",
    "yue": "han",
    "en": "latin",
    "es": "latin",
    "fr": "latin",
    "de": "latin",
    "it": "latin",
    "pt": "latin",
    "ru": "cyrillic",
}
PATTERNS = {
    "hangul": re.compile(r"[가-힣]"),
    "kana": re.compile(r"[぀-ヿ]"),
    "han": re.compile(r"[一-鿿]"),
    "latin": re.compile(r"[A-Za-z]"),
    "cyrillic": re.compile(r"[Ѐ-ӿ]"),
}
SHARE = 0.6


def text_language(text: str, candidates: tuple[str, ...]) -> str | None:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return None
    if "ja" in candidates and any(PATTERNS["kana"].match(ch) for ch in letters):
        return "ja"  # Japanese text often carries English words in Latin letters
    counts = {s: sum(bool(p.match(ch)) for ch in letters) for s, p in PATTERNS.items()}
    if "ja" in candidates and not any(SCRIPT.get(c) == "han" for c in candidates):
        counts["kana"] += counts.pop("han")  # kanji in a session without Chinese
    script = max(counts, key=counts.get)
    if counts[script] < SHARE * len(letters):
        return None
    if script == "han" and "ja" in candidates:
        return None  # kanji: Japanese or Chinese
    owners = [c for c in candidates if SCRIPT.get(c) == script or (script == "kana" and c == "ja")]
    return owners[0] if len(owners) == 1 else None
