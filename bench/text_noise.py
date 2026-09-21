"""Seeded ASR-like perturbation of a transcript.

Each word is hit by one error with probability ``level``. The kind of error is drawn
in the proportions ASR error analyses report -- substitutions first, then deletions,
then insertions -- so the realised word error rate lands close to ``level`` (Korean
spacing errors add to it). Substitutions come from a homophone/confusion table, a
changed digit, or a sound-alike edit: a Korean syllable with a confusable vowel,
final consonant or tense initial, an English plural/tense ending or vowel. The
English vowel edit can produce a non-word, which a real recognizer would not, so the
synthetic curve supplements real ASR output rather than replacing it.
"""
from __future__ import annotations

import hashlib
import random
import re


NOISE_VERSION = "asr-text-noise-2026-09-21.2"

_ERROR_MIX = (("substitution", .6), ("deletion", .25), ("insertion", .15))

_EN_CONFUSION_PAIRS = (
    ("to", "two"), ("too", "to"), ("there", "their"), ("they're", "there"),
    ("your", "you're"), ("its", "it's"), ("then", "than"), ("where", "were"),
    ("wear", "where"), ("weather", "whether"), ("for", "four"), ("right", "write"),
    ("here", "hear"), ("no", "know"), ("new", "knew"), ("one", "won"), ("by", "buy"),
    ("see", "sea"), ("meet", "meat"), ("week", "weak"), ("hour", "our"), ("a", "the"),
    ("an", "and"), ("in", "and"), ("can", "can't"), ("would", "wood"),
    ("thirteen", "thirty"), ("fourteen", "forty"), ("fifteen", "fifty"),
    ("sixteen", "sixty"), ("seventeen", "seventy"), ("eighteen", "eighty"),
    ("nineteen", "ninety"),
)
_EN_CONFUSIONS: dict[str, str] = {}
for _left, _right in _EN_CONFUSION_PAIRS:
    _EN_CONFUSIONS.setdefault(_left, _right)
    _EN_CONFUSIONS.setdefault(_right, _left)

_KO_CONFUSIONS = {
    "네": "내", "내": "네", "안": "않", "않": "안", "되": "돼", "돼": "되",
    "맞아": "마자", "같이": "가치", "낫다": "났다", "몇": "멧",
    "일": "이", "이": "일", "삼": "상", "사": "싸",
}

# Hangul syllable = 0xAC00 + (initial * 21 + medial) * 28 + final.
_KO_INITIAL_SWAPS = {0: 1, 1: 0, 3: 4, 4: 3, 7: 8, 8: 7, 9: 10, 10: 9, 12: 13, 13: 12, 14: 12}
# ㅐ↔ㅔ, ㅒ↔ㅖ, ㅙ→ㅚ→ㅞ→ㅙ, ㅓ↔ㅗ, ㅜ↔ㅡ
_KO_MEDIAL_SWAPS = {1: 5, 5: 1, 3: 7, 7: 3, 10: 11, 11: 15, 15: 10, 4: 8, 8: 4, 13: 18, 18: 13}
# ㄴ↔ㅇ, ㅁ→ㄴ, ㄷ→ㅅ, ㅅ↔ㅆ, ㅈ→ㅅ, ㅂ↔ㅍ, ㄱ↔ㅋ, ㄲ→ㄱ, ㅎ→(none), ㄹ→(none)
_KO_FINAL_SWAPS = {4: 21, 21: 4, 16: 4, 7: 19, 19: 20, 20: 19, 22: 19, 17: 26, 26: 17,
                   1: 24, 24: 1, 2: 1, 27: 0, 8: 0}

_EN_VOWEL_SWAPS = {"a": "e", "e": "i", "i": "e", "o": "u", "u": "o"}


def _seed(seed: int, item_id: str, level: float) -> int:
    raw = f"{seed}|{item_id}|{level:.8f}|{NOISE_VERSION}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+(?:['’]\w+)?|[^\w\s]|\s+", text, re.UNICODE)


def _word(token: str) -> bool:
    return bool(re.fullmatch(r"\w+(?:['’]\w+)?", token, re.UNICODE))


def _match_case(original: str, replacement: str) -> str:
    return replacement[:1].upper() + replacement[1:] if original[:1].isupper() else replacement


def _corrupt_number(token: str, rng: random.Random) -> str:
    indexes = [index for index, char in enumerate(token) if char.isdigit()]
    index = rng.choice(indexes)
    replacement = str((int(token[index]) + rng.choice((1, 2, 8, 9))) % 10)
    return token[:index] + replacement + token[index + 1:]


def _ko_sound_alike(token: str, rng: random.Random) -> str | None:
    edits = []
    for index, char in enumerate(token):
        code = ord(char) - 0xAC00
        if not 0 <= code < 11172:
            continue
        initial, medial, final = code // 588, (code % 588) // 28, code % 28
        for part, table, value in (("initial", _KO_INITIAL_SWAPS, initial),
                                   ("medial", _KO_MEDIAL_SWAPS, medial),
                                   ("final", _KO_FINAL_SWAPS, final)):
            if value in table:
                edits.append((index, part, table[value], (initial, medial, final)))
    if not edits:
        return None
    index, part, new, (initial, medial, final) = rng.choice(edits)
    initial = new if part == "initial" else initial
    medial = new if part == "medial" else medial
    final = new if part == "final" else final
    return token[:index] + chr(0xAC00 + (initial * 21 + medial) * 28 + final) + token[index + 1:]


def _en_sound_alike(token: str, rng: random.Random) -> str | None:
    lowered = token.lower()
    options = []
    if len(token) > 3 and lowered.endswith("s") and not lowered.endswith("ss"):
        options.append(token[:-1])
    elif len(token) > 2:
        options.append(token + "s")
    if len(token) > 4 and lowered.endswith("ed"):
        options.append(token[:-2])
    vowels = [index for index, char in enumerate(lowered) if index and char in _EN_VOWEL_SWAPS]
    if vowels:
        index = rng.choice(vowels)
        swap = _EN_VOWEL_SWAPS[lowered[index]]
        options.append(token[:index] + (swap.upper() if token[index].isupper() else swap)
                       + token[index + 1:])
    return rng.choice(options) if options else None


def _substitute(token: str, lang: str, rng: random.Random) -> tuple[str, str] | None:
    confusions = _KO_CONFUSIONS if lang == "ko" else _EN_CONFUSIONS
    lowered = token.casefold()
    if lowered in confusions:
        return _match_case(token, confusions[lowered]), "confusion"
    if any(char.isdigit() for char in token):
        return _corrupt_number(token, rng), "number"
    edited = (_ko_sound_alike if lang == "ko" else _en_sound_alike)(token, rng)
    if edited and edited != token:
        return edited, "sound_alike"
    return None


def _error_kind(rng: random.Random) -> str:
    draw = rng.random()
    for kind, share in _ERROR_MIX:
        if draw < share:
            return kind
        draw -= share
    return _ERROR_MIX[-1][0]


def perturb_transcript(text: str, *, lang: str, level: float,
                       seed: int = 0, item_id: str = "") -> tuple[str, list[dict]]:
    if lang not in {"ko", "en"}:
        raise ValueError("ASR text noise supports only ko and en")
    if not 0 <= level <= 1:
        raise ValueError("noise level must be in [0, 1]")
    if level == 0 or not text:
        return text, []
    rng = random.Random(_seed(seed, item_id, level))
    source = _tokens(text)
    output = []
    operations = []

    for index, token in enumerate(source):
        if token.isspace():
            if lang == "ko" and rng.random() < level * 0.18:
                operations.append({"type": "spacing_deletion", "index": index, "text": token})
                continue
            output.append(token)
            continue
        if not _word(token):
            if rng.random() < level:
                operations.append({"type": "punctuation_deletion", "index": index, "text": token})
                continue
            output.append(token)
            continue
        if rng.random() >= level:
            output.append(token)
            continue
        kind = _error_kind(rng)
        substituted = _substitute(token, lang, rng) if kind == "substitution" else None
        if kind == "deletion" or (kind == "substitution" and substituted is None):
            operations.append({"type": "word_deletion", "index": index, "text": token})
            continue
        if substituted is not None:
            replacement, method = substituted
            operations.append({"type": "substitution", "method": method, "index": index,
                               "before": token, "after": replacement})
            output.append(replacement)
            continue
        output.append(token + " " + token)
        operations.append({"type": "word_repetition", "index": index, "text": token})

    noisy = "".join(output)
    if lang == "ko" and noisy and rng.random() < level * 0.25:
        matches = list(re.finditer(r"[가-힣]{4,}", noisy))
        if matches:
            match = rng.choice(matches)
            cut = rng.randrange(match.start() + 1, match.end())
            noisy = noisy[:cut] + " " + noisy[cut:]
            operations.append({"type": "spacing_insertion", "index": cut})
    noisy = re.sub(r"[ \t]+", " ", noisy).strip()

    if noisy == text.strip():
        words = [index for index, token in enumerate(source) if _word(token)]
        if words:
            index = words[-1]
            token = source[index]
            noisy = (text.strip() + " " + token).strip()
            operations.append({"type": "word_repetition", "index": index, "text": token,
                               "forced": True})
        else:
            noisy = ""
            operations.append({"type": "deletion", "forced": True})
    return noisy, operations


__all__ = ["NOISE_VERSION", "perturb_transcript"]
