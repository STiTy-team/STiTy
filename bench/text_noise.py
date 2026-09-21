from __future__ import annotations

import hashlib
import random
import re


NOISE_VERSION = "asr-text-noise-2026-09-21.1"

_EN_CONFUSIONS = {
    "to": "two", "two": "to", "too": "to", "there": "their", "their": "there",
    "your": "you're", "you're": "your", "weather": "whether", "whether": "weather",
    "four": "for", "for": "four", "right": "write", "write": "right",
    "here": "hear", "hear": "here", "no": "know", "know": "no",
}

_KO_CONFUSIONS = {
    "네": "내", "내": "네", "안": "않", "않": "안", "되": "돼", "돼": "되",
    "맞아": "마자", "같이": "가치", "낫다": "났다", "몇": "멧",
    "일": "이", "이": "일", "삼": "상", "사": "싸",
}


def _seed(seed: int, item_id: str, level: float) -> int:
    raw = f"{seed}|{item_id}|{level:.8f}|{NOISE_VERSION}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+(?:['’]\w+)?|[^\w\s]|\s+", text, re.UNICODE)


def _word(token: str) -> bool:
    return bool(re.fullmatch(r"\w+(?:['’]\w+)?", token, re.UNICODE))


def _corrupt_number(token: str, rng: random.Random) -> str:
    indexes = [index for index, char in enumerate(token) if char.isdigit()]
    if not indexes:
        return token
    index = rng.choice(indexes)
    replacement = str((int(token[index]) + rng.choice((1, 2, 8, 9))) % 10)
    return token[:index] + replacement + token[index + 1:]


def perturb_transcript(text: str, *, lang: str, level: float,
                       seed: int = 0, item_id: str = "") -> tuple[str, list[dict]]:
    if lang not in {"ko", "en"}:
        raise ValueError("ASR text noise supports only ko and en")
    if not 0 <= level <= 1:
        raise ValueError("noise level must be in [0, 1]")
    if level == 0 or not text:
        return text, []
    rng = random.Random(_seed(seed, item_id, level))
    confusion = _KO_CONFUSIONS if lang == "ko" else _EN_CONFUSIONS
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
            if rng.random() < min(1.0, level * 1.8):
                operations.append({"type": "punctuation_deletion", "index": index, "text": token})
                continue
            output.append(token)
            continue
        draw = rng.random()
        if draw < level * 0.28:
            operations.append({"type": "word_deletion", "index": index, "text": token})
            continue
        replacement = token
        lowered = token.casefold()
        if draw < level * 0.58 and lowered in confusion:
            replacement = confusion[lowered]
            if token[:1].isupper():
                replacement = replacement[:1].upper() + replacement[1:]
            operations.append({"type": "confusion_substitution", "index": index,
                               "before": token, "after": replacement})
        elif draw < level * 0.72 and any(char.isdigit() for char in token):
            replacement = _corrupt_number(token, rng)
            operations.append({"type": "number_substitution", "index": index,
                               "before": token, "after": replacement})
        output.append(replacement)
        if rng.random() < level * 0.12:
            output.append(" " + replacement)
            operations.append({"type": "word_repetition", "index": index, "text": replacement})

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

