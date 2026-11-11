"""Emission-schedule labels: source tokenization, clause ends, LLM JSON parsing and the gate.

A schedule is a list of (after_word, text) items: once source word `after` has been heard,
`text` is appended to the translation. Pure functions only; no I/O.
"""

import os
import re
import sys
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_prefix import content  # noqa: E402  (content-word stems; import has no side effects)

_EN_CONJ = {"and", "but", "so", "because", "if", "when", "while", "that", "which"}
_EN_PUNCT_END = re.compile(r"[.!?;:,][\"'”’)\]]*$")
_KO_PUNCT_END = re.compile(r"[.!?,][\"'”’)\]]*$")
_KO_CONNECTIVE_END = ("고", "서", "면", "데", "지만", "니까")


def words(src: str, lang: str) -> list[str]:
    """Whitespace tokens (ko eojeol / en words); punctuation stays attached."""
    return src.split()


def clause_ends(tokens: list[str], lang: str) -> list[int]:
    """Sorted unique 1-based indices of tokens that end a clause; the last token always does."""
    ends = set()
    n = len(tokens)
    for i, t in enumerate(tokens, 1):
        if lang == "ko":
            if _KO_PUNCT_END.search(t) or t.rstrip(".!?,\"'”’)]").endswith(_KO_CONNECTIVE_END):
                ends.add(i)
        else:
            if _EN_PUNCT_END.search(t):
                ends.add(i)
            if i < n and tokens[i].lower().strip(".!?;:,\"'") in _EN_CONJ:
                ends.add(i)
    if n:
        ends.add(n)
    return sorted(ends)


def parse_schedule(obj) -> list[tuple[int, str]]:
    """{"schedule": [{"after": k, "text": s}, ...]} -> [(k, s), ...]; ValueError("format: ...") if malformed."""
    if not isinstance(obj, dict) or not isinstance(obj.get("schedule"), list):
        raise ValueError("format: expected {'schedule': [...]}")
    out = []
    for it in obj["schedule"]:
        if not isinstance(it, dict) or "after" not in it or "text" not in it:
            raise ValueError(f"format: item needs 'after' and 'text': {it!r}")
        after, text = it["after"], it["text"]
        if isinstance(after, bool) or not isinstance(after, int):
            raise ValueError(f"format: 'after' must be an int: {after!r}")
        if not isinstance(text, str):
            raise ValueError(f"format: 'text' must be a string: {text!r}")
        out.append((after, text))
    return out


def check_schedule(
    items,
    tokens: list[str],
    lang: str,
    tgt: str,
    *,
    known_at: Callable[[int], set],
    full_known: set,
    N: int,
    clause_end: list[int],
) -> str | None:
    """Reason string if the schedule violates a rule, else None.

    Order: range, all-empty, lookahead, clause, deferral. `known_at(k)` is the set of target
    content stems already supported by source words 1..k; `full_known` is the same for the whole source.
    """
    K = len(tokens)
    if not isinstance(items, (list, tuple)):
        return "range: items is not a list"
    prev = 0
    for it in items:
        try:
            after, _ = it
        except (TypeError, ValueError):
            return f"range: bad item {it!r}"
        if isinstance(after, bool) or not isinstance(after, int):
            return f"range: after is not an int: {after!r}"
        if not 1 <= after <= K:
            return f"range: after={after} outside 1..{K}"
        if after < prev:
            return f"range: after={after} decreases (previous {prev})"
        prev = after

    if not any(text.strip() for _, text in items):
        return "all additions are empty"

    for after, text in items:
        ahead = sorted((content(text, tgt) - known_at(after)) & full_known)
        if ahead:
            return f"lookahead: {' '.join(ahead)} - not yet heard at word {after}"

    first_seen = {}
    for after, text in items:
        for w in content(text, tgt):
            if w not in first_seen:
                first_seen[w] = next((k for k in range(1, K + 1) if w in known_at(k)), None)
    ends = sorted(clause_end)
    for after, text in items:
        for w in sorted(content(text, tgt)):
            j = first_seen[w]
            if j is None:
                continue
            c = next((e for e in ends if e >= j), K)
            if after > c:
                return f"clause: {w} heard at word {j} (clause ends {c}) but emitted after {after}"
    for after, text in items:
        for w in sorted(content(text, tgt)):
            j = first_seen[w]
            if j is not None and after - j > N:
                return f"deferral: {w} heard at word {j} but emitted after {after} (> {N} words late)"
    return None


def eojeol_end_times(tokens: list[str], word_times: list[dict]) -> list[float]:
    """End time of each token, mapping by cumulative alphanumeric counts (aligner words may split/merge tokens)."""
    counts = [sum(c.isalnum() for c in w["w"]) for w in word_times]
    cum, total = [], 0
    for c in counts:
        total += c
        cum.append(total)
    out, need, wi = [], 0, 0
    for t in tokens:
        need += sum(c.isalnum() for c in t)
        while wi < len(cum) - 1 and cum[wi] < need:
            wi += 1
        out.append(word_times[wi]["end"] if word_times else 0.0)
    return out
