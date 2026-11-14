"""Candidate decision points for the prefix labeler (prefix16).

The labeler is asked "what can you add now?" only at these source-word positions, so it cannot pick an
earlier spot on its own. English: clause ends (schedule_gate.clause_ends - punctuation, before a
conjunction) plus the sentence end. Korean: clause ends plus phrase ends - an eojeol whose last morpheme
is a topic/auxiliary particle (JX: 은/는/도/부터/까지) or an adverbial particle (JKB: 에서/에/로/으로/께), or
that ends in a comma. Subject (이/가), genitive (의) and object (을/를) particles are not candidates: a
piece opened there would be a bare subject. Two candidates closer than `min_gap` tokens keep the later
one (more has been heard there); the last token is always a candidate and never displaces another."""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schedule_gate import _tags, clause_ends  # noqa: E402

_PHRASE_TAGS = {"JX", "JKB"}
_TRAIL = re.compile(r"[,.!?\"'”’)\]]+$")


def ko_phrase_end(tok: str) -> bool:
    """The eojeol ends a phrase the labeler may be asked at (see the module docstring)."""
    if tok.endswith(","):
        return True
    core = _TRAIL.sub("", tok)
    if not core:
        return False
    tags = _tags(core)
    return bool(tags) and tags[-1] in _PHRASE_TAGS


def candidate_points(tokens: list[str], lang: str, min_gap: int = 2) -> list[int]:
    """Sorted unique 1-based positions where the labeler is asked; the last token is always one."""
    n = len(tokens)
    if not n:
        return []
    pts = set(clause_ends(tokens, lang))
    if lang == "ko":
        pts |= {i for i, t in enumerate(tokens, 1) if ko_phrase_end(t)}
    pts.discard(n)
    out: list[int] = []
    for p in sorted(pts):
        if out and p - out[-1] < min_gap:
            out[-1] = p
        else:
            out.append(p)
    if out and n - out[-1] < 1:
        out.pop()
    out.append(n)
    return out
