"""Candidate decision points for the prefix labeler (prefix16).

The labeler is asked "what can you add now?" only at these source-word positions, so it cannot pick an
earlier spot on its own. English: clause ends (schedule_gate.clause_ends - punctuation, before a
conjunction) plus phrase ends - the token before a preposition, relativizer or subordinator (never before "of") -
plus the sentence end. Korean: clause ends plus phrase ends - an eojeol whose last morpheme
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
# English phrase ends: the token BEFORE one of these starts a new phrase/clause. "of" is left out - it continues
# the noun phrase ("the oldest skeleton of Britain") and a cut there would leave a bare noun.
_EN_PHRASE_STARTERS = {"in", "on", "at", "to", "for", "with", "by", "from", "into", "onto", "about", "over", "under",
                       "between", "through", "during", "before", "after", "as", "than", "without", "against", "toward",
                       "towards", "across", "around", "within", "upon", "via", "per",
                       "that", "which", "who", "whom", "whose", "where", "when", "while", "because", "although",
                       "though", "if", "unless", "since", "until", "so", "and", "but", "or"}
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
    else:
        pts |= {i for i in range(1, n) if tokens[i].lower().strip("\"'“‘(") in _EN_PHRASE_STARTERS}
    pts.discard(n)
    kept: list[int] = []
    for p in sorted(pts, reverse=True):  # later point wins; a kept point never moves
        if not kept or kept[-1] - p >= min_gap:
            kept.append(p)
    return sorted(kept) + [n]
