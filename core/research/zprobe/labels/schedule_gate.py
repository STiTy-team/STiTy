"""Emission-schedule labels: source tokenization, clause ends, LLM JSON parsing and the gate.

A schedule is a list of (after_word, text) items: once source word `after` has been heard,
`text` is appended to the translation. An item may also carry `src`, the source word numbers it
translates (self-reported alignment); with it the lookahead, clause and deferral rules are judged
against max(src) instead of Hy prefix-translation stems, which misfire on synonyms (서한/편지).
Pure functions only; no I/O.
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
# Particles that end in a connective-looking syllable (학교에서, 선생님께서, 학생으로서).
_KO_PARTICLE_END = ("에서", "께서", "로서")


def _ko_connective(tok: str) -> bool:
    """Heuristic: eojeol ends in a clausal connective. Drops particle endings and one-syllable
    remainders (최고, 광고, 측면). Sino-Korean nouns with a longer stem (예: 홍보광고) still slip
    through; that residual noise is accepted rather than pulling in a tagger."""
    t = tok.rstrip(".!?,\"'”’)]")
    if t.endswith(_KO_PARTICLE_END):
        return False
    for suf in _KO_CONNECTIVE_END:
        if t.endswith(suf):
            stem = t[: len(t) - len(suf)]
            if len(stem) >= 2:
                return True
            # One-syllable stems: keep the clearly verbal ones (하고/해서/되면, and past tense 갔고/했고).
            return len(stem) == 1 and (stem in _KO_VERBAL_1 or _has_ss_final(stem))
    return False


_KO_VERBAL_1 = {"하", "해", "되", "돼"}


def _has_ss_final(syl: str) -> bool:
    """Hangul syllable whose final consonant is ㅆ (past-tense marker: 갔, 했, 왔)."""
    code = ord(syl) - 0xAC00
    return 0 <= code < 11172 and code % 28 == 20


# --- piece shape (prefix15 rules 2-3): Korean pieces may split an eojeol (응답률 | 도) - the row builder glues a
# piece that starts with a particle/ending - but a non-final piece must not leave a bare noun hanging without its
# particle (이 문장 | 특별히 ...); English pieces must not end in a dangling function word (..., so | I introduced).
_KO_CONTINUATION = ("J", "E", "XS", "VCP", "VCN")
_KO_BARE_NOUN = ("NNG", "NNP", "NNB", "NR", "NP", "SL", "SN", "XSN", "SH")
_EN_ALONE_OK = {"and", "or", "but", "so"}  # a coordinating conjunction may stand alone as a piece
_EN_DANGLING = {"a", "an", "the", "of", "to", "in", "on", "at", "by", "for", "with", "from", "into", "onto",
                "about", "over", "under", "between", "through", "during", "before", "after", "and", "or", "but",
                "so", "because", "although", "though", "that", "which", "who", "whom", "whose", "if", "when",
                "while", "than", "as", "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
                "do", "does", "did", "will", "would", "can", "could", "should", "may", "might", "must", "not",
                "very"}
_KO_DETERMINERS = {"이", "그", "저", "이런", "그런", "저런", "어느", "무슨", "몇", "어떤", "새", "옛", "온갖"}
_kiwi = None


def _tags(text: str) -> list[str]:
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    return [t.tag for t in _kiwi.tokenize(text.strip())]


def ko_continues(text: str) -> bool:
    """The piece starts with a particle/ending/suffix: glue it to the previous text without a space."""
    t = text.strip()
    if not t or not ("\uac00" <= t[0] <= "\ud7a3"):
        return False
    # a bare particle has no context for the tagger (도 alone reads as a noun): tag it after a dummy noun,
    # one consonant-final (것은/것도) and one vowel-final (사과와/사과를)
    _tags("")
    # a multi-eojeol piece that starts with a determiner (이 숫자를, 그 도구들은) is a new word, not the particle 이;
    # the tagger cannot tell a lone syllable apart (에/이/와 alone all tag IC), so the list is explicit
    if " " in t and t.split()[0] in _KO_DETERMINERS:
        return False
    for ctx in ("것", "사과"):
        toks = _kiwi.tokenize(ctx + t)
        if len(toks) >= 2 and toks[0].form == ctx and toks[1].tag.startswith(_KO_CONTINUATION):
            return True
    return False


def _piece_tags(text: str) -> list[str]:
    """Tags of a piece; a continuation piece is tagged in context so 도 reads as a particle, not a noun."""
    t = text.strip()
    if ko_continues(t):
        for ctx in ("것", "사과"):
            toks = _kiwi.tokenize(ctx + t)
            if len(toks) >= 2 and toks[0].form == ctx:
                return [x.tag for x in toks[1:]]
    return _tags(t)


def ends_bare(text: str) -> bool:
    """The text ends in a noun-like morpheme without its particle (응답률, 팔과 턱, T1R2, 발견한 것)."""
    t = text.strip()
    if not t or _KO_PUNCT_END.search(t):
        return False
    tags = _tags(t)
    return bool(tags) and tags[-1] in _KO_BARE_NOUN


def glues(prev: str, text: str) -> bool:
    """`text` continues the last eojeol of `prev` (응답률 | 도, 팔과 턱 | 에 상처를): a continuation piece after a
    bare noun. Both sides are required - "이 숫자를" after "빠르게" starts with the determiner 이, not the particle."""
    return bool(prev) and ko_continues(text) and ends_bare(prev)


def join_texts(texts: list[str]) -> str:
    """Pieces joined with one space, except a continuation piece which is glued (응답률 + 도 -> 응답률도)."""
    out = ""
    for t in texts:
        t = t.strip()
        if not t:
            continue
        out = out + t if not out or glues(out, t) else out + " " + t
    return out


_EN_WORD_SPLIT = re.compile(r"^\s*('(s|re|ve|ll|d|m)|n't)\b")


def shape_check(items, tgt: str) -> str | None:
    """Reason string if a piece has a bad shape, else None. Non-final pieces with text are judged; an English
    piece that starts with a clitic ('s, n't) is a split word wherever it sits."""
    texts = [t for _, t in items if t.strip()]
    if tgt != "ko":
        for i, t in enumerate(texts):
            if _EN_WORD_SPLIT.match(t):
                return f"shape: piece {i + 1} \"{t.strip()}\" splits a word - join it to the previous piece"
    for i, t in enumerate(texts[:-1]):
        t = t.strip()
        nxt = texts[i + 1]
        if _KO_PUNCT_END.search(t) if tgt == "ko" else _EN_PUNCT_END.search(t):
            continue
        if tgt == "ko":
            tags = _piece_tags(t)
            if tags and tags[-1] in _KO_BARE_NOUN and not glues(t, nxt):
                return (f"shape: piece {i + 1} \"{t}\" ends in a bare noun - attach its particle (e.g. 은/는/이/가/을/를) "
                        "or fold it into the next piece (a title or appositive goes in one piece with its name)")
        else:
            ws = t.split()
            last = ws[-1].lower().strip("\"'\u201d\u2019()[],")
            if len(ws) == 1 and last in _EN_ALONE_OK:
                continue
            if last in _EN_DANGLING:
                return f"shape: piece {i + 1} \"{t}\" ends in a dangling word \"{last}\" - end the piece before it or carry it into the next piece"
    return None


_LEADING_PUNCT = re.compile(r"^\s*([,.;:!?\u2026]+)\s*")


def fix_leading_punct(items):
    """A piece that starts with punctuation (", namely,") gives that punctuation to the previous piece."""
    out = []
    for after, text in items:
        m = _LEADING_PUNCT.match(text) if text else None
        if m and out and out[-1][1].strip():
            pa, pt = out[-1]
            out[-1] = (pa, pt.rstrip() + m.group(1))
            text = text[m.end():]
            if text and not text.startswith(" "):
                text = " " + text
        out.append((after, text))
    return out


def merge_same_after(items, covers=None):
    """Consecutive pieces with the same `after` are one emission: merge their text (and src lists)."""
    out, cov_out = [], []
    covers = list(covers) if covers is not None else [None] * len(items)
    for (after, text), cov in zip(items, covers):
        if out and out[-1][0] == after and text.strip() and out[-1][1].strip():
            out[-1] = (after, join_texts([out[-1][1], text]))
            if cov_out[-1] is not None and cov is not None:
                cov_out[-1] = set(cov_out[-1]) | set(cov)
            continue
        out.append((after, text))
        cov_out.append(cov)
    return out, cov_out


def words(src: str, lang: str) -> list[str]:
    """Whitespace tokens (ko eojeol / en words); punctuation stays attached."""
    return src.split()


def clause_ends(tokens: list[str], lang: str) -> list[int]:
    """Sorted unique 1-based indices of tokens that end a clause; the last token always does."""
    ends = set()
    n = len(tokens)
    for i, t in enumerate(tokens, 1):
        if lang == "ko":
            if _KO_PUNCT_END.search(t) or _ko_connective(t):
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


def parse_covers(obj) -> list:
    """Per item of parse_schedule(obj): the set of source word numbers in its "src" list, or None when absent.
    ValueError("format: ...") when "src" is present but not a list of ints."""
    out = []
    for it in obj["schedule"]:
        src = it.get("src")
        if src is None:
            out.append(None)
            continue
        if not isinstance(src, list) or any(isinstance(x, bool) or not isinstance(x, int) for x in src):
            raise ValueError(f"format: 'src' must be a list of ints: {src!r}")
        out.append(set(src))
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
    covers=None,
) -> str | None:
    """Reason string if the schedule violates a rule, else None.

    Order: range, all-empty, lookahead, clause, deferral. `known_at(k)` is the set of target
    content stems already supported by source words 1..k; `full_known` is the same for the whole source.
    `covers` (parse_covers): items with a src set are judged by it - lookahead = any src word > after,
    clause/deferral measured from max(src); items without one fall back to the Hy stems.
    """
    K = len(tokens)
    covers = list(covers) if covers is not None else [None] * len(items) if isinstance(items, (list, tuple)) else []
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
    if len(covers) != len(items):
        return f"range: {len(covers)} src lists for {len(items)} items"

    for (after, text), cov in zip(items, covers):
        if cov is not None:
            if text.strip() and not cov:
                return f"range: item after={after} has text but an empty src list"
            bad = sorted(x for x in cov if not 1 <= x <= K)
            if bad:
                return f"range: src words {bad} outside 1..{K}"
            late = sorted(x for x in cov if x > after)
            if late:
                return f"lookahead: src words {late} are after word {after}"
            continue
        ahead = sorted((content(text, tgt) - known_at(after)) & full_known)
        if ahead:
            return f"lookahead: {' '.join(ahead)} - not yet heard at word {after}"

    ends = sorted(clause_end)
    for (after, text), cov in zip(items, covers):  # self-aligned items: clause and deferral from max(src)
        if cov is None or not text.strip():
            continue
        j = max(cov)
        c = next((e for e in ends if e >= j), K)
        if after > c:
            return f"clause: item covering words up to {j} (clause ends {c}) emitted after {after}"
        if after - j > N:
            return f"deferral: item covering words up to {j} emitted after {after} (> {N} words late)"
    items = [it for it, cov in zip(items, covers) if cov is None]  # the rest: Hy-stem rules below

    first_seen = {}
    for after, text in items:
        for w in content(text, tgt):
            if w not in first_seen:
                first_seen[w] = next((k for k in range(1, K + 1) if w in known_at(k)), None)
    # A stem is checked only on the item that first emits it: a repeated content word
    # ("... analyzed the data") is a later mention, not a deferred first emission.
    first_emit = []
    emitted = set()
    for after, text in items:
        for w in sorted(content(text, tgt)):
            if w not in emitted:
                emitted.add(w)
                first_emit.append((after, w))
    for after, w in first_emit:
        j = first_seen[w]
        if j is None:
            continue
        c = next((e for e in ends if e >= j), K)
        if after > c:
            return f"clause: {w} heard at word {j} (clause ends {c}) but emitted after {after}"
    for after, w in first_emit:
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
