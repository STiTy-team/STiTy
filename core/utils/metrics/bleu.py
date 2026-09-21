"""Translation quality (BLEU).

Two corpus numbers, deliberately:

  bleu      correctly-routed segments only
  bleu_all  every segment

In a multi-language run a misrouted utterance gets compared against a reference in
a language it was never translated into. That scores near zero for reasons that
have nothing to do with translation quality, so reporting only the combined figure
blends a language-detection failure into the translation number and makes both
unreadable.

Each target language is scored separately and then averaged by pair count, because
a single corpus BLEU across mixed targets is not a meaningful number -- the
tokenizer differs per language.

Go through these functions rather than constructing a sacrebleu metric directly.
`_metric` caches by tokenizer: building a fresh one per call makes `ja-mecab` and
`ko-mecab` create a MeCab Tagger each time, which mmaps four ipadic dictionaries
and never releases them. Scoring sentence by sentence then accumulates mappings
linearly until `vm.max_map_count` (65530 by default) is exhausted -- measured at
16,276 Taggers, 65,104 mappings -- after which thread stack allocation fails with
`RuntimeError: can't allocate lock`, the pool's workers die, and their futures
never complete, so the run hangs rather than crashing. sacrebleu's BLEU keeps no
state between calls, so reuse is safe.
"""
import re
from functools import lru_cache

from . import routing

# sacrebleu tokenizers. Scores produced with different tokenizers cannot be
# compared, so the signature travels with every number.
DEFAULT_TOKENIZE = {
    "de": "13a", "en": "13a", "es": "13a", "fr": "13a", "it": "13a",
    "nl": "13a", "pt": "13a", "ro": "13a", "ru": "13a", "cs": "13a",
    "zh": "zh", "ja": "ja-mecab", "ko": "ko-mecab",
}

# Subtitles carry event markers like "(Laughter)" / "(Gelächter)". ASR does not
# produce them, so leaving them in a reference deducts BLEU unfairly. Only known
# event words are removed -- dropping every parenthesis would damage references
# that legitimately contain one, such as "(2005)".
_NONSPEECH_WORDS = (
    "laughter", "laughs", "applause", "cheers", "cheering", "music", "singing",
    "video", "audio", "recording", "silence", "sighs", "beat", "sniffs",
    "laughter and applause", "clapping", "boos", "gasps",
    "gelächter", "lachen", "applaus", "beifall", "musik", "gesang", "jubel",
    "stille", "seufzt", "klatschen",
)
_NONSPEECH_RE = re.compile(
    r"[\(\[]\s*(?:" + "|".join(re.escape(w) for w in _NONSPEECH_WORDS) + r")\s*[\)\]]",
    re.IGNORECASE,
)
_WS = re.compile(r"\s+")


def strip_nonspeech(text: str) -> str:
    if not text:
        return ""
    return _WS.sub(" ", _NONSPEECH_RE.sub(" ", text)).strip()


def resolve_tokenize(target_lang: str) -> str:
    return DEFAULT_TOKENIZE.get((target_lang or "").lower(), "13a")


@lru_cache(maxsize=None)
def _resolve(tokenize: str, effective_order: bool):
    """(metric, signature_suffix) on success, (None, reason) on failure.

    Failures are cached too. `lru_cache` does not memoize exceptions, so caching a
    constructor that raises caches nothing: a missing ja/ko dictionary would be
    retried -- and its fallback rebuilt -- on every call, which is the repeated
    construction this cache exists to prevent.

    A missing MeCab dictionary falls back to the character tokenizer and says so in
    the signature. That fallback changes the score, so it must never pass silently.
    """
    try:
        import sacrebleu  # noqa: F401
    except ImportError:
        return None, "sacrebleu-not-installed"
    try:
        return _build(tokenize, effective_order), ""
    except Exception as exc:  # noqa: BLE001 - ja-mecab / ko-mecab not installed
        if tokenize not in ("ja-mecab", "ko-mecab"):
            return None, f"sacrebleu-error: {exc}"
    try:
        return _build("char", effective_order), " [fallback:char]"
    except Exception as exc:  # noqa: BLE001
        return None, f"sacrebleu-error: {exc}"


def _build(tokenize: str, effective_order: bool):
    import sacrebleu

    return sacrebleu.metrics.BLEU(tokenize=tokenize, effective_order=effective_order)


def _clean(text: str, strip_events: bool) -> str:
    return strip_nonspeech(text) if strip_events else (text or "")


def for_pairs(pairs, *, target_lang: str, tokenize: str | None = None,
              strip_events: bool = True) -> dict:
    """pairs = [(hypothesis, reference)] sharing one target language."""
    tok = tokenize or resolve_tokenize(target_lang)
    hyps = [_clean(h, strip_events) for h, _ in pairs]
    refs = [_clean(r, strip_events) for _, r in pairs]
    if not hyps:
        return {"bleu": None, "bleu_signature": None, "bleu_tokenize": tok, "n_pairs": 0}
    metric, note = _resolve(tok, False)
    if metric is None:
        return {"bleu": None, "bleu_signature": note, "bleu_tokenize": tok,
                "n_pairs": len(hyps)}
    score = metric.corpus_score(hyps, [refs])
    return {"bleu": score.score, "bleu_signature": str(metric.get_signature()) + note,
            "bleu_tokenize": tok, "n_pairs": len(hyps)}


def by_target(pairs_by_target, *, strip_events: bool = True) -> dict:
    per_lang = {target: for_pairs(pairs, target_lang=target, strip_events=strip_events)
                for target, pairs in sorted(pairs_by_target.items())}
    scored = [(v["n_pairs"], v["bleu"]) for v in per_lang.values()
              if v["bleu"] is not None]
    total = sum(n for n, _ in scored)
    weighted = (sum(n * s for n, s in scored) / total) if total else None
    # `n_pairs` counts the pairs that produced a score; `n_built` counts the pairs
    # that existed. They differ when the scorer could not run, which is exactly the
    # case where "were there pairs at all?" decides what to tell the reader.
    return {"bleu": weighted, "by_target": per_lang, "n_pairs": total,
            "n_built": sum(v["n_pairs"] for v in per_lang.values())}


def sentence(hypothesis: str, reference: str, *, target_lang: str,
             strip_events: bool = True) -> float | None:
    """One utterance's BLEU. Secondary to the corpus figure -- it swings hard on
    short sentences -- but it is what ranks utterances for inspection, so it takes
    the character fallback rather than returning None and dropping the utterance
    out of the ranking entirely."""
    hyp = _clean(hypothesis, strip_events)
    ref = _clean(reference, strip_events)
    if not hyp or not ref:
        return None
    metric, _ = _resolve(resolve_tokenize(target_lang), True)
    if metric is None:
        return None
    try:
        return metric.sentence_score(hyp, [ref]).score
    except Exception:  # noqa: BLE001
        return None


def _pairs(items, languages, *, only_routed: bool):
    """One pair per utterance per target language, not one per segment.

    The segments of an utterance are joined back into a single hypothesis before
    scoring. Pairing each segment against the utterance's whole reference instead
    compares a fragment to the complete sentence, and BLEU's brevity penalty then
    scales the score by roughly exp(1 - N) for an utterance committed in N pieces --
    0.37x at two segments, 0.14x at three. Since a more eager commit policy produces
    more segments, that deflation falls hardest on exactly the axis a run is
    measuring, and a policy would lose on segment count rather than on translation.

    An utterance whose segments went to different targets contributes one pair to
    each, which is what keeps the routed-vs-all split meaningful: routing is decided
    per commit, so a single utterance can be partly correct.

    A segment that produced no translation still joins its group. Dropping it would
    hide a failure the reference says should have been translated.
    """
    by_lang: dict[str, list] = {}
    no_reference: set[str] = set()
    for item in items:
        groups: dict[str, list[str]] = {}
        for segment in item.segments:
            if not segment.target_lang:
                continue
            if only_routed and not routing.is_correctly_routed(item, segment, languages):
                continue
            groups.setdefault(segment.target_lang, []).append(segment.translation)
        for target, parts in groups.items():
            reference = item.reference_translation(target)
            if not reference:
                no_reference.add(target)
                continue
            hypothesis = " ".join(p for p in parts if p).strip()
            by_lang.setdefault(target, []).append((hypothesis, reference))
    return by_lang, no_reference


def corpus(items, *, languages, strip_events: bool = True) -> tuple[dict, dict]:
    """(values, unavailable).

    `bleu` covers correctly-routed output only and `bleu_all` covers everything. In
    a multi-language run a misrouted utterance is compared against a reference in a
    language it was never translated into, which scores near zero for reasons that
    have nothing to do with translation quality -- so reporting only the combined
    figure blends a language-detection failure into the translation number and makes
    both unreadable.

    A target language the dataset carries no reference for is simply absent from the
    breakdown, and named in the reason. That is the normal case for a dataset with
    transcripts but translations in only some languages.
    """
    items = list(items)
    routed, routed_missing = _pairs(items, languages, only_routed=True)
    everything, all_missing = _pairs(items, languages, only_routed=False)

    scored = by_target(routed, strip_events=strip_events)
    combined = by_target(everything, strip_events=strip_events)

    values, unavailable = {}, {}
    if scored["bleu"] is not None:
        values["bleu"] = scored["bleu"]
        values["bleu_by_target"] = {k: v for k, v in scored["by_target"].items()
                                    if v["bleu"] is not None}
    if combined["bleu"] is not None:
        values["bleu_all"] = combined["bleu"]

    # A target only counts as missing when it produced no pair at all. An item
    # without a reference is a pair that could not be built, and the breakdown's
    # `n_pairs` already says how many were -- reporting the language as unavailable
    # while its score sits in `values` would put one metric in both places.
    uncovered = (routed_missing | all_missing) - set(routed) - set(everything)
    if scored["bleu"] is None:
        unavailable["bleu"] = _why_no_bleu(items, scored, combined, uncovered)
    elif uncovered:
        unavailable["bleu_by_target"] = (
            "no reference translation for " + ", ".join(sorted(uncovered)))
    return values, unavailable


def _why_no_bleu(items, scored, combined, missing) -> str:
    """Why there is no BLEU, in the order that tells the reader what to fix.

    Whether pairs were built at all comes before anything about references: if they
    were, the references are plainly there and the scorer is what stopped -- saying
    "no reference translation" then sends the reader to look at their dataset for a
    missing package.
    """
    if not any(item.segments for item in items):
        return "no segment was committed"
    if combined["bleu"] is not None:
        return "no correctly-routed segment had a reference translation"
    if scored["n_built"] or combined["n_built"]:
        reasons = sorted({
            v["bleu_signature"] for v in
            list(scored["by_target"].values()) + list(combined["by_target"].values())
            if v["bleu"] is None and v["bleu_signature"]})
        if "sacrebleu-not-installed" in reasons:
            return "sacrebleu is not installed (pip install 'sacrebleu[ja,ko]')"
        return "; ".join(reasons) if reasons else "the scorer returned no score"
    if missing:
        return "no reference translation for " + ", ".join(sorted(missing))
    return "no segment reported a target language"
