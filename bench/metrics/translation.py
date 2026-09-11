"""Translation quality (BLEU).

Two numbers, deliberately:

  bleu      over correctly-routed segments only
  bleu_all  over everything

In a multi-language run a misrouted utterance is compared against a reference in
a language it was never translated into, which produces a near-zero score that
has nothing to do with translation quality. Reporting only the combined number
would blend a language-detection failure into the translation figure and make
both unreadable.

corpus_bleu_score must be called once per run per language pair, never per
sentence: building a ja/ko MeCab tokenizer per sentence has leaked 65,104 mmaps
and hung a thread pool. metrics_ast._bleu_metric's lru_cache is what prevents
that, so go through these functions rather than constructing metrics directly.
"""
from evaluation.ast.metrics_ast import corpus_bleu_score, resolve_tokenize, strip_nonspeech


def available() -> bool:
    try:
        import sacrebleu  # noqa: F401
    except ImportError:
        return False
    return True


def _clean(text: str, strip_events: bool) -> str:
    text = text or ""
    return strip_nonspeech(text) if strip_events else text


def bleu_for_pairs(pairs, *, target_lang: str, tokenize: str | None = None,
                   strip_events: bool = True) -> dict:
    """pairs = [(hypothesis, reference)] that share one target language."""
    tok = tokenize or resolve_tokenize(target_lang)
    hyps = [_clean(h, strip_events) for h, _ in pairs]
    refs = [_clean(r, strip_events) for _, r in pairs]
    if not hyps:
        return {"bleu": None, "bleu_signature": None, "bleu_tokenize": tok, "n_pairs": 0}
    score, signature = corpus_bleu_score(hyps, refs, tokenize=tok)
    return {"bleu": score, "bleu_signature": signature, "bleu_tokenize": tok,
            "n_pairs": len(hyps)}


def bleu_by_target(pairs_by_target, *, strip_events: bool = True) -> dict:
    """Scores each target language separately, then reports a weighted mean.

    A single corpus BLEU across mixed target languages is not a meaningful
    number -- the tokenizer differs per language.
    """
    per_lang = {}
    for target_lang, pairs in sorted(pairs_by_target.items()):
        per_lang[target_lang] = bleu_for_pairs(pairs, target_lang=target_lang,
                                               strip_events=strip_events)
    scored = [(v["n_pairs"], v["bleu"]) for v in per_lang.values() if v["bleu"] is not None]
    total = sum(n for n, _ in scored)
    weighted = (sum(n * s for n, s in scored) / total) if total else None
    return {"bleu": weighted, "by_target": per_lang, "n_pairs": total}
