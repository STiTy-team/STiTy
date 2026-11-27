"""Pure decision parts of the audio langid: score normalisation, the previous-language prior,
HMM smoothing over time steps and the fallback span."""

import math

# Qwen3-ForcedAligner's languages, by project code.
ALIGNER_LANGS = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
}


def _softmax(logp: dict[str, float]) -> dict[str, float] | None:
    finite = {k: v for k, v in logp.items() if v != -math.inf and not math.isnan(v)}
    if not finite:
        return None
    top = max(finite.values())
    exp = {k: math.exp(v - top) if k in finite else 0.0 for k, v in logp.items()}
    total = sum(exp.values())
    return {k: v / total for k, v in exp.items()}


def normalise(scores: dict[str, float], candidates: tuple[str, ...]) -> dict[str, float] | None:
    """Log-scores restricted to the candidates, as probabilities summing to 1."""
    return _softmax({c: scores.get(c, -math.inf) for c in candidates})


def with_prior(logp: dict[str, float], previous: str | None, stay: float) -> dict[str, float]:
    """Add a log prior: `stay` on the previous language, the rest shared by the others."""
    if previous is None or previous not in logp or len(logp) < 2:
        return dict(logp)
    rest = (1.0 - stay) / (len(logp) - 1)
    return {k: v + math.log(stay if k == previous else rest) for k, v in logp.items()}


def decide(probs: dict[str, float] | None, previous: str | None, conf: float) -> str | None:
    """The top candidate if it is the unique top and reaches `conf`; otherwise the previous
    language (or None). A tie means no evidence either way (e.g. a uniform posterior)."""
    if not probs:
        return previous
    ranked = sorted(probs.values(), reverse=True)
    best = max(probs, key=probs.get)
    if len(ranked) > 1 and ranked[0] <= ranked[1]:
        return previous
    return best if probs[best] >= conf else previous


def _transitions(p: float, k: int) -> list[list[float]]:
    stay = math.log(p)
    move = math.log((1.0 - p) / (k - 1)) if k > 1 else -math.inf
    return [[stay if i == j else move for j in range(k)] for i in range(k)]


def smooth(
    steps: list[dict[str, float] | None],
    candidates: tuple[str, ...],
    p_stay: float,
    p_gap: float | None = None,
    gaps: list[bool] | None = None,
) -> list[dict[str, float]]:
    """Forward-backward posteriors per step. Each step holds log-likelihoods per candidate
    (None = no evidence); the language stays with probability `p_stay` between steps, or
    `p_gap` into a step that follows a silence (`gaps[i]` true)."""
    n, k = len(steps), len(candidates)
    if n == 0:
        return []
    usual = _transitions(p_stay, k)
    across = _transitions(p_gap, k) if p_gap is not None else usual
    tr = [across if gaps and gaps[t] else usual for t in range(n)]
    emit = [[0.0] * k if s is None else [s.get(c, -math.inf) for c in candidates] for s in steps]

    def lse(xs):
        top = max(xs)
        return top if top == -math.inf else top + math.log(sum(math.exp(x - top) for x in xs))

    fwd = [[-math.log(k) + emit[0][j] for j in range(k)]]
    for t in range(1, n):
        fwd.append(
            [lse([fwd[t - 1][i] + tr[t][i][j] for i in range(k)]) + emit[t][j] for j in range(k)]
        )
    bwd = [[0.0] * k for _ in range(n)]
    for t in range(n - 2, -1, -1):
        bwd[t] = [
            lse([tr[t + 1][i][j] + emit[t + 1][j] + bwd[t + 1][j] for j in range(k)])
            for i in range(k)
        ]
    out = []
    for t in range(n):
        post = _softmax({c: fwd[t][j] + bwd[t][j] for j, c in enumerate(candidates)})
        out.append(post or {c: 1.0 / k for c in candidates})
    return out


def lag_span(prev_decision: float, decision: float, lag: float) -> tuple[float, float]:
    """The fallback span: the audio between decisions, moved back by `lag` seconds."""
    return max(0.0, prev_decision - lag), max(0.0, decision - lag)


def span_mass(
    steps: list[tuple[float, dict[str, float] | None]],
    posts: list[dict[str, float]],
    span: tuple[float, float],
    half: float,
    candidates: tuple[str, ...],
    read: str = "mean",
    read_sec: float = 1.0,
) -> dict[str, float] | None:
    """Posterior mass over the timeline steps whose centre (t - half) lies in the span. Steps in
    the span without speech give no verdict (None); a span holding no step centre at all reads
    the nearest step. `read` picks which in-span steps count: all (`mean`), those in the last
    `read_sec` of the span (`tail`), or only the last one with speech (`end`)."""
    if not steps:
        return None
    centres = [t - half for t, _ in steps]
    lo = max(span[0], span[1] - read_sec) if read == "tail" else span[0]
    in_span = [i for i, c in enumerate(centres) if lo <= c <= span[1]]
    if read == "tail" and not any(steps[i][1] for i in in_span):
        in_span = [i for i, c in enumerate(centres) if span[0] <= c <= span[1]]
    if in_span:
        inside = [i for i in in_span if steps[i][1]]
        if not inside:
            return None
        if read == "end":
            inside = inside[-1:]
    else:
        inside = [min(range(len(centres)), key=lambda i: abs(centres[i] - span[1]))]
    mass = {c: sum(posts[i][c] for i in inside) for c in candidates}
    total = sum(mass.values()) or 1.0
    return {c: v / total for c, v in mass.items()}


def combine(
    per_scorer: dict[str, dict[str, float]],
    candidates: tuple[str, ...],
    how: str = "mean",
    weights: dict[str, float] | None = None,
) -> dict[str, float] | None:
    """One log-score per candidate from several scorers, each first renormalised over the
    candidates: `mean` = weighted mean of log-probabilities, `max` = the scorer whose top
    candidate is most probable. None when no scorer has a verdict."""
    probs = {}
    for name, scores in per_scorer.items():
        p = normalise(scores, candidates)
        if p:
            probs[name] = p
    if not probs:
        return None
    if how == "max":
        best = max(probs, key=lambda n: max(probs[n].values()))
        return {c: math.log(v) if v > 0 else -math.inf for c, v in probs[best].items()}
    weights = weights or {}
    total = sum(weights.get(n, 1.0) for n in probs)
    return {
        c: sum(weights.get(n, 1.0) * math.log(max(p[c], 1e-12)) for n, p in probs.items()) / total
        for c in candidates
    }
