"""Folding verbatim duplicates out of a translation stream, against a transcript.

A direct-translation stream passes same-language speech through verbatim, so the
viewer would read the sentence twice: once on the transcription pane and once on
the translation pane. These helpers decide, from text alone, what of a translation
commit is already carried by the transcript near it — no language identification,
so a wrong language verdict can never eat content.

`region_fold` works at token level with gap absorption: a token matches when its
normalized form appears in the transcript window, a single mismatching token
BETWEEN matches folds with its run (the two models disagree on proper nouns and
spelling), and a run only folds when it has at least two matches and four
normalized characters. Folding whole commits was measured and rejected twice: it
clips the head of the next translated sentence when a commit straddles a language
switch, and any short-commit containment rule collides with conversational echo
(speakers repeating each other's words across languages). Measured on northstar:
region fold keeps COMET within 0.005 of the unfolded stream while whole-commit
folding cost 0.014-0.025.
"""
import re

_NORM = re.compile(r"[^0-9a-z가-힣]")

MATCH_WINDOW_BACK_SEC = 8.0
MATCH_WINDOW_AHEAD_SEC = 2.0
MIN_RUN_MATCHES = 2
MIN_RUN_CHARS = 4
MIN_TOKEN_CHARS = 2


def norm(text: str) -> str:
    return _NORM.sub("", text.lower())


def window_text(transcript: list[tuple[float, str]], at_sec: float) -> str:
    """The normalized transcript heard around `at_sec`, as one string.

    `transcript` rows are `(decision_audio_sec, normalized_text)`; callers keep only
    what has actually arrived, so causality is theirs to enforce.
    """
    return "".join(
        text for sec, text in transcript
        if at_sec - MATCH_WINDOW_BACK_SEC <= sec <= at_sec + MATCH_WINDOW_AHEAD_SEC
    )


def region_fold(text: str, window: str) -> str:
    """`text` with the runs the transcript already carries removed.

    `""` means the whole commit was a duplicate. Returning `text` unchanged means
    the transcript (so far) carries none of it — the caller may ask again when
    more transcript arrives.
    """
    if not window:
        return text
    tokens = text.split()
    hit = [len(norm(tok)) >= MIN_TOKEN_CHARS and norm(tok) in window for tok in tokens]
    keep = list(hit)
    for i in range(1, len(tokens) - 1):
        if not keep[i] and hit[i - 1] and hit[i + 1]:
            keep[i] = True
    out, i = [], 0
    while i < len(tokens):
        if not keep[i]:
            out.append(tokens[i])
            i += 1
            continue
        j = i
        while j < len(tokens) and keep[j]:
            j += 1
        run = tokens[i:j]
        matches = sum(1 for k in range(i, j) if hit[k])
        if not (matches >= MIN_RUN_MATCHES and len(norm("".join(run))) >= MIN_RUN_CHARS):
            out.extend(run)
        i = j
    return re.sub(r"^[\s,.!?…]+", "", " ".join(out))
