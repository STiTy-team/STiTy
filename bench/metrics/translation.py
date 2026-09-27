import re

from sacrebleu.metrics import BLEU

from .common import (
    macro,
    resegmented,
    sentence_lang,
    src_code,
    succeeded,
    translated,
    translating,
)


BLEU_TOKENIZE = {"zh": "zh", "ja": "ja-mecab", "ko": "ko-mecab"}
NONSPEECH = re.compile(
    r"[\(\[]\s*(?:laughter and applause|laughter|laughs|applause|cheers|cheering|music|"
    r"singing|video|audio|recording|silence|sighs|beat|sniffs|clapping|boos|gasps|"
    r"gelächter|lachen|applaus|beifall|musik|gesang|jubel|stille|seufzt|klatschen)\s*[\)\]]",
    re.IGNORECASE,
)


def bleu(rows: list[dict], target: str) -> float | None:
    return macro(bleu_pairs(rows, target))


def bleu_pairs(rows: list[dict], target: str) -> dict[str, float] | None:
    by_pair: dict[str, list[dict]] = {}
    for sentence in translation_sentences(rows, target):
        by_pair.setdefault(sentence["pair"], []).append(sentence)

    scores = {}
    for pair, sentences in sorted(by_pair.items()):
        target = pair.rsplit("-", 1)[1]
        metric = BLEU(
            tokenize=BLEU_TOKENIZE.get(target, "13a"), effective_order=len(sentences) == 1
        )
        score = metric.corpus_score([s["mt"] for s in sentences], [[s["ref"] for s in sentences]])
        scores[pair] = score.score
    return scores or None


def translation_sentences(rows: list[dict], target: str) -> list[dict]:
    sentences = []
    for row in translating(succeeded(rows), target):
        if row.get("reference_segmentation"):
            found = [
                (
                    sentence_lang(row, sentence),
                    sentence.get("transcript") or "",
                    instance.prediction,
                    instance.reference,
                )
                for sentence, instance in resegmented(row, target)
            ]
        elif row["reference_translations"].get(target):
            found = [
                (
                    src_code(row),
                    row["reference"] or "",
                    _translation(row, target),
                    row["reference_translations"][target],
                )
            ]
        else:
            found = []
        sentences += [
            {"pair": f"{lang}-{target}", "src": _clean(src), "mt": _clean(mt), "ref": _clean(ref)}
            for lang, src, mt, ref in found
        ]
    return sentences


def _clean(text: str) -> str:
    return " ".join(NONSPEECH.sub(" ", text).split())


def _translation(row: dict, target: str) -> str:
    return " ".join(
        s["translation"]
        for s in translated(row["records"])
        if s.get("translation") and (s.get("target_lang") or target) == target
    )
