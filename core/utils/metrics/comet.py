from functools import lru_cache

from . import routing

MODEL = "Unbabel/wmt22-comet-da"
BATCH_SIZE = 16


@lru_cache(maxsize=None)
def _model(name: str):
    from comet import download_model, load_from_checkpoint

    return load_from_checkpoint(download_model(name))


def _triples(items, languages) -> list[dict]:
    triples = []
    for item in items:
        target = languages.expected_target(item.src_lang)
        reference = item.reference_translation(target)
        if not item.reference or not reference:
            continue
        parts = [s.translation for s in item.segments
                 if s.target_lang and routing.is_correctly_routed(item, s, languages)]
        triples.append({"src": item.reference,
                        "mt": " ".join(p for p in parts if p).strip(),
                        "ref": reference})
    return triples


def corpus(items, *, languages, model: str = MODEL) -> tuple[dict, dict]:
    items = list(items)
    triples = _triples(items, languages)
    if not triples:
        return {}, {"comet": "no item carried both a reference transcript and a "
                             "reference translation in its expected target"}
    try:
        scorer = _model(model)
    except ImportError:
        return {}, {"comet": "unbabel-comet is not installed; run this through bench/comet (see bench/README.md)"}

    import torch

    output = scorer.predict(triples, batch_size=BATCH_SIZE,
                            gpus=1 if torch.cuda.is_available() else 0,
                            progress_bar=False)
    return {"comet": output.system_score, "comet_model": model,
            "n_comet_pairs": len(triples)}, {}
