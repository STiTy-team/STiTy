import argparse
import json
from pathlib import Path
from statistics import mean

from core.utils.stream import read

COMET_MODEL = "Unbabel/wmt22-comet-da"
INPUTS = "comet_inputs.jsonl"
KEYS = ("comet", "comet_by_pair", "comet_model")


def comet(sentences: list[dict]) -> dict[str, float] | None:
    by_pair: dict[str, list[float]] = {}
    for sentence, value in zip(sentences, sentence_scores(sentences)):
        by_pair.setdefault(sentence["pair"], []).append(value)
    return {pair: mean(values) for pair, values in sorted(by_pair.items())} or None


def sentence_scores(sentences: list[dict]) -> list[float]:
    translated = [s for s in sentences if s["mt"]]
    predicted = iter(_predict(translated) if translated else [])
    return [next(predicted) if s["mt"] else 0.0 for s in sentences]


def _predict(sentences: list[dict]) -> list[float]:
    import torch
    from comet import download_model, load_from_checkpoint

    model = load_from_checkpoint(download_model(COMET_MODEL))
    output = model.predict([{k: s[k] for k in ("src", "mt", "ref")} for s in sentences],
                           batch_size=16, gpus=1 if torch.cuda.is_available() else 0,
                           progress_bar=False)
    return list(output.scores)


def score(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    inputs_path = run_dir / INPUTS
    sentences = ([s for s in read(inputs_path) if s["src"]] if inputs_path.exists() else [])

    by_pair = comet(sentences)
    for key in KEYS:
        summary["metrics"].pop(key, None)
        summary["unavailable"].pop(key, None)
    if by_pair is None:
        reason = (f"{INPUTS} is missing; rerun the bench" if not inputs_path.exists() else
                  "no sentence had a reference transcript and a reference translation "
                  "in its target language")
        summary["unavailable"].update({"comet": reason, "comet_by_pair": reason})
    else:
        summary["metrics"].update({"comet": mean(by_pair.values()), "comet_by_pair": by_pair,
                                   "comet_model": COMET_MODEL})
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
    return {key: summary["metrics"].get(key) for key in ("comet", "comet_by_pair")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bench.comet",
        description="bench 실행 하나에 COMET 을 채점해 summary.json 에 더한다",
    )
    parser.add_argument("run_dir", type=Path, help="bench/runs/<이름>")
    args = parser.parse_args(argv)
    print(json.dumps(score(args.run_dir), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
