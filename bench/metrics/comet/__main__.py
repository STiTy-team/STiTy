import json
import logging
from argparse import Namespace
from pathlib import Path
from statistics import mean

from bench import notify, store
from bench.settings import BenchSettings
from core.utils import cli, env
from core.utils.json import read_json, read_jsonl, write_json

COMET_MODEL = "Unbabel/wmt22-comet-da"
INPUTS = "comet_inputs.jsonl"
SCORES = "comet_scores.jsonl"
KEYS = ("comet", "comet_by_pair", "comet_model")

log = logging.getLogger(__name__)


def comet(sentences: list[dict], scores: list[float] | None = None) -> dict[str, float] | None:
    by_pair: dict[str, list[float]] = {}
    for sentence, value in zip(
        sentences, scores if scores is not None else sentence_scores(sentences)
    ):
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
    output = model.predict(
        [{k: s[k] for k in ("src", "mt", "ref")} for s in sentences],
        batch_size=16,
        gpus=1 if torch.cuda.is_available() else 0,
        progress_bar=False,
    )
    return list(output.scores)


def score(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    summary = read_json(summary_path)
    inputs_path = run_dir / INPUTS
    sentences = [s for s in read_jsonl(inputs_path) if s["src"]] if inputs_path.exists() else []

    scores = sentence_scores(sentences) if sentences else []
    by_pair = comet(sentences, scores)
    # One score per scored sentence, in `comet_inputs.jsonl` order -- the replay
    # dashboard draws the run's COMET distribution from it, not just the mean.
    with open(run_dir / SCORES, "w", encoding="utf-8") as f:
        for sentence, value in zip(sentences, scores):
            f.write(json.dumps({"pair": sentence["pair"], "comet": value}) + "\n")
    for key in KEYS:
        summary["metrics"].pop(key, None)
        summary["unavailable"].pop(key, None)
    if by_pair is None:
        reason = (
            f"{INPUTS} is missing; rerun the bench"
            if not inputs_path.exists()
            else "no sentence had a reference transcript and a reference translation "
            "in its target language"
        )
        summary["unavailable"].update({"comet": reason, "comet_by_pair": reason})
    else:
        summary["metrics"].update(
            {"comet": mean(by_pair.values()), "comet_by_pair": by_pair, "comet_model": COMET_MODEL}
        )
    write_json(summary_path, summary)
    return {key: summary["metrics"].get(key) for key in ("comet", "comet_by_pair")}


def main(args: Namespace) -> int:
    print(json.dumps(score(args.run_dir), ensure_ascii=False))
    summary = read_json(args.run_dir / "summary.json")
    notify.success(summary, run_id=upload(args.run_dir, summary))
    return 0


def upload(run_dir: Path, summary: dict) -> str | None:
    settings = BenchSettings.load()
    bucket = settings.bucket()
    if bucket is None:
        log.warning("[UPLOAD-SKIPPED] S3 not configured, the run stays on this machine")
        return None
    already_shared = store.shared_run_id(run_dir)
    if already_shared:
        log.warning("[UPLOAD-SKIPPED] already uploaded as %s", already_shared)
        return already_shared
    try:
        run = store.upload_run(bucket, run_dir, settings.stity_host)
    except Exception as e:
        log.warning("[UPLOAD-FAILED] the run stays on this machine: %s", e)
        return None
    log.info("[UPLOADED] %s", run.folder)
    return run.run_id


if __name__ == "__main__":
    args = cli.parse(
        [
            {
                "name": "run-dir",
                "type": Path,
                "help": "bench/runs/<이름>",
            }
        ],
        prog="python -m bench.metrics.comet",
        description="bench 실행 하나에 COMET 을 채점해 summary.json 에 더한다",
    )
    env.load()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        raise SystemExit(main(args))
    except Exception as e:
        summary_path = args.run_dir / "summary.json"
        notify.failure(e, summary=read_json(summary_path) if summary_path.exists() else None)
        raise
