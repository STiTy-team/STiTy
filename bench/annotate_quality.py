from __future__ import annotations

import argparse
import asyncio
import copy
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from core.utils import metrics
from core.utils.metrics import critical_information, fluency

from .quality_annotators import (
    OpenAIJsonJudge,
    QUALITY_ANNOTATOR_VERSION,
    annotate_critical_information,
    annotate_fluency,
)


DEFAULT_LM = {
    "en": "FacebookAI/roberta-base",
    "ko": "klue/roberta-base",
}


class KoEnLanguages:
    def expected_target(self, src_lang: str) -> str:
        return "ko" if str(src_lang).lower() == "en" else "en"


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as source:
        for lineno, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{lineno} is not a JSON object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _source_items(path: str | Path) -> tuple[Path, Path]:
    candidate = Path(path).resolve()
    items = candidate if candidate.is_file() else candidate / "items.jsonl"
    if not items.is_file():
        raise ValueError(f"items.jsonl not found: {items}")
    return items.parent, items


def _target_lang(row: dict) -> str:
    source = str(row.get("src_lang") or "").lower()
    if source not in {"ko", "en"}:
        raise ValueError(f"quality annotation supports only ko<->en, got {source!r}")
    return "en" if source == "ko" else "ko"


def _contexts(rows: list[dict]) -> dict[int, list[str]]:
    history: dict[str, list[str]] = {}
    output = {}
    for index, row in enumerate(rows):
        group = str(row.get("group") or row.get("id") or index)
        output[index] = list(history.get(group, []))[-3:]
        candidate = str(row.get("hypothesis_translation") or "").strip()
        if candidate:
            history.setdefault(group, []).append(candidate)
    return output


async def annotate_rows(rows: list[dict], *, judge, concurrency: int = 4,
                        annotate_critical: bool = True,
                        annotate_spoken_fluency: bool = True) -> list[dict]:
    contexts = _contexts(rows)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def annotate(index: int, source_row: dict) -> dict:
        row = copy.deepcopy(source_row)
        if row.get("status") not in (None, "ok"):
            return row
        source_lang = str(row.get("src_lang") or "").lower()
        target_lang = _target_lang(row)
        source = str(row.get("hypothesis") or row.get("reference") or "").strip()
        candidate = str(row.get("hypothesis_translation") or "").strip()
        reference = str((row.get("reference_translations") or {}).get(target_lang) or "").strip()
        if not candidate:
            return row
        metric_inputs = copy.deepcopy(row.get("metric_inputs") or {})
        annotation_errors = []
        async with semaphore:
            if annotate_critical and source and reference:
                gold = ((metric_inputs.get("critical_information") or {})
                        .get("reference_spans"))
                try:
                    metric_inputs["critical_information"] = await annotate_critical_information(
                        judge=judge, source=source, reference=reference, candidate=candidate,
                        source_lang=source_lang, target_lang=target_lang,
                        gold_reference_spans=gold,
                    )
                except Exception as exc:
                    annotation_errors.append({"axis": "critical_information",
                                              "error": type(exc).__name__, "detail": str(exc)})
            if annotate_spoken_fluency:
                try:
                    metric_inputs["fluency"] = await annotate_fluency(
                        judge=judge, candidate=candidate, target_lang=target_lang,
                        previous_turns=contexts[index],
                    )
                except Exception as exc:
                    annotation_errors.append({"axis": "fluency", "error": type(exc).__name__,
                                              "detail": str(exc)})
        row["metric_inputs"] = metric_inputs
        if annotation_errors:
            row["quality_annotation_errors"] = annotation_errors
        return row

    return list(await asyncio.gather(*(annotate(index, row)
                                       for index, row in enumerate(rows))))


def add_pseudo_perplexity(rows: list[dict], *, models: dict[str, str],
                          device: str | None = None, batch_size: int = 16) -> None:
    grouped: dict[str, list[dict]] = {"ko": [], "en": []}
    for row in rows:
        candidate = str(row.get("hypothesis_translation") or "").strip()
        if candidate and row.get("status") in (None, "ok"):
            target = _target_lang(row)
            grouped[target].append({"id": str(row.get("id") or ""), "text": candidate})
    by_id = {str(row.get("id") or ""): row for row in rows}
    for lang, texts in grouped.items():
        if not texts:
            continue
        model = models.get(lang)
        if not model:
            continue
        try:
            result = fluency.target_lm_pseudo_perplexity(
                texts, model_name=model, device=device, batch_size=batch_size)
            for item_id, value in result["per_item"].items():
                row = by_id[item_id]
                block = row.setdefault("metric_inputs", {}).setdefault("fluency", {})
                block["target_lm_pseudo_perplexity"] = value
                block["target_lm_model"] = model
                block["target_lang"] = lang
        finally:
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


def score_quality(rows: list[dict]) -> metrics.RunScore:
    items = [metrics.Utterance.from_row(row) for row in rows
             if row.get("status") in (None, "ok")]
    aggregate = {}
    unavailable = {}
    for values, missing in (critical_information.corpus(items), fluency.corpus(items)):
        aggregate.update(values)
        unavailable.update(missing)
    return metrics.RunScore(aggregate=aggregate, unavailable=unavailable)


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    targets = [path / name for name in (
        "items.jsonl", "summary.json", "quality_config.json", "source_config.yml")]
    existing = [target for target in targets if target.exists()]
    if existing and not overwrite:
        raise ValueError(f"output already exists; pass --overwrite: {path}")
    for target in existing:
        target.unlink()


def _summary(*, rows: list[dict], score: metrics.RunScore, source_run: Path,
             args, started: datetime, finished: datetime) -> dict:
    errors = sum(len(row.get("quality_annotation_errors") or []) for row in rows)
    return {
        "mode": "quality_annotation",
        "status": "degraded" if errors else "ok",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "source_run": str(source_run),
        "counts": {"items": len(rows),
                   "annotated": sum(bool(row.get("metric_inputs")) for row in rows),
                   "annotation_errors": errors},
        "metrics": score.aggregate,
        "unavailable": score.unavailable,
        "annotation": {
            "version": QUALITY_ANNOTATOR_VERSION,
            "judge_model": args.judge_model,
            "critical_information": not args.skip_critical,
            "spoken_fluency_mqm": not args.skip_judge,
            "pseudo_perplexity": not args.skip_pseudo_perplexity,
            "lm": {"en": args.lm_en, "ko": args.lm_ko},
        },
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m bench.annotate_quality",
        description="한영 중요 정보와 목표 언어 유창성 주석을 기존 run에 생성합니다.",
    )
    parser.add_argument("source_run", help="원본 run 디렉터리 또는 items.jsonl")
    parser.add_argument("output", help="보강된 items.jsonl과 summary.json을 저장할 디렉터리")
    parser.add_argument("--judge-model", default="gpt-5.4-mini")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--lm-en", default=DEFAULT_LM["en"])
    parser.add_argument("--lm-ko", default=DEFAULT_LM["ko"])
    parser.add_argument("--device")
    parser.add_argument("--lm-batch-size", type=int, default=16)
    parser.add_argument("--skip-critical", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--skip-pseudo-perplexity", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = datetime.now(timezone.utc)
    source_run, source_items = _source_items(args.source_run)
    output = Path(args.output).resolve()
    if output == source_run:
        raise ValueError("quality annotations must be written to a derived run directory")
    _prepare_output(output, overwrite=args.overwrite)
    rows = _read_jsonl(source_items)
    judge = None
    if not args.skip_critical or not args.skip_judge:
        judge = OpenAIJsonJudge(model=args.judge_model)
    rows = asyncio.run(annotate_rows(
        rows, judge=judge, concurrency=args.concurrency,
        annotate_critical=not args.skip_critical,
        annotate_spoken_fluency=not args.skip_judge,
    ))
    if not args.skip_pseudo_perplexity:
        add_pseudo_perplexity(
            rows, models={"en": args.lm_en, "ko": args.lm_ko},
            device=args.device, batch_size=args.lm_batch_size)
    _write_jsonl(output / "items.jsonl", rows)
    source_config = source_run / "config.yml"
    if source_config.is_file():
        shutil.copyfile(source_config, output / "source_config.yml")
    score = score_quality(rows)
    finished = datetime.now(timezone.utc)
    summary = _summary(rows=rows, score=score, source_run=source_run,
                       args=args, started=started, finished=finished)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (output / "quality_config.json").write_text(
        json.dumps(summary["annotation"], indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": score.aggregate,
                      "unavailable": score.unavailable}, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
