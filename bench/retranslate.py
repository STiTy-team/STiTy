"""Replay stored ASR commits through a different translation backend.

Usage::

    python -m bench.retranslate bench/runs/<source> configs/retranslate.yml

No audio, VAD or ASR code is loaded.  The source run's committed
``segments[*].original`` values are translated in their original order.  The new
run therefore changes the translator while freezing ASR and segmentation.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from core.utils import metrics


RUNS_DIR = Path(__file__).resolve().parent / "runs"
RUN_FILES = ("events.jsonl", "items.jsonl", "summary.json", "config.yml")
QUALITY_AXES = {
    "meaning", "critical_information", "fluency", "context",
    "asr_robustness", "intent",
}

_STALE_ITEM_METRICS = {
    "wer", "cer", "avg_fsl_sec", "n_seg_with_fsl", "laal_ms", "laal_ca_ms",
    "laal_uncapped_ms", "sentence_bleu", "route_errors", "n_segments",
}


def _gold_metric_inputs(raw) -> dict:
    """Keep human/reference annotations; discard values tied to the old output."""
    source = copy.deepcopy(dict(raw or {}))
    result = {key: value for key, value in source.items()
              if key not in {"meaning", "critical_information", "fluency", "context",
                             "asr_robustness", "intent"}}

    critical = source.get("critical_information") or {}
    if "reference_spans" in critical:
        result["critical_information"] = {
            "reference_spans": copy.deepcopy(critical["reference_spans"]),
        }

    intent = {}
    for name, labels in (source.get("intent") or {}).items():
        if isinstance(labels, dict) and labels.get("reference") is not None:
            intent[name] = {"reference": copy.deepcopy(labels["reference"])}
    if intent:
        result["intent"] = intent
    return result


def _source_segments(row: dict) -> tuple[list[dict], str]:
    segments = [copy.deepcopy(segment) for segment in (row.get("segments") or [])
                if (segment.get("original") or "").strip()]
    if segments:
        return segments, "committed_segments"
    hypothesis = (row.get("hypothesis") or "").strip()
    if not hypothesis:
        return [], "empty"
    return ([{"original": hypothesis, "language": row.get("src_lang") or "",
              "commit_reason": "utterance_fallback"}], "utterance_fallback")


def _detach_source_timing(segment: dict) -> dict:
    timing = {}
    for key in ("decision_audio_sec", "recv_elapsed_sec"):
        if segment.get(key) is not None:
            timing[key] = segment.pop(key)
    if timing:
        segment["source_timing"] = timing
    return segment


async def retranslate_row(row: dict, *, translator, languages,
                          context: list[str] | None = None,
                          on_segment=None) -> dict:
    """Translate one stored item while preserving ASR text and commit boundaries."""
    output = copy.deepcopy(row)
    for key in _STALE_ITEM_METRICS:
        output.pop(key, None)
    output["source_status"] = row.get("status") or ""
    output["metric_inputs"] = _gold_metric_inputs(row.get("metric_inputs"))
    output["translation_errors"] = []

    if row.get("status") not in (None, "ok"):
        output["translation_status"] = "skipped_source_error"
        output["hypothesis_translation"] = ""
        output["segments"] = []
        return output

    source_lang = str(row.get("src_lang") or "").lower()
    target_lang = languages.expected_target(source_lang)
    translator.start(language=source_lang)
    running_context = context if context is not None else []
    source_segments, segmentation = _source_segments(row)
    translated_segments = []
    total_elapsed = 0.0

    if not source_segments:
        output["translation_status"] = "skipped_empty_source"
        output["hypothesis_translation"] = ""
        output["segments"] = []
        output["translation_calls"] = 0
        output["translation_runtime_sec"] = 0.0
        output["translation_source"] = segmentation
        return output

    for index, source in enumerate(source_segments):
        segment = _detach_source_timing(source)
        original = (segment.get("original") or "").strip()
        detected_source = (segment.get("language") or source_lang).lower()
        started = time.perf_counter()
        try:
            translation, detected = await translator.translate(
                original, target_lang, detected_source,
                context=list(running_context),
            )
        except Exception as exc:  # noqa: BLE001 - preserve the rest of the run
            translation, detected = "", ""
            output["translation_errors"].append(
                {"segment_index": index, "error": type(exc).__name__, "detail": str(exc)})
        elapsed = time.perf_counter() - started
        total_elapsed += elapsed

        if original and not (translation or "").strip() and not any(
                error["segment_index"] == index for error in output["translation_errors"]):
            output["translation_errors"].append(
                {"segment_index": index, "error": "empty_translation"})

        segment["translation"] = translation or ""
        segment["language"] = segment.get("language") or detected or source_lang
        segment["target_lang"] = target_lang
        segment["translation_elapsed_sec"] = elapsed
        translated_segments.append(segment)
        if on_segment is not None:
            on_segment(segment, index)
        if original:
            running_context.append(original)

    output["segments"] = translated_segments
    output["hypothesis_translation"] = " ".join(
        segment["translation"].strip() for segment in translated_segments
        if segment["translation"].strip()
    )
    output["translation_status"] = (
        "degraded" if output["translation_errors"] else "ok")
    output["translation_calls"] = len(translated_segments)
    output["translation_runtime_sec"] = total_elapsed
    output["translation_source"] = segmentation

    reference = (output.get("reference_translations") or {}).get(target_lang, "")
    sentence_bleu = metrics.bleu.sentence(
        output["hypothesis_translation"], reference, target_lang=target_lang)
    if sentence_bleu is not None:
        output["sentence_bleu"] = sentence_bleu
    return output


async def retranslate_rows(rows, *, translator, languages,
                           context_scope: str = "item",
                           before_item=None, after_item=None,
                           on_segment=None) -> list[dict]:
    """Replay rows in source order; optionally carry source-text context by group."""
    if context_scope not in ("item", "group"):
        raise ValueError("context_scope must be 'item' or 'group'")
    group_contexts: dict[str, list[str]] = {}
    output = []
    for row in rows:
        if before_item is not None:
            before_item(row)
        group = str(row.get("group") or row.get("id") or "")
        context = [] if context_scope == "item" else group_contexts.setdefault(group, [])
        translated = await retranslate_row(
            row, translator=translator, languages=languages, context=context,
            on_segment=on_segment)
        output.append(translated)
        if after_item is not None:
            after_item(translated)
    return output


def score_translation_rows(rows, *, languages) -> metrics.RunScore:
    """Score translation quality only; do not relabel inherited ASR latency."""
    score = metrics.score_run(
        [row for row in rows if row.get("status") == "ok"], languages=languages)
    aggregate = {
        key: value for key, value in score.aggregate.items()
        if key in QUALITY_AXES or key in {"bleu", "bleu_all", "bleu_by_target"}
    }
    unavailable = {
        key: value for key, value in score.unavailable.items()
        if key.startswith("bleu") or key.split(".", 1)[0] in QUALITY_AXES
    }
    return metrics.RunScore(aggregate=aggregate, unavailable=unavailable,
                            misrouted_items=score.misrouted_items)


def _source_run(path: str | Path) -> tuple[Path, Path]:
    candidate = Path(path).resolve()
    items = candidate if candidate.is_file() else candidate / "items.jsonl"
    if not items.is_file():
        raise ValueError(f"source items.jsonl not found: {items}")
    return items.parent, items


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_run_dir(run_dir: Path, *, source_run: Path, config_path: Path) -> None:
    if run_dir.resolve() == source_run.resolve():
        raise ValueError("output run must differ from source run")
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:
        (run_dir / name).unlink(missing_ok=True)
    shutil.copyfile(config_path, run_dir / "config.yml")


def _runtime_metric(rows) -> dict:
    elapsed = [float(segment.get("translation_elapsed_sec") or 0)
               for row in rows for segment in (row.get("segments") or [])]
    if not elapsed:
        return {"calls": 0, "total_sec": 0.0, "average_sec": 0.0}
    return {"calls": len(elapsed), "total_sec": sum(elapsed),
            "average_sec": sum(elapsed) / len(elapsed), "max_sec": max(elapsed)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m bench.retranslate",
        description="저장된 ASR commit을 번역기만 바꿔 다시 평가합니다.",
    )
    parser.add_argument("source_run", help="원본 run 디렉터리 또는 items.jsonl")
    parser.add_argument("config", help="translation-only config.yml")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    # Heavy/optional bench dependencies stay out of module import so the replay
    # transform itself remains CPU-unit-testable.
    args = parse_args(argv)
    from core.errors import ConfigError
    from core.pipelines.translation import translators
    from core.utils import env, logging
    from . import config, report

    env.load()
    logging.configure()
    try:
        cfg = config.load_retranslate(args.config)
        source_run, source_items = _source_run(args.source_run)
        source_rows = list(logging.read_stream(source_items))
        if not source_rows:
            raise ConfigError(f"{source_items} has no readable rows")

        run_dir = RUNS_DIR / cfg.name
        _prepare_run_dir(run_dir, source_run=source_run,
                         config_path=Path(args.config).resolve())
        logging.attach_stream(run_dir / "events.jsonl")
        logging.bind(run=cfg.name)

        translator_cls = translators.get(cfg.translation.name)
        translator = translator_cls(cfg.translation.options, cfg=cfg)
        writer = report.ItemWriter(run_dir / "items.jsonl")
        started = datetime.now(timezone.utc)
        logging.emit("run_open", name=cfg.name, mode="translation_only",
                     source_run=str(source_run), n_items=len(source_rows))

        def before(row):
            logging.bind(item=str(row.get("id") or ""),
                         session=str(row.get("group") or ""))
            logging.start_clock()
            logging.emit("item_open", mode="translation_only")

        def after(row):
            logging.emit("item_close", mode="translation_only",
                         translation_status=row.get("translation_status"),
                         translation_calls=row.get("translation_calls", 0))
            logging.stop_clock()
            writer.write(row)

        def segment_event(segment, index):
            payload = {key: value for key, value in segment.items()
                       if key not in {"t", "type", "item", "session", "audio"}}
            logging.emit("final", replay_segment_index=index, **payload)

        async def execute():
            try:
                await translator.load()
                return await retranslate_rows(
                    source_rows, translator=translator, languages=cfg.languages,
                    context_scope=cfg.context_scope,
                    before_item=before, after_item=after,
                    on_segment=segment_event)
            finally:
                try:
                    await translator.close()
                except Exception as exc:  # noqa: BLE001 - retain the primary failure
                    logging.getLogger("bench").warning("closing translator failed: %s", exc)

        failure = None
        try:
            rows = asyncio.run(execute())
        except KeyboardInterrupt:
            logging.getLogger("bench").warning(
                "interrupted; scoring completed translations")
            rows = writer.read_back()
            failure = "interrupted"
        except Exception as exc:  # noqa: BLE001 - preserve partial replay output
            logging.getLogger("bench").error("translation replay failed: %s", exc)
            rows = writer.read_back()
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            writer.close()

        finished = datetime.now(timezone.utc)
        quality = score_translation_rows(rows, languages=cfg.languages)
        quality.aggregate["translation_runtime"] = _runtime_metric(rows)
        source_summary_path = source_run / "summary.json"
        source_summary = {}
        if source_summary_path.is_file():
            try:
                source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        source_info = {
            "path": str(source_run),
            "items_sha256": _sha256(source_items),
            "name": source_summary.get("name"),
            "commit": (source_summary.get("environment") or {}).get("commit"),
            "dataset": source_summary.get("dataset"),
        }
        report.write_translation_only(
            cfg=cfg, score=quality, rows=rows, source=source_info,
            started=started, finished=finished, run_dir=run_dir,
            component={"backend": cfg.translation.name, **cfg.translation.options},
            failure=failure)
        logging.bind(item="", session="")
        logging.emit("run_close", n_rows=len(rows),
                     status="failed" if failure else "ok")
        return 1 if failure else 0
    except (ConfigError, ValueError) as exc:
        logging.getLogger("bench").error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
