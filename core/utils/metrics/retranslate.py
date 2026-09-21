"""Translate a bench run's committed ASR segments again with another translator."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from bench.report import environment, print_summary
from core.pipelines.translation.dialogue import Dialogue
from core.utils import metrics

from .derived import BENCH_RUNS
from .metric_inputs import QUALITY_AXES, gold_inputs


RUN_FILES = ("events.jsonl", "items.jsonl", "summary.json", "config.yml")
USAGE_LOG = "translation_usage.jsonl"

_STALE_ITEM_METRICS = {
    "wer", "cer", "avg_fsl_sec", "n_seg_with_fsl", "laal_ms", "laal_ca_ms",
    "laal_uncapped_ms", "sentence_bleu", "route_errors", "n_segments",
}


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
                          dialogue: Dialogue | None = None,
                          on_segment=None) -> dict:
    output = copy.deepcopy(row)
    for key in _STALE_ITEM_METRICS:
        output.pop(key, None)
    output["source_status"] = row.get("status") or ""
    output["metric_inputs"] = gold_inputs(row.get("metric_inputs"))
    output["translation_errors"] = []

    if row.get("status") not in (None, "ok"):
        output["translation_status"] = "skipped_source_error"
        output["hypothesis_translation"] = ""
        output["segments"] = []
        return output

    source_lang = str(row.get("src_lang") or "").lower()
    target_lang = languages.expected_target(source_lang)
    translator.start(language=source_lang)
    dialogue = dialogue if dialogue is not None else Dialogue()
    dialogue.open(group=str(row.get("group") or row.get("id") or ""))
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
        speaker = str(segment.get("speaker") or "")
        started = time.perf_counter()
        try:
            translation, detected = await translator.translate(
                original, target_lang, detected_source,
                context=dialogue.context(target_lang),
                speaker=dialogue.label(speaker),
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
        dialogue.said(original, translation or "", lang=detected_source,
                      target=target_lang, speaker=speaker)

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
    if context_scope not in ("item", "group"):
        raise ValueError("context_scope must be 'item' or 'group'")
    dialogues: dict[str, Dialogue] = {}
    output = []
    for row in rows:
        if before_item is not None:
            before_item(row)
        group = str(row.get("group") or row.get("id") or "")
        dialogue = Dialogue() if context_scope == "item" else dialogues.setdefault(
            group, Dialogue())
        translated = await retranslate_row(
            row, translator=translator, languages=languages, dialogue=dialogue,
            on_segment=on_segment)
        output.append(translated)
        if after_item is not None:
            after_item(translated)
    return output


def score_translation_rows(rows, *, languages) -> metrics.RunScore:
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


def _prepare_run_dir(run_dir: Path, *, source_run: Path, config_path: Path) -> None:
    if run_dir.resolve() == source_run.resolve():
        raise ValueError("output run must differ from source run")
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:
        (run_dir / name).unlink(missing_ok=True)
    shutil.copyfile(config_path, run_dir / "config.yml")


def _cuda_memory(loaded: dict | None = None) -> dict | None:
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return None
    gib = 1024 ** 3
    if loaded is None:
        torch.cuda.reset_peak_memory_stats()
        return {"loaded_allocated_gib": torch.cuda.memory_allocated() / gib}
    return {**loaded, "inference_peak_reserved_gib": torch.cuda.max_memory_reserved() / gib}


def _runtime_metric(rows) -> dict:
    elapsed = [float(segment.get("translation_elapsed_sec") or 0)
               for row in rows for segment in (row.get("segments") or [])]
    if not elapsed:
        return {"calls": 0, "total_sec": 0.0, "average_sec": 0.0}
    return {"calls": len(elapsed), "total_sec": sum(elapsed),
            "average_sec": sum(elapsed) / len(elapsed), "max_sec": max(elapsed)}


def write_summary(*, cfg, score, rows, source: dict, started: datetime,
                  finished: datetime, run_dir: Path,
                  component: dict, usage: dict | None = None,
                  failure: str | None = None) -> Path:
    errored = [row for row in rows if row.get("translation_status") not in
               ("ok", "skipped_source_error", "skipped_empty_source")]
    skipped_source_error = [row for row in rows
                            if row.get("translation_status") == "skipped_source_error"]
    skipped_empty_source = [row for row in rows
                            if row.get("translation_status") == "skipped_empty_source"]
    empty = [row for row in rows if row.get("translation_calls", 0)
             and not (row.get("hypothesis_translation") or "").strip()]
    failed_ids = list(dict.fromkeys(
        str(row.get("id") or "")
        for row in errored + skipped_source_error + skipped_empty_source + empty))
    wall_sec = (finished - started).total_seconds()
    status = "failed" if failure else ("degraded" if failed_ids else "ok")
    payload = {
        "name": cfg.name,
        "mode": "translation_only",
        "stamp": started.strftime("%Y%m%dT%H%M%S"),
        "status": status,
        "failure": failure,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "metrics": score.aggregate,
        "unavailable": score.unavailable,
        "counts": {
            "items": len(rows),
            "sessions": len({row.get("group") for row in rows if row.get("group")}),
            "segments": sum(len(row.get("segments") or []) for row in rows),
            "translation_errors": sum(len(row.get("translation_errors") or [])
                                      for row in rows),
            "empty_translation": len(empty),
            "source_error_skips": len(skipped_source_error),
            "empty_source_skips": len(skipped_empty_source),
            "wall_sec": round(wall_sec, 2),
        },
        "failed_items": failed_ids,
        "misrouted_items": score.misrouted_items,
        "config": cfg.raw,
        "components": {"translation": component},
        "source_run": source,
        "measurement_scope": {
            "asr": "frozen from source run; WER/FSL/LAAL not rescored",
            "segmentation": "frozen committed segments from source run",
            "translation": "executed in this run",
        },
        "environment": environment(),
    }
    if usage is not None:
        payload["usage"] = usage
    out_path = run_dir / "summary.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    print_summary(payload, out_path)
    return out_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m core.utils.metrics.retranslate",
        description="저장된 ASR commit을 번역기만 바꿔 다시 평가합니다.",
    )
    parser.add_argument("source_run", help="원본 run 디렉터리 또는 items.jsonl")
    parser.add_argument("config", help="translation-only config.yml")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from core.errors import ConfigError
    from core.pipelines.translation import translators
    from core.utils import env, logging
    from bench import report

    from . import derived, translation_config

    env.load()
    logging.configure()
    try:
        cfg = translation_config.load(args.config)
        source_run, source_items = derived.source_items(args.source_run)
        source_rows = list(logging.read_stream(source_items))
        if not source_rows:
            raise ConfigError(f"{source_items} has no readable rows")

        run_dir = BENCH_RUNS / cfg.name
        _prepare_run_dir(run_dir, source_run=source_run,
                         config_path=Path(args.config).resolve())
        logging.attach_stream(run_dir / "events.jsonl")
        logging.bind(run=cfg.name)

        translator_cls = translators.get(cfg.translation.name)
        translator = translator_cls(cfg.translation.options, cfg=cfg)
        translator.usage_log = run_dir / USAGE_LOG
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

        memory = {}

        async def execute():
            try:
                await translator.load()
                memory["loaded"] = _cuda_memory()
                rows = await retranslate_rows(
                    source_rows, translator=translator, languages=cfg.languages,
                    context_scope=cfg.context_scope,
                    before_item=before, after_item=after,
                    on_segment=segment_event)
                if memory["loaded"] is not None:
                    memory["final"] = _cuda_memory(memory["loaded"])
                return rows
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
        if memory.get("final"):
            quality.aggregate["translation_runtime"]["cuda_memory"] = memory["final"]
        source_summary_path = source_run / "summary.json"
        source_summary = {}
        if source_summary_path.is_file():
            try:
                source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        source_info = {
            "path": str(source_run),
            "items_sha256": derived.sha256(source_items),
            "name": source_summary.get("name"),
            "commit": (source_summary.get("environment") or {}).get("commit"),
            "dataset": source_summary.get("dataset"),
        }
        write_summary(
            cfg=cfg, score=quality, rows=rows, source=source_info,
            started=started, finished=finished, run_dir=run_dir,
            component={"backend": cfg.translation.name, **cfg.translation.options},
            usage=translator.usage(), failure=failure)
        logging.bind(item="", session="")
        logging.emit("run_close", n_rows=len(rows),
                     status="failed" if failure else "ok")
        return 1 if failure else 0
    except (ConfigError, ValueError) as exc:
        logging.getLogger("bench").error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
