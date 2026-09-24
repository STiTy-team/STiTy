import argparse
import asyncio
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from core.errors import ConfigError, DataError, STiTyError
from core.utils import audio as audio_mod
from core.utils import env, logging, metrics, stream

from . import config
from . import dataset as datasets
from . import report
from core.components import Final, Partial, Speech, Transcribed
from core.pipeline import build as build_pipeline
from core.pipeline import describe as describe_pipeline

from .config import BenchConfig

log = logging.getLogger(__name__)

RUNS_DIR = Path(__file__).resolve().parent / "runs"

CHUNK_SIZE_MS = 200
TRAILING_SILENCE_MS = 4000
REALTIME = True


def _row(item, *, status: str, **fields) -> dict:
    return {
        "id": item.id,
        "status": status,
        "group": item.group,
        "src_lang": item.src_lang,
        "speaker": item.speaker,
        "duration_sec": item.duration,
        "audio_sec": 0.0,
        "reference": item.transcript,
        "hypothesis": "",
        "hypothesis_translation": "",
        "reference_translations": dict(item.translations),
        "segments": [],
        **fields,
    }


def _written(produced: list) -> list[dict]:
    finals = []
    for made in produced:
        if isinstance(made, Final):
            row = asdict(made)
            stream.record("final", **row)
            finals.append(row)
        elif isinstance(made, Transcribed):
            stream.record("transcribed", **asdict(made))
        elif isinstance(made, Partial):
            stream.record("partial", **asdict(made))
        elif isinstance(made, Speech):
            stream.record("vad_speech_end", at=round(made.ended_at, 3),
                          started_at=round(made.started_at, 3))
    return finals


async def _stream_item(pipeline, item, cfg) -> dict:
    try:
        audio = audio_mod.load_window(
            item.audio, offset=item.offset, duration=item.duration
        )
    except DataError as e:
        stream.start_clock()
        stream.record("item_error", error="audio_load_failed", detail=str(e))
        return _row(item, status="error", error=str(e))

    pcm = audio_mod.to_pcm_bytes(audio)
    bytes_per_chunk = max(
        2, int(audio_mod.SAMPLING_RATE * CHUNK_SIZE_MS / 1000) * 2
    )
    audio_sec = len(pcm) / 2 / audio_mod.SAMPLING_RATE
    target_lang = cfg.languages.expected_target(item.src_lang)

    stream.start_clock()
    stream.record(
        "item_open",
        audio_sec=round(audio_sec, 3),
        src_lang=item.src_lang,
        target_lang=target_lang,
        trailing_silence_ms=TRAILING_SILENCE_MS,
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    pipeline.start(src_lang=item.src_lang, target_lang=target_lang)

    origin = time.perf_counter()
    sent_samples = 0
    segments: list[dict] = []

    async def feed(chunk: bytes, *, silence: bool) -> None:
        nonlocal sent_samples
        sent_samples += len(chunk) // 2
        stream.audio_position(sent_samples / audio_mod.SAMPLING_RATE)
        if REALTIME:
            delay = (
                origin + sent_samples / audio_mod.SAMPLING_RATE - time.perf_counter()
            )
            if delay > 0:
                await asyncio.sleep(delay)
        segments.extend(_written(await pipeline.listen(chunk)))
        stream.record("chunk", silence=silence)

    try:
        for start in range(0, len(pcm), bytes_per_chunk):
            await feed(pcm[start : start + bytes_per_chunk], silence=False)

        silence_left = TRAILING_SILENCE_MS
        while silence_left > 0:
            step = min(CHUNK_SIZE_MS, silence_left)
            silence_left -= step
            await feed(audio_mod.silence_bytes(step), silence=True)

        segments.extend(_written(await pipeline.finish()))
    finally:
        stream.stop_clock()
        stream.audio_position(None)

    hypothesis = " ".join((s.get("original") or "").strip() for s in segments).strip()
    hyp_translation = " ".join(
        (s.get("translation") or "").strip() for s in segments
    ).strip()

    stream.record("item_close", n_finals=len(segments),
                  empty_hypothesis=not hypothesis)

    return _row(item, status="ok", audio_sec=round(audio_sec, 3),
                hypothesis=hypothesis, hypothesis_translation=hyp_translation,
                segments=segments)


def score_item(row: dict, cfg) -> dict:
    if row.get("status") == "ok":
        row.update(metrics.score_item(row, languages=cfg.languages))
    return row


def score_run(rows, cfg) -> metrics.RunScore:
    score = metrics.score_run([r for r in rows if r.get("status") == "ok"],
                              languages=cfg.languages)
    score.unavailable["comet"] = "scored separately by python -m bench.comet"
    return score


async def _run(cfg, dataset, pipeline, writer) -> list[dict]:
    rows: list[dict] = []
    current_group = None
    await pipeline.load()
    for index, item in enumerate(dataset.items, start=1):
        stream.bind(item=item.id, session=item.group)
        if item.group != current_group:
            current_group = item.group
            stream.record("session_open", group=item.group)
        row = await _stream_item(pipeline, item, cfg)
        row = score_item(row, cfg)
        rows.append(row)
        writer.write(row)
        log.info(
            "[ITEM] %d/%d %s wer=%s segments=%d",
            index,
            len(dataset.items),
            item.id,
            "-" if row.get("wer") is None else f"{row['wer']:.3f}",
            len(row.get("segments") or []),
        )
    return rows


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="python -m bench",
        description="STiTy 벤치마크",
    )

    parser.add_argument(
        "--config",
        required=True,
        help="파이프라인 설정 이름 (configs/pipelines/<이름>.yml)",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="데이터셋 설정 이름 (configs/datasets/<이름>.yml)",
    )

    args = parser.parse_args(argv)
    return args


def load_config(args) -> BenchConfig:
    try:
        return config.load(args.config, args.dataset)
    except ConfigError as e:
        log.error("[FAILED] %s", e)
        raise SystemExit(2)


DATA_ROOT_ENV = "STITY_DATA_ROOT"


def data_root() -> Path:
    root = env.path(DATA_ROOT_ENV)
    if root is None:
        raise ConfigError(
            f"{DATA_ROOT_ENV} is not set. Point it at the directory holding the "
            f"converted datasets, e.g. export {DATA_ROOT_ENV}=~/datasets"
        )
    return root


RUN_FILES = ("events.jsonl", "items.jsonl", "summary.json")


def open_run_dir(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:
        (run_dir / name).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args)

    failure = None
    try:
        dataset = datasets.load(cfg.dataset, data_root())
        pipeline = build_pipeline(cfg)

        started = datetime.now(timezone.utc)
        stamp = started.strftime("%Y%m%dT%H%M%S")
        run_dir = RUNS_DIR / cfg.name
        open_run_dir(run_dir)

        stream.attach(run_dir / "events.jsonl")
        stream.bind(run=cfg.name)
        stream.record(
            "run_open",
            name=cfg.name,
            dataset=dataset.name,
            n_items=len(dataset.items),
            n_sessions=dataset.n_sessions,
            started_at=started.isoformat(),
        )

        status = "ok"
        writer = report.ItemWriter(run_dir / "items.jsonl")
        try:
            rows = asyncio.run(_run(cfg, dataset, pipeline, writer))
        except KeyboardInterrupt:
            log.warning("[INTERRUPTED] scoring what was written so far")
            status = "degraded"
            rows = writer.read_back()
        except ConfigError:
            raise
        except STiTyError as e:
            log.error("[FAILED] %s", e)
            status = "failed"
            failure = str(e)
            rows = writer.read_back()
        finally:
            writer.close()
            asyncio.run(pipeline.close())

        stream.bind(item="", session="")
        stream.record("run_close", n_rows=len(rows), status=status)

        report.write_all(
            cfg=cfg,
            dataset=dataset,
            score=score_run(rows, cfg),
            rows=rows,
            stamp=stamp,
            status=status,
            started=started,
            finished=datetime.now(timezone.utc),
            run_dir=run_dir,
            components=describe_pipeline(cfg),
            pacing={"chunk_size_ms": CHUNK_SIZE_MS,
                    "trailing_silence_ms": TRAILING_SILENCE_MS,
                    "realtime": REALTIME},
            failure=failure,
        )
        if failure is not None:
            return 1
    except ConfigError as e:
        log.error("[FAILED] %s", e)
        return 2
    except STiTyError as e:
        log.error("[FAILED] %s", e)
        return 1

    return 0


if __name__ == "__main__":
    env.load()
    logging.configure()
    raise SystemExit(main())
