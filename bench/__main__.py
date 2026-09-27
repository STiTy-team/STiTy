import asyncio
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

from core.errors import STiTyError
from core.utils import audio
from core.utils import cli, clock, env, logging, stream
from core.utils.json import JsonlWriter, read_jsonl, write_jsonl

from . import augment, config
from .metrics.common import transcribed, translated
from .metrics import score
from .metrics.translation import translation_sentences
from . import dataset as datasets
from . import report
from core.pipeline import build as build_pipeline


log = logging.getLogger(__name__)


CHUNK_SIZE_MS = 200
TRAILING_SILENCE_MS = 4000


def _row(item, *, status: str, **fields) -> dict:
    return {
        **asdict(item),
        "status": status,
        "audio_sec": 0.0,
        "compute_sec": 0.0,
        "transcription_output": "",
        "records": [],
        **fields,
    }


async def _stream_item(pipeline, augmenter, item, cfg, languages: list[str]) -> dict:
    try:
        samples = audio.load_window(item.audio, offset=item.offset, duration=item.duration_sec)
        samples, augmented = augmenter.apply(samples, key=augment.item_key(cfg.dataset.name, item.id))
    except STiTyError as e:
        stream.start_clock()
        stream.record("item_error", error="audio_load_failed", detail=str(e))
        return _row(item, status="error", error=str(e))

    audio_sec = len(samples) / audio.SAMPLING_RATE
    pcm = audio.to_pcm_bytes(samples) + audio.silence_bytes(TRAILING_SILENCE_MS)
    records, compute_sec, heard_sec = [], 0.0, 0.0

    async def run(step) -> None:
        nonlocal compute_sec
        started = clock.monotonic()
        produced = await step
        compute_sec += clock.elapsed_since(started)
        records.extend(stream.record(made.type, **asdict(made)) for made in produced)

    stream.start_clock()
    stream.record(
        "item_open",
        audio_sec=round(audio_sec, 3),
        src_lang=item.src_lang,
        target_lang=cfg.target,
        augment=augmented,
    )
    pipeline.start(languages=languages, target_lang=cfg.target)
    try:
        async for chunk in audio.stream_realtime(pcm, chunk_ms=CHUNK_SIZE_MS):
            silence = heard_sec >= audio_sec
            heard_sec += len(chunk) / 2 / audio.SAMPLING_RATE
            stream.audio_position(heard_sec)
            await run(pipeline.listen(chunk))
            stream.record("chunk", silence=silence)
        await run(pipeline.finish())
    finally:
        stream.stop_clock()
        stream.audio_position(None)
    stream.record("item_close")

    transcription_output = " ".join(record["original"].strip() for record in transcribed(records))
    return _row(
        item,
        status="ok",
        audio_sec=round(audio_sec, 3),
        compute_sec=round(compute_sec, 4),
        transcription_output=transcription_output.strip(),
        records=records,
        augment=augmented,
    )


def score_item(row: dict, cfg) -> dict:
    if row.get("status") == "ok":
        row.update(score.score_row(row, cfg.target))
    return row


def score_run(rows, cfg) -> tuple[dict, dict]:
    scores, unavailable = score.score_run(rows, cfg.target)
    unavailable["comet"] = "scored separately by python -m bench.metrics.comet"
    return scores, unavailable


def write_comet_inputs(rows, cfg, path: Path) -> None:
    write_jsonl(path, translation_sentences(rows, cfg.target))


async def _run(cfg, dataset, pipeline, augmenter, writer) -> None:
    current_group = None
    await pipeline.load()
    for index, item in enumerate(dataset.items, start=1):
        stream.bind(item=item.id, session=item.group)
        if item.group != current_group:
            current_group = item.group
            stream.record("session_open", group=item.group)
        row = await _stream_item(pipeline, augmenter, item, cfg, dataset.languages)
        row = score_item(row, cfg)
        writer.write(row)
        log.info(
            "[ITEM] %d/%d %s wer=%s segments=%d",
            index,
            len(dataset.items),
            item.id,
            "-" if row.get("wer") is None else f"{row['wer']:.3f}",
            len(translated(row.get("records") or [])),
        )


RUN_FILES = (
    "events.jsonl",
    "items.jsonl",
    "summary.json",
    "comet_inputs.jsonl",
    "comet_scores.jsonl",
)


def reset_run_dir(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:
        (run_dir / name).unlink(missing_ok=True)


def main(args: Namespace) -> None:
    cfg = config.load(args.config, args.dataset)
    data_root = config.get_data_root()
    dataset = datasets.load(cfg.dataset, data_root)
    augmenter = augment.build(cfg.augment, data_root=data_root)
    pipeline = build_pipeline(cfg)

    started = clock.now()
    timer = clock.monotonic()
    run_dir = config.get_runs_dir() / cfg.name
    reset_run_dir(run_dir)

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

    status, failure = "ok", None
    writer = JsonlWriter(run_dir / "items.jsonl")
    try:
        asyncio.run(_run(cfg, dataset, pipeline, augmenter, writer))
    except KeyboardInterrupt:
        log.warning("[INTERRUPTED] scoring what was written so far")
        status = "degraded"
    except STiTyError as e:
        status, failure = "failed", e
    finally:
        writer.close()
        asyncio.run(pipeline.close())
    rows = list(read_jsonl(writer.path))

    stream.bind(item="", session="")
    stream.record("run_close", n_rows=len(rows), status=status)

    write_comet_inputs(rows, cfg, run_dir / "comet_inputs.jsonl")
    report.write_all(
        cfg=cfg,
        dataset=dataset,
        scores=score_run(rows, cfg),
        rows=rows,
        status=status,
        started=started,
        wall_sec=clock.elapsed_since(timer),
        run_dir=run_dir,
        pacing={"chunk_size_ms": CHUNK_SIZE_MS, "trailing_silence_ms": TRAILING_SILENCE_MS},
        failure=None if failure is None else str(failure),
    )
    if failure is not None:
        raise failure


if __name__ == "__main__":
    args = cli.parse(
        [
            {
                "name": "config",
                "help": "파이프라인 설정 이름 (configs/pipelines/<이름>.yml)",
            },
            {
                "name": "dataset",
                "help": "데이터셋 설정 이름 (configs/datasets/<이름>.yml)",
            },
        ],
        prog="python -m bench",
        description="STiTy 벤치마크",
    )
    env.load()
    logging.configure()

    try:
        main(args)
    except STiTyError as e:
        log.error("[FAILED] %s", e)
        raise SystemExit(e.exit_code)
