"""VAD alone over a dataset, on the CPU: how it cuts the audio, compared with the script.

    uv run --project bench python -m bench.vad --dataset northstar_en+ko-en \\
        --config asr.qwen-seg-v2-ko+mt.qwen3.5-4b_vad-ramp [--config ...]

Only the pipeline's `vad` (and `enhancement`, if it has one) is built, so no model
is loaded and no GPU is needed. Each config is fed the same audio as `python -m bench`
(augmentation included, trailing silence included) in the same 200 ms chunks.

What it reports, per config:

- segment length p50 / p90 / max -- conversation needs short segments (WORKLOG N1)
- coverage: share of reference-sentence time inside a VAD segment; `missed` counts
  sentences less than half covered (a missed turn start under noise)
- `turn_end_recall`: share of reference sentence ends followed within `--tolerance`
  seconds by a VAD segment end. A segment that runs through a speaker change counts
  as a miss here; that is the window-level language label problem (B7) at its source.
- `end_precision`: share of VAD segment ends that fall within `--tolerance` seconds
  after some reference sentence end. Low precision means cuts inside sentences, which
  the ASR pays for with fragments; read it together with `turn_end_recall`.
- `end_delay_p50`: seconds from a reference sentence end to the VAD end that matched it

Results go to `bench/vad_runs/<dataset>/<config>.json`.
"""
import asyncio
from statistics import median

import numpy as np

from core.errors import ConfigError
from core.pipeline import build_part
from core.utils import audio, cli, env, logging
from core.utils.json import write_json
from core.utils.paths import get_project_root

from . import augment, config
from . import dataset as datasets
from .__main__ import CHUNK_SIZE_MS, TRAILING_SILENCE_MS

log = logging.getLogger(__name__)


def reference_spans(item) -> list[tuple[float, float]]:
    if item.reference_segmentation:
        return sorted((s["offset"], s["offset"] + s["duration"])
                      for s in item.reference_segmentation)
    return [(0.0, item.duration_sec)]


def overlap(a: tuple[float, float], spans: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(a[1], b[1]) - max(a[0], b[0])) for b in spans)


def cut(cfg, item, augmenter, detector, enhancer) -> list[tuple[float, float]]:
    samples = audio.load_window(item.audio, offset=item.offset, duration=item.duration_sec)
    samples, _ = augmenter.apply(samples, key=augment.item_key(cfg.dataset.name, item.id))
    pcm = audio.to_pcm_bytes(samples) + audio.silence_bytes(TRAILING_SILENCE_MS)
    detector.start(languages=[], target_lang="")
    if enhancer is not None:
        enhancer.start()
    step = int(audio.SAMPLING_RATE * CHUNK_SIZE_MS / 1000) * 2
    for start in range(0, len(pcm), step):
        chunk = pcm[start:start + step]
        detector.detect(enhancer.enhance(chunk).vad if enhancer is not None else chunk)
    heard = len(pcm) / 2 / audio.SAMPLING_RATE
    return [(s, heard if e is None else e) for s, e in detector.spans]


def score(items, cut_by_item: dict, tolerance: float) -> dict:
    lengths, covered, total, missed, ends, hits, delays = [], 0.0, 0.0, 0, 0, 0, []
    cuts = placed = 0
    for item in items:
        segments = cut_by_item[item.id]
        lengths += [e - s for s, e in segments]
        seg_ends = [e for _, e in segments]
        ref_ends = [end for _, end in reference_spans(item)]
        cuts += len(seg_ends)
        placed += sum(any(0.0 <= e - end <= tolerance for end in ref_ends) for e in seg_ends)
        for ref in reference_spans(item):
            inside = overlap(ref, segments)
            covered += inside
            total += ref[1] - ref[0]
            missed += inside < 0.5 * (ref[1] - ref[0])
            ends += 1
            after = [e - ref[1] for e in seg_ends if 0.0 <= e - ref[1] <= tolerance]
            if after:
                hits += 1
                delays.append(min(after))
    return {
        "segments": len(lengths),
        "segment_p50_sec": float(median(lengths)) if lengths else None,
        "segment_p90_sec": float(np.percentile(lengths, 90)) if lengths else None,
        "segment_max_sec": float(max(lengths)) if lengths else None,
        "coverage": covered / total if total else None,
        "missed_sentences": missed,
        "reference_sentences": ends,
        "turn_end_recall": hits / ends if ends else None,
        "end_precision": placed / cuts if cuts else None,
        "end_delay_p50_sec": float(median(delays)) if delays else None,
    }


async def sweep(dataset_name: str, configs: list[str], tolerance: float) -> None:
    data_root = config.get_data_root()
    out_dir = get_project_root() / "bench" / "vad_runs" / dataset_name
    for name in configs:
        cfg = config.load(name, dataset_name)
        data = datasets.load(cfg.dataset, data_root)
        augmenter = augment.build(cfg.augment, data_root=data_root)
        detector = await build_part(cfg, "vad")
        enhancer = await build_part(cfg, "enhancement")
        if detector is None:
            raise ConfigError(f"pipeline config {name!r} has no vad")
        cut_by_item = {item.id: cut(cfg, item, augmenter, detector, enhancer)
                       for item in data.items}
        result = {"dataset": dataset_name, "config": name, "tolerance_sec": tolerance,
                  **score(data.items, cut_by_item, tolerance),
                  "segments_by_item": {k: [[round(s, 3), round(e, 3)] for s, e in v]
                                       for k, v in cut_by_item.items()}}
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / f"{name}.json", result)
        shown = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in result.items()
                 if k not in ("segments_by_item", "dataset", "config")}
        print(f"{name}: {shown}")


if __name__ == "__main__":
    args = cli.parse(
        [
            {"name": "dataset", "help": "데이터셋 설정 이름 (configs/datasets/<이름>.yml)"},
            {"name": "config", "action": "append",
             "help": "파이프라인 설정 이름, 여러 번 줄 수 있다"},
            {"name": "tolerance", "type": float, "default": 1.5,
             "help": "문장 끝 뒤 몇 초 안에 VAD 끝이 오면 맞춘 것으로 보나"},
        ],
        prog="python -m bench.vad",
        description="VAD 만 CPU 로 돌려 데이터셋을 어떻게 자르는지 본다",
    )
    env.load()
    logging.configure()
    asyncio.run(sweep(args.dataset, args.config, args.tolerance))
