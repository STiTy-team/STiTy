"""Replay commits through the audio langid with several scorers at once and keep every raw score.

Like replay.py, but the component runs an ensemble of scorers with `keep_raw`, so each stored step
holds per-scorer, per-window log-scores and a silence flag. sweep2.py then rebuilds any scorer
subset, combine rule, window subset, temperature, p_stay / p_gap and read mode offline.

usage: uv run --project bench python core/research/lid_module/replay2.py --mode timeline \
         [--centered] --scorers whisper-token ecapa mms (--run items.jsonl | --mix DIR) \
         --langs ko en --span-cache spans.jsonl --out out.jsonl
"""

import argparse
import asyncio
import json
import os
import sys
import time

import librosa

sys.path.insert(0, os.getcwd())

from core.utils import env  # noqa: E402

env.load()

from core.components.langid.audio.component import AudioLanguageDetector  # noqa: E402
from core.components.langid.audio.scorers import build_scorer  # noqa: E402
from core.components.langid.audio.span import AlignerSpan  # noqa: E402
from core.components.registry import Transcribed  # noqa: E402
from core.research.lid_module.replay import MemoSpan, mix_commits, run_commits  # noqa: E402
from core.utils import audio as audio_mod  # noqa: E402

SR = audio_mod.SAMPLING_RATE
CHUNK_SEC = 0.5
KEEP_STEPS = 80


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("segment", "timeline"), required=True)
    ap.add_argument("--centered", action="store_true")
    ap.add_argument("--scorers", nargs="+", required=True, help="name or name@model")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run")
    src.add_argument("--mix")
    ap.add_argument("--data-root", default="/home/mobility/STiTy-team/datasets")
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--span-cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    commits = list(run_commits(a.run, a.data_root) if a.run else mix_commits(a.mix))
    settings = {
        "mode": a.mode,
        "scorers": tuple(a.scorers),
        "centered": a.centered,
        "align_from": "prev_end",
        "short_text_sec": 0.0,
        "keep_raw": True,
    }
    det = AudioLanguageDetector(settings, cfg=None)
    scorers = {}
    for entry in a.scorers:
        name, _, model = entry.partition("@")
        scorers[entry if model else name] = build_scorer(name, model or None)
    det.scorers = scorers
    det.spanner = MemoSpan(AlignerSpan(), a.span_cache)
    out = open(a.out, "w", encoding="utf-8")
    t0, audio_total = time.time(), 0.0
    for conv, wav_path, items in commits:
        wav, _ = librosa.load(wav_path, sr=SR)
        audio_total += len(wav) / SR
        det.spanner.conv = conv
        det.start(languages=list(a.langs))
        fed, chunk = 0, int(CHUNK_SEC * SR)
        for decision, text, truth in items:
            while fed < min(len(wav), int(decision * SR)):
                det.hear(audio_mod.to_pcm_bytes(wav[fed : fed + chunk]))
                fed += chunk
            commit = Transcribed(
                original=text,
                language="",
                commit_reason="seg",
                decision_audio_sec=decision,
                committed_elapsed_sec=decision,
            )
            detected = asyncio.run(det.detect(commit))
            tr = det.trace[-1]
            row = {
                "conv": conv,
                "decision": decision,
                "text": text,
                "truth": truth,
                "span": tr["span"],
                "src": tr["src"],
                "detected": detected,
            }
            if a.mode == "segment":
                row["raw"] = tr["raw"]
            else:
                steps = [
                    [t, q, r]
                    for (t, _), q, r in zip(det.steps, det.silent, det.raw)
                    if t <= decision + 1e-6
                ]
                steps += [[p["t"], p["silent"], p["raw"]] for p in tr["provisional"]]
                row["steps"] = steps[-KEEP_STEPS:]
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
    took = time.time() - t0
    print(
        json.dumps(
            {
                "out": a.out,
                "convs": len(commits),
                "audio_sec": round(audio_total, 1),
                "compute_sec": round(took, 1),
                "rtf": round(took / max(1, audio_total), 3),
            }
        )
    )
    print("REPLAY_DONE", flush=True)


if __name__ == "__main__":
    main()
