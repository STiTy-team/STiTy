"""Replay commits through the audio langid (core/components/langid/audio) and keep its raw evidence.

Commits come from a bench run (`--run items.jsonl`, real ASR commits, truth = the reference sentence
the commit text matches, bench.metrics.asr._routes) or are simulated from a mix dataset's word
times (`--mix <dataset dir>`): each turn is cut into pieces of 3-10 words, decided 1.0-2.5 s after
the piece's last word, truth = the turn's language. The component is fed audio in 0.5 s chunks and
asked to detect at each decision time, exactly as a pipeline would.

Writes one JSONL row per commit (truth, span, span source, raw segment scores or the timeline steps
known at the decision) so decision settings can be swept offline by sweep.py.

usage: uv run --project bench python core/research/lid_module/replay.py --mode segment \
         --scorer ecapa (--run items.jsonl | --mix DIR) --langs ko en --out out.jsonl
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time

import librosa
import numpy as np

sys.path.insert(0, os.getcwd())

from core.utils import env  # noqa: E402

env.load()

from bench.metrics import asr  # noqa: E402
from bench.metrics.common import translated  # noqa: E402
from core.components.langid.audio.component import AudioLanguageDetector  # noqa: E402
from core.components.langid.audio.scorers import build_scorer  # noqa: E402
from core.components.langid.audio.span import AlignerSpan  # noqa: E402
from core.components.registry import Transcribed  # noqa: E402
from core.utils import audio as audio_mod  # noqa: E402
from core.utils.json import read_jsonl  # noqa: E402

SR = audio_mod.SAMPLING_RATE
CHUNK_SEC = 0.5
NO_SPACE = ("ja", "zh")


def run_commits(path: str, data_root: str):
    """(conversation id, wav path, [(decision, text, truth)]) from a bench run."""
    for row in read_jsonl(path):
        wav = row["audio"]
        if not os.path.isabs(wav):
            wav = os.path.join(data_root, wav)
        commits = [
            (c["decision_audio_sec"], c["original"], truth)
            for (truth, _), c in zip(asr._routes(row), translated(row["records"]))
        ]
        yield row["id"], wav, commits


def mix_commits(mix_dir: str, seed: int = 0):
    """Simulated commits from a mix dataset's word times."""
    rng = random.Random(seed)
    words = {r["id"]: r["words"] for r in read_jsonl(os.path.join(mix_dir, "alignment.jsonl"))}
    convs: dict[str, list] = {}
    for item in read_jsonl(os.path.join(mix_dir, "manifest.jsonl")):
        convs.setdefault(item["group"], []).append(item)
    for conv, items in sorted(convs.items()):
        commits, last = [], 0.0
        for item in items:
            ws = words.get(item["id"]) or []
            i = 0
            while i < len(ws):
                n = rng.randint(3, 10)
                piece = ws[i : i + n]
                i += n
                sep = "" if item["src_lang"] in NO_SPACE else " "
                text = sep.join(w["word"] for w in piece)
                end = item["offset"] + piece[-1]["end"]
                last = max(last, end + rng.uniform(1.0, 2.5))
                commits.append((round(last, 2), text, item["src_lang"]))
        yield conv, os.path.join(mix_dir, items[0]["audio"]), sorted(commits)


class MemoSpan:
    """Aligner spans are the same for every scorer: compute each once."""

    def __init__(self, inner, cache_path):
        self.inner, self.path = inner, cache_path
        self.memo = {}
        if os.path.exists(cache_path):
            self.memo = {r["k"]: r["v"] for r in read_jsonl(cache_path)}
        self.out = open(cache_path, "a", encoding="utf-8")

    def find(self, audio, start, text, candidates):
        k = f"{self.conv}|{start:.2f}|{len(audio)}|{text}"
        if k not in self.memo:
            v = self.inner.find(audio, start, text, candidates)
            self.memo[k] = list(v) if v else None
            self.out.write(json.dumps({"k": k, "v": self.memo[k]}, ensure_ascii=False) + "\n")
            self.out.flush()
        v = self.memo[k]
        return tuple(v) if v else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("segment", "timeline"), required=True)
    ap.add_argument("--scorer", required=True)
    ap.add_argument("--model")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run")
    src.add_argument("--mix")
    ap.add_argument("--data-root", default="/home/mobility/STiTy-team/datasets")
    ap.add_argument("--langs", nargs="+", required=True)
    ap.add_argument("--span-cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    commits = list(run_commits(a.run, a.data_root) if a.run else mix_commits(a.mix))
    # The settings of the first grid (2026-11-20 night), before the component's defaults changed.
    legacy = {"align_from": "decision", "conf": 0.7}
    if a.mode == "timeline":
        legacy.update(centered=False, read="mean", p_stay=0.9, temperature=1.0)
    det = AudioLanguageDetector(
        {"mode": a.mode, "scorer": a.scorer, "model": a.model, **legacy}, cfg=None
    )
    det.scorer = build_scorer(a.scorer, a.model)
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
                "probs": tr["probs"],
            }
            if a.mode == "segment":
                row["raw"] = tr["raw"]
            else:
                row["steps"] = [[t, s] for t, s in det.steps if t <= decision + 1e-6][-80:]
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
