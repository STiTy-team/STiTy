"""Short-clip language accuracy of audio LID scorers, by clip length.

Clips: FLEURS test cut at word boundaries with the aligner word times (runs of whole words whose
span falls in a length bucket, VAD-trimmed like the component does), and YODAS2 held-out crops
(short_data.py --split test). Each scorer is asked over all 7 languages and over en + one other
language (the pair a two-language session would use).

usage: uv run --project bench python core/research/lid_module/short_eval.py \
         --scorers whisper-token whisper-head --out logs/audio-lid/eval/short_eval.md
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from statistics import mean

sys.path.insert(0, os.getcwd())

from core.utils import env  # noqa: E402

env.load()

import librosa  # noqa: E402
import numpy as np  # noqa: E402

from core.components.langid.audio.scorers import build_scorer  # noqa: E402
from core.components.langid.audio.vad import speech_only  # noqa: E402

LANGS = ["ko", "en", "ja", "es", "zh", "fr", "de"]
FLEURS_DIRS = {
    "ko": "ko_kr",
    "en": "en_us",
    "ja": "ja_jp",
    "es": "es_419",
    "zh": "cmn_hans_cn",
    "fr": "fr_fr",
    "de": "de_de",
}
BUCKETS = [(0.3, 0.6), (0.6, 1.0), (1.0, 1.5), (1.5, 4.0)]
SR = 16000


def bucket_of(d):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= d < hi:
            return i
    return None


def fleurs_clips(root, lang, per_bucket, rng):
    base = os.path.join(root, "fleurs", FLEURS_DIRS[lang])
    if not os.path.exists(os.path.join(base, "alignment.jsonl")):
        return []
    audio_of = {}
    for line in open(os.path.join(base, "manifest.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        audio_of[r["id"]] = os.path.join(base, r["audio"])
    want = {b: [] for b in range(len(BUCKETS))}
    rows = [
        json.loads(line) for line in open(os.path.join(base, "alignment.jsonl"), encoding="utf-8")
    ]
    rng.shuffle(rows)
    for r in rows:
        words = [w for w in r.get("words", []) if w["end"] > w["start"]]
        if not words or r["id"] not in audio_of:
            continue
        wav = None
        for b, (lo, hi) in enumerate(BUCKETS):
            if len(want[b]) >= per_bucket:
                continue
            starts = list(range(len(words)))
            rng.shuffle(starts)
            for i in starts:
                j = i
                while j < len(words) and words[j]["end"] - words[i]["start"] < lo:
                    j += 1
                if j < len(words) and words[j]["end"] - words[i]["start"] < hi:
                    if wav is None:
                        wav, _ = librosa.load(audio_of[r["id"]], sr=SR)
                    clip = speech_only(wav[int(words[i]["start"] * SR) : int(words[j]["end"] * SR)])
                    if len(clip) >= int(0.2 * SR):
                        want[b].append(clip)
                    break
        if all(len(v) >= per_bucket for v in want.values()):
            break
    return [(b, c) for b, cs in want.items() for c in cs]


def yodas_clips(data_dir, lang, per_bucket, rng):
    path = os.path.join(data_dir, f"yodas_test_{lang}.npz")
    if not os.path.exists(path):
        return []
    d = np.load(path)
    offs = np.concatenate([[0], np.cumsum(d["lengths"])])
    by = defaultdict(list)
    for i in range(len(d["lengths"])):
        b = bucket_of(d["lengths"][i] / SR)
        if b is not None:
            by[b].append(i)
    out = []
    for b, idx in by.items():
        rng.shuffle(idx)
        for i in idx[:per_bucket]:
            out.append((b, d["audio"][offs[i] : offs[i + 1]].astype(np.float32) / 32767))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorers", nargs="+", required=True, help="name or name@model")
    ap.add_argument("--data-root", default="/home/mobility/STiTy-team/datasets")
    ap.add_argument("--yodas-dir", default="/home/mobility/STiTy-team/STiTy/models/lidhead/data")
    ap.add_argument("--per-bucket", type=int, default=150)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(0)
    sets = {}
    for lang in LANGS:
        sets[("fleurs-test", lang)] = fleurs_clips(a.data_root, lang, a.per_bucket, rng)
        sets[("yodas-test", lang)] = yodas_clips(a.yodas_dir, lang, a.per_bucket, rng)
        print(lang, {k[0]: len(v) for k, v in sets.items() if k[1] == lang}, flush=True)
    results = {}
    for entry in a.scorers:
        name, _, model = entry.partition("@")
        scorer = build_scorer(name, model or None)
        for (src, lang), clips in sets.items():
            for b, clip in clips:
                full = scorer.scores(clip, tuple(LANGS))
                ok7 = max(full, key=full.get) == lang
                pairs = []
                for other in LANGS:
                    if other == lang or "en" not in (lang, other):
                        continue
                    pair = {c: full[c] for c in (lang, other)}
                    pairs.append(max(pair, key=pair.get) == lang)
                results.setdefault((entry, src, b, lang), []).append(
                    (ok7, mean(pairs) if pairs else None)
                )
        print(f"scored {entry}", flush=True)
    lines = [
        "| set | length | scorer | 7-way mean | en+X pair mean | " + " | ".join(LANGS) + " |",
        "|---|---|---|---|---|" + "---|" * len(LANGS),
    ]
    for src in ("fleurs-test", "yodas-test"):
        for b, (lo, hi) in enumerate(BUCKETS):
            for entry in a.scorers:
                per = {lang: results.get((entry, src, b, lang), []) for lang in LANGS}
                acc7 = {lang: mean(x for x, _ in v) for lang, v in per.items() if v}
                accp = [p for v in per.values() for _, p in v if p is not None]
                if not acc7:
                    continue
                cols = " | ".join(f"{acc7[lang]:.3f}" if lang in acc7 else "-" for lang in LANGS)
                lines.append(
                    f"| {src} | {lo}-{hi} s | {entry} | {mean(acc7.values()):.3f} | "
                    f"{mean(accp):.3f} | {cols} |"
                )
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("SHORT_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
