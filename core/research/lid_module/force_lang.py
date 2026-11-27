"""Re-transcribe each northstar commit with the ASR language prefix left free, forced to the
langid decision, and forced to the other language, and compare CER against the reference words
inside the commit's audio span.

Offline check (not streaming): one commit = the aligner span of its text (spans cache from
replay2.py), padded a little, decoded alone by the same ASR the bench run used.

usage: uv run --project bench python core/research/lid_module/force_lang.py \
         --run bench/runs/northstar_ko+en-ko@v4/common_audiolid-timeline/items.jsonl \
         --spans logs/audio-lid/eval/spans2.northstar.jsonl --out logs/audio-lid/eval/force_lang.jsonl
"""

import argparse
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.getcwd())

from core.utils import env  # noqa: E402

env.load()

import librosa  # noqa: E402

from bench.metrics.asr import _routes  # noqa: E402

SR = 16000
PAD = 0.15
NAME = {"ko": "Korean", "en": "English"}
OTHER = {"ko": "en", "en": "ko"}


def norm(t: str) -> str:
    t = re.sub(r"<[^>]+>", "", t).lower()
    return re.sub(r"[\s\W_]+", "", t)


def cer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, b in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a != b))
    return d[len(h)], len(r)


def script_lang(t: str) -> str:
    h, l = len(re.findall("[가-힣]", t)), len(re.findall("[A-Za-z]", t))
    return "ko" if h > l else "en" if l > h else "?"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--spans", required=True)
    ap.add_argument("--model", default="Doo12/Qwen3-ASR-1.7B-koen-seg-mix-merged")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
    import torch

    model = Qwen3ASRModel.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda:0")
    spans = {}
    for line in open(a.spans, encoding="utf-8"):
        r = json.loads(line)
        sc, _, _, txt = r["k"].split("|", 3)
        spans[(sc, txt)] = r["v"]

    out = open(a.out, "w", encoding="utf-8")
    for line in open(a.run, encoding="utf-8"):
        row = json.loads(line)
        wav, _ = librosa.load(row["audio"], sr=SR)
        words = row["reference_alignment"]
        trans = [f for f in row["records"] if f["type"] == "translated"]
        heads = [f for f in row["records"] if f["type"] == "transcribed"]
        truths = [t for t, _ in _routes(row)]
        for f, h, truth in zip(trans, heads, truths):
            span = spans.get((row["id"], f["original"]))
            if not span:
                continue
            s, e = span
            ref = " ".join(w["word"] for w in words if s - PAD <= (w["start"] + w["end"]) / 2 <= e + PAD)
            clip = wav[int(max(0, s - PAD) * SR) : int((e + PAD) * SR)]
            lid = f["language"]
            hyp = {}
            for cond, lang in (("auto", None), ("lid", lid), ("other", OTHER.get(lid))):
                if cond != "auto" and lang not in NAME:
                    continue
                res = await model.transcribe(audio=[(clip, SR)], language=NAME[lang] if lang else None)
                hyp[cond] = {"lang": res[0].language, "text": res[0].text}
            rec = {
                "item": row["id"], "audio": f["audio"], "span": span, "truth": truth, "lid": lid,
                "asr_tag": h["language"], "stream": f["original"], "ref": ref, "hyp": hyp,
                "cer": {k: cer(ref, v["text"]) for k, v in hyp.items()} | {"stream": cer(ref, f["original"])},
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
        print(f"done {row['id']}", flush=True)
    print("FORCE_LANG_DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
