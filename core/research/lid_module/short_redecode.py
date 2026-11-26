"""Why do short utterances go missing after a langid slot reset: the shorter audio, or the
other language's header?

For every short northstar reference sentence (<= MAX_SEC) spoken right after a sentence in the
other language - the spot where a langid reset puts it at the start of a fresh slot - decode:
  long        - CONTEXT_SEC of preceding audio + the sentence, header left to the model
                (what the slot heard before any reset)
  short       - the sentence alone, header left to the model
  short-same  - the sentence alone, header forced to its own language
  short-other - the sentence alone, header forced to the other language (a reset that put it
                in the wrong slot)
and score how much of the sentence's text comes out (character recall, difflib).

usage: uv run --project bench python core/research/lid_module/short_redecode.py \
         --out logs/audio-lid/eval/short_redecode.jsonl
"""

import argparse
import difflib
import json
import os
import re
import sys
from statistics import mean

sys.path.insert(0, os.getcwd())

from core.utils import env  # noqa: E402

env.load()

import librosa  # noqa: E402
import torch  # noqa: E402

ITEMS = "bench/runs/northstar_ko+en-ko@v4/common_audiolid-timeline/items.jsonl"
SR = 16000
MAX_SEC = 2.5
CONTEXT_SEC = 8.0
PAD = 0.1
NAME = {"ko": "Korean", "en": "English"}
OTHER = {"ko": "en", "en": "ko"}


def norm(t: str) -> str:
    return re.sub(r"[\s\W_]+", "", re.sub(r"<[^>]+>", "", t).lower())


def recall(target: str, out: str) -> float:
    a, b = norm(target), norm(out)
    if not a:
        return 1.0
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(n for _, _, n in sm.get_matching_blocks()) / len(a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from qwen_asr.inference.qwen3_asr import Qwen3ASRModel

    gpu = torch.cuda.is_available()
    m = Qwen3ASRModel.from_pretrained(
        "Doo12/Qwen3-ASR-1.7B-koen-seg-mix-merged",
        dtype=torch.bfloat16 if gpu else torch.float32,
        device_map="cuda:0" if gpu else "cpu",
        max_new_tokens=128,
    )
    base = m._build_text_prompt(context="", force_language=None)

    def decode(wav, prefix):
        m._build_text_prompt = lambda context, force_language: base + prefix
        return prefix + m._infer_asr_transformers([""], [wav], [None])[0]

    out = open(a.out, "w", encoding="utf-8")
    scores = {}
    for line in open(ITEMS, encoding="utf-8"):
        row = json.loads(line)
        wav, _ = librosa.load(row["audio"], sr=SR)
        sents = row["reference_segmentation"]
        for prev, cur in zip(sents, sents[1:]):
            lang, before = cur["src_lang"], prev["src_lang"]
            if lang not in NAME or before == lang or cur["duration"] > MAX_SEC:
                continue
            s, e = cur["offset"], cur["offset"] + cur["duration"]
            short = wav[int(max(0, s - PAD) * SR) : int((e + PAD) * SR)]
            long = wav[int(max(0, s - CONTEXT_SEC) * SR) : int((e + PAD) * SR)]
            hyp = {
                "long": decode(long, ""),
                "short": decode(short, ""),
                "short-same": decode(short, f"language {NAME[lang]}<asr_text>"),
                "short-other": decode(short, f"language {NAME[OTHER[lang]]}<asr_text>"),
            }
            rec = {
                "item": row["id"], "lang": lang, "start": s, "dur": cur["duration"],
                "ref": cur["transcript"], "hyp": hyp,
                "recall": {k: round(recall(cur["transcript"], v), 3) for k, v in hyp.items()},
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            for k, v in rec["recall"].items():
                scores.setdefault((lang, k), []).append(v)
            print(f"{row['id']} {s:6.1f} {lang} {rec['recall']} {cur['transcript'][:40]}", flush=True)
    print("| sentence lang | n | long | short | short-same | short-other |")
    for lang in NAME:
        n = len(scores.get((lang, "long"), []))
        cells = " | ".join(f"{mean(scores[(lang, k)]):.3f}" for k in ("long", "short", "short-same", "short-other")) if n else "-"
        print(f"| {lang} | {n} | {cells} |")
    print("SHORT_REDECODE_DONE", flush=True)


if __name__ == "__main__":
    main()
