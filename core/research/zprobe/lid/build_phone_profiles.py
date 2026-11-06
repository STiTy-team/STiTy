"""Fit the phonotactic LID profiles the runtime fused LID reads (core/utils/lid.py).

Phone unigram frequencies per language, counted over FLEURS train utterances with the
wav2vec2-lv-60-espeak-cv-ft phone recognizer. The tokenizer's phonemizer backend is bypassed -
CTC argmax ids are mapped through vocab.json and collapsed - so no espeak install is needed.
Fitted on read speech, judged on conversational commits: the 283-window northstar bench put the
profile verdict at 0.876 overall and 0.912 on short commits, the whisper-small blind spot.

usage: .venv/bin/python core/research/zprobe/lid/build_phone_profiles.py [--langs ko en] [--utts 200]"""
import argparse
import csv
import json
import sys

import soundfile as sf
import torch

sys.path.insert(0, "core/research/zprobe")
import probe_z as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--langs", nargs="+", default=["ko", "en"])
ap.add_argument("--utts", type=int, default=200)
ap.add_argument("--model", default="facebook/wav2vec2-lv-60-espeak-cv-ft")
ap.add_argument("--out", default="models/zprobe/data/lid_phone_profiles.json")
a = ap.parse_args()
SR = 16000

from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC  # noqa: E402
fe = Wav2Vec2FeatureExtractor.from_pretrained(a.model)
pm = Wav2Vec2ForCTC.from_pretrained(a.model, dtype=torch.float16).to("cuda").eval()
vocab = json.load(open(hf_hub_download(a.model, "vocab.json")))
id2tok = {v: k for k, v in vocab.items()}
special = {v for k, v in vocab.items() if k.startswith("<")}


def phones(y):
    inp = fe(y, sampling_rate=SR, return_tensors="pt").input_values.to("cuda", torch.float16)
    with torch.no_grad():
        ids = pm(inp).logits.argmax(-1)[0].tolist()
    out, prev = [], None
    for i in ids:
        if i != prev and i not in special:
            out.append(id2tok[i])
        prev = i
    return out


profiles = {}
for lang in a.langs:
    tsv = P.FLEURS / P.LANGS[lang][0] / "train.tsv"
    counts, used = {}, 0
    for r in csv.reader(open(tsv, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE):
        if used >= a.utts:
            break
        wav = P.FLEURS / P.LANGS[lang][0] / "audio" / "train" / r[1]
        if not wav.exists():
            continue
        y, sr = sf.read(str(wav), dtype="float32")
        if sr != SR or len(y) < SR:
            continue
        for ph in phones(y[: 8 * SR]):
            counts[ph] = counts.get(ph, 0) + 1
        used += 1
    total = sum(counts.values())
    profiles[lang] = {ph: c / total for ph, c in counts.items()}
    print(f"{lang}: {used} utts, {len(counts)} phones, {total} tokens", flush=True)

json.dump({"model": a.model, "profiles": profiles}, open(a.out, "w"))
print(f"{a.out} written", flush=True)
print("BUILD_PHONE_PROFILES_DONE", flush=True)
