"""Fit phone-frequency profiles for the audio langid's `phone` scorers, for any FLEURS language.

The wav2vec2-lv-60-espeak-cv-ft phone recognizer turns each FLEURS *train* utterance (the test
split is the evaluation data) into a phone sequence (CTC argmax ids through vocab.json, collapsed,
no espeak install needed); unigram and bigram counts per language are written with add-0.5
smoothing. The train archives are streamed and read only up to `--utts` clips per language.

usage: uv run --project bench python core/research/lid_module/build_phone_profiles.py \
         --langs ko_kr en_us ja_jp es_419 --utts 300 --out models/lidprofiles/phone_profiles.json
"""

import argparse
import csv
import io
import json
import os
import sys
import tarfile
from collections import Counter

sys.path.insert(0, os.getcwd())

from core.utils import env  # noqa: E402

env.load()

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import HfFileSystem, hf_hub_download  # noqa: E402
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC  # noqa: E402

FLEURS_CODES = {
    "ko_kr": "ko",
    "en_us": "en",
    "ja_jp": "ja",
    "es_419": "es",
    "cmn_hans_cn": "zh",
    "fr_fr": "fr",
    "de_de": "de",
}
REPO = "google/fleurs"


def phone_model(name):
    fe = Wav2Vec2FeatureExtractor.from_pretrained(name)
    pm = Wav2Vec2ForCTC.from_pretrained(name, dtype=torch.float16).to("cuda").eval()
    vocab = json.load(open(hf_hub_download(name, "vocab.json")))
    id2tok = {v: k for k, v in vocab.items()}
    special = {v for k, v in vocab.items() if k.startswith("<")}
    return fe, pm, id2tok, special


def phones(audio, fe, pm, id2tok, special):
    inp = fe(audio, sampling_rate=16000, return_tensors="pt").input_values.to("cuda", torch.float16)
    with torch.no_grad():
        ids = pm(inp).logits.argmax(-1)[0].tolist()
    out, prev = [], None
    for i in ids:
        if i != prev and i not in special:
            out.append(id2tok[i])
        prev = i
    return out


def train_clips(fleurs_lang, n):
    tsv = hf_hub_download(REPO, f"data/{fleurs_lang}/train.tsv", repo_type="dataset")
    wanted = {
        r[1]
        for r in csv.reader(open(tsv, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)
        if len(r) > 1
    }
    fs = HfFileSystem()
    with fs.open(f"datasets/{REPO}/data/{fleurs_lang}/audio/train.tar.gz", "rb") as raw:
        with tarfile.open(fileobj=raw, mode="r|gz") as tar:
            got = 0
            for m in tar:
                name = os.path.basename(m.name)
                if not m.isfile() or name not in wanted:
                    continue
                audio, sr = sf.read(io.BytesIO(tar.extractfile(m).read()), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(1)
                if sr != 16000:
                    import librosa

                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                yield audio
                got += 1
                if got >= n:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", required=True, help="FLEURS language dirs, e.g. ko_kr")
    ap.add_argument("--utts", type=int, default=300)
    ap.add_argument("--model", default="facebook/wav2vec2-lv-60-espeak-cv-ft")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fe, pm, id2tok, special = phone_model(a.model)
    inventory = sorted({t for i, t in id2tok.items() if i not in special})
    profiles = {}
    for fl in a.langs:
        uni, bi, n = Counter(), Counter(), 0
        for audio in train_clips(fl, a.utts):
            seq = ["<s>"] + phones(audio, fe, pm, id2tok, special)
            uni.update(seq[1:])
            bi.update(f"{x} {y}" for x, y in zip(seq, seq[1:]))
            n += 1
        total_u = sum(uni.values()) + 0.5 * len(inventory)
        prof_u = {ph: (uni[ph] + 0.5) / total_u for ph in inventory}
        heads = Counter()
        for k, v in bi.items():
            heads[k.split(" ")[0]] += v
        prof_b = {k: v / heads[k.split(" ")[0]] for k, v in bi.items()}
        profiles[FLEURS_CODES.get(fl, fl)] = {
            "uni": prof_u,
            "bi": prof_b,
            "utts": n,
            "phones": sum(uni.values()),
        }
        print(f"{fl}: {n} utts, {sum(uni.values())} phones, {len(bi)} bigrams", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"model": a.model, "profiles": profiles}, open(a.out, "w"), ensure_ascii=False)
    print("PROFILES_DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
