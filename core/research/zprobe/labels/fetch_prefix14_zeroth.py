"""Zeroth-Korean train utterances with their kosp2e English translations, for the prefix14 ko->en track.

kosp2e only ships the English side; its zeroth part is the public Zeroth-Korean corpus, whose parquet
mirror (kresnik/zeroth_korean) carries audio + Korean transcript under the same utterance ids. Join the
two on id, sample --n, transcode the flac bytes to 16 kHz mono wav, and write a manifest.

usage: .venv/bin/python core/research/zprobe/labels/fetch_prefix14_zeroth.py \
         --parquet-dir models/zprobe/data/prefix14/zeroth_train/data \
         --xlsx <kosp2e split/zeroth_train.xlsx> --n 3600 --out-dir models/zprobe/data/prefix14/kosp2e_zeroth
"""
import argparse
import glob
import io
import json
import os
import random

import librosa
import openpyxl
import pyarrow.parquet as pq
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--parquet-dir", required=True)
ap.add_argument("--xlsx", required=True)
ap.add_argument("--n", type=int, default=3600)
ap.add_argument("--seed", type=int, default=14)
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()

sheet = openpyxl.load_workbook(a.xlsx, read_only=True).active
en = {str(r[1]): str(r[2]).strip() for r in list(sheet.iter_rows(values_only=True))[1:] if r[1] and r[2]}
print(f"kosp2e zeroth translations: {len(en)}", flush=True)

wav_dir = os.path.join(a.out_dir, "wav16k")
os.makedirs(wav_dir, exist_ok=True)
manifest = os.path.join(a.out_dir, "manifest.jsonl")
done = set()
if os.path.exists(manifest):
    done = {json.loads(l)["id"] for l in open(manifest, encoding="utf-8")}

shards = sorted(glob.glob(os.path.join(a.parquet_dir, "train-*.parquet")))
if not shards:
    raise SystemExit(f"no train-*.parquet under {a.parquet_dir}")
ids = []
for s in shards:
    ids += [i for i in pq.read_table(s, columns=["id"]).column("id").to_pylist() if i in en]
print(f"zeroth train rows with a translation: {len(ids)}", flush=True)
rng = random.Random(a.seed)
rng.shuffle(ids)
want = set(ids[: a.n])
todo = want - done
print(f"want {len(want)}, already {len(want) - len(todo)}, todo {len(todo)}", flush=True)

out = open(manifest, "a", encoding="utf-8")
n_ok = 0
for s in shards:
    t = pq.read_table(s)
    for r in t.to_pylist():
        if r["id"] not in todo:
            continue
        wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        path = os.path.join(wav_dir, f"{r['id']}.wav")
        sf.write(path, wav, 16000, subtype="PCM_16")
        out.write(json.dumps({"id": r["id"], "audio": os.path.abspath(path), "lang": "ko",
                              "transcript": r["text"].strip(), "en": en[r["id"]],
                              "duration": round(len(wav) / 16000, 3)}, ensure_ascii=False) + "\n")
        out.flush()
        n_ok += 1
        if n_ok % 500 == 0:
            print(f"{n_ok}/{len(todo)}", flush=True)
print(f"FETCH_ZEROTH_DONE wrote {n_ok}, manifest rows {len(done) + n_ok}", flush=True)
