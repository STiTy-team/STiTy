"""Short speech crops for training / checking the `whisper-head` language scorer.

Sources (audio only, streamed from the HF archives and read only as far as needed):
  fleurs  - FLEURS train / dev utterances (read speech)
  yodas   - YODAS2 YouTube audio (espnet/yodas2, CC-BY-3.0): one long wav per video; random 20 s
            stretches are VAD-trimmed and cropped
Each crop keeps only speech samples (silero VAD, as the langid component does) and has a duration
drawn from the buckets below. Output: one npz per (source, split, language) with int16 crops and
their lengths.

usage: uv run --project bench python core/research/lid_module/short_data.py --source fleurs \
         --split train --lang ko_kr --items 1500 --crops 3 --out-dir models/lidhead/data
       ... --source yodas --split train --lang ko --shard 0 --items 75 --crops 60
"""

import argparse
import csv
import io
import os
import random
import sys
import tarfile

sys.path.insert(0, os.getcwd())

from core.utils import env  # noqa: E402

env.load()

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from huggingface_hub import HfFileSystem, hf_hub_download  # noqa: E402

from core.components.langid.audio.vad import speech_only  # noqa: E402

SR = 16000
BUCKETS = [((0.3, 0.6), 0.3), ((0.6, 1.0), 0.3), ((1.0, 1.5), 0.2), ((1.5, 4.0), 0.2)]
FLEURS_LANG = {
    "ko_kr": "ko",
    "en_us": "en",
    "ja_jp": "ja",
    "es_419": "es",
    "cmn_hans_cn": "zh",
    "fr_fr": "fr",
    "de_de": "de",
}
STRETCH_SEC = 20.0


def duration(rng: random.Random) -> float:
    ((lo, hi),) = rng.choices([b for b, _ in BUCKETS], weights=[w for _, w in BUCKETS])
    return rng.uniform(lo, hi)


def to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(1)
    if sr != SR:
        import librosa

        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SR)
    return audio.astype(np.float32)


def crops_from(speech: np.ndarray, n: int, rng: random.Random) -> list[np.ndarray]:
    out = []
    for _ in range(n):
        d = int(duration(rng) * SR)
        if len(speech) < d:
            if len(speech) >= int(0.3 * SR):
                out.append(speech)
            continue
        start = rng.randrange(0, len(speech) - d + 1)
        out.append(speech[start : start + d])
    return out


def fleurs_items(lang: str, split: str, n: int):
    repo = "google/fleurs"
    tsv = hf_hub_download(repo, f"data/{lang}/{split}.tsv", repo_type="dataset")
    wanted = {
        r[1]
        for r in csv.reader(open(tsv, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)
        if len(r) > 1
    }
    fs = HfFileSystem()
    got = 0
    with fs.open(f"datasets/{repo}/data/{lang}/audio/{split}.tar.gz", "rb") as raw:
        with tarfile.open(fileobj=raw, mode="r|gz") as tar:
            for m in tar:
                if not m.isfile() or os.path.basename(m.name) not in wanted:
                    continue
                audio, sr = sf.read(io.BytesIO(tar.extractfile(m).read()), dtype="float32")
                yield to_16k(audio, sr)
                got += 1
                if got >= n:
                    return


def yodas_items(lang: str, shard: int, n: int):
    fs = HfFileSystem()
    path = f"datasets/espnet/yodas2/data/{lang}000/audio/{shard:08d}.tar.gz"
    got = 0
    with fs.open(path, "rb") as raw:
        with tarfile.open(fileobj=raw, mode="r|gz") as tar:
            for m in tar:
                if not m.isfile() or not m.name.endswith(".wav"):
                    continue
                audio, sr = sf.read(io.BytesIO(tar.extractfile(m).read()), dtype="float32")
                yield to_16k(audio, sr)
                got += 1
                if got >= n:
                    return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("fleurs", "yodas"), required=True)
    ap.add_argument("--split", required=True, help="train / dev / test (output name)")
    ap.add_argument("--lang", required=True, help="FLEURS dir (ko_kr) or YODAS2 code (ko)")
    ap.add_argument("--shard", type=int, default=0, help="YODAS2 shard")
    ap.add_argument("--items", type=int, required=True)
    ap.add_argument("--crops", type=int, required=True, help="crops per utterance / video")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(f"{a.source}-{a.split}-{a.lang}-{a.seed}")
    code = FLEURS_LANG.get(a.lang, a.lang)
    crops = []
    if a.source == "fleurs":
        for audio in fleurs_items(a.lang, "dev" if a.split == "dev" else a.split, a.items):
            crops += crops_from(speech_only(audio), a.crops, rng)
    else:
        for i, audio in enumerate(yodas_items(a.lang, a.shard, a.items)):
            stretch = int(STRETCH_SEC * SR)
            for _ in range(max(1, a.crops // 4)):
                if len(audio) <= stretch:
                    piece = audio
                else:
                    s = rng.randrange(0, len(audio) - stretch)
                    piece = audio[s : s + stretch]
                crops += crops_from(speech_only(piece), 4, rng)
            if i % 10 == 0:
                print(f"{a.lang} video {i}: {len(crops)} crops", flush=True)
    lengths = np.array([len(c) for c in crops], dtype=np.int32)
    data = (
        np.concatenate([(np.clip(c, -1, 1) * 32767).astype(np.int16) for c in crops])
        if crops
        else np.zeros(0, np.int16)
    )
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"{a.source}_{a.split}_{code}.npz")
    np.savez(out, audio=data, lengths=lengths, lang=code)
    print(f"DONE {out}: {len(crops)} crops, {lengths.sum() / SR / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
