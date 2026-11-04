"""AmberNet (NeMo langid_ambernet) — lid_bench.py 와 같은 180클립·창."""
import time, glob
import numpy as np
import torch
import librosa

SR = 16000
LANGS = {"ko_kr": "ko", "en_us": "en", "es_419": "es"}
WINDOWS = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
ROOT = "/home/mobility/stity-datasets/fleurs/data"
N_PER_LANG = 60

from silero_vad import load_silero_vad, get_speech_timestamps
vad = load_silero_vad()

clips = []
for d, code in LANGS.items():
    for p in sorted(glob.glob(f"{ROOT}/{d}/audio/test/*.wav"))[:N_PER_LANG]:
        wav, _ = librosa.load(p, sr=SR, mono=True)
        ts = get_speech_timestamps(torch.from_numpy(wav), vad, sampling_rate=SR)
        if not ts:
            continue
        start = ts[0]["start"]
        if len(wav) - start < int(0.3 * SR):
            continue
        clips.append((code, wav[start:]))
print(f"clips: {len(clips)}", flush=True)

import nemo.collections.asr as nemo_asr

m = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("langid_ambernet").to("cuda").eval()
labels = list(m.cfg.train_ds.labels) if hasattr(m.cfg.train_ds, "labels") else list(m.cfg.labels)
pick_ids = {c: labels.index(c) for c in LANGS.values()}
vram = torch.cuda.memory_allocated() / 2**20

for w in WINDOWS:
    hits, times = [], []
    for code, wav in clips:
        seg = torch.from_numpy(wav[: int(w * SR)]).unsqueeze(0).to("cuda")
        length = torch.tensor([seg.shape[1]], device="cuda")
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, _ = m(input_signal=seg, input_signal_length=length)
        times.append(time.perf_counter() - t0)
        lg = logits[0].float()
        pick = max(pick_ids, key=lambda c: lg[pick_ids[c]].item())
        hits.append(pick == code)
    print(f"ambernet  w={w:>3}s  acc={np.mean(hits):.3f}  n={len(hits)}  "
          f"infer={np.median(times)*1000:.0f}ms  vram={vram:.0f}MiB", flush=True)
print("LID_AMBERNET_DONE", flush=True)
