"""Cache whisper-small encoder frames for the short crops (short_data.py output).

For each data npz writes features/<same name>.pt: `frames` (all crops' frames concatenated, fp16,
[N, 768]), `n_frames` per crop (ceil(samples / 320): one encoder frame is 20 ms), `lang`, `dur`.
Whisper pads every clip to 30 s, so only the frames covering the clip are kept.

usage: uv run --project bench python core/research/lid_module/short_features.py DATA_DIR OUT_DIR
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

MODEL = "openai/whisper-small"
SR = 16000
HOP = 320
BATCH = 32


def encoder(model=MODEL, device="cuda"):
    from transformers import WhisperFeatureExtractor, WhisperModel

    fe = WhisperFeatureExtractor.from_pretrained(model)
    enc = WhisperModel.from_pretrained(model, dtype=torch.float16).encoder.to(device).eval()
    return fe, enc


@torch.inference_mode()
def encode(clips, fe, enc, device="cuda"):
    """List of float32 clips -> list of [frames, 768] fp16 tensors (CPU)."""
    feats = fe(clips, sampling_rate=SR, return_tensors="pt").input_features.to(
        device, torch.float16
    )
    out = enc(feats).last_hidden_state
    return [out[i, : max(1, math.ceil(len(c) / HOP))].cpu() for i, c in enumerate(clips)]


def main(data_dir, out_dir):
    fe, enc = encoder()
    os.makedirs(out_dir, exist_ok=True)
    for npz in sorted(Path(data_dir).glob("*.npz")):
        dst = Path(out_dir) / (npz.stem + ".pt")
        if dst.exists():
            continue
        d = np.load(npz)
        lengths = d["lengths"]
        audio = d["audio"].astype(np.float32) / 32767
        offs = np.concatenate([[0], np.cumsum(lengths)])
        clips = [audio[offs[i] : offs[i + 1]] for i in range(len(lengths))]
        frames = []
        for i in range(0, len(clips), BATCH):
            frames += encode(clips[i : i + BATCH], fe, enc)
        torch.save(
            {
                "frames": torch.cat(frames) if frames else torch.zeros(0, 768, dtype=torch.float16),
                "n_frames": torch.tensor([f.shape[0] for f in frames], dtype=torch.int32),
                "lang": str(d["lang"]),
                "dur": torch.tensor(lengths / SR, dtype=torch.float32),
            },
            dst,
        )
        print(f"{npz.name}: {len(clips)} crops -> {dst.name}", flush=True)
    print("FEATURES_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
