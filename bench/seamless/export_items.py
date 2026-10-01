"""Write the exact audio bench streams (loaded and volume-normalized the bench way)
to wav files, with a manifest, so SeamlessStreaming hears the same input.

Run from the STiTy-bench root with the bench env:
    PYTHONPATH=. bench/.venv/bin/python <this> --dataset fleurs_en-ko_vol23 --out DIR
"""

import argparse
import json
from pathlib import Path

import soundfile as sf

from bench import augment, config
from bench import dataset as datasets
from core.utils import audio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = config.load("mock", args.dataset)
    data_root = config.get_data_root()
    dataset = datasets.load(cfg.dataset, data_root)
    augmenter = augment.build(cfg.augment, data_root=data_root)

    out = Path(args.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for item in dataset.items:
            samples = audio.load_window(item.audio, offset=item.offset, duration=item.duration_sec)
            samples, augmented = augmenter.apply(
                samples, key=augment.item_key(cfg.dataset.name, item.id)
            )
            wav = out / "wav" / f"{item.id}.wav"
            sf.write(str(wav), samples, audio.SAMPLING_RATE, subtype="FLOAT")
            f.write(
                json.dumps(
                    {
                        "utt_id": item.id,
                        "wav": str(wav),
                        "src_lang": item.src_lang,
                        "tgt_lang": cfg.target,
                        "src_text": item.reference,
                        "tgt_text": item.reference_translations.get(cfg.target, ""),
                        "duration_sec": len(samples) / audio.SAMPLING_RATE,
                        "augment": augmented,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"{len(dataset.items)} items -> {out}")


if __name__ == "__main__":
    main()
