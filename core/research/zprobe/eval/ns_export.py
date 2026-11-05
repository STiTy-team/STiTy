"""Cut the northstar r1 recording (real mixed ko/en conversation, 6 scenes, gold transcripts and translations) into
per-utterance wavs and write zprobe-style exports: ns-ko-en (72 Korean utterances) and ns-en-ko (48 English ones).
The scenes are 16 kHz mono already, so rows are plain [offset, offset+duration] slices.

usage: python core/research/zprobe/eval/ns_export.py  (idempotent; --force rewrites)"""

import argparse, json, os

import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="/home/mobility/stity-ns-lid/manifest_r1.jsonl")
ap.add_argument("--audio_dir", default="/home/mobility/stity-ns-lid/data/northstar-260928")
ap.add_argument(
    "--cond",
    default="clean",
    help="clean, or noisy (the restaurant-noise render the LID runs used)",
)
ap.add_argument("--noisy_dir", default="/home/mobility/stity-ns-lid/audio/noisy")
ap.add_argument("--out", default="/home/mobility/zprobe_bench/export")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8")]
scenes = {}
for r in rows:
    if r["audio"] not in scenes:
        p = (
            os.path.join(a.audio_dir, r["audio"])
            if a.cond == "clean"
            else os.path.join(a.noisy_dir, os.path.basename(r["audio"]))
        )
        wav, sr = sf.read(p, dtype="float32")
        assert sr == 16000
        scenes[r["audio"]] = wav

tag = "ns" if a.cond == "clean" else f"ns-{a.cond}"
for pair, src in [("ko-en", "ko"), ("en-ko", "en")]:
    d = os.path.join(a.out, f"{tag}-{pair}")
    man = os.path.join(d, "manifest.jsonl")
    if os.path.exists(man) and not a.force:
        print(f"{man} exists")
        continue
    os.makedirs(os.path.join(d, "wav"), exist_ok=True)
    n = 0
    with open(man, "w", encoding="utf-8") as f:
        for r in rows:
            if r["src_lang"] != src:
                continue
            tgt = pair.split("-")[1]
            wav = scenes[r["audio"]][
                int(r["offset"] * 16000) : int((r["offset"] + r["duration"]) * 16000)
            ]
            p = os.path.join(d, "wav", f"{r['id']}.wav")
            sf.write(p, wav, 16000)
            f.write(
                json.dumps(
                    {
                        "utt_id": r["id"],
                        "wav": p,
                        "src_lang": src,
                        "tgt_lang": tgt,
                        "src_text": r["reference"]["transcript"],
                        "tgt_text": r["reference"]["translations"][tgt],
                        "duration_sec": round(r["duration"], 3),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    print(f"{man}: {n} rows")
