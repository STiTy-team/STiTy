"""SeamlessStreaming on an exported manifest: speech-to-speech, or ASR only.

s2s: the teammate's S2S path (s2s_synthesize_seamless.py, SeamlessStreamingS2STAgent
with emit_audio) unchanged, fed 200 ms chunks at real-time pace. Each text segment
is logged with the audio position fed so far (what bench calls decision_audio_sec)
as well as the wall-clock time, so YAAL can be scored the bench way afterwards.
The synthesized speech is saved for the ASR round trip.

asr: the same engine with task=asr (tgt_lang = src_lang), for WER/CER. Fed at
real-time pace too, so the streaming policy sees the audio as it would live.

Run with the asr-seamless conda env. Appends per item, so a crash keeps what is done.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# engine_seamless.py lives on the feat/omni-seamless-backends work tree, not in this branch.
BACKENDS = os.environ.get(
    "SEAMLESS_BACKENDS", "/home/skkai/bench-wt/omni-seamless-backends/evaluation/backends"
)
sys.path.insert(0, BACKENDS)
from engine_seamless import SAMPLING_RATE, SeamlessStreamingEngine  # noqa: E402

STEP = int(0.2 * SAMPLING_RATE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--task", choices=["s2s", "asr"], required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    utts = [json.loads(l) for l in Path(args.manifest).read_text(encoding="utf-8").splitlines()]
    if args.limit:
        utts = utts[: args.limit]
    src, tgt = utts[0]["src_lang"], utts[0]["tgt_lang"]

    out = Path(args.out_dir)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    index = out / f"{args.task}.jsonl"
    done = set()
    if index.exists():
        done = {json.loads(l)["utt_id"] for l in index.read_text(encoding="utf-8").splitlines()}

    if args.task == "s2s":
        eng = SeamlessStreamingEngine(task="translate", src_lang=src, tgt_lang=tgt, emit_audio=True)
    else:
        eng = SeamlessStreamingEngine(task="asr", src_lang=src)

    with open(index, "a", encoding="utf-8") as f:
        for i, u in enumerate(utts):
            if u["utt_id"] in done:
                continue
            audio, sr = sf.read(u["wav"], dtype="float32")
            assert sr == SAMPLING_RATE, (u["wav"], sr)
            t0 = time.perf_counter()
            deadline = t0
            segments = []  # (text, audio_fed_sec, wall_sec)
            try:
                eng.start(src)
                for j in range(0, len(audio), STEP):
                    deadline += STEP / SAMPLING_RATE
                    now = time.perf_counter()
                    if now < deadline:
                        time.sleep(deadline - now)
                    text = eng.feed(audio[j : j + STEP])
                    if text:
                        fed = min(j + STEP, len(audio)) / SAMPLING_RATE
                        segments.append((text, fed, time.perf_counter() - t0))
                text = eng.finish()
                if text:
                    segments.append((text, len(audio) / SAMPLING_RATE, time.perf_counter() - t0))
                rec = {
                    "utt_id": u["utt_id"],
                    "ok": True,
                    "text": " ".join(s[0] for s in segments).strip(),
                    "segments": segments,
                    "wall_sec": round(time.perf_counter() - t0, 3),
                }
                if args.task == "s2s":
                    chunks = eng.last_audio()
                    if not chunks:
                        raise RuntimeError("no audio chunks produced")
                    wav = np.concatenate(
                        [np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks]
                    )
                    path = out / "wav" / f"{u['utt_id']}.wav"
                    sf.write(str(path), wav, SAMPLING_RATE)
                    rec["synth_wav"] = str(path)
                    rec["synth_sec"] = round(len(wav) / SAMPLING_RATE, 3)
            except Exception as e:  # noqa: BLE001 - recorded and counted as an error
                rec = {"utt_id": u["utt_id"], "ok": False, "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"[{i + 1}/{len(utts)}] {u['utt_id']} ok={rec['ok']} segs={len(segments)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
