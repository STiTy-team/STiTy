"""2단 LID 시뮬레이션 — 음향(whisper-small) 잠정 판정 + ASR 텍스트 교정, 북극성 120발화.

음향: whisper-small {ko,en} 제한 argmax, 창 0.5 / 1.0초.
텍스트: koen ASR(models/Qwen3-ASR-1.7B-koen-seg-mix-merged)로 발화 앞 t초(1.0/1.5/2.0)를
전사해 한글/라틴 글자 비율로 판정. 글자 6자 이상 & 우세 비율 0.8 이상일 때만 확정,
확정되면 음향 판정을 덮어쓴다 (아니면 음향 유지).
보고: 단독/결합 정확도, 뒤집기 횟수(옳게/그르게), 뒤집힌 발화의 추가 지연.
"""

import json, re, time, unicodedata
import numpy as np
import torch
import librosa

SR = 16000
NS = "/home/mobility/stity-datasets/northstar-260928"
ACOUSTIC_W = [0.5, 1.0]
TEXT_T = [1.0, 1.5, 2.0]
KOEN = "models/Qwen3-ASR-1.7B-koen-seg-mix-merged"

from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()

scenes, clips = {}, []
for line in open(f"{NS}/manifest.jsonl"):
    it = json.loads(line)
    a = it["audio"]
    if a not in scenes:
        scenes[a], _ = librosa.load(f"{NS}/{a}", sr=SR, mono=True)
    seg = scenes[a][int(it["offset"] * SR) : int((it["offset"] + it["duration"]) * SR)]
    ts = get_speech_timestamps(torch.from_numpy(seg), vad, sampling_rate=SR)
    if not ts or len(seg) - ts[0]["start"] < int(0.3 * SR):
        continue
    clips.append((it["src_lang"], it["id"], seg[ts[0]["start"] :]))
print(f"clips: {len(clips)}", flush=True)

from transformers import WhisperForConditionalGeneration, WhisperProcessor

proc = WhisperProcessor.from_pretrained("openai/whisper-small")
wmodel = (
    WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", dtype=torch.float16)
    .to("cuda")
    .eval()
)
tok = proc.tokenizer
allowed = {c: tok.convert_tokens_to_ids(f"<|{c}|>") for c in ("ko", "en")}
sot = tok.convert_tokens_to_ids("<|startoftranscript|>")

acoustic = {}
for w in ACOUSTIC_W:
    picks = []
    for code, uid, wav in clips:
        feats = proc(wav[: int(w * SR)], sampling_rate=SR, return_tensors="pt").input_features.to(
            "cuda", torch.float16
        )
        with torch.no_grad():
            logits = (
                wmodel(feats, decoder_input_ids=torch.tensor([[sot]], device="cuda"))
                .logits[0, -1]
                .float()
            )
        picks.append(max(allowed, key=lambda c: logits[allowed[c]].item()))
    acoustic[w] = picks
    acc = np.mean([p == c for p, (c, _, _) in zip(picks, clips)])
    print(f"acoustic whisper-small w={w}s acc={acc:.3f}", flush=True)
del wmodel
torch.cuda.empty_cache()


def script_verdict(text):
    text = re.sub(r"<[^>]*>", " ", text)
    h = l = 0
    for ch in text:
        if "가" <= ch <= "힣" or "ㄱ" <= ch <= "ㅣ":
            h += 1
        elif ch.isalpha() and unicodedata.name(ch, "").startswith("LATIN"):
            l += 1
    total = h + l
    if total < 6:
        return None
    if h / total >= 0.8:
        return "ko"
    if l / total >= 0.8:
        return "en"
    return None


import asyncio
from qwen_asr import Qwen3ASRModel

asr = Qwen3ASRModel.from_pretrained(
    KOEN, dtype=torch.bfloat16, device_map="cuda:0", max_inference_batch_size=32, max_new_tokens=64
)

text_verdicts = {}
for t in TEXT_T:
    batch = [(wav[: int(t * SR)], SR) for _, _, wav in clips]
    t0 = time.perf_counter()
    results = asyncio.run(asr.transcribe(audio=batch, language=None, return_time_stamps=False))
    dt = (time.perf_counter() - t0) / len(batch)
    verdicts = [script_verdict(r.text) for r in results]
    text_verdicts[t] = verdicts
    dec = sum(v is not None for v in verdicts)
    acc_dec = (
        np.mean([v == c for v, (c, _, _) in zip(verdicts, clips) if v is not None]) if dec else 0
    )
    print(
        f"text t={t}s decisive={dec}/{len(clips)} acc(decisive)={acc_dec:.3f} "
        f"asr_infer={dt * 1000:.0f}ms/utt",
        flush=True,
    )

print(
    "\n| 음향 w | 텍스트 t | 음향 단독 | 결합 | 뒤집음(옳/그름) | 뒤집힌 발화 추가지연 |",
    flush=True,
)
print("|---|---|---|---|---|---|", flush=True)
for w in ACOUSTIC_W:
    base = acoustic[w]
    base_acc = np.mean([p == c for p, (c, _, _) in zip(base, clips)])
    for t in TEXT_T:
        comb, good, bad = [], 0, 0
        for a, v, (c, _, _) in zip(base, text_verdicts[t], clips):
            f = v if v is not None else a
            comb.append(f)
            if v is not None and v != a:
                if v == c:
                    good += 1
                else:
                    bad += 1
        acc = np.mean([p == c for p, (c, _, _) in zip(comb, clips)])
        print(
            f"| {w}s | {t}s | {base_acc:.3f} | {acc:.3f} | {good}/{bad} | +{t - w:.1f}s (해당 {good + bad}건) |",
            flush=True,
        )

for t in TEXT_T:
    miss = [
        (uid, c, a, v)
        for (c, uid, _), a, v in zip(clips, acoustic[1.0], text_verdicts[t])
        if (v if v is not None else a) != c
    ]
    print(f"t={t}s 결합(w=1.0) 오답 {len(miss)}: {[m[0] for m in miss]}", flush=True)
print("LID_TEXT_DONE", flush=True)
