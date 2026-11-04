"""텍스트 교정 천장 — 발화 전체를 koen ASR 로 전사했을 때의 확정률·정확도.

lid_text_correct.py 의 t 상한(2초)을 없앤 극한값. 결합 = 음향 w=1.0 잠정 +
전체 발화 텍스트가 확정이면 덮어씀. 발화별 "확정이 서는 시점"도 prefix 이분으로 잰다.
"""
import asyncio, json, re, time, unicodedata
import numpy as np
import torch
import librosa

SR = 16000
NS = "/home/mobility/stity-datasets/northstar-260928"
KOEN = "models/Qwen3-ASR-1.7B-koen-seg-mix-merged"

from silero_vad import load_silero_vad, get_speech_timestamps
vad = load_silero_vad()

scenes, clips = {}, []
for line in open(f"{NS}/manifest.jsonl"):
    it = json.loads(line)
    a = it["audio"]
    if a not in scenes:
        scenes[a], _ = librosa.load(f"{NS}/{a}", sr=SR, mono=True)
    seg = scenes[a][int(it["offset"] * SR): int((it["offset"] + it["duration"]) * SR)]
    ts = get_speech_timestamps(torch.from_numpy(seg), vad, sampling_rate=SR)
    if not ts or len(seg) - ts[0]["start"] < int(0.3 * SR):
        continue
    clips.append((it["src_lang"], it["id"], seg[ts[0]["start"]:]))
print(f"clips: {len(clips)}", flush=True)

from transformers import WhisperForConditionalGeneration, WhisperProcessor
proc = WhisperProcessor.from_pretrained("openai/whisper-small")
wmodel = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small", dtype=torch.float16).to("cuda").eval()
tok = proc.tokenizer
allowed = {c: tok.convert_tokens_to_ids(f"<|{c}|>") for c in ("ko", "en")}
sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
acoustic = []
for code, uid, wav in clips:
    feats = proc(wav[: SR], sampling_rate=SR, return_tensors="pt").input_features.to("cuda", torch.float16)
    with torch.no_grad():
        logits = wmodel(feats, decoder_input_ids=torch.tensor([[sot]], device="cuda")).logits[0, -1].float()
    acoustic.append(max(allowed, key=lambda c: logits[allowed[c]].item()))
del wmodel
torch.cuda.empty_cache()
print(f"acoustic w=1.0s acc={np.mean([p == c for p, (c, _, _) in zip(acoustic, clips)]):.3f}", flush=True)


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


from qwen_asr import Qwen3ASRModel
asr = Qwen3ASRModel.from_pretrained(KOEN, dtype=torch.bfloat16, device_map="cuda:0",
                                    max_inference_batch_size=32, max_new_tokens=128)


def verdicts_at(sec_list):
    batch, idx = [], []
    for i, (_, _, wav) in enumerate(clips):
        n = int(sec_list[i] * SR)
        batch.append((wav[: n] if n > 0 else wav, SR))
        idx.append(i)
    results = asyncio.run(asr.transcribe(audio=batch, language=None, return_time_stamps=False))
    return [script_verdict(r.text) for r in results]


full_secs = [len(w) / SR for _, _, w in clips]
tv = verdicts_at(full_secs)
dec = sum(v is not None for v in tv)
acc_dec = np.mean([v == c for v, (c, _, _) in zip(tv, clips) if v is not None])
print(f"full-utterance text: decisive={dec}/{len(clips)} acc(decisive)={acc_dec:.3f}", flush=True)

comb, good, bad = [], 0, 0
for a, v, (c, _, _) in zip(acoustic, tv, clips):
    f = v if v is not None else a
    comb.append(f)
    if v is not None and v != a:
        good, bad = good + (v == c), bad + (v != c)
acc = np.mean([p == c for p, (c, _, _) in zip(comb, clips)])
print(f"combined (acoustic 1.0s + full text): acc={acc:.3f} flips good/bad={good}/{bad}", flush=True)
miss = [(uid, c, a, v) for (c, uid, _), a, v in zip(clips, acoustic, tv)
        if (v if v is not None else a) != c]
print("오답:", [(m[0], f"gt={m[1]}", f"음향={m[2]}", f"텍스트={m[3]}") for m in miss], flush=True)

grid = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]
settle = [None] * len(clips)
for t in grid:
    tvt = verdicts_at([min(t, s) for s in full_secs])
    for i, v in enumerate(tvt):
        if settle[i] is None and v is not None and v == clips[i][0]:
            settle[i] = t
settled = [s for s in settle if s is not None]
print(f"옳은 확정 도달: {len(settled)}/{len(clips)}, 중앙값 {np.median(settled):.1f}s, "
      f"p90 {np.percentile(settled, 90):.1f}s", flush=True)
print("LID_TEXT_FULL_DONE", flush=True)
