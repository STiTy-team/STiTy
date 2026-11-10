"""2단 LID 를 프로덕션 조건으로 북극성에 — 장면 연속 오디오 + silero VAD 분할.

lid_text_correct/full 과 달리 발화 경계를 매니페스트(오라클)가 아니라 VAD 가 만든다
(min_silence 800ms, bench silero 설정과 동일). 판정 단위는 VAD 발화.

정책(채택안 그대로):
  음향: whisper-small, 발화 시작 1.0초 창, {ko,en} 제한 argmax = 잠정 판정
  텍스트: koen ASR prefix 를 1.5초부터 0.5초 간격(격자 상한 7초)으로 재검,
          한글/라틴 비율(특수토큰 제거, 글자 6+, 우세 0.8+) 확정 시 1회 뒤집기 후 고정
진실: 매니페스트 발화 스팬과의 겹침 최대 언어. 두 언어 겹침이 각각 1초 이상이면 mixed 로 따로 센다.
보고: 발화 수·정확도(건수/길이가중)·뒤집기 옳/그름·확정 시점·오답 목록·mixed 상세.
"""

import asyncio, json, re, time, unicodedata
import numpy as np
import torch
import librosa

SR = 16000
NS = "/home/mobility/stity-datasets/northstar-260928"
KOEN = "models/Qwen3-ASR-1.7B-koen-seg-mix-merged"
TEXT_GRID = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]

from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()

truth_spans = {}
for line in open(f"{NS}/manifest.jsonl"):
    it = json.loads(line)
    truth_spans.setdefault(it["audio"], []).append(
        (it["offset"], it["offset"] + it["duration"], it["src_lang"], it["id"])
    )

utts = []
for audio, spans in sorted(truth_spans.items()):
    wav, _ = librosa.load(f"{NS}/{audio}", sr=SR, mono=True)
    ts = get_speech_timestamps(
        torch.from_numpy(wav), vad, sampling_rate=SR, min_silence_duration_ms=800
    )
    for seg in ts:
        s, e = seg["start"] / SR, seg["end"] / SR
        if e - s < 0.3:
            continue
        ov = {}
        for a, b, lang, uid in spans:
            d = min(e, b) - max(s, a)
            if d > 0:
                ov[lang] = ov.get(lang, 0.0) + d
        if not ov:
            continue
        truth = max(ov, key=ov.get)
        mixed = len([l for l, d in ov.items() if d >= 1.0]) > 1
        utts.append(
            {
                "scene": audio,
                "s": s,
                "e": e,
                "wav": wav[seg["start"] : seg["end"]],
                "truth": truth,
                "mixed": mixed,
                "ov": ov,
            }
        )
print(f"VAD utterances: {len(utts)} (mixed: {sum(u['mixed'] for u in utts)})", flush=True)

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
for u in utts:
    feats = proc(u["wav"][:SR], sampling_rate=SR, return_tensors="pt").input_features.to(
        "cuda", torch.float16
    )
    with torch.no_grad():
        logits = (
            wmodel(feats, decoder_input_ids=torch.tensor([[sot]], device="cuda"))
            .logits[0, -1]
            .float()
        )
    u["acoustic"] = max(allowed, key=lambda c: logits[allowed[c]].item())
del wmodel
torch.cuda.empty_cache()
aacc = np.mean([u["acoustic"] == u["truth"] for u in utts])
print(f"acoustic w=1.0s acc={aacc:.3f}", flush=True)


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

asr = Qwen3ASRModel.from_pretrained(
    KOEN, dtype=torch.bfloat16, device_map="cuda:0", max_inference_batch_size=32, max_new_tokens=96
)

for u in utts:
    u["text_v"] = {}
for t in TEXT_GRID:
    todo = [u for u in utts if t <= len(u["wav"]) / SR + 0.49]
    if not todo:
        continue
    batch = [(u["wav"][: int(min(t, len(u["wav"]) / SR) * SR)], SR) for u in todo]
    results = asyncio.run(asr.transcribe(audio=batch, language=None, return_time_stamps=False))
    for u, r in zip(todo, results):
        u["text_v"][t] = script_verdict(r.text)
    print(f"text t={t}s done ({len(todo)} utts)", flush=True)

good = bad = 0
settle, errors = [], []
for u in utts:
    cur, flipped, settled = u["acoustic"], False, None
    if cur == u["truth"]:
        settled = 1.0
    for t in TEXT_GRID:
        v = u["text_v"].get(t)
        if v is None:
            continue
        if not flipped and v != cur:
            if v == u["truth"]:
                good += 1
            else:
                bad += 1
            cur, flipped = v, True
        if settled is None and cur == u["truth"]:
            settled = t
    u["final"] = cur
    if settled is not None and cur == u["truth"]:
        settle.append(settled)
    if cur != u["truth"]:
        errors.append(u)

n = len(utts)
acc = np.mean([u["final"] == u["truth"] for u in utts])
dur = np.array([len(u["wav"]) / SR for u in utts])
wacc = float(np.sum(dur * [u["final"] == u["truth"] for u in utts]) / dur.sum())
pure = [u for u in utts if not u["mixed"]]
pacc = np.mean([u["final"] == u["truth"] for u in pure])
print(
    f"\nfinal acc={acc:.3f} ({n} utts)  duration-weighted={wacc:.3f}  "
    f"pure-only={pacc:.3f} ({len(pure)})",
    flush=True,
)
print(f"flips good/bad={good}/{bad}", flush=True)
if settle:
    print(
        f"옳은 확정: {len(settle)}/{n}, 중앙값 {np.median(settle):.1f}s, "
        f"p90 {np.percentile(settle, 90):.1f}s",
        flush=True,
    )
print("오답:", flush=True)
for u in errors:
    print(
        f"  {u['scene']} [{u['s']:.1f}-{u['e']:.1f}] truth={u['truth']} "
        f"음향={u['acoustic']} final={u['final']} mixed={u['mixed']} ov={ {k: round(v, 1) for k, v in u['ov'].items()} }",
        flush=True,
    )
mixed_utts = [u for u in utts if u["mixed"]]
if mixed_utts:
    macc = np.mean([u["final"] == u["truth"] for u in mixed_utts])
    print(f"mixed {len(mixed_utts)}건 acc={macc:.3f} (진실=겹침최대 언어 기준)", flush=True)
print("LID_NS_STREAM_DONE", flush=True)
