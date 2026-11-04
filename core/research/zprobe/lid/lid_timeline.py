"""LID 타임라인 생성 — koen 캐스케이드 런 커밋마다 인과적 판정 셋을 남긴다.

커밋 시각 t 에 쓸 수 있는 증거만으로:
  ac  = 음향 단독 (whisper-small {ko,en} 제한 argmax, 커밋 창 뒤 4초 상한; 창이 짧으면 직전 유지)
  two = 2단 (커밋 텍스트 한글/라틴 비율 확정 우선, 비확정이면 ac, 그래도 없으면 직전 유지)
산출: lid_timeline.json  {scene: [[t, ac, two], ...]}  (t = decision_audio_sec 오름차순)
"""
import json, re, unicodedata
import torch
import librosa

SR = 16000
NS = "/home/mobility/stity-datasets/northstar-260928"
RUN = "/home/mobility/STiTy-bench/bench/runs/northstar_ko+en-ko@v2/asr.qwen-seg-koen+mt.qwen3.5-4b"
OUT = "/home/mobility/STiTy/core/research/zprobe/lid/lid_timeline.json"


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


from transformers import WhisperForConditionalGeneration, WhisperProcessor
proc = WhisperProcessor.from_pretrained("openai/whisper-small")
wmodel = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small", dtype=torch.float16).to("cuda").eval()
tok = proc.tokenizer
allowed = {c: tok.convert_tokens_to_ids(f"<|{c}|>") for c in ("ko", "en")}
sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
wavs = {}


def acoustic(scene, a, b):
    if scene not in wavs:
        wavs[scene], _ = librosa.load(f"{NS}/audio/{scene}.wav", sr=SR, mono=True)
    a = max(a, b - 4.0)
    seg = wavs[scene][int(a * SR): int(b * SR)]
    if len(seg) < int(0.3 * SR):
        return None
    feats = proc(seg, sampling_rate=SR, return_tensors="pt").input_features.to("cuda", torch.float16)
    with torch.no_grad():
        logits = wmodel(feats, decoder_input_ids=torch.tensor([[sot]], device="cuda")).logits[0, -1].float()
    return max(allowed, key=lambda c: logits[allowed[c]].item())


timeline = {}
for line in open(f"{RUN}/items.jsonl"):
    row = json.loads(line)
    scene = row["id"]
    heard_from, prev_ac, prev_two = 0.0, None, None
    out = []
    for c in row["records"]:
        if c.get("type") != "transcribed":
            continue
        t = float(c.get("decision_audio_sec") or heard_from)
        ac = acoustic(scene, heard_from, t) or prev_ac
        tv = script_verdict(c.get("original") or "")
        two = tv or acoustic(scene, heard_from, t) or prev_two
        out.append([t, ac, two])
        prev_ac, prev_two = ac or prev_ac, two or prev_two
        heard_from = t
    timeline[scene] = out
    print(f"{scene}: {len(out)} commits", flush=True)

json.dump(timeline, open(OUT, "w"), ensure_ascii=False)
print("LID_TIMELINE_DONE", flush=True)
