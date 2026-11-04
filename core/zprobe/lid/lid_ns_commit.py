"""2단 LID 를 bench 커밋 단위로 — qwen-seg-koen 캐스케이드 런 post-hoc 재판정.

bench 의 lang_detect_accuracy 와 같은 틀(커밋마다 [직전 결정 시각, 이번 결정 시각] 창,
진실 = 매니페스트 발화 스팬 겹침 최대 언어)로, 커밋의 language 를 2단으로 바꿔 단다:
  1) 커밋 텍스트의 한글/라틴 비율 (특수토큰 제거, 글자 6+, 우세 0.8+)
  2) 비확정이면 whisper-small {ko,en} 제한 argmax, 그 커밋 오디오 창(뒤 4초 상한)
  3) 그래도 없으면 같은 장면의 직전 커밋 결정 유지 (첫 커밋이면 음향 argmax 그대로)
보고: 기준선(모델 자가보고) 대비 정확도·혼동, 어느 단이 판정했는지 분포.
"""
import ast, json, re, unicodedata
from collections import Counter
import numpy as np
import torch
import librosa

SR = 16000
NS = "/home/mobility/stity-datasets/northstar-260928"
RUN = "/home/mobility/STiTy-bench/bench/runs/northstar_ko+en-ko@v2/asr.qwen-seg-koen+mt.qwen3.5-4b"

spans = {}
for line in open(f"{NS}/manifest.jsonl"):
    it = json.loads(line)
    scene = it["audio"].split("/")[-1].removesuffix(".wav")
    spans.setdefault(scene, []).append((it["offset"], it["offset"] + it["duration"], it["src_lang"]))


def truth_of(scene, a, b):
    best, lang = -1.0, "?"
    for s, e, l in spans[scene]:
        d = min(b, e) - max(a, s)
        if d > best:
            best, lang = d, l
    return lang


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


rows = [json.loads(l) for l in open(f"{RUN}/items.jsonl")]
base_pairs, new_pairs, stage_count, errors = [], [], Counter(), []
for row in rows:
    recs = row["records"]
    recs = ast.literal_eval(recs) if isinstance(recs, str) else recs
    scene = row["id"]
    heard_from, prev = 0.0, None
    for c in recs:
        if c.get("type") != "transcribed":
            continue
        heard_to = float(c.get("decision_audio_sec") or heard_from)
        truth = truth_of(scene, heard_from, heard_to)
        base_pairs.append((truth, c.get("language") or "?"))
        v = script_verdict(c.get("original") or "")
        if v is not None:
            stage_count["text"] += 1
        else:
            v = acoustic(scene, heard_from, heard_to)
            if v is not None:
                stage_count["acoustic"] += 1
            else:
                v = prev
                stage_count["carry" if v else "none"] += 1
        if v is None:
            v = "?"
        new_pairs.append((truth, v))
        if truth != v:
            errors.append((scene, heard_from, heard_to, truth, v,
                           ("text" if script_verdict(c.get("original") or "") else "ac/carry"),
                           (c.get("original") or "")[:60]))
        prev = v if v != "?" else prev
        heard_from = heard_to

for name, pairs in (("자가보고(koen)", base_pairs), ("2단 post-hoc", new_pairs)):
    acc = np.mean([t == g for t, g in pairs])
    conf = dict(sorted(Counter(f"{t}->{g}" for t, g in pairs).items()))
    print(f"{name}: acc={acc:.3f} n={len(pairs)} {conf}", flush=True)
print("판정 출처:", dict(stage_count), flush=True)

print(f"2단 오답 {len(errors)}건 (출처별 {Counter(e[5] for e in errors)}):", flush=True)
for scene, a, b, t, g, st, txt in errors:
    print(f"  {scene} [{a:.1f}-{b:.1f}] truth={t} got={g} via={st} text={txt!r}", flush=True)
print("LID_NS_COMMIT_DONE", flush=True)
