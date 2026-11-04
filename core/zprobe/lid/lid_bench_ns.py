"""북극성 120발화 LID 벤치 — 어댑터/전사 라우팅 게이트 관점.

항목마다 장면 wav 의 [offset, offset+duration] 을 떼고, silero VAD 로 그 안의
발화 시작을 찾아 창을 자른다. 정답은 manifest 의 src_lang (ko 72 / en 48).
판정은 {ko,en} 제한 argmax — 프로덕션 라우팅과 같은 조건.
모델: voxlingua107-ecapa / whisper-small(현행) / whisper-medium.
"""
import json, time
import numpy as np
import torch
import librosa

SR = 16000
WINDOWS = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
NS = "/home/mobility/stity-datasets/northstar-260928"

from silero_vad import load_silero_vad, get_speech_timestamps
vad = load_silero_vad()

scenes = {}
clips = []  # (src_lang, id, wav-from-speech-start)
for line in open(f"{NS}/manifest.jsonl"):
    it = json.loads(line)
    a = it["audio"]
    if a not in scenes:
        scenes[a], _ = librosa.load(f"{NS}/{a}", sr=SR, mono=True)
    s = int(it["offset"] * SR)
    e = int((it["offset"] + it["duration"]) * SR)
    seg = scenes[a][s:e]
    ts = get_speech_timestamps(torch.from_numpy(seg), vad, sampling_rate=SR)
    if not ts:
        print(f"no-speech {it['id']}", flush=True)
        continue
    start = ts[0]["start"]
    if len(seg) - start < int(0.3 * SR):
        continue
    clips.append((it["src_lang"], it["id"], seg[start:]))
n_ko = sum(1 for c, _, _ in clips if c == "ko")
print(f"clips: {len(clips)} (ko {n_ko} / en {len(clips)-n_ko})", flush=True)


def run(name, predict):
    for w in WINDOWS:
        res = {"ko": [], "en": []}
        times = []
        wrong = []
        for code, uid, wav in clips:
            seg = wav[: int(w * SR)]
            t0 = time.perf_counter()
            pick = predict(seg)
            times.append(time.perf_counter() - t0)
            res[code].append(pick == code)
            if pick != code:
                wrong.append(uid)
        acc = np.mean(res["ko"] + res["en"])
        print(f"{name}  w={w:>3}s  acc={acc:.3f}  ko={np.mean(res['ko']):.3f}  en={np.mean(res['en']):.3f}  "
              f"infer={np.median(times)*1000:.0f}ms  wrong={','.join(wrong[:8])}{'...' if len(wrong) > 8 else ''}",
              flush=True)


def bench_ecapa():
    from speechbrain.inference.classifiers import EncoderClassifier
    m = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa",
                                       run_opts={"device": "cuda"})
    labels = [m.hparams.label_encoder.ind2lab[i] for i in range(len(m.hparams.label_encoder.ind2lab))]
    pick_ids = {c: next(i for i, lab in enumerate(labels) if lab == c or lab.startswith(c + ":"))
                for c in ("ko", "en")}

    def predict(seg):
        with torch.no_grad():
            out = m.classify_batch(torch.from_numpy(seg).unsqueeze(0).to("cuda"))
        probs = out[0][0].float()
        return max(pick_ids, key=lambda c: probs[pick_ids[c]].item())

    run("voxlingua107-ecapa", predict)
    del m
    torch.cuda.empty_cache()


def bench_whisper(name, tag):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    proc = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name, dtype=torch.float16).to("cuda").eval()
    tok = proc.tokenizer
    allowed = {c: tok.convert_tokens_to_ids(f"<|{c}|>") for c in ("ko", "en")}
    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")

    def predict(seg):
        feats = proc(seg, sampling_rate=SR, return_tensors="pt").input_features.to("cuda", torch.float16)
        dec = torch.tensor([[sot]], device="cuda")
        with torch.no_grad():
            logits = model(feats, decoder_input_ids=dec).logits[0, -1].float()
        return max(allowed, key=lambda c: logits[allowed[c]].item())

    run(tag, predict)
    del model
    torch.cuda.empty_cache()


for fn in [bench_ecapa,
           lambda: bench_whisper("openai/whisper-small", "whisper-small"),
           lambda: bench_whisper("openai/whisper-medium", "whisper-medium")]:
    try:
        fn()
    except Exception as e:
        print(f"SKIP: {type(e).__name__}: {e}", flush=True)
print("LID_NS_DONE", flush=True)
