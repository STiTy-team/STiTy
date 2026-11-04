"""LID 전용 모델 벤치마크 — lid_bench.py 와 같은 180클립·창에서 whisper 외 모델.

모델마다 try/except — 하나가 깨져도 나머지는 돈다. 판정은 whisper 벤치와 같은
규칙: 모델의 전체 라벨 위 확률에서 {ko,en,es} 에 해당하는 라벨만 보고 argmax.
"""
import time, glob
import numpy as np
import torch
import librosa

SR = 16000
LANGS = {"ko_kr": "ko", "en_us": "en", "es_419": "es"}
WINDOWS = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
ROOT = "/home/mobility/stity-datasets/fleurs/data"
N_PER_LANG = 60

from silero_vad import load_silero_vad, get_speech_timestamps
vad = load_silero_vad()

clips = []
for d, code in LANGS.items():
    for p in sorted(glob.glob(f"{ROOT}/{d}/audio/test/*.wav"))[:N_PER_LANG]:
        wav, _ = librosa.load(p, sr=SR, mono=True)
        ts = get_speech_timestamps(torch.from_numpy(wav), vad, sampling_rate=SR)
        if not ts:
            continue
        start = ts[0]["start"]
        if len(wav) - start < int(0.3 * SR):
            continue
        clips.append((code, wav[start:]))
print(f"clips: {len(clips)}", flush=True)


def run(name, predict, vram_fn=lambda: torch.cuda.memory_allocated() / 2**20):
    for w in WINDOWS:
        hits, times = [], []
        for code, wav in clips:
            seg = wav[: int(w * SR)]
            t0 = time.perf_counter()
            pick = predict(seg)
            times.append(time.perf_counter() - t0)
            hits.append(pick == code)
        print(f"{name}  w={w:>3}s  acc={np.mean(hits):.3f}  n={len(hits)}  "
              f"infer={np.median(times)*1000:.0f}ms  vram={vram_fn():.0f}MiB", flush=True)


def bench_mms256():
    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
    name = "facebook/mms-lid-256"
    fe = AutoFeatureExtractor.from_pretrained(name)
    m = Wav2Vec2ForSequenceClassification.from_pretrained(name, dtype=torch.float16).to("cuda").eval()
    idx = {c: i for i, c in m.config.id2label.items()}
    pick_ids = {"ko": idx["kor"], "en": idx["eng"], "es": idx["spa"]}

    def predict(seg):
        x = fe(seg, sampling_rate=SR, return_tensors="pt").input_values.to("cuda", torch.float16)
        with torch.no_grad():
            logits = m(x).logits[0].float()
        return max(pick_ids, key=lambda c: logits[pick_ids[c]].item())

    run("mms-lid-256", predict)
    del m
    torch.cuda.empty_cache()


def bench_speechbrain(src, tag, names):
    from speechbrain.inference.classifiers import EncoderClassifier
    m = EncoderClassifier.from_hparams(source=src, run_opts={"device": "cuda"})
    labels = [m.hparams.label_encoder.ind2lab[i] for i in range(len(m.hparams.label_encoder.ind2lab))]
    pick_ids = {}
    for c, name in names.items():
        for i, lab in enumerate(labels):
            if lab == c or lab.startswith(c + ":") or lab.startswith(c + " ") or name.lower() in lab.lower():
                pick_ids[c] = i
                break
    assert len(pick_ids) == 3, f"{tag} labels missing: {pick_ids} from {labels[:12]}"

    def predict(seg):
        with torch.no_grad():
            out = m.classify_batch(torch.from_numpy(seg).unsqueeze(0).to("cuda"))
        probs = out[0][0].float()
        return max(pick_ids, key=lambda c: probs[pick_ids[c]].item())

    run(tag, predict)
    del m
    torch.cuda.empty_cache()


def bench_silero():
    model, lang_dict, lang_group_dict, utils = torch.hub.load(
        "snakers4/silero-vad", "silero_lang_detector_95", onnx=False, trust_repo=True)
    get_language_and_group = utils[0] if isinstance(utils, (list, tuple)) else utils
    names = {i: lang_dict[str(i)] for i in range(len(lang_dict))} if isinstance(lang_dict, dict) else dict(enumerate(lang_dict))
    pick_ids = {}
    for i, n in names.items():
        low = str(n).lower()
        for c, key in (("ko", "ko"), ("en", "en"), ("es", "es")):
            if low.startswith(key + ",") or low == key or low.startswith(key + "-"):
                pick_ids.setdefault(c, i)
    assert len(pick_ids) == 3, f"silero labels missing: {pick_ids} from {list(names.values())[:10]}"

    def predict(seg):
        with torch.no_grad():
            lang_logits, _ = model(torch.from_numpy(seg).unsqueeze(0))
        logits = lang_logits[0].float()
        return max(pick_ids, key=lambda c: logits[pick_ids[c]].item())

    run("silero-lang95(cpu)", predict, vram_fn=lambda: 0.0)


for fn in [bench_mms256,
           lambda: bench_speechbrain("speechbrain/lang-id-voxlingua107-ecapa", "voxlingua107-ecapa",
                                     {"ko": "Korean", "en": "English", "es": "Spanish"}),
           lambda: bench_speechbrain("speechbrain/lang-id-commonlanguage_ecapa", "commonlanguage-ecapa",
                                     {"ko": "Korean", "en": "English", "es": "Spanish"}),
           bench_silero]:
    try:
        fn()
    except Exception as e:
        print(f"SKIP {fn}: {type(e).__name__}: {e}", flush=True)

print("LID_BENCH2_DONE", flush=True)
