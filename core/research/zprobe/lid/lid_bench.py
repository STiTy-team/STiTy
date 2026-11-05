"""LID 창 길이 벤치마크 — whisper 체급별 ko/en/es 단창 정확도.

FLEURS test ko_kr/en_us/es_419 각 60클립. silero VAD 로 발화 시작을 찾고
그 지점부터 창을 잘라 whisper 언어 토큰 1스텝 디코드. lid_router 와 같은 방식
(전체 언어 softmax 후 허용집합 {ko,en,es} 안에서 argmax).
"""

import sys, time, glob
import numpy as np
import torch
import librosa

SR = 16000
LANGS = {"ko_kr": "ko", "en_us": "en", "es_419": "es"}
WINDOWS = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
MODELS = sys.argv[1:] or [
    "openai/whisper-small",
    "openai/whisper-medium",
    "openai/whisper-large-v3-turbo",
]
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
print(
    f"clips: {len(clips)}",
    {c: sum(1 for x, _ in clips if x == c) for c in set(x for x, _ in clips)},
    flush=True,
)

from transformers import WhisperForConditionalGeneration, WhisperProcessor

for name in MODELS:
    proc = WhisperProcessor.from_pretrained(name)
    model = (
        WhisperForConditionalGeneration.from_pretrained(name, dtype=torch.float16).to("cuda").eval()
    )
    tok = proc.tokenizer
    allowed = {code: tok.convert_tokens_to_ids(f"<|{code}|>") for code in LANGS.values()}
    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
    mem = torch.cuda.memory_allocated() / 2**20
    for w in WINDOWS:
        hits, times = [], []
        for code, wav in clips:
            seg = wav[: int(w * SR)]
            feats = proc(seg, sampling_rate=SR, return_tensors="pt").input_features.to(
                "cuda", torch.float16
            )
            dec = torch.tensor([[sot]], device="cuda")
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(feats, decoder_input_ids=dec).logits[0, -1].float()
            times.append(time.perf_counter() - t0)
            pick = max(allowed, key=lambda c: logits[allowed[c]].item())
            hits.append(pick == code)
        print(
            f"{name}  w={w:>3}s  acc={np.mean(hits):.3f}  n={len(hits)}  "
            f"infer={np.median(times) * 1000:.0f}ms  vram={mem:.0f}MiB",
            flush=True,
        )
    del model
    torch.cuda.empty_cache()
print("LID_BENCH_DONE", flush=True)
