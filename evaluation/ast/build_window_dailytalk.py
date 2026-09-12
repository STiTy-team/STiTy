"""DailyTalk 발화를 이어붙여 70초짜리 in-domain 장문 wav 를 만든다.

ACL 학술 독백과 **같은 길이**를 in-domain 오디오로 재현해, 오디오 버퍼가 자라는 원인이
길이인지 도메인인지 가르는 대조군이다. 결과는 RESULTS.md 「오디오 버퍼가 자라는 원인」
절에 있다 — 같은 69초인데 버퍼가 8초를 안 넘겼다.

클립 사이 300ms 무음을 넣는다(covost2_concat 과 같은 규칙). 원본이 44.1kHz 라 16kHz 로
내린다(학습 창 생성기와 같다).

사용:
    .venv/bin/python evaluation/ast/build_window_dailytalk.py \
        evaluation/ast/samples/dailytalk_window70.wav

출력 wav 와 함께 매니페스트 한 줄에 쓸 정보(duration, n_utts, src_text)를 JSON 으로
찍는다. 매니페스트는 `acl6060_dev_en-ja.jsonl` 의 행 형식을 그대로 쓰고 `tgt_text` 는
영어 원문을 넣는다 — 이 대조군은 커밋 시각만 읽으므로 BLEU 는 보지 않는다."""
import json, re, sys
import librosa, numpy as np, soundfile as sf

BASE = "Qwen3-ASR/finetuning/data/DailyTalk"
TARGET_SEC, GAP_SEC, SR = 70.0, 0.3, 16000

rows = {}
for l in open(f"{BASE}/train.jsonl"):
    d = json.loads(l)
    if "/audio/" in d["audio"] or d["audio"].startswith("audio/"):
        rows[d["audio"].split("/")[-1]] = d["text"].split("<asr_text>", 1)[-1].strip()

def key(fn):
    m = re.match(r"(\d+)_(\d+)_d(\d+)\.wav", fn)
    return (int(m.group(3)), int(m.group(1))) if m else (10**9, 0)

files = sorted(rows, key=key)
start = files.index("0_0_d1000.wav") if "0_0_d1000.wav" in files else 0
gap = np.zeros(int(GAP_SEC * SR), dtype=np.float32)
chunks, texts, total = [], [], 0.0
for fn in files[start:]:
    # 원본이 44.1kHz 라 16kHz 로 내린다(학습 창 생성기와 같은 규칙).
    a, _ = librosa.load(f"{BASE}/audio/{fn}", sr=SR, mono=True)
    if total + len(a) / SR > TARGET_SEC: break
    if chunks: chunks.append(gap); total += GAP_SEC
    chunks.append(a); total += len(a) / SR
    texts.append(re.sub(r"\s*<SEG>\s*", " ", rows[fn]).strip())

out = sys.argv[1]
sf.write(out, np.concatenate(chunks), SR)
print(json.dumps({"wav": out, "duration": round(total, 3), "n_utts": len(texts),
                  "src_text": " ".join(texts)}, ensure_ascii=False))
