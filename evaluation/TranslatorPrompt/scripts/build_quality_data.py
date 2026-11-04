"""2.3 데이터 → data/quality_eval.jsonl, data/fewshot.json

평가:
- ted: LongContextMT 의 Gehry 강연 398문장 (en→ko)
- wmt24: WMT24 Chat en-ko test 1,982문장 (고객 ko→en, 상담원 en→ko). 사람이 쓴 정답.
  https://github.com/WMT-Chat-task/chat-task-2024-results (test/en-ko.csv), CC-BY-NC-4.0

예시(S4) — 방향마다 3쌍을 직접 썼다. 코퍼스 정답에서 무작위로 뽑았더니 오타('가입시키지 았습니다'),
번호 누락, 맥락 없이는 뜻이 안 통하는 의역이 섞여 예시가 번역 방식을 잘못 가르칠 수 있었다.
실시간 서비스는 대화 상황을 미리 알 수 없으므로 평가 데이터의 도메인(건축 강연, 계정 상담)과 무관한
여러 상황(학술 발표, 일상, 회의, 길 묻기)에서 골랐고, 모든 데이터셋에 같은 예시를 쓴다.
"""
import csv
import io
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
TALK = HERE.parent / "LongContextMT" / "data" / "talk231.jsonl"
WMT_TEST = "https://raw.githubusercontent.com/WMT-Chat-task/chat-task-2024-results/main/test/en-ko.csv"
FEWSHOT = {
    "en-ko": [
        ["Let me walk you through the main findings of our study.",
         "저희 연구의 주요 결과를 차례로 설명드리겠습니다."],
        ["Honestly, I didn't expect the results to be this clear.",
         "솔직히 결과가 이렇게 분명하게 나올 줄은 몰랐어요."],
        ["Can we move the meeting to Thursday afternoon?", "회의를 목요일 오후로 옮길 수 있을까요?"],
    ],
    "ko-en": [
        ["내일 비 온다던데 우산 챙겨.", "I heard it's going to rain tomorrow, so take an umbrella."],
        ["이 부분은 다음 강의에서 자세히 다루겠습니다.", "We'll cover this part in detail in the next lecture."],
        ["혹시 이 근처에 약국이 있나요?", "Excuse me, is there a pharmacy around here?"],
    ],
}


def fetch_csv(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return list(csv.DictReader(io.StringIO(r.read().decode("utf-8"))))


def main():
    rows = []
    for line in TALK.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows.append({"dataset": "ted", "id": f"ted{r['idx']:03d}", "src_lang": "en", "tgt_lang": "ko",
                     "src": r["en"], "ref": r["ko"]})
    test = fetch_csv(WMT_TEST)
    for i, r in enumerate(test):
        if not (r["source"] or "").strip() or not (r["reference"] or "").strip():
            continue
        rows.append({"dataset": "wmt24", "id": f"wmt{i:04d}", "src_lang": r["source_language"],
                     "tgt_lang": r["target_language"], "src": r["source"].strip(),
                     "ref": r["reference"].strip(), "doc_id": r["doc_id"], "sender": r["sender"]})
    out = HERE / "data" / "quality_eval.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    (HERE / "data" / "fewshot.json").write_text(json.dumps(FEWSHOT, ensure_ascii=False, indent=2) + "\n")
    by = {}
    for r in rows:
        k = f"{r['dataset']}:{r['src_lang']}-{r['tgt_lang']}"
        by[k] = by.get(k, 0) + 1
    print(f"wrote {out}: {len(rows)} rows {by}")


if __name__ == "__main__":
    main()
