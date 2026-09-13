# 라벨 후보 F/G 를 다른 환경에서 돌리기

정답(오라클) 라벨을 바꿔 보는 실험이다. **LLM 호출이 없다 — API 비용 0, GPU 만 든다.**

## 무엇을 재나

지금 라벨은 경계마다 세 값을 따로 재서 곱한다: `(1 − contra) × (adq_l + adq_r)/2`.
조각을 따로 채점하므로 "이어 붙이면 말이 되나" 는 아무도 안 본다. 최종 지표(gold)는
조각 번역을 이어 붙여 참조와 COMET 으로 비교한다. 라벨을 그 모양으로 맞춘 것이 F/G 다.

| 후보 | src | mt | ref |
|---|---|---|---|
| **F** 참조 COMET (`wmt22-comet-da`) | 원문 전체 | MT(앞조각) ⊕ MT(뒷조각) | MT(원문 전체) — madlad |
| **G** 무참조 QE (`wmt22-cometkiwi-da`) | 원문 전체 | 같은 이어붙임 | 없음 |

참조를 사람 번역이 아니라 로컬 번역기로 만드는 이유는 gold 번역이 없는 데이터셋에도
붙이기 위해서다. 조각과 참조가 같은 엔진이라 둘의 차이는 절단이 낸 손해뿐이다.

## 판정 기준

test 100 gold(3타깃 COMET 평균, T 격자 4/6/8/12):

| 조건 | 격자평균 |
|---|---|
| 무절단 (자르지 않은 문장) | 0.8770 |
| **라벨 B 오라클 (현행 정답)** | **0.7685** |
| minimal_tgt 프롬프트 (현행 최고 정책) | 0.7686 |
| run24 v0 프롬프트 | 0.7465 |

**F 오라클이 0.7685 를 넘으면 라벨을 바꿀 근거다.** 못 넘으면 `metrics.py` 에 적힌 기각
사유("참조를 full 번역으로 두면 어순을 단조화한 좋은 분절이 감점된다")가 실측으로 맞은
것이고, 그 자체가 결론이다.

## 준비물

- 이 저장소, 브랜치 `autoseg-distill`
- `.venv` (COMET 계열은 반드시 `.venv/bin/python` — 기본 python 은 `comet` 이 없다)
- GPU 9 GB 이상. madlad-3b → wmt22-comet-da → cometkiwi 를 순서대로 올린다
- **CometKiwi 는 HF 게이트 모델이다.** huggingface.co 에서 라이선스 동의 후 `hf auth login`.
  안 되어 있으면 G 만 실패한다 (F 는 게이트가 없다)
- `.env` 는 필요 없다 (LLM 호출 없음)

## 실행

```bash
tmux new-session -d -s pseudoref -c <저장소> \
  "bash core/meaning_segmentator/tools/autoseg_en2x/run24/probe/run_pseudoref.sh"
tail -f core/meaning_segmentator/experiment/artifacts/en2x/logs/run24_pseudoref.log
```

체인이 하는 일: dev 라벨 F/G 계산 → 타깃 간 일치·기존 프롬프트와의 overlap 분석 →
test 라벨 → 오라클 절단 emit → madlad 번역 + BLEU → COMET → run24 v0 대비 부트스트랩 표.

각 GPU 단계 앞에 VRAM 폴링이 있다 (9 GB 이상이 30초 간격 2회). **다른 프로세스를 죽이지
않는다** — 이 기계의 GPU 는 다른 작업과 공유한다.

## 걸리는 시간

번역 캐시(`run21/cache/translate_{de,ja,zh}.json`)가 있으면 30~40분. 캐시가 없으면
madlad 로 dev+test 의 prefix/suffix 를 다 번역해야 해서 몇 시간이다. 캐시는 저장소에
같이 넣어 뒀다 (`.gitignore` 예외로 강제 추가).

## 결과 보는 곳

| 파일 | 내용 |
|---|---|
| `.../logs/run24_pseudoref.log` | 전 과정. dev 분석표가 여기 찍힌다 |
| `run24/pseudoref_{dev,test}.json` | 경계별 F/G 원값 (타깃별) |
| `run24_gold_F_pseudoref/compare.md` | **F 오라클 gold 표 — 이게 판정** |
| `run24_gold_G_qeconcat/compare.md` | G 오라클 gold 표 |

dev 분석표에서 같이 볼 것: `타깃겹침`(세 타깃이 같은 자리를 고르는 비율. A 0.412 / B 0.507),
`타깃상관`(A +0.332 / B +0.609), 그리고 기존 프롬프트(v0, minimal_tgt)와의 overlap.
라벨이 좋아졌다면 타깃 간 일치가 오르고 gold 오라클이 오른다. 둘이 어긋나면 gold 를 믿는다.

## 도중에 죽으면

`pseudoref_{dev,test}.json` 이 있으면 그 단계는 건너뛴다. 그대로 다시 띄우면 된다.
번역·COMET 캐시도 남으므로 재실행이 싸다.

---

## 추가: 오라클 확정안 H

`H = G × (1 − 소스 contra)`. G 는 이어 붙인 뒤 채점하므로 앞 조각의 오해를 뒤 조각이
메우면 통과한다 — 사용자는 그때 이미 틀린 것을 읽었다. 소스 contra 가 그 자리를 잡는다.
dev 실측으로 두 신호는 겹치지 않는다 (소스만 모순 84자리, 번역만 모순 174자리).

```bash
tmux new-session -d -s H -c <저장소> \
  "bash core/meaning_segmentator/tools/autoseg_en2x/run24/probe/run_H.sh"
```

`pseudoref_test.json` 이 있으면 라벨 계산을 건너뛰고 emit 부터 한다 (GPU 20분).
결과는 `run24_gold_H_qeXcontra/compare.md`.

| 비교 대상 | 격자평균 |
|---|---|
| 무절단 | 0.8770 |
| G 오라클 | 0.7823 |
| B 오라클 | 0.7685 |
| min_tgt 정책 | 0.7686 |
