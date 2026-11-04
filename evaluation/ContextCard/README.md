# ContextCard — 가공한 맥락(요약·고유명사·상황)을 문맥으로 줄 때 en→ko 번역 품질

1.1(LongContextMT)은 앞 번역을 그대로 N개 넣었다. 여기서는 앞 내용을 **추려서** 넣는다.
추출기가 8문장마다 맥락 카드를 고치고, 번역기는 그 시점에 준비된 최신 카드를 받는다.

- 데이터: 1.1 과 같은 TED 강연 한 편 — Frank Gehry, "My days as a young rebel" (talkid 231, 398문장)
- 번역기·추출기: gpt-6-luna (`reasoning_effort: none`). 번역 문면은 1.1 의 SYSTEM 에 `<notes>` 안내 한 문장만 더한다
- 비동기 흉내: 문장 i 까지 번역한 뒤 (i+1 이 8의 배수이면) 그 8문장의 원문과 **그 조건이 낸 번역**으로 카드를 고치고,
  고친 카드는 한 문장 늦게(i+2 부터) 쓴다
- 비교군: 1.1 의 N=0·8·16 체인을 그대로 가져온다 (`baseline_translations.jsonl`). 같은 번역 문면이다

| 조건 | 번역기에 보여 주는 칸 | 추출 |
|---|---|---|
| K1 | 요약 | V1 — 매번 카드 전체를 다시 쓴다 |
| K2 | 용어·인물 (30개 상한) | V1 |
| K3 | 상황: 장르·화자·청중·말투 | V1 |
| K4 | 전부 | V1 |
| K2b | 고유명사만. 30개가 차면 원문에 가장 오래 안 나온 것부터 뺀다 | V2 — 바뀐 칸만 낸다 (요약·상황은 안 바뀌면 null, 용어는 새 항목만) |
| K3b | 상황에서 말투 칸을 뺀다. 말투는 번역기가 문장마다 정한다 | V2 |
| K5 | 효과가 있던 칸을 합친다: 요약 + 고유명사(K2b 방식) + 말투를 포함한 상황 | V2, 상황에 말투 칸 허용 |

채점: 첫 카드가 쓰이는 10번째 문장부터(389문장), 그리고 1.1 과 같은 257번째부터(142문장). COMET, MetricX-24,
Doc-COMET-w2(앞 2문장의 원문·정답을 문맥으로), chrF++, 말투 일치(문장 끝으로 합니다체·해요체·기타를 가르고 정답과
같은 비율), 합니다체 비율, 입력 토큰. N0 대비 같은 문장끼리 차이와 부트스트랩 95% 구간.
XCOMET-XL 은 이 머신의 HF 토큰으로 받을 수 없어(403) 뺐다.

## 디렉터리

| 경로 | 내용 |
|---|---|
| `configs/card.yml` | 조건, 갱신 주기(8), 늦춤(1), 비교군, 단가, 예산 |
| `scripts/card_prompts.py` | 번역 문면, 추출 문면(V1·V2), 카드 합치기와 보여 주기 |
| `scripts/card_run.py` | 조건마다 체인을 병렬로 돈다. 문장마다·카드 판마다 붙이고, 다시 돌리면 이어 간다 |
| `scripts/card_doc_comet.py` | Doc-COMET-w2 채점 (`doc2_cache.jsonl`) |
| `scripts/card_aggregate.py` | `summary.{json,md}` |
| `scripts/run_card.sh` | tmux 체인: 번역 → COMET·MetricX(1.1 의 `score.py`, 1.1 점수 캐시를 가져온다) → Doc-COMET → 집계 |
| `results/<run_id>/` | `translations.jsonl`, `cards.jsonl`, `baseline_translations.jsonl`, `api_usage.jsonl`(호출마다 비용, `kind` 는 translate·extract), 채점 캐시, 요약 |

## 다시 돌리기

저장소 루트에서. 번역은 transformers 5.x 이상이 아니어도 되지만(API 만 쓴다) 1.1 의 `lcmt` 모듈을 import 한다.
채점은 unbabel-comet 과 metricx24 가 있는 환경(`../LongContextMT/README.md` 의 "환경").

```bash
tmux new-session -d -s card -c "$PWD" "TRANS_PY=<번역 환경>/bin/python METRICS_PY=<채점 환경>/bin/python bash evaluation/ContextCard/scripts/run_card.sh"
```
