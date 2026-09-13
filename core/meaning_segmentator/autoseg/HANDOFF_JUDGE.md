# judge01 을 다른 환경에서 이어 돌리기

판단형 루프(`loop_judge`)의 첫 실행을 띄우는 절차. **여기까지 끝났고, 남은 것은 라벨 두
분할과 루프 실행이다.**

## 지금 상태

| | 상태 |
|---|---|
| 코드 | `loop_judge.py`, `runtime/hset.py`, `runtime/agents_judge.py` + 단위 테스트 49개 |
| 라벨 정의 | `cohesion × (1 − 소스 contra)`, cohesion 은 zh/ja/de/**es** 4타깃 평균 |
| dev 라벨 | 4타깃 완료 (`run24/pseudoref_dev.json`) |
| test·train 라벨 | **미완** — 스페인어를 넣어 다시 만들어야 한다 |
| run25 라벨 파일 | 아직 3타깃. 위 둘이 끝난 뒤 재생성한다 |
| 번역 캐시 | zh/ja/de 24,378건 + es 8,619건을 저장소에 넣어 뒀다 (`run24/cache/translate_*.json`) |

## 준비물

- 브랜치 `autoseg-judge`
- `.venv` (COMET 계열은 반드시 `.venv/bin/python`)
- GPU 9 GB 이상 — madlad-3b → CometKiwi 순으로 올린다
- CometKiwi 는 HF 게이트 모델: 라이선스 동의 + `hf auth login`
- `.env` 에 `OPENAI_API_KEY` (루프만 쓴다. 라벨 단계는 API 0)

## 1. 남은 라벨 (GPU만, API 0, 15~20분)

```bash
export PYTHONPATH=.
for sp in test train; do
  .venv/bin/python -u -m core.meaning_segmentator.tools.autoseg_en2x.run24.probe.pseudoref \
      --split $sp --targets Chinese Japanese German Spanish
done
.venv/bin/python -u -m core.meaning_segmentator.tools.autoseg_en2x.run24.probe.make_run25
```

`make_run25` 가 run25 의 `oracle_labels_{dev,test,train}.json` 을 4타깃으로 다시 굽고
`config.json` 의 `targets` 도 갱신한다. 확인:

```bash
.venv/bin/python -c "import json;print(json.load(open('core/meaning_segmentator/experiment/artifacts/en2x/en-multi/run25/config.json'))['targets'])"
# ['Chinese', 'Japanese', 'German', 'Spanish']
```

## 2. 루프 (약 $22~26, 1시간 반)

```bash
tmux new-session -d -s judge01 -c <저장소> \
  "bash core/meaning_segmentator/tools/autoseg_en2x/run_judge01.sh"
tail -f core/meaning_segmentator/experiment/artifacts/en2x/logs/judge01.log
```

설정은 스크립트 안에 있다: dev-A 150 / dev-B 265 / k 1~10 / min_gap 1 / 5 이터 /
2 이터마다 체크포인트 / 예산 $35 / v0 는 `--generate-v0` 로 만든다.

## 무엇을 보나

- `[v0] 후보 H_set [...] → N 채택` — 생성된 v0 두 개 중 어느 쪽이 이겼나.
  **프롬프트가 판단형으로 나왔는지 `prompt_v0.txt` 를 눈으로 볼 것.** 표면형 규칙 목록이
  나왔으면 Writer 지시문이 안 먹은 것이다.
- `[iter N] 사례 12 (경계 x / 상호작용 y)` — 상호작용이 많으면 목표를 집합 탐색으로 바꿔야
  한다 (`--target-sets`).
- `[iter N] Δ H_set +x [lo, hi] 짝 n / 문장 m → accept|confirm|reject`
- `[체크포인트] dev-B ...` — 여기서 나빠지면 롤백된다. dev-A 만 오르고 dev-B 가 안 오르면
  dev-A 를 외우는 중이다.

기준선 (test 100, gold COMET-DA 격자평균, min_gap 3 시절 값):
무분절 0.8770 / 탐색 최적 0.7919 / 그리디 오라클 0.7780 / 정책 0.7686.
**min_gap 1 과 4타깃으로 바뀌었으므로 이 값들은 다시 재야 같은 자에서 비교된다.**

## 걸려 넘어졌던 것

- `pseudoref` 는 번역 캐시를 `--mt-cache-from`(기본 run21)에 쓰고 `loop_judge` 는 run24
  캐시를 본다. 두 곳이 갈리면 같은 조각을 두 번 번역한다. 지금은 run24 로 합쳐 뒀다.
- 체인을 로그의 `DONE`/`FAILED` 문자열로 잇지 말 것. 실패한 옛 줄이 남아 있으면 다음
  실행이 그것을 보고 죽는다. 마커 파일로 잇는다.
- 오래 도는 것은 반드시 tmux. 에이전트 백그라운드로 띄우면 세션이 끝날 때 같이 죽는다.
