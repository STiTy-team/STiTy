#!/bin/bash
# run23 — 문장당 k=3 회 채점, 순위 평균 (`--k-samples 3`). run22 와 이것 하나만 다르다.
#
# 근거 (run22 실측, 2026-09-13):
#   같은 v0 를 dev 에서 두 번 재면 남긴 경계의 28% 가 옮겨가고 쌍체 Δ +0.023±0.017 —
#   무변경이 채택 문턱(1 se)을 넘는다. 개정의 churn(30~37%)과 거의 같은 크기라 지금까지의
#   Δ 는 규칙 효과가 아니라 샘플링 잡음이었다. 관문 통과 규칙은 dev 3,399 경계 중 3 자리에만
#   걸린다(완벽 적용 상한 +0.006). Claude API 도 결정론이 안 됐다 (opus-5 temperature 거부,
#   sonnet-4-6/haiku-4-5 temperature 0 도 5회 중 3~5가지 출력). 그래서 평균이다.
#
# v0·분할·라벨은 run22(=run21=run16 분할) 그대로. 캐시:
#   segment.json    = run22 캐시 (v0 dev/train/holdout 1회분, 샘플 0)
#   segment_s1.json = run22 retest_v0a 캐시 (v0 dev 재채점, 샘플 1)
#   → iter 0 의 dev 는 샘플 2 만 새로 든다.
#
# 비용: 분절이 90% 라 이터당 약 3배 (~$8). 예산 $30 하드스톱. 무개선 3회면 조기 종료.
#
# 볼 것:
#   - dev 쌍체 se 가 0.017 → ~0.010 으로 주나 (샘플링 잡음이 주였다면)
#   - 이터별 후보 Δ 가 규칙 도달률과 맞는 크기(≤0.01)로 내려앉나
#   - 채택이 나오면 test 에서 재현되나 (run18/21 은 안 됐다)
set -u

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1

set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=.

if [ -x .venv-autoseg/bin/python ]; then PY=.venv-autoseg/bin/python; else PY=.venv/bin/python; fi

S="core/meaning_segmentator/tools/autoseg_en2x/run23"
LOG="core/meaning_segmentator/experiment/artifacts/en2x/logs/run23.log"
mkdir -p "$(dirname "$LOG")"

echo "== $(date '+%F %T') run23 start (PY=$PY)" >> "$LOG"

$PY -u -m core.meaning_segmentator.autoseg.loop_distill \
    --dataset fleurs-en-multi-x \
    --pair-id en2x/en-multi --run-id run23 \
    --split-from en2x/en-multi/run16 \
    --labels-from en2x/en-multi/run21 \
    --train 100 --train-pool 200 \
    --n-cases 24 \
    --rule-min-matches 5 --rule-min-t 2.0 --rule-min-support 2 \
    --revision-rounds 1 --pool-size 0 \
    --k-samples 3 \
    --provider openai --model gpt-5-mini \
    --iterations 5 --budget 30 >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') loop_distill exit=$rc" >> "$LOG"
[ $rc -eq 0 ] && touch "$S/loop.done" || touch "$S/loop.failed"

echo "== $(date '+%F %T') cost_report" >> "$LOG"
$PY -u -m core.meaning_segmentator.autoseg.infra.cost_report \
    --run-id en2x/en-multi/run23 --budget 30 >> "$LOG" 2>&1

touch "$S/chain.done"
echo "== $(date '+%F %T') DONE" >> "$LOG"
