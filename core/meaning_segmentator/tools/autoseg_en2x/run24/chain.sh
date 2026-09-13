#!/bin/bash
# run24 — 라벨의 contra 를 소스 NLI 로 (`--contra-source source`). run23 대비 바뀐 것:
#   1) 라벨: (1 − 소스NLI contra) × adq.  adq 는 run21 번역 라벨 재사용, contra 만 원문 NLI 로 덮어씀
#      근거: 번역 contra 는 타깃 간 순위상관 0.21 (라벨 성분 중 최악), 소스 NLI 는 3타깃 평균과 +0.42,
#      라벨 타깃 간 일치 0.33→0.61, gold(test 100 COMET) 손해 없음 (run23/gold_test.md)
#   2) 프롬프트 문구: [Output Rules] 와 [Core Principles] 의 (c) 절이 "번역의 모순" 에서
#      "원문 전체 vs 원문 앞부분" 으로. 그래서 v0 가 run23 과 글자가 달라 캐시 적중 없음 — iter 0 부터 든다.
# 나머지(k=3, 분할, 격자, 관문, 예산 $30)는 run23 과 같다.
# overlap 수치는 run23 과 직접 비교 불가 (정답이 다르다). 비교는 gold 로.
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=.
if [ -x .venv-autoseg/bin/python ]; then PY=.venv-autoseg/bin/python; else PY=.venv/bin/python; fi
S="core/meaning_segmentator/tools/autoseg_en2x/run24"
LOG="core/meaning_segmentator/experiment/artifacts/en2x/logs/run24.log"
mkdir -p "$(dirname "$LOG")"
echo "== $(date '+%F %T') run24 start (PY=$PY)" >> "$LOG"
$PY -u -m core.meaning_segmentator.autoseg.loop_distill \
    --dataset fleurs-en-multi-x \
    --pair-id en2x/en-multi --run-id run24 \
    --split-from en2x/en-multi/run16 \
    --labels-from en2x/en-multi/run21 --contra-source source \
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
    --run-id en2x/en-multi/run24 --budget 30 >> "$LOG" 2>&1
touch "$S/chain.done"
echo "== $(date '+%F %T') DONE" >> "$LOG"
