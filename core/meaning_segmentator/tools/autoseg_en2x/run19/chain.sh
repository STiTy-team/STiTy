#!/bin/bash
# run19 — 사례 계층화 + 규칙 일반화 관문의 효과. run18 대비 셋만 바뀐다.
#
#   1) 사례 계층화 + --n-cases 24   실패 종류(contra / frag_left / frag_right)별로 8개씩.
#      run18 은 손실 상위 8개라 contra 6 / frag_right 2 / frag_left 0 이었다.
#   2) 규칙 일반화 관문              제안 규칙을 train 전 경계(약 1400)에 대고 재서,
#      걸린 자리의 라벨이 주장한 방향으로 유의한 것만 프롬프트에 넣는다. LLM 호출 0.
#   3) 풀 중복 제거                 채택본이 풀 한 칸을 먹던 것 (run18 에서 D 가 절반만 걸림)
#
# **v0·분할·라벨·캐시는 run18 과 동일하다.** iter 0 은 캐시 적중으로 비용 0 이고 수치가
# run18 과 글자 그대로 같아야 한다 — 다르면 통제가 깨진 것이다.
#
# 이번에 볼 것:
#   - `규칙 관문 N/24 통과` 가 몇으로 찍히나. 0 이면 관문이 너무 빡빡하다 (--rule-min-t 1.5)
#   - 채택 개정의 규칙이 여전히 4~9 경계짜리인가. run18 은 그랬고 test 에서 2.6se 로 졌다
#   - 이터별 홀드아웃 최댓값이 우상향하나 (run18: 0.4277 / 0.3842 / 0.4086 / 0.4379)
set -u

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1

set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=.

if [ -x .venv-autoseg/bin/python ]; then PY=.venv-autoseg/bin/python; else PY=.venv/bin/python; fi

S="core/meaning_segmentator/tools/autoseg_en2x/run19"
LOG="core/meaning_segmentator/experiment/artifacts/en2x/logs/run19.log"
mkdir -p "$(dirname "$LOG")"

echo "== $(date '+%F %T') run19 start (PY=$PY)" >> "$LOG"

$PY -u -m core.meaning_segmentator.autoseg.loop_distill \
    --dataset fleurs-en-multi-x \
    --pair-id en2x/en-multi --run-id run19 \
    --split-from en2x/en-multi/run16 \
    --labels-from en2x/en-multi/run18 \
    --train 100 --train-pool 200 \
    --n-cases 24 \
    --rule-min-matches 8 --rule-min-t 2.0 \
    --revision-rounds 1 --pool-size 2 \
    --provider openai --model gpt-5-mini \
    --iterations 5 --budget 12 >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') loop_distill exit=$rc" >> "$LOG"
[ $rc -eq 0 ] && touch "$S/loop.done" || touch "$S/loop.failed"

echo "== $(date '+%F %T') cost_report" >> "$LOG"
$PY -u -m core.meaning_segmentator.autoseg.infra.cost_report \
    --run-id en2x/en-multi/run19 --budget 12 >> "$LOG" 2>&1

touch "$S/chain.done"
echo "== $(date '+%F %T') DONE" >> "$LOG"
