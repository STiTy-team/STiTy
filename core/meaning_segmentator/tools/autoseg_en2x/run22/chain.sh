#!/bin/bash
# run22 — 후보 풀을 끈다. run21 과 이것 하나만 다르다 (--pool-size 2 -> 0).
#
# 풀은 기각된 후보를 다음 이터의 출발점으로 쓰는 장치였다. 세 런(run19~21)의 실측:
#
#   승격 6건 중 5건이 `grow` = 항상 `best` 에서 출발한 후보
#   풀에서 출발해 승격된 것은 run19 iter2 의 neutral 하나뿐이고 dev 에서 기각(+0.0002)
#   채택 4건은 전부 grow
#
# **게다가 풀이 분량 제약을 의도보다 세게 만들었다.** 풀 항목은 대체로 shrink/neutral 이
# 만든 짧은 프롬프트라 거기서 출발하면 한계선이 같이 내려간다 — run21 iter2·3 에서
# best 가 12,032(뒤엔 16,005)인데 neutral·shrink 는 9,935 이하로 쓰라는 요구를 받았다.
# "순증 금지"가 실제로는 "best 보다 2,000~6,000자 짧게"가 된다. neutral 의 분량 탈락
# 10건(전체 12건 중)에 이 원인이 섞여 있다.
#
# 그래서 풀을 먼저 끄고, neutral 모드의 성적은 공정한 조건에서 다시 잰다. 지금 데이터로
# neutral 을 없애면 풀 탓인지 모드 탓인지 못 가른 채 없애는 것이 된다.
#
# **v0·분할·라벨·캐시는 run21 과 동일하다.** iter 0 은 캐시 적중으로 비용 0.
#
# 볼 것: 세 모드가 전부 best 에서 출발할 때
#   - neutral 의 분량 탈락이 줄어드나 (run21 은 2건)
#   - neutral/shrink 가 승격까지 가나 (run19~21 통틀어 1건)
#   - 채택과 test 가 run21(+0.0333, 1.5se)과 비교해 어떤가
set -u

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1

set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=.

if [ -x .venv-autoseg/bin/python ]; then PY=.venv-autoseg/bin/python; else PY=.venv/bin/python; fi

S="core/meaning_segmentator/tools/autoseg_en2x/run22"
LOG="core/meaning_segmentator/experiment/artifacts/en2x/logs/run22.log"
mkdir -p "$(dirname "$LOG")"

echo "== $(date '+%F %T') run22 start (PY=$PY)" >> "$LOG"

$PY -u -m core.meaning_segmentator.autoseg.loop_distill \
    --dataset fleurs-en-multi-x \
    --pair-id en2x/en-multi --run-id run22 \
    --split-from en2x/en-multi/run16 \
    --labels-from en2x/en-multi/run21 \
    --train 100 --train-pool 200 \
    --n-cases 24 \
    --rule-min-matches 5 --rule-min-t 2.0 --rule-min-support 2 \
    --revision-rounds 1 --pool-size 0 \
    --provider openai --model gpt-5-mini \
    --iterations 5 --budget 12 >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') loop_distill exit=$rc" >> "$LOG"
[ $rc -eq 0 ] && touch "$S/loop.done" || touch "$S/loop.failed"

echo "== $(date '+%F %T') cost_report" >> "$LOG"
$PY -u -m core.meaning_segmentator.autoseg.infra.cost_report \
    --run-id en2x/en-multi/run22 --budget 12 >> "$LOG" 2>&1

touch "$S/chain.done"
echo "== $(date '+%F %T') DONE" >> "$LOG"
