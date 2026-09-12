#!/bin/bash
# run18 — 후보 분포가 우상향하는가. A(성공 방향) · C(채택 귀속) · D(후보 풀) 의 효과만 본다.
#
# run17 은 개정을 처음 채택했지만 후보 분포 자체는 안 올랐다 — 이터별 홀드아웃 overlap
# 최댓값이 0.393 -> 0.474 -> 0.439 -> 0.387 로 올랐다 내려온다. 이 런의 판정 기준은
# **채택 여부가 아니라 그 수열이 단조 증가에 가까워지는가** 다.
#
# **v0 를 run17 것으로 고정한다.** run17 의 v0 는 run16 보다 dev 0.0465 낮게 뽑혔고 그
# 격차가 결과 해석을 계속 흐렸다. 같은 v0 에서 출발하면 run17 <-> run18 차이가 A·C·D
# 로만 읽힌다. 분절 캐시도 같이 옮겼으므로 iter 0 은 run17 과 글자 그대로 같은 출발점
# 이고 비용이 0 이다.
#
# 재추출은 껐다 (--revision-rounds 1). run17 실측에서 r1 이 r0 보다 나은 것이 3이터 중
# 1회뿐이라 평균적으로 값을 못 했고, 이터당 후보를 3 -> 5~6 으로 늘려 비용의 주범이었다.
# 관문(홀드아웃 Δ>0)은 그대로 둔다 — dev 평가 4회를 아낀 쪽은 그것이다.
#
# --adopt-se-mult 는 기본값 1.0 그대로. 문턱을 낮추면 잡음을 채택할 위험이 같이 온다.
set -u

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1

set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=.

if [ -x .venv-autoseg/bin/python ]; then PY=.venv-autoseg/bin/python; else PY=.venv/bin/python; fi

S="core/meaning_segmentator/tools/autoseg_en2x/run18"
LOG="core/meaning_segmentator/experiment/artifacts/en2x/logs/run18.log"
mkdir -p "$(dirname "$LOG")"

echo "== $(date '+%F %T') run18 start (PY=$PY)" >> "$LOG"

$PY -u -m core.meaning_segmentator.autoseg.loop_distill \
    --dataset fleurs-en-multi-x \
    --pair-id en2x/en-multi --run-id run18 \
    --split-from en2x/en-multi/run16 \
    --labels-from en2x/en-multi/run17 \
    --train 100 --train-pool 200 \
    --revision-rounds 1 --pool-size 2 \
    --provider openai --model gpt-5-mini \
    --iterations 5 --budget 12 >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') loop_distill exit=$rc" >> "$LOG"
[ $rc -eq 0 ] && touch "$S/loop.done" || touch "$S/loop.failed"

echo "== $(date '+%F %T') cost_report" >> "$LOG"
$PY -u -m core.meaning_segmentator.autoseg.infra.cost_report \
    --run-id en2x/en-multi/run18 --budget 12 >> "$LOG" 2>&1

touch "$S/chain.done"
echo "== $(date '+%F %T') DONE" >> "$LOG"
