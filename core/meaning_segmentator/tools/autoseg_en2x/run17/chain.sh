#!/bin/bash
# run17 — 개정이 실제로 개선되는지 본다 (승격 관문 · 거부 부검 · 근거 확대).
#
# run16 대비 바뀐 것 넷. 프롬프트 쪽은 안 건드렸다 — 루프 기계장치만 바꿨다.
#
#   1) 승격 관문      홀드아웃 Δ>0 을 넘는 후보만 dev 로. 못 넘으면 재추출(최대 2라운드),
#                     그래도 없으면 dev 평가를 건너뛴다 (--revision-rounds 2)
#   2) 거부 부검      진 개정본의 낙폭 상위 문장 + 원인 진단을 다음 Critic·PE 에 넘긴다
#   3) 사례 격자 전체 main_T 하나가 아니라 문장마다 가장 크게 진 T 에서 사례를 싣는다
#   4) 근거 100문장   train 배치 30 -> 100, 홀드아웃 60 -> 100
#
# **평가 분할은 run16 것을 그대로 쓴다** (--split-from). 매니페스트를 loop905 로 늘렸는데
# 층화가 층마다 셔플을 해서 그냥 쓰면 test/dev 가 통째로 뒤바뀐다 (실측: +70 에 test
# 22/100 잔존). 고정하면 순서까지 동일하고 새 문장은 전부 train 으로만 간다.
#
# 라벨은 dev/test 만 run15 에서 재사용된다 (id 목록이 순서까지 같아야 재사용된다 —
# labels.py 의 조건). train 200문장 라벨은 새로 만든다: madlad 번역 + NLI + CometKiwi 라
# **LLM 비용 0, GPU 20~30분**이다.
set -u

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1

set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=.

if [ -x .venv-autoseg/bin/python ]; then PY=.venv-autoseg/bin/python; else PY=.venv/bin/python; fi

S="core/meaning_segmentator/tools/autoseg_en2x/run17"
LOG="core/meaning_segmentator/experiment/artifacts/en2x/logs/run17.log"
mkdir -p "$(dirname "$LOG")"

# 크래시 후 재실행에서도 이전 로그를 지우지 않는다 — 지우면 그 실행의 지출이 복원 불가다.
echo "== $(date '+%F %T') run17 start (PY=$PY)" >> "$LOG"

$PY -u -m core.meaning_segmentator.autoseg.loop_distill \
    --dataset fleurs-en-multi-x \
    --pair-id en2x/en-multi --run-id run17 \
    --split-from en2x/en-multi/run16 \
    --labels-from en2x/en-multi/run15 \
    --train 100 --train-pool 200 \
    --revision-rounds 2 \
    --provider openai --model gpt-5-mini \
    --iterations 5 --budget 20 >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') loop_distill exit=$rc" >> "$LOG"
[ $rc -eq 0 ] && touch "$S/loop.done" || touch "$S/loop.failed"

echo "== $(date '+%F %T') cost_report" >> "$LOG"
$PY -u -m core.meaning_segmentator.autoseg.infra.cost_report \
    --run-id en2x/en-multi/run17 --budget 20 >> "$LOG" 2>&1

touch "$S/chain.done"
echo "== $(date '+%F %T') DONE" >> "$LOG"
