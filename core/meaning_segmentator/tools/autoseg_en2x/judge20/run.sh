#!/bin/bash
# judge20 — 판정 집합을 키우고 선별 단계를 없앤다.
#   tmux new-session -d -s judge20 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge20/run.sh"
#
# judge15~19 가 실패한 자리를 오늘 실측으로 짚었다: 후보를 25~60문장에서 좁혔는데 그 표본의
# 잡음 반폭이 0.0325~0.0503 이고 찾는 효과가 0.008 이다. 좁히기가 무작위였고, 잘 만든 후보가
# 본채점까지 못 갔다. 그래서 **좁히지 않는다** — 후보를 넷만 만들고 넷 다 dev 600 에서 잰다.
#
#   dev 600      채택 판정, 반폭 0.0075 (judge19 는 200 / 0.0130)
#   후보 4       prune ×2 + narrow_rule + rewrite, 전원 Critic 1순위 finding
#   유형 게이트  없음 (--type-select 를 주지 않는다)
#   체크포인트   없음 — 다섯 런에서 롤백 0회, 분할에 dev-B 가 없어 자동 해제된다
#   채택 문턱    하한 > 0 (--adopt-strong 0). 판정이 정밀해진 만큼 조인다
#   v0           A — sel 300 에서 잰 세 갈래가 구분되지 않아 손대지 않은 생성물을 쓴다
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge20.log
mkdir -p "$(dirname "$LOG")"
echo "== $(date '+%F %T') judge20 start" >> $LOG
.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run28 --run-id en2x/en-multi/judge20 \
    --generate-v0 --v0-candidates 1 \
    --candidate-roles prune,prune,narrow_rule,rewrite --pe-candidates 4 \
    --candidates-cap 4 --findings-max 1 --full-score-max 4 \
    --adopt-rule lo --adopt-strong 0 --guard-bin 0.01 \
    --labeled-examples --growth-per-iter 0.15 --n-cases 60 \
    --case-exclude bin --case-alloc loss \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 3 --iterations 3 \
    --workers 128 --extra-key-envs OPENAI_API_KEY_2 \
    --provider openai --model gpt-5-mini --budget ${BUDGET:-130} ${RESUME:+--resume} >> $LOG 2>&1
echo "== $(date '+%F %T') exit=$? DONE" >> $LOG
