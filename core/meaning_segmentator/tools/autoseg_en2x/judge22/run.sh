#!/bin/bash
# judge22 — 채택 판정을 ≤3 구간으로 옮긴다.
#
# judge21 이 남긴 것: 세 개정 모두 dev 에서 양수인데 test 660 에서 음수였다. 구간을 펴 보면
# 순서 규칙(fallback)만 test ≤3 에서 +0.0055 로 살아남았고, H_set 평균(-0.0019)은 나머지 네
# 구간의 번짐에 상쇄돼 기각이 됐다. 최종 지표인 CoVoST2 등지연 COMET 에서 지는 자리가 짧은
# 지연이므로 거기서 가른다. 다른 구간의 퇴행은 --guard-bin 이 따로 막는다.
#
# v0 는 judge21 것을 그대로 쓴다 — 새로 뽑으면 비교가 끊기고, 같으면 iter 0 채점이 캐시로 0원이다.
# 역할에서 narrow_rule 은 뺐다(세 이터 -0.0005/-0.0009/-0.0050, 마지막은 퇴행 가드).
# examples_only 도 뺐다(dev 최고였으나 test 에서 ≤3 조차 -0.0087).
#
#   tmux new-session -d -s judge22 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge22/run.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V0=core/meaning_segmentator/experiment/artifacts/en2x/en-multi/judge21/prompt_v0.txt
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge22.log

.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run28 --run-id en2x/en-multi/judge22 \
    --prompt "$V0" \
    --candidate-roles fallback,prune,single_small --pe-candidates 3 \
    --candidates-cap 3 --findings-max 1 --full-score-max 3 \
    --adopt-rule lo --adopt-bin "≤3" --adopt-strong 0 --guard-bin 0.01 --no-near-miss \
    --labeled-examples --growth-per-iter 0.15 --n-cases 45 --inversions-max 15 \
    --case-exclude bin --case-alloc loss \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 --iterations 4 \
    --workers 128 --extra-key-envs OPENAI_API_KEY_2 \
    --provider openai --model gpt-5-mini --budget "${BUDGET:-60}" >> "$LOG" 2>&1
rc=$?
echo "== $(date '+%F %T') judge22 exit=$rc" >> "$LOG"
