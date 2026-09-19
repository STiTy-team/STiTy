#!/bin/bash
# judge21 — 순위의 **아래쪽**을 겨냥한다.
#   tmux new-session -d -s judge21 -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/judge21/run.sh"
#
# judge20 은 아홉 후보 중 하나도 채택하지 못했고, 개선이 매번 긴 지연에 몰렸다. 진단이 이유를
# 짚었다 (`diag/rank_depth.json`): 프롬프트 순위는 1위에서 무작위 대비 6.5배지만 5위 1.6배,
# 8위 1.2배, 13위 1.00배로 무너진다. 짧은 지연은 8위까지 소비하고(≤3 의 평균 절단 수 8.5)
# 긴 지연은 1위만 쓴다. 원칙 여덟이 전부 "여기서 자르지 마라" 라 위험한 자리를 아래로 밀 뿐,
# **밀려난 자리들끼리의 서열을 아무도 안 정한다.**
#
#   fallback        서열을 적는 새 역할 — 비교문만 받고 금지문·결속문은 코드가 거부한다
#   examples_only   시연으로 같은 구멍을 친다 — 예시는 문장의 전체 순위를 한 번에 보여준다
#   역전 쌍 15      깊은 구간에서 프롬프트와 라벨이 뒤집힌 쌍을 Critic 에 넘긴다
#   깊이 배수 로그  개정이 순위의 어디를 고쳤는지 이터마다·부검마다 남긴다
#   --no-near-miss  근소 기각본을 기반으로 안 삼는다 — judge20 은 통과율 0.50 짜리 기각본이
#                   이터 2·3 의 기반이 되면서 후속 후보가 0.18 까지 떨어졌다
#   --k-samples 1   **배포 조건과 맞춘다.** CoVoST2 라벨링은 문장당 1회만 분절한다. k=3 병합은
#                   루프 안에서만 쓰는 앙상블이라, 그 조건에서 고른 프롬프트가 1회 분절에서도
#                   나은지는 보장되지 않는다. 비용도 1/3 이고, 짝 비교 반폭은 거의 그대로다 —
#                   k=3 에서 0.0100(judge20 아홉 후보 평균), k=1 추정 0.0103. k=3 이 주는 것은
#                   정밀도가 아니라 절대 수준(+0.0129)이었다.
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge21.log
mkdir -p "$(dirname "$LOG")"
echo "== $(date '+%F %T') judge21 start" >> $LOG
.venv-autoseg/bin/python -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run28 --run-id en2x/en-multi/judge21 \
    --generate-v0 --v0-candidates 1 \
    --candidate-roles fallback,narrow_rule,examples_only,prune --pe-candidates 4 \
    --candidates-cap 4 --findings-max 1 --full-score-max 4 \
    --adopt-rule lo --adopt-strong 0 --guard-bin 0.01 --no-near-miss \
    --labeled-examples --growth-per-iter 0.15 --n-cases 45 --inversions-max 15 \
    --case-exclude bin --case-alloc loss \
    --min-gap 1 --min-chunk 2 --max-k 99 --k-samples 1 --iterations 3 \
    --workers 128 --extra-key-envs OPENAI_API_KEY_2 \
    --provider openai --model gpt-5-mini --budget ${BUDGET:-130} ${RESUME:+--resume} >> $LOG 2>&1
echo "== $(date '+%F %T') exit=$? DONE" >> $LOG
