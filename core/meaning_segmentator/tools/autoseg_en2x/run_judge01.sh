#!/bin/bash
# judge01 — 판단형 루프 첫 실행. 목표는 그리디 오라클(경계별 H 상위 k), 판정은 (문장,k) 짝의
# H_set 을 문장 단위로 재추출한 부트스트랩 CI 하한. min_gap 1 — 짧은 조각은 H_set 이 벌한다.
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/judge01.log
mkdir -p "$(dirname "$LOG")"
echo "== $(date '+%F %T') judge01 start" >> $LOG
$PY -u -m core.meaning_segmentator.autoseg.loop_judge \
    --from-run en2x/en-multi/run25 --run-id en2x/en-multi/judge01 \
    --prompt core/meaning_segmentator/tools/autoseg_en2x/run24/probe/v0_coh4.txt \
    --dev-a 150 --min-gap 1 --min-chunk 2 --max-k 10 \
    --iterations 5 --n-cases 12 --checkpoint-every 2 \
    --provider openai --model gpt-5-mini --budget 25 >> $LOG 2>&1
echo "== $(date '+%F %T') exit=$? DONE" >> $LOG
