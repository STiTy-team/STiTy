#!/bin/bash
# GPU 를 같이 쓰는 다른 작업이 자리를 비울 때까지 기다렸다가 로컬 번역 → 채점 → 집계.
# 공유 GPU 라 남의 작업은 건드리지 않는다. 여유가 NEED_MIB 이상이면 시작한다.
#   tmux new-session -d -s qual-s5-local -c "$PWD" "bash evaluation/TranslatorPrompt/scripts/run_local_when_free.sh S5"
set -uo pipefail
root=evaluation/TranslatorPrompt
export PYTHONPATH=$PWD
# 채점·집계용 파이썬 — unbabel-comet 과 metricx24 가 있는 환경. 번역은 PATH 의 python 으로 돈다.
METRICS_PY=${METRICS_PY:-python}
export PYTORCH_ALLOC_CONF=expandable_segments:True
run_id=${RUN_ID:-qual-20260930}
need=${NEED_MIB:-10000}
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$need" ]; do sleep 60; done
echo "[$(date +%T)] GPU free — local translate $*"
python -u $root/scripts/quality_run.py --model qwen3.5-4b-bf16 --run-id $run_id --batch 16 --conditions "$@" \
  >> $root/logs/quality_qwen3.5-4b-bf16.log 2>&1 || echo "local FAILED"
until ! tmux has-session -t qual-s5 2>/dev/null; do sleep 30; done
echo "[$(date +%T)] score"
"$METRICS_PY" -u evaluation/LongContextMT/scripts/score.py --run-dir $root/results/$run_id \
  --glob "*/quality.jsonl" --only comet kiwi metricx >> $root/logs/quality_score.log 2>&1 || echo "score FAILED"
echo "[$(date +%T)] aggregate"
"$METRICS_PY" -u $root/scripts/quality_aggregate.py $run_id >> $root/logs/quality_aggregate.log 2>&1 \
  || echo "aggregate FAILED"
echo "[$(date +%T)] all done"
