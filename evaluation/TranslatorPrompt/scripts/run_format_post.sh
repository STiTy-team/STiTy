#!/bin/bash
# 2.4 번역 두 모델이 끝나면 채점 → 집계.
#   tmux new-session -d -s fmt-post -c "$PWD" "bash evaluation/TranslatorPrompt/scripts/run_format_post.sh >> evaluation/TranslatorPrompt/logs/post.log 2>&1"
set -uo pipefail
root=evaluation/TranslatorPrompt
m=$root/logs/markers
export PYTHONPATH=$PWD
# 채점·집계용 파이썬 — unbabel-comet 과 metricx24 가 있는 환경. 번역은 PATH 의 python 으로 돈다.
METRICS_PY=${METRICS_PY:-python}
until [ -f $m/format_gpt-6-luna.done ] && [ -f $m/format_qwen3.5-4b-bf16.done ]; do sleep 30; done
echo "[$(date +%T)] score"
"$METRICS_PY" -u evaluation/LongContextMT/scripts/score.py \
  --run-dir $root/results/fmt-20260930 --glob "*/format.jsonl" --only comet kiwi metricx \
  >> $root/logs/score.log 2>&1 && touch $m/score.done || echo "score FAILED"
echo "[$(date +%T)] aggregate"
"$METRICS_PY" -u $root/scripts/format_aggregate.py >> $root/logs/aggregate.log 2>&1 \
  && touch $m/aggregate.done || echo "aggregate FAILED"
echo "[$(date +%T)] all done"
