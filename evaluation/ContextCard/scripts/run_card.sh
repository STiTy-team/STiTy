#!/bin/bash
# 맥락 카드: 번역(K1~K4) → 채점 → 집계. 저장소 루트에서 tmux 로 띄운다.
#   tmux new-session -d -s card -c "$PWD" "TRANS_PY=<번역 환경>/bin/python METRICS_PY=<채점 환경>/bin/python bash evaluation/ContextCard/scripts/run_card.sh"
set -uo pipefail
root=evaluation/ContextCard
export PYTHONPATH=$PWD PYTORCH_ALLOC_CONF=expandable_segments:True
TRANS_PY=${TRANS_PY:-python}
METRICS_PY=${METRICS_PY:-python}
run_id=${RUN_ID:-card-20260930}
run=$root/results/$run_id
echo "[$(date +%T)] translate"
"$TRANS_PY" -u $root/scripts/card_run.py --run-id $run_id >> $root/logs/card_run.log 2>&1 || { echo "translate FAILED"; exit 1; }
# 1.1 에서 이미 잰 점수는 가져온다 (같은 (원문, 번역, 정답)이면 같은 키)
[ -f $run/scores_cache.jsonl ] || cp evaluation/LongContextMT/results/lcmt-20260930/scores_cache.jsonl $run/scores_cache.jsonl
echo "[$(date +%T)] score"
"$METRICS_PY" -u evaluation/LongContextMT/scripts/score.py --run-dir $run --glob "*/*translations.jsonl" \
  --only comet metricx >> $root/logs/card_score.log 2>&1 || { echo "score FAILED"; exit 1; }
echo "[$(date +%T)] doc-comet"
"$METRICS_PY" -u $root/scripts/card_doc_comet.py $run_id >> $root/logs/card_score.log 2>&1 || { echo "doc-comet FAILED"; exit 1; }
echo "[$(date +%T)] aggregate"
"$METRICS_PY" -u $root/scripts/card_aggregate.py $run_id >> $root/logs/card_aggregate.log 2>&1 \
  && touch $root/logs/markers/card_${run_id}.done || echo "aggregate FAILED"
echo "[$(date +%T)] all done"
