#!/bin/bash
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=.
PY=.venv/bin/python; S=core/meaning_segmentator/tools/autoseg_en2x/run24/probe
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/run24_gold.log
echo "== $(date '+%F %T') emit" >> $LOG
$PY -u $S/gold_llm.py >> $LOG 2>&1 || { echo "EMIT FAILED" >> $LOG; exit 1; }
$PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval --run-id en2x/en-multi/run24_gold_llm_k3 --label llm_k3 --split test \
   --dataset fleurs --manifest-tag multi_loop405 --targets de ja zh --translate-engine local --wordtimes interp --no-auto-greedy --no-sentence-bleu --workers 16 >> $LOG 2>&1 || echo "BLEU FAILED" >> $LOG
$PY -u -m core.meaning_segmentator.autoseg.scoring.comet_eval --run-id en2x/en-multi/run24_gold_llm_k3 --label llm_k3 --split test \
   --manifest-tag multi_loop405 --targets de ja zh >> $LOG 2>&1 || echo "COMET FAILED" >> $LOG
$PY -u $S/gold_llm.py --compare >> $LOG 2>&1
echo "== $(date '+%F %T') DONE" >> $LOG
