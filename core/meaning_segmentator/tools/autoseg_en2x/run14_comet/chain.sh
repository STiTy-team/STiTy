#!/bin/bash
cd /home/mobility/STiTy
set -a; . ./.env; set +a
export PYTHONPATH=.
PY=.venv-autoseg/bin/python
S=core/meaning_segmentator/tools/autoseg_en2x/run14_comet
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/run14_comet.log
echo "== $(date +%T) segment v3 v0 v1 v2 v4 v5" >> $LOG
$PY -u $S/seg_versions.py >> $LOG 2>&1 || { echo "SEG FAILED" >> $LOG; exit 1; }
touch $S/seg.done
until $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; do
  echo "== $(date +%T) CUDA unavailable — waiting" >> $LOG; sleep 120
done
for v in 3 0 1 2 4 5; do
  echo "== $(date +%T) bleu_eval v$v" >> $LOG
  $PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
     --run-id en2x/en-multi/run14_v$v --label v$v --split test \
     --dataset fleurs --manifest-tag multi_loop405 --targets de ja zh \
     --translate-engine local --wordtimes interp --no-auto-greedy --no-sentence-bleu \
     --workers 16 >> $LOG 2>&1 || echo "BLEU v$v FAILED" >> $LOG
  echo "== $(date +%T) comet_eval v$v" >> $LOG
  $PY -u -m core.meaning_segmentator.autoseg.scoring.comet_eval \
     --run-id en2x/en-multi/run14_v$v --label v$v --split test \
     --manifest-tag multi_loop405 --targets de ja zh >> $LOG 2>&1 || echo "COMET v$v FAILED" >> $LOG
  touch $S/v$v.done
done
$PY $S/summarize.py >> $LOG 2>&1
touch $S/chain.done
echo "== $(date +%T) DONE" >> $LOG
