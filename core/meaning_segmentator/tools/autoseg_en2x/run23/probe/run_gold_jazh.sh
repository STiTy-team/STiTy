#!/bin/bash
# ja/zh 만 다시: 첫 실행은 sacrebleu[ja](MeCab) 이 없어 ja 부트스트랩에서 죽었다. de 산출은 그대로 둔다.
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=.
PY=.venv/bin/python
S=core/meaning_segmentator/tools/autoseg_en2x/run23/probe
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/run23_gold.log
for p in llm_k3 A_contra_adqLR B_srcnli_adqLR D_adqLR; do
  echo "== $(date '+%F %T') bleu_eval $p (ja zh)" >> $LOG
  $PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
     --run-id en2x/en-multi/run23_gold_$p --label $p --split test \
     --dataset fleurs --manifest-tag multi_loop405 --targets ja zh \
     --translate-engine local --wordtimes interp --no-auto-greedy --no-sentence-bleu \
     --workers 16 >> $LOG 2>&1 || echo "BLEU $p FAILED" >> $LOG
  echo "== $(date '+%F %T') comet_eval $p (ja zh)" >> $LOG
  $PY -u -m core.meaning_segmentator.autoseg.scoring.comet_eval \
     --run-id en2x/en-multi/run23_gold_$p --label $p --split test \
     --manifest-tag multi_loop405 --targets ja zh >> $LOG 2>&1 || echo "COMET $p FAILED" >> $LOG
done
echo "== $(date '+%F %T') summarize" >> $LOG
$PY -u $S/summarize_gold.py >> $LOG 2>&1
echo "== $(date '+%F %T') DONE2" >> $LOG
