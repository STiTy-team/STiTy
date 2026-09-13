#!/bin/bash
# 라벨 B(소스 NLI contra) 의 gold 검증 — test 100, FLEURS multi_loop405 (de/ja/zh), madlad, COMET-da.
# 1) emit_gold.py 로 정책 4개 절단을 prompt_eval 형식으로 (LLM 0, NLI 만)
# 2) 정책마다 bleu_eval(madlad 번역) + comet_eval   3) summarize_gold.py → run23/gold_test.md
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=.
PY=.venv/bin/python
S=core/meaning_segmentator/tools/autoseg_en2x/run23/probe
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/run23_gold.log
POLICIES="llm_k3 A_contra_adqLR B_srcnli_adqLR D_adqLR"
echo "== $(date '+%F %T') emit" >> $LOG
$PY -u $S/emit_gold.py >> $LOG 2>&1 || { echo "EMIT FAILED" >> $LOG; exit 1; }
for p in $POLICIES; do
  echo "== $(date '+%F %T') bleu_eval $p" >> $LOG
  $PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
     --run-id en2x/en-multi/run23_gold_$p --label $p --split test \
     --dataset fleurs --manifest-tag multi_loop405 --targets de ja zh \
     --translate-engine local --wordtimes interp --no-auto-greedy --no-sentence-bleu \
     --workers 16 >> $LOG 2>&1 || echo "BLEU $p FAILED" >> $LOG
  echo "== $(date '+%F %T') comet_eval $p" >> $LOG
  $PY -u -m core.meaning_segmentator.autoseg.scoring.comet_eval \
     --run-id en2x/en-multi/run23_gold_$p --label $p --split test \
     --manifest-tag multi_loop405 --targets de ja zh >> $LOG 2>&1 || echo "COMET $p FAILED" >> $LOG
  touch $S/gold_$p.done
done
echo "== $(date '+%F %T') summarize" >> $LOG
$PY -u $S/summarize_gold.py >> $LOG 2>&1
touch $S/gold.done
echo "== $(date '+%F %T') DONE" >> $LOG
