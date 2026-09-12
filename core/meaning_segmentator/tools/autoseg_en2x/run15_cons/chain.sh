#!/bin/bash
# consistency 를 라벨 축으로 더할 값이 있는가 — gold 참조로 가른다.
#
# 1) oracle_label 로 정책별 절단을 prompt_eval 형식으로 내보낸다 (LLM 호출 0)
# 2) 정책마다 bleu_eval + comet_eval — FLEURS gold 참조로 채점
#
# 정책 넷이 답하는 것:
#   contra_adqLR         현행 라벨. 기준선
#   contra_cons_adqLR    세 축 다. consistency 가 현행에 무엇을 더하나
#   cons_adqLR           consistency 가 contradiction 을 대체할 수 있나
#   cons                 consistency 단독으로 얼마나 가나
cd /home/mobility/STiTy
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=.
PY=.venv-autoseg/bin/python
S=core/meaning_segmentator/tools/autoseg_en2x/run15_cons
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/run15_cons_gold.log
POLICIES="contra_adqLR contra_cons_adqLR cons_adqLR cons"

echo "== $(date +%T) labels+emit (test)" >> $LOG
$PY -u -m core.meaning_segmentator.autoseg.gates.oracle_label \
   --run-id en2x/en-multi/run15 --split test --llm-rows \
   core/meaning_segmentator/experiment/artifacts/en2x/en-multi/run15/test_rows.json \
   --emit $POLICIES >> $LOG 2>&1 || { echo "EMIT FAILED" >> $LOG; exit 1; }
touch $S/emit.done

for p in $POLICIES; do
  echo "== $(date +%T) bleu_eval $p" >> $LOG
  $PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
     --run-id en2x/en-multi/run15_oracle_$p --label $p --split test \
     --dataset fleurs --manifest-tag multi_loop405 --targets de ja zh \
     --translate-engine local --wordtimes interp --no-auto-greedy --no-sentence-bleu \
     --workers 16 >> $LOG 2>&1 || echo "BLEU $p FAILED" >> $LOG
  echo "== $(date +%T) comet_eval $p" >> $LOG
  $PY -u -m core.meaning_segmentator.autoseg.scoring.comet_eval \
     --run-id en2x/en-multi/run15_oracle_$p --label $p --split test \
     --manifest-tag multi_loop405 --targets de ja zh >> $LOG 2>&1 \
     || echo "COMET $p FAILED" >> $LOG
  touch $S/$p.done
done
touch $S/chain.done
echo "== $(date +%T) DONE" >> $LOG
