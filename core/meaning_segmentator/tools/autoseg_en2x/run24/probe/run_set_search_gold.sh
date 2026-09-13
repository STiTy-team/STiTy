#!/bin/bash
# test 에서 집합 탐색 → 탐색 최적 절단집합의 gold(COMET-DA). T별 상한을 채운다.
# H_set(무참조 QE) 최대화가 사람 참조 기준으로도 이기는지 감사하는 자리이기도 하다.
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; S=core/meaning_segmentator/tools/autoseg_en2x/run24/probe
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/run25_set_search_gold.log
TAG=S_search
echo "== $(date '+%F %T') test 집합 탐색 + emit" >> $LOG
$PY -u -m core.meaning_segmentator.tools.autoseg_en2x.run24.probe.set_search \
    --split test --n 100 --m 8 --max-sets 200 --emit >> $LOG 2>&1 || { echo "SEARCH FAILED" >> $LOG; exit 1; }
echo "== $(date '+%F %T') bleu" >> $LOG
$PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
    --run-id en2x/en-multi/run24_gold_$TAG --label $TAG --split test \
    --dataset fleurs --manifest-tag multi_loop405 --targets de ja zh \
    --translate-engine local --wordtimes interp --no-auto-greedy --no-sentence-bleu \
    --workers 16 >> $LOG 2>&1 || { echo "BLEU FAILED" >> $LOG; exit 1; }
echo "== $(date '+%F %T') comet" >> $LOG
$PY -u -m core.meaning_segmentator.autoseg.scoring.comet_eval \
    --run-id en2x/en-multi/run24_gold_$TAG --label $TAG --split test \
    --manifest-tag multi_loop405 --targets de ja zh >> $LOG 2>&1 || { echo "COMET FAILED" >> $LOG; exit 1; }
$PY -u $S/compare_gold.py $TAG >> $LOG 2>&1
echo "== $(date '+%F %T') DONE" >> $LOG
