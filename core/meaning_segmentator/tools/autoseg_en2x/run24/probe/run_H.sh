#!/bin/bash
# 오라클 확정안 H = G × (1 − 소스 contra) 의 gold. pseudoref_test.json 이 있으면 라벨 계산은 건너뛴다.
# 비교 대상: G 단독 0.7823 / B 0.7685 / min_tgt 정책 0.7686 / 무절단 0.8770 (격자평균).
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; S=core/meaning_segmentator/tools/autoseg_en2x/run24/probe
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/run24_H.log
NEED=9000
wait_gpu() {
  local ok=0 free
  for i in $(seq 1 720); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -ge "$NEED" ]; then ok=$((ok+1)); else ok=0; fi
    [ $ok -ge 2 ] && return 0
    sleep 30
  done
  echo "GPU WAIT TIMEOUT" >> $LOG; return 1
}
echo "== $(date '+%F %T') emit (F/G/H)" >> $LOG
wait_gpu || exit 1
$PY -u -m core.meaning_segmentator.tools.autoseg_en2x.run24.probe.pseudoref --split test --emit >> $LOG 2>&1 \
  || { echo "EMIT FAILED" >> $LOG; exit 1; }
TAG=H_qeXcontra
echo "== $(date '+%F %T') $TAG bleu" >> $LOG
wait_gpu || exit 1
$PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
    --run-id en2x/en-multi/run24_gold_$TAG --label $TAG --split test \
    --dataset fleurs --manifest-tag multi_loop405 --targets de ja zh \
    --translate-engine local --wordtimes interp --no-auto-greedy --no-sentence-bleu \
    --workers 16 >> $LOG 2>&1 || { echo "BLEU FAILED" >> $LOG; exit 1; }
echo "== $(date '+%F %T') $TAG comet" >> $LOG
$PY -u -m core.meaning_segmentator.autoseg.scoring.comet_eval \
    --run-id en2x/en-multi/run24_gold_$TAG --label $TAG --split test \
    --manifest-tag multi_loop405 --targets de ja zh >> $LOG 2>&1 || { echo "COMET FAILED" >> $LOG; exit 1; }
$PY -u $S/compare_gold.py $TAG >> $LOG 2>&1
echo "== $(date '+%F %T') DONE" >> $LOG
