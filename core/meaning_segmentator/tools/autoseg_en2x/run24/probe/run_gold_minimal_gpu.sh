#!/bin/bash
# GPU 가 비면 gold 의 GPU 단계만 잇는다 (test 채점은 이미 끝났다 — prompt_eval 존재).
# funasr 벤치가 vLLM 으로 15 GB 를 물고 있어 madlad 가 OOM 났다. 남의 프로세스는 건드리지 않는다.
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; S=core/meaning_segmentator/tools/autoseg_en2x/run24/probe
LOG=core/meaning_segmentator/experiment/artifacts/en2x/logs/run24_gold_minimal.log
TAG=min_tgt_k3
NEED=9000   # MiB. madlad-3b + COMET 이 6.4 GB 를 쓰고 여유가 필요하다
echo "== $(date '+%F %T') GPU 대기 (>= ${NEED} MiB 가 2회 연속)" >> $LOG
ok=0
for i in $(seq 1 720); do   # 최대 6시간
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ "$free" -ge "$NEED" ]; then ok=$((ok+1)); else ok=0; fi
  [ $ok -ge 2 ] && break
  sleep 30
done
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "== $(date '+%F %T') 대기 끝 free=${free}MiB" >> $LOG
[ "$free" -lt "$NEED" ] && { echo "GPU WAIT TIMEOUT" >> $LOG; exit 1; }
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
echo "== $(date '+%F %T') compare" >> $LOG
$PY -u $S/compare_gold.py $TAG >> $LOG 2>&1
echo "== $(date '+%F %T') DONE" >> $LOG
