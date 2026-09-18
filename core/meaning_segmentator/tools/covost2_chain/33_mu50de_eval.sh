#!/bin/bash
# de 의 MU n=50 라벨이 완성되면 그 조건만 BLEU·COMET 에 추가한다.
# bleu_eval 은 파일을 통째로 덮어쓰므로 백업 후 merge_conditions.py 로 되얹는다.
set -u
cd /home/skkai/STiTy || exit 1
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
RUN=en2x/covost2/full
F=core/meaning_segmentator/experiment/artifacts/$RUN
BK=$F/bleu_backup_pre_mu50de
L=$F/logs
# 라벨 최종본이 나올 때까지 기다린다 (진행분이 아니라 완성본).
while [ ! -s "$F/baselines/mu_prefix_mad_n50_de_test.json" ]; do sleep 60; done
sleep 10
mkdir -p $BK; cp $F/bleu/de.json $BK/
echo "백업 -> $BK  $(date '+%F %T')"

$PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
  --run-id $RUN --label none_use_manifest --split test \
  --dataset covost2 --manifest-tag full --targets de \
  --t-grid 2 3 4 6 --src-spaced 1 \
  --translate-engine local --local-mt-model google/madlad400-3b-mt --mt-batch 48 \
  --workers 24 --baselines mu_prefix_mad_n50 --baselines-native mu_prefix_mad_n50 \
  --conditions mu_prefix_mad_n50 --bootstrap 0 --no-sentence-bleu --no-auto-greedy \
  > $L/bleu_eval_mu50de.log 2>&1
rc=$?
echo "  bleu exit=$rc $(date '+%F %T')"
[ $rc -ne 0 ] && { cp $BK/de.json $F/bleu/; exit 1; }

$PY core/meaning_segmentator/tools/covost2_chain/merge_conditions.py $BK $F/bleu de || {
  cp $BK/de.json $F/bleu/; exit 1; }

$PY -u -m core.meaning_segmentator.autoseg.baselines.comet_score \
  --run-id $RUN --dataset covost2 --manifest-tag full \
  --label none_use_manifest --split test --targets de --only-missing \
  --model Unbabel/wmt22-comet-da --batch-size 32 > $L/comet_mu50de.log 2>&1
echo "  comet exit=$? $(date '+%F %T')"
touch $F/baselines/mu50de_eval.done
echo "===== 완료 $(date '+%F %T') ====="
