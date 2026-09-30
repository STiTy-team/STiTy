#!/bin/bash
# 우리 정책(auto) 을 비교군과 같은 15,530문장 위에서 채점한다.
# 라벨은 auto_judge13_iter3 — prompt_eval 산출이 15,530행이라 비교군과 ID 집합이 같다
# (옛 auto_run13_mg1 은 15,430이라 못 쓴다).
#
# 우리 축은 T 격자다. 조각 경계가 비교군과 달라 번역 캐시가 많이 빗나가므로 이 실행은
# 앞선 것들보다 오래 걸린다 — 첫 조건 시간을 보고 판단할 것.
set -u
cd /home/skkai/STiTy || exit 1
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
RUN=en2x/covost2/full
F=core/meaning_segmentator/experiment/artifacts/$RUN
BK=$F/bleu_backup_pre_auto
L=$F/logs
mkdir -p $BK $L
cp $F/bleu/de.json $F/bleu/ja.json $F/bleu/zh.json $BK/
echo "백업 -> $BK  $(date '+%F %T')"

for t in de ja zh; do
  echo "===== bleu_eval $t $(date '+%F %T') ====="
  $PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
    --run-id $RUN --label auto_judge13_iter3 --split test \
    --dataset covost2 --manifest-tag full --targets $t \
    --t-grid 2 3 4 6 --src-spaced 1 \
    --translate-engine local --local-mt-model google/madlad400-3b-mt --mt-batch 48 \
    --workers 24 --conditions auto_T2 auto_T3 auto_T4 auto_T6 \
    --bootstrap 0 --no-sentence-bleu --no-auto-greedy \
    > $L/bleu_eval_auto_$t.log 2>&1
  rc=$?
  echo "  $t exit=$rc $(date '+%F %T')"
  [ $rc -ne 0 ] && { echo "!! 실패 — 백업 되돌림"; cp $BK/*.json $F/bleu/; exit 1; }
done

echo "===== 비교군 조건과 병합 $(date '+%F %T') ====="
$PY core/meaning_segmentator/tools/covost2_chain/merge_conditions.py $BK $F/bleu de ja zh || {
  echo "!! 병합 실패"; cp $BK/*.json $F/bleu/; exit 1; }

echo "===== COMET $(date '+%F %T') ====="
for t in de ja zh; do
  $PY -u -m core.meaning_segmentator.autoseg.baselines.comet_score \
    --run-id $RUN --dataset covost2 --manifest-tag full \
    --label auto_judge13_iter3 --split test --targets $t --only-missing \
    --model Unbabel/wmt22-comet-da --batch-size 32 > $L/comet_auto_$t.log 2>&1 &
done
wait
touch $F/baselines/auto_eval.done
echo "===== 완료 $(date '+%F %T') ====="
