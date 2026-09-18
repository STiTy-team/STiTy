#!/bin/bash
# madlad 비교군(AlignAtt f 스윕 + MU n 스윕) 참조 기반 평가 — BLEU 그다음 COMET.
#
# --baselines-native 로 coarsen T 격자를 끈다. f 와 n 이 이미 정책 자기 노브라 우리 T 를
# 또 얹으면 노브가 두 겹이 된다 (09_alignatt_native_eval.sh 와 같은 이유).
#
# **15,530 전체를 쓴다.** --label 에 없는 이름을 주면 bleu_eval 이 prompt_eval 대신
# 매니페스트에서 문장을 읽는다. 종전 조건 30개는 auto_run13_mg1(15,430) 위에서 난 값이라
# 같은 표에 못 섞지만, 비교가 필요 없다는 판단이라 새 자로 다시 낸다.
#
# bleu_eval 은 bleu/<타깃>.json 을 통째로 덮어쓴다. 종전 값은 bleu_backup_premadlad/ 에
# 그대로 두고 병합하지 않는다 — 문장 집합이 다른 값을 합치면 표 안에서 자가 갈린다.
#
# 번역 캐시가 비어 있어(다른 기계에서 돌아 .gitignore 로 안 따라왔다) 조각 번역을 처음부터
# 한다 — 시간의 대부분이 이것이다. 타깃은 순차로 돈다. 병목은 프로세스 수가 아니라 워커다.
set -u
cd /home/skkai/STiTy || exit 1
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
RUN=en2x/covost2/full
F=core/meaning_segmentator/experiment/artifacts/$RUN
BK=$F/bleu_backup_premadlad
L=$F/logs
mkdir -p $BK $L
cp $F/bleu/de.json $F/bleu/ja.json $F/bleu/zh.json $BK/ 2>/dev/null
echo "백업 -> $BK"

AA="alignatt_mad_f2 alignatt_mad_f4 alignatt_mad_f6 alignatt_mad_f8"

for t in de ja zh; do
  # de n=50 은 GB10 이 아직 돌고 있다. 들어오면 그 조건만 따로 돌려 합치면 된다.
  MU="mu_prefix_mad_n2 mu_prefix_mad_n10"
  [ $t != de ] && MU="$MU mu_prefix_mad_n50"
  B="$AA $MU"
  echo "===== bleu_eval $t $(date '+%F %T') ====="
  $PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
    --run-id $RUN --label none_use_manifest --split test \
    --dataset covost2 --manifest-tag full --targets $t \
    --t-grid 2 3 4 6 --src-spaced 1 \
    --translate-engine local --local-mt-model google/madlad400-3b-mt --mt-batch 48 \
    --workers 24 --baselines $B --baselines-native $B \
    --conditions $B unsegmented \
    --bootstrap 0 --no-sentence-bleu --no-auto-greedy \
    > $L/bleu_eval_madlad_$t.log 2>&1
  rc=$?
  echo "  $t exit=$rc $(date '+%F %T')"
  [ $rc -ne 0 ] && { echo "!! 실패 — 백업 되돌림"; cp $BK/*.json $F/bleu/; exit 1; }
done

echo "===== COMET (없는 조건만) $(date '+%F %T') ====="
$PY -u -m core.meaning_segmentator.autoseg.baselines.comet_score \
  --run-id $RUN --dataset covost2 --manifest-tag full \
  --label none_use_manifest --split test --targets de ja zh --only-missing \
  --model Unbabel/wmt22-comet-da --batch-size 32 > $L/comet_madlad.log 2>&1
echo "  COMET exit=$? $(date '+%F %T')"
touch $F/baselines/madlad_eval.done
echo "===== 완료 $(date '+%F %T') ====="
