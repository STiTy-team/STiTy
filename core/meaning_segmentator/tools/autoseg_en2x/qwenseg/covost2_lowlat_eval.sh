#!/bin/bash
# 스트리밍 qwen <SEG> 곡선의 저지연 구간 — θ 0.001~0.02 를 추가한다 (θ=0.05 가 조각 2.35 라 T2~T3 구간이 비었다).
# CoVoST2 500문장 탐색: θ 0.001/0.003/0.005/0.01/0.02 → 조각 4.59/4.14/3.85/3.39/2.91 (min_words 2).
#   tmux new-session -d -s qwenlowlat -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/qwenseg/covost2_lowlat_eval.sh"
#
# 별도 런(full_qwenseg_lowlat)에서 재고, 끝에 full_qwenseg_offline/bleu(기존 조건 + 스트리밍 + 오프라인 + judge13)를
# 이 런 쪽으로 합쳐 그린다. 다른 런 파일은 읽기만 한다. 지금 GPU 를 쓰는 다른 작업이 없으므로 캐시는 full 것을 공유한다.
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
A=core/meaning_segmentator/experiment/artifacts
SRC=en2x/covost2/full
RUN=en2x/covost2/full_qwenseg_lowlat
F=$A/$RUN
L=$F/logs
TH="0.001 0.003 0.005 0.01 0.02"
B=$(for t in $TH; do printf "qwenseg_th%s " $t; done)
mkdir -p $L
[ -e $F/cache ] || ln -s ../full/cache $F/cache
ts () { date '+%F %T'; }

echo "===== 분절 $(ts) ====="
$PY -u -m core.meaning_segmentator.autoseg.gates.qwen_seg_baselines \
  --src-run $SRC --run-id $RUN --thresholds $TH > $L/baselines.log 2>&1
rc=$?; echo "  분절 exit=$rc $(ts)"; grep "조각" $L/baselines.log
[ $rc -eq 0 ] || exit 1

for t in de ja zh; do
  echo "===== bleu_eval $t $(ts) ====="
  $PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
    --run-id $RUN --label none_use_manifest --split test \
    --dataset covost2 --manifest-tag full --targets $t \
    --t-grid 2 3 4 6 --src-spaced 1 \
    --translate-engine local --local-mt-model google/madlad400-3b-mt --mt-batch 48 \
    --workers 24 --baselines $B --baselines-native $B \
    --bootstrap 0 --no-sentence-bleu --no-auto-greedy \
    > $L/bleu_eval_$t.log 2>&1
  rc=$?; echo "  $t exit=$rc $(ts)"
  [ $rc -eq 0 ] || exit 1
done

echo "===== COMET $(ts) ====="
$PY -u -m core.meaning_segmentator.autoseg.baselines.comet_score \
  --run-id $RUN --dataset covost2 --manifest-tag full \
  --label none_use_manifest --split test --targets de ja zh --only-missing \
  --model Unbabel/wmt22-comet-da --batch-size 32 > $L/comet.log 2>&1
rc=$?; echo "  COMET exit=$rc $(ts)"; grep "조건 채점" $L/comet.log
[ $rc -eq 0 ] || exit 1

echo "===== 합침 $(ts) ====="
cp -r $F/bleu $F/bleu_lowlat_only
$PY core/meaning_segmentator/tools/covost2_chain/merge_conditions.py $A/en2x/covost2/full_qwenseg_offline/bleu $F/bleu de ja zh || exit 1

echo "===== 그래프 $(ts) ====="
COMMON="--run-id $RUN --targets de ja zh --t-grid 2 3 4 6 --point-labels none --solid --panel-width 7.5 --legend-ncol 4 --no-cite"
for M in bleu comet; do
  S=$([ $M = comet ] && echo _comet || echo "")
  # 스트리밍 판: qwen θ 곡선 + 비교군 (오프라인 qwen·judge13·LR·구두점 제외)
  $PY -m core.meaning_segmentator.autoseg.scoring.plot_tradeoff $COMMON --metric $M \
    --drop auto alignatt mu_prefix alignatt_native judge13 qwenp0 qwenlrall punct --out tradeoff_stream$S 2>&1 | tail -1
  # 오프라인 판: 같은 스타일 (qwen θ 대신 qwen p0)
  $PY -m core.meaning_segmentator.autoseg.scoring.plot_tradeoff $COMMON --metric $M \
    --drop auto alignatt mu_prefix alignatt_native judge13 qwenseg qwenlrall punct --out tradeoff_offline$S 2>&1 | tail -1
done
touch $F/covost2_lowlat_eval.done
echo "===== 완료 $(ts) ====="
