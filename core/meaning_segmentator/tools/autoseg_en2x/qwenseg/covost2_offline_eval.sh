#!/bin/bash
# qwen <SEG> 확률을 오프라인 점수로 써서 auto_T 와 같은 절단기(T 격자, min_gap 1)로 자른 조건을
# CoVoST2 전체에서 BLEU·COMET 으로 잰다. 스트리밍 체인(covost2_eval.sh)과 동시에 돈다.
#   tmux new-session -d -s qwenoffline -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/qwenseg/covost2_offline_eval.sh"
#
# **번역 캐시는 복사본을 쓴다.** 스트리밍 체인이 full/cache 를 쓰는 중이고 JsonCache 는 같은 .tmp 이름으로
# 원자 교체를 하므로 두 프로세스가 한 파일을 쓰면 충돌한다.
# 마지막에 스트리밍 체인의 완료 마커(full_qwenseg/covost2_eval.done)를 기다렸다가 셋(full 기존 조건 +
# 스트리밍 θ + 오프라인 T)을 이 런 쪽으로 합쳐 그린다. full·full_qwenseg 는 읽기만 한다.
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
A=core/meaning_segmentator/experiment/artifacts
SRC=en2x/covost2/full
RUN=en2x/covost2/full_qwenseg_offline
F=$A/$RUN
L=$F/logs
GRID="2 3 4 6"
B=$(for k in p0 lrall; do for t in $GRID; do printf "qwen%s_T%s " $k $t; done; done)
mkdir -p $L
[ -e $F/cache ] || cp -r $A/$SRC/cache $F/cache
ts () { date '+%F %T'; }

echo "===== 오프라인 점수·절단 $(ts) ====="
$PY -u -m core.meaning_segmentator.autoseg.gates.qwen_seg_offline_baselines \
  --src-run $SRC --run-id $RUN --t-grid $GRID > $L/baselines.log 2>&1
rc=$?; echo "  exit=$rc $(ts)"; grep "조각" $L/baselines.log
[ $rc -eq 0 ] || exit 1

for t in de ja zh; do
  echo "===== bleu_eval $t $(ts) ====="
  $PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
    --run-id $RUN --label none_use_manifest --split test \
    --dataset covost2 --manifest-tag full --targets $t \
    --t-grid $GRID --src-spaced 1 \
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

echo "===== 스트리밍 체인 완료 대기 $(ts) ====="
while [ ! -f $A/en2x/covost2/full_qwenseg/covost2_eval.done ]; do sleep 60; done
cp -r $F/bleu $F/bleu_offline_only
# full_qwenseg/bleu 는 이미 full 의 기존 조건 + 스트리밍 θ 가 합쳐진 상태다
$PY core/meaning_segmentator/tools/covost2_chain/merge_conditions.py $A/en2x/covost2/full_qwenseg/bleu $F/bleu de ja zh || exit 1

echo "===== 그래프 $(ts) ====="
for M in bleu comet; do
  S=$([ $M = comet ] && echo _comet || echo "")
  $PY -m core.meaning_segmentator.autoseg.scoring.plot_tradeoff \
    --run-id $RUN --targets de ja zh --metric $M --t-grid 2 3 4 6 \
    --drop auto alignatt mu_prefix alignatt_native --out tradeoff_qwenseg_all$S 2>&1 | tail -2
done
touch $F/covost2_offline_eval.done
echo "===== 완료 $(ts) ====="
