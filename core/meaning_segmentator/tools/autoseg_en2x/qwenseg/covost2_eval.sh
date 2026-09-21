#!/bin/bash
# Qwen3-ASR <SEG> 텍스트 스트리밍 분절을 CoVoST2 전체(15,530)에서 BLEU·COMET 으로 재고, 기존 madlad
# 비교군 그림에 곡선 하나로 얹는다.
#   tmux new-session -d -s qwencovost -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/qwenseg/covost2_eval.sh"
#
# **기존 en2x/covost2/full 은 읽기만 한다.** bleu_eval 은 run 의 bleu/<tgt>.json 을 통째로 덮어쓰므로
# 별도 런(full_qwenseg)에서 돌리고, 끝에 full 의 조건을 그 사본 쪽으로 합친다 (merge_conditions 는
# 두 번째 인자 쪽만 쓴다). 번역 캐시만 full 과 공유한다 — 키가 원문 조각이라 섞여도 결과가 같다.
#
# 번역 madlad, COMET wmt22-comet-da, 지연 laal_ms(강제정렬) — 30_madlad_eval.sh 와 같은 인자다.
# 비용 0 (전부 로컬).
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
A=core/meaning_segmentator/experiment/artifacts
SRC=en2x/covost2/full
RUN=en2x/covost2/full_qwenseg
F=$A/$RUN
L=$F/logs
TH="0.05 0.1 0.2 0.3 0.5"
B=$(for t in $TH; do printf "qwenseg_th%s " $t; done)
mkdir -p $L
[ -e $F/cache ] || ln -s ../full/cache $F/cache
ts () { date '+%F %T'; }

# run27 H_set 스윕이 GPU 를 쓰는 동안은 기다린다 (Qwen + madlad + COMET 동시 적재는 OOM 위험)
SWEEP_LOG=$A/en2x/logs/qwenseg01_sweep.log
[ -n "${SKIP_WAIT:-}" ] || while ! grep -q "exit=.*DONE" $SWEEP_LOG 2>/dev/null; do sleep 30; done
echo "== $(ts) 스윕 끝남 확인, 시작"

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

echo "===== 기존 조건과 합침 (full 은 읽기만) $(ts) ====="
cp -r $F/bleu $F/bleu_qwenseg_only
$PY core/meaning_segmentator/tools/covost2_chain/merge_conditions.py $A/$SRC/bleu $F/bleu de ja zh || exit 1

echo "===== 그래프 $(ts) ====="
for M in bleu comet; do
  S=$([ $M = comet ] && echo _comet || echo "")
  $PY -m core.meaning_segmentator.autoseg.scoring.plot_tradeoff \
    --run-id $RUN --targets de ja zh --metric $M --t-grid 2 3 4 6 \
    --drop auto alignatt mu_prefix alignatt_native --out tradeoff_qwenseg$S 2>&1 | tail -2
done
touch $F/covost2_eval.done
echo "===== 완료 $(ts) ====="
