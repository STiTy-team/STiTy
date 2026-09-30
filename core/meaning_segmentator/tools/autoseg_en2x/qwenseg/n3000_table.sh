#!/bin/bash
# 임시 등지연 표 — CoVoST2 n3000(full 의 부분집합) 위에서 디코더 θ 곡선과 AlignAtt f 곡선을
# 같은 번역기·같은 COMET·같은 강제정렬 지연으로 맞댄다. full 체인의 축소판이다.
#   tmux new-session -d -s qwentable -c <저장소> "bash core/meaning_segmentator/tools/autoseg_en2x/qwenseg/n3000_table.sh"
# 비용 0 (전부 로컬). ACL AST 런이 GPU 를 쓰는 동안은 기다린다.
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-.venv-autoseg/bin/python}
A=core/meaning_segmentator/experiment/artifacts/en2x/covost2
SRC=en2x/covost2/full
RUN=en2x/covost2/n3000_qwenseg_voxmix
F=$A/n3000_qwenseg_voxmix
L=$F/logs
MODEL=${MODEL:-models/Qwen3-ASR-1.7B-en-covost2-vox-mix-t91-c320-merged}
TH=${TH:-"0.003 0.005 0.01 0.02 0.03 0.05 0.1 0.2"}
ALIGN="alignatt_mad_f2 alignatt_mad_f4 alignatt_mad_f6 alignatt_mad_f8"
B="$(for t in $TH; do printf "qwenseg_th%s " $t; done)$ALIGN"
mkdir -p $L $F/baselines
[ -e $F/cache ] || ln -s ../full/cache $F/cache
ts () { date '+%F %T'; }

# GPU 를 쓰는 선행 작업(ACL AST 런)이 끝날 때까지 기다린다. 마커는 tmux 세션 존재 여부가
# 아니라 로그의 완료 줄이다 — 세션이 죽어도 로그는 남는다.
WAIT_LOG=${WAIT_LOG:-}
if [ -n "$WAIT_LOG" ]; then
  echo "== $(ts) 선행 런 대기: $WAIT_LOG"
  while ! grep -q "결과:" "$WAIT_LOG" 2>/dev/null; do sleep 60; done
  echo "== $(ts) 선행 런 완료 확인"
  sleep 30
fi

# AlignAtt 라벨은 full 런에 있다. 같은 문장 집합의 부분집합이라 그대로 읽는다.
for p in $ALIGN; do for t in de ja zh; do
  ln -sf ../../full/baselines/${p}_${t}_test.json $F/baselines/${p}_${t}_test.json
done; done

echo "===== 분절 $(ts) ====="
$PY -u -m core.meaning_segmentator.autoseg.gates.qwen_seg_baselines \
  --src-run $SRC --run-id $RUN --model "$MODEL" --thresholds $TH > $L/baselines.log 2>&1
rc=$?; echo "  분절 exit=$rc $(ts)"; grep "조각" $L/baselines.log
[ $rc -eq 0 ] || exit 1

for t in de ja zh; do
  echo "===== bleu_eval $t $(ts) ====="
  $PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
    --run-id $RUN --label none_use_manifest --split test \
    --dataset covost2 --manifest-tag n3000 --targets $t \
    --t-grid 2 3 4 6 --src-spaced 1 --wordtimes qwen \
    --translate-engine local --local-mt-model google/madlad400-3b-mt --mt-batch 48 \
    --workers 24 --baselines $B --baselines-native $B \
    --bootstrap 0 --no-sentence-bleu --no-auto-greedy \
    > $L/bleu_eval_$t.log 2>&1
  rc=$?; echo "  $t exit=$rc $(ts)"
  [ $rc -eq 0 ] || exit 1
done

echo "===== COMET $(ts) ====="
$PY -u -m core.meaning_segmentator.autoseg.baselines.comet_score \
  --run-id $RUN --dataset covost2 --manifest-tag n3000 \
  --label none_use_manifest --split test --targets de ja zh --only-missing \
  --model Unbabel/wmt22-comet-da --batch-size 32 > $L/comet.log 2>&1
rc=$?; echo "  COMET exit=$rc $(ts)"; grep "조건 채점" $L/comet.log
[ $rc -eq 0 ] || exit 1

echo "===== 등지연 표 $(ts) ====="
$PY core/meaning_segmentator/tools/covost2_chain/en2x_table.py \
  --run-id $RUN --targets zh de ja --metric comet \
  --row 'alignatt_mad_f:AlignAtt~\cite{papi-2023}' \
  --row 'qwenseg_th:Decoder \texttt{SEG} threshold (Ours)' \
  --outside unsegmented --outside-first 2>&1 | tail -30
touch $F/n3000_table.done
echo "===== 완료 $(ts) ====="
