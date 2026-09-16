#!/bin/bash
# judge13 (iter3 채택본 / iter0 v0) 를 CoVoST2 full 15530 에서 비교군 5종과 함께 평가한다.
#   bleu_eval 은 <run-id>/bleu/<tgt>.json 을 통째로 덮어쓰므로 run13_mg1 결과(full/bleu, n=15430)를
#   보존하려고 별도 run-id (full_judge13, full_judge13v0) 를 쓴다. cache/baselines/prompt_eval 은
#   full 로 심링크 — 번역 캐시(비교군 prefix 는 이미 있음)를 그대로 재사용한다.
#   선행: baselines/rebuild_15530.done (비교군 15530), x2en de test 라벨링 종료(GPU 여유).
#   tmux new-session -d -s judge13-eval -c <저장소> "bash core/meaning_segmentator/tools/covost2_label/eval_judge13_full.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-.venv-autoseg/bin/python}
A=core/meaning_segmentator/experiment/artifacts
FULL=$A/en2x/covost2/full
GRID="2 3 4 6"
BASE="punct alignatt mu_prefix causal_align syntax"
LOG=$FULL/logs/eval_judge13.log
ts () { date '+%F %T'; }
echo "== $(ts) start" >> $LOG

while [ ! -f $FULL/baselines/rebuild_15530.done ]; do sleep 60; done
echo "== $(ts) baselines 15530 ok" >> $LOG
while ! grep -q ' test$' $A/x2en/de-multi/run03/labels.done 2>/dev/null; do sleep 60; done
echo "== $(ts) de test labels done, GPU free" >> $LOG

run_one () {   # <run-id suffix> <label> <title>
  local rid=en2x/covost2/$1 label=$2 title=$3 D=$A/en2x/covost2/$1
  mkdir -p $D/logs
  for d in cache baselines prompt_eval; do [ -e $D/$d ] || ln -s ../full/$d $D/$d; done
  for t in zh de ja; do
    # 재실행 대비: 타깃 결과가 이미 있으면 건너뛴다 (bleu_eval 은 타깃 끝에 bleu/<tgt>.json 을 쓴다).
    [ -s $D/bleu/$t.json ] && { echo "== $(ts) $1 bleu_eval $t skip (있음)" >> $LOG; continue; }
    echo "== $(ts) $1 bleu_eval $t" >> $LOG
    $PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
      --run-id $rid --label $label --split test \
      --dataset covost2 --manifest-tag full --targets $t \
      --t-grid $GRID --src-spaced 1 \
      --translate-engine local --local-mt-model google/madlad400-3b-mt --mt-batch 48 \
      --workers 24 --baselines $BASE --bootstrap 0 --no-sentence-bleu --no-auto-greedy \
      > $D/logs/bleu_eval_$t.log 2>&1
    echo "== $(ts) $1 bleu_eval $t exit=$?" >> $LOG
  done
  [ -f $D/eval.done ] && { echo "== $(ts) $1 이미 완료" >> $LOG; return 0; }
  echo "== $(ts) $1 comet" >> $LOG
  $PY -u -m core.meaning_segmentator.autoseg.baselines.comet_score \
    --run-id $rid --dataset covost2 --manifest-tag full --src en \
    --label $label --split test --targets zh de ja --only-missing \
    --model Unbabel/wmt22-comet-da --batch-size 32 > $D/logs/comet.log 2>&1
  echo "== $(ts) $1 comet exit=$?" >> $LOG
  $PY -m core.meaning_segmentator.autoseg.scoring.plot_tradeoff \
    --run-id $rid --targets zh de ja --metric comet \
    --out tradeoff_covost2_$1_comet --title "$title" --no-header >> $D/logs/plot.log 2>&1
  echo "== $(ts) $1 plot exit=$?" >> $LOG
  touch $D/eval.done
}

run_one full_judge13   auto_judge13_iter3 "CoVoST2 EN->X test 15530 judge13 iter3 (min_gap=1)"
run_one full_judge13v0 auto_judge13_iter0 "CoVoST2 EN->X test 15530 judge13 v0 (min_gap=1)"
echo "== $(ts) ALL DONE" >> $LOG
