#!/bin/bash
# judge44 의 v0 와 5스텝 채택본을 CoVoST2 full 15,530 에서 평가한다 — BLEU·chrF·LAAL·COMET.
#
# **지연 노브는 점수 임계값이다** (`--score-grid`). T 격자는 문장 길이가 절단 수를 정하는데
# (`k = 어절/T`) 그 문장에 좋은 자리가 있는지와 무관하다. 임계값은 점수가 정하고, 올리면
# 무분절로 수렴해 곡선의 끝점이 된다. 우리 프롬프트의 `auto_T*` 는 `--no-auto-t` 로 끈다.
#
# **`--t-grid` 는 그래도 준다** — 비교군(punct·alignatt·mu_prefix·causal_align·syntax)의 지연
# 노브가 그것이다. 비우면 비교군이 한 점으로 줄어 곡선이 사라진다. 두 노브는 x 축을 실측 LAAL
# 로 두면 같은 그림에 놓인다(`laal_words`·`laal_ms` 가 이미 계산된다).
#
# 선행: ja 오라클 라벨이 끝나야 GPU 가 빈다 — **로그 grep 이 아니라 마커 파일**로 기다린다
# (오늘 체인 로그 한 줄에 기댔다가 두 번 물렸다: 대기 줄이 자기 패턴에 걸려 즉시 통과했고,
#  다음엔 완료 줄이 안 찍혀 한 시간을 헛기다렸다).
#
# 비용: **API $0.** 번역은 로컬 madlad, COMET 도 로컬 GPU 다.
# 시간: judge13 이 같은 코퍼스에서 비교군 5종까지 3타깃 × 2프롬프트를 48분에 끝냈다
#       (번역 캐시를 ../full 에서 물려받았다). 우리 조건은 점수 격자 4개뿐이고 그 조각의 77%가
#       T 격자 조각과 문자열이 겹쳐(3,000문장 실측) 번역 캐시에 걸린다 — 1시간 안팎을 본다.
#
#   tmux new-session -d -s j44eval -c <저장소> "bash core/meaning_segmentator/tools/covost2_chain/14b_judge44_eval.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LANGSMITH_TRACING=0
PY=${PY:-.venv-autoseg/bin/python}
A=core/meaning_segmentator/experiment/artifacts
SRC=$A/en2x/covost2/full_judge44          # 라벨이 있는 곳
FULL=$A/en2x/covost2/full                 # 캐시·비교군·prompt_eval 을 물려받을 곳
WAIT=${WAIT:-$A/x2en/ja-multi/run04/labels.done}
TGRID="${TGRID:-2 3 4 6}"
SGRID="${SGRID:-20 40 60 80}"
BASE="${BASE:-punct alignatt mu_prefix causal_align syntax}"
LOG=$SRC/logs/eval.log
ts () { date '+%F %T'; }

echo "== $(ts) eval 대기: $WAIT" >> $LOG
until [ -f "$WAIT" ]; do sleep 30; done
echo "== $(ts) 선행 완료 확인 — 시작" >> $LOG

one () {   # <run 디렉토리 이름> <라벨 이름> <라벨 jsonl>
  local rid=en2x/covost2/$1 label=$2 src=$3 D=$A/en2x/covost2/$1
  mkdir -p $D/logs $D/bleu
  # 번역 캐시·비교군·prompt_eval 은 full 과 공유한다 — 비교군 조각은 이미 번역돼 있다.
  for d in cache baselines prompt_eval; do [ -e $D/$d ] || ln -s ../full/$d $D/$d; done
  if [ ! -s $FULL/prompt_eval/${label}_test.json ]; then
    echo "== $(ts) $1 변환 (tag-order score, min-gap 1)" >> $LOG
    # **점수제다.** 기본값 rank 로 돌리면 `<SEG:1>` 을 최고 확신으로 읽어 절단이 거꾸로 돈다.
    $PY -u core/meaning_segmentator/tools/covost2_label/to_prompt_eval.py \
      --labels $src --run-id $rid --label $label --split test \
      --min-gap 1 --t-grid $TGRID --tag-order score --spaced yes \
      >> $D/logs/to_prompt_eval.log 2>&1 || { echo "변환 FAIL" >> $LOG; return 1; }
  fi
  for t in zh de ja; do
    [ -s $D/bleu/$t.json ] && { echo "== $(ts) $1 $t skip (있음)" >> $LOG; continue; }
    echo "== $(ts) $1 bleu_eval $t" >> $LOG
    $PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
      --run-id $rid --label $label --split test \
      --dataset covost2 --manifest-tag full --targets $t \
      --t-grid $TGRID --score-grid $SGRID --no-auto-t --src-spaced 1 \
      --translate-engine local --local-mt-model google/madlad400-3b-mt --mt-batch 48 \
      --workers 24 --baselines $BASE --bootstrap 0 --no-sentence-bleu --no-auto-greedy \
      > $D/logs/bleu_eval_$t.log 2>&1
    echo "== $(ts) $1 bleu_eval $t exit=$?" >> $LOG
  done
  echo "== $(ts) $1 comet" >> $LOG
  $PY -u -m core.meaning_segmentator.autoseg.baselines.comet_score \
    --run-id $rid --dataset covost2 --manifest-tag full --src en \
    --label $label --split test --targets zh de ja --only-missing \
    --model Unbabel/wmt22-comet-da --batch-size 32 > $D/logs/comet.log 2>&1
  echo "== $(ts) $1 comet exit=$?" >> $LOG
  date '+%F %T' >> $D/eval.done
}

one full_j44v0   auto_j44v0   $SRC/labels/v0.jsonl   || exit 1
one full_j44best auto_j44best $SRC/labels/best.jsonl || exit 1
echo "== $(ts) ALL DONE" >> $LOG
