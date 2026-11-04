#!/usr/bin/env bash
# DialogueContext 전체 체인: main 번역 → (S1 번역 ∥ main 채점) → 전체 채점 → 집계.
# 저장소 루트에서 tmux 로 띄운다 (claude 세션이 끝나도 살아 있게):
#   tmux new-session -d -s dctx -c /home/skkai/stity-langswitch/STiTy "bash evaluation/DialogueContext/scripts/run_all.sh"
# 진행: tail -f evaluation/DialogueContext/logs/run_all.log  (단계별 로그는 logs/*.log)
# 다시 띄우면 끝난 단계(logs/markers/*.done)는 건너뛰고, 단계 안에서도 끝난 job_id 는 건너뛴다.
#
# 순서와 이유
#   1) main: 전 모델·전 조건 1,040개를 한 프로세스에서 섞어 차례로 돈다(동시성 1). 지연을 재므로
#      이 단계에는 다른 번역·채점 프로세스를 겹치지 않는다.
#   2) main 이 끝나면 S1 번역과 main 채점을 겹쳐 돌린다.
#      - S1 은 모델마다 프로세스를 따로 띄운다. S1 행은 모두 latency_valid=false 로 남는다(지연 분석 대상 아님).
#        로컬 모델 둘은 GPU 여유가 되면 따로, 모자라면 한 프로세스에서 차례로 돈다.
#      - 판정(judge, API 만 씀)은 바로 시작한다.
#      - 참조 채점(COMET·XCOMET, GPU 약 14GB)은 S1 로컬 모델이 다 올라간 뒤 GPU 여유가
#        SCORE_NEED_MIB 이상이면 곧바로, 아니면 S1 이 끝난 뒤에 시작한다.
#   3) S1 과 main 채점이 끝나면 채점 둘을 한 번 더 돌린다(캐시가 있어 S1 행만 새로 채점된다).
#   4) aggregate.py 로 표와 그래프를 만든다.
#
# 환경 변수
#   CONFIG=...          설정 파일 (기본 configs/experiment.yml)
#   S1_PARALLEL=auto    auto | yes | no  (no: S1 전체를 한 프로세스에서 순차로)
#   SCORING=yes         no 면 번역만 하고 끝낸다
#   JUDGE=yes           no 면 판정(유료 API)을 건너뛰고 참조 채점만 한다
#   JUDGE_BUDGET=12     판정 예산(USD). judge_context.py --budget-usd 로 넘긴다
#   RESULTS_DIR=...     결과 기준 디렉터리를 바꾼다(시험용). 결과는 <RESULTS_DIR>/<run_id>/
#   TRANSLATE_ARGS=...  run_translations.py 에 더 넘길 인자(시험용, 예: "--limit-instances 1")
REPO=/home/skkai/stity-langswitch/STiTy
W=$REPO/evaluation/DialogueContext
CONFIG=${CONFIG:-$W/configs/experiment.yml}
S1_PARALLEL=${S1_PARALLEL:-auto}
SCORING=${SCORING:-yes}
JUDGE=${JUDGE:-yes}
JUDGE_BUDGET=${JUDGE_BUDGET:-12}
NEED_MIB_PER_LOCAL=6000   # 4B 4bit 하나가 올라가 돌 때 잡는 양(실측 약 3~4.5 GiB)에 여유를 더한 값
MARGIN_MIB=3000
SCORE_NEED_MIB=16000      # XCOMET-XL 채점이 잡는 양(약 14 GB)에 여유를 더한 값

LOGS=${LOGS_DIR:-$W/logs}   # LOGS_DIR 로 바꿀 수 있다(시험용)
M=$LOGS/markers
mkdir -p $M
cd $REPO
export PYTHONPATH=$REPO
export PYTORCH_ALLOC_CONF=expandable_segments:True
# API 키(OPENAI_API_KEY, DEEPL_API_KEY, HF_TOKEN)는 각 파이썬 스크립트가 .env 에서 직접 읽는다.
# .env 는 'KEY = value' 꼴이라 셸에서 source 할 수 없다.
cfg() { sed -n "s/^ *$1: *//p" $CONFIG | head -1 | sed 's/ *#.*//'; }
TF5=$(cfg venv-tf5)/bin/python
METRICS=$(cfg venv-metrics)/bin/python
RESULTS_DIR=${RESULTS_DIR:-$REPO/$(cfg results_dir)}
RUN_DIR=$RESULTS_DIR/$(cfg run_id)
RUN="$TF5 -u $W/scripts/run_translations.py --config $CONFIG --results-dir $RESULTS_DIR $TRANSLATE_ARGS"

gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
log() { echo "=== $(date '+%F %T') $* (gpu_used_mib=$(gpu_used))" | tee -a $LOGS/run_all.log; }
models_of() {  # models_of local|api
  $TF5 -c "import yaml; c = yaml.safe_load(open('$CONFIG'))
print(' '.join(k for k, v in c['models'].items() if (v['kind'] == 'local') == ('$1' == 'local')))"
}
step() {  # step <marker> <log> <command...>: 마커가 없을 때만 돌리고, 성공하면 마커를 남긴다
  local mark=$1 out=$2; shift 2
  [ -f $M/$mark.done ] && return 0
  rm -f $M/$mark.failed
  log "$mark start"
  if "$@" >> $LOGS/$out.log 2>&1; then
    touch $M/$mark.done; log "$mark done"
  else
    touch $M/$mark.failed; log "$mark FAILED (see logs/$out.log)"; return 1
  fi
}
xcomet_status() { $TF5 -c "import json; print(json.load(open('$RUN_DIR/xcomet_status.json')).get('status'))" 2>/dev/null; }
score_ref() {  # score_ref <marker>: XCOMET 이 GPU 부족(다른 세션)으로 실패하면 비워질 때까지 기다렸다 다시 한다
  local mark=$1 try
  for try in 1 2 3 4 5 6; do
    step $mark score_reference $METRICS -u $W/scripts/score_reference.py --run-dir $RUN_DIR || return 1
    [ "$(xcomet_status)" = failed ] || return 0
    rm -f $M/$mark.done
    log "$mark: XCOMET failed (see xcomet_status.json); waiting for ${SCORE_NEED_MIB} MiB free GPU, try $try/6"
    timeout 3600 bash -c "until [ \$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) -ge $SCORE_NEED_MIB ]; do sleep 30; done"
  done
  touch $M/$mark.failed; log "$mark: XCOMET still failing; rerun later (COMET and lexical scores are saved)"
  return 1
}
judge() { [ "$JUDGE" = yes ] || return 0; step $1 judge $TF5 -u $W/scripts/judge_context.py --run-dir $RUN_DIR --budget-usd $JUDGE_BUDGET; }

log "run_all start; run dir $RUN_DIR"

# 1) main
step main translate_main $RUN --phase main || exit 1

# 2) S1 ∥ main 채점
[ "$SCORING" = yes ] && { judge judge_main & }

s1_pids=()
s1_labels=()
s1_launch() {  # s1_launch <label> <models,...> [--parallel]
  local label=$1 models=$2; shift 2
  s1_labels+=($label)
  echo "=== launch $(date '+%F %T')" >> $LOGS/translate_s1_$label.log
  step s1_$label translate_s1_$label $RUN --phase s1 --models $models "$@" &
  s1_pids+=($!)
}
LOCAL=$(models_of local)
API=$(models_of api)
if [ ! -f $M/s1.done ]; then
  n_local=$(echo $LOCAL | wc -w)
  need=$((NEED_MIB_PER_LOCAL * n_local + MARGIN_MIB))
  free=$(gpu_free)
  mode=$S1_PARALLEL
  if [ "$mode" = auto ]; then
    if [ "$free" -ge "$need" ]; then mode=yes; else mode=no; fi
  fi
  log "s1 mode=$mode (gpu free ${free} MiB, need ${need} MiB to run $n_local local models apart)"
  if [ "$mode" = no ]; then
    s1_launch all $(echo $LOCAL $API | tr ' ' ',') --parallel   # 채점과 겹칠 수 있어 지연은 무효
  else
    for m in $LOCAL; do s1_launch $m $m --parallel; sleep 5; done
    for m in $API; do s1_launch $m $m --parallel; done
  fi
fi

if [ "$SCORING" = yes ] && [ ! -f $M/score_ref_main.done ]; then
  # 로컬 S1 프로세스가 모델을 다 올리고 워밍업을 마칠 때까지 기다린다("[s1] N chains" 줄)
  for label in "${s1_labels[@]}"; do
    case " $API " in *" $label "*) continue ;; esac
    until awk '/^=== launch/{f=0} /^\[s1\]/{f=1} END{exit !f}' $LOGS/translate_s1_$label.log \
          || [ -f $M/s1_$label.done ] || [ -f $M/s1_$label.failed ]; do sleep 10; done
  done
  free=$(gpu_free)
  if [ "$free" -ge "$SCORE_NEED_MIB" ]; then
    log "reference scoring of main starts alongside s1 (gpu free ${free} MiB)"
  else
    log "gpu free ${free} MiB < ${SCORE_NEED_MIB}; reference scoring waits for s1"
    for pid in "${s1_pids[@]}"; do wait $pid; done
  fi
  score_ref score_ref_main &
fi

for pid in "${s1_pids[@]}"; do wait $pid; done
if [ ! -f $M/s1.done ]; then
  ok=1
  for label in "${s1_labels[@]}"; do [ -f $M/s1_$label.done ] || ok=0; done
  [ $ok = 1 ] && touch $M/s1.done && log "s1 done"
fi
wait   # main 채점이 아직 돌고 있으면 끝날 때까지

[ -f $M/s1.done ] || { log "s1 has failures; rerun to resume (scoring of s1 skipped)"; exit 1; }
[ "$SCORING" = yes ] || { log "translation chain finished (scoring disabled)"; exit 0; }

# 3) S1 행까지 채점 (캐시에 없는 것만 새로 돈다)
score_ref score_ref_all
judge judge_all
# 4) 집계
step aggregate aggregate $METRICS -u $W/scripts/aggregate.py --run-dir $RUN_DIR
log "run_all finished"
