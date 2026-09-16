#!/bin/bash
# covost2/full 비교군 5종을 새 매니페스트(15530)에 맞춰 증분 재생성한다.
#   기존 최종 json(15430행)을 .partial.jsonl 로 흘려 넣고 build --resume 으로 없는 100문장만 돈다.
#   --label 은 주지 않는다: 기본값 auto_best 가 covost2/full 에 없어 매니페스트에서 읽는다
#   (auto_run13_mg1 을 주면 15430 으로 돌아가 todo 0).
#   tmux new-session -d -s covost2-baselines -c <저장소> "bash core/meaning_segmentator/tools/covost2_label/rebuild_baselines_full.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-.venv-autoseg/bin/python}
RUN=en2x/covost2/full
F=core/meaning_segmentator/experiment/artifacts/$RUN
LOGD=$F/logs/baselines
mkdir -p "$LOGD"
LOG=$LOGD/rebuild_15530.log
echo "== $(date '+%F %T') seed partial" >> "$LOG"
for f in "$F"/baselines/*_test.json; do
  p="${f%.json}.partial.jsonl"
  [ -s "$p" ] && continue
  $PY -c "
import json; d=json.load(open('$f'))
open('$p','w').write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in d['rows']))
print('$p', len(d['rows']))" >> "$LOG" 2>&1 || { echo "seed FAIL $f" >> "$LOG"; exit 1; }
done
B="$PY -u -m core.meaning_segmentator.autoseg.baselines.build --run-id $RUN --dataset covost2 --manifest-tag full --resume"
run() { local pol=$1; shift
  echo "== $(date '+%F %T') $pol start" >> "$LOG"
  $B --policy "$pol" "$@" >> "$LOGD/${pol}_15530.log" 2>&1 || { echo "== $(date '+%F %T') $pol FAIL" >> "$LOG"; return 1; }
  echo "== $(date '+%F %T') $pol done" >> "$LOG"
}
run punct
run syntax
run causal_align --targets de ja zh
run mu_prefix    --targets de ja zh
run alignatt     --targets de ja zh
echo "== $(date '+%F %T') ALL DONE" >> "$LOG"
touch "$F/baselines/rebuild_15530.done"
