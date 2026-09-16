#!/bin/bash
# SASST 네이티브 노브 스윕 — `max_chunk` (원논문의 "maximum span of seven tokens").
#   지금 곡선은 max_chunk=7 라벨 한 벌을 coarsen 으로 쓸어 만든 것이라 축이 우리 T 다.
#   원논문 노브로 지연을 만들면 축이 그쪽 것이 된다.
#   spaCy 는 GPU 를 안 쓴다 (prefer_gpu 호출 없음) — judge 루프와 자원이 안 겹친다.
#   tmux new-session -d -s sweep-syntax -c <저장소> "bash core/meaning_segmentator/tools/covost2_chain/22_native_syntax.sh"
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
export PYTHONPATH=.
PY=${PY:-.venv-autoseg/bin/python}
RUN=en2x/covost2/full
F=core/meaning_segmentator/experiment/artifacts/$RUN
LOG=$F/logs/baselines/native_syntax.log
mkdir -p "$(dirname "$LOG")"
for mc in 3 5 10; do
  echo "== $(date '+%F %T') syntax max_chunk=$mc start" >> $LOG
  $PY -u -m core.meaning_segmentator.autoseg.baselines.build \
    --run-id $RUN --dataset covost2 --manifest-tag full \
    --policy syntax --max-chunk $mc --out-name syntax_mc$mc --resume \
    >> $F/logs/baselines/syntax_mc$mc.log 2>&1
  echo "== $(date '+%F %T') syntax max_chunk=$mc exit=$?" >> $LOG
done
echo "== $(date '+%F %T') ALL DONE" >> $LOG
touch $F/baselines/native_syntax.done
