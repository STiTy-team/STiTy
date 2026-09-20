#!/bin/bash
# judge-de 가 끝나면 ja·zh judge 루프를 **동시에** 띄운다.
#
#   tmux new-session -d -s judge-queue -c <저장소> "bash core/meaning_segmentator/tools/autoseg_x2en/queue_ja_zh.sh"
#
# 대기는 tmux 세션 유무로 본다 — PID 로 기다리면 세션이 죽을 때 의미가 없어진다.
# 설정은 de(judge01)와 같게 준다. 세 언어를 같은 자로 읽으려면 EXTRA 를 바꾸지 말 것.
# GPU 는 런당 madlad-3b + CometKiwi 로 7~9.5 GB 다. 둘이면 19 GB 안쪽 — 24 GB 에 들어간다.
# 셋이 겹치면 터진다 (2026-09-16 22:21 de OOM: 9.39 + 6.97 + 6.97 GB).
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
WAIT_FOR=${WAIT_FOR:-judge-de}
RUN=${RUN:-judge01}
BUDGET=${BUDGET:-60}
EXTRA=${EXTRA:---pe-candidates 4 --screen-n 50 --screen-skip --labeled-examples --candidate-roles free,free,examples_only --confirm-dev-b}
LOG=core/meaning_segmentator/experiment/artifacts/x2en/logs/queue_ja_zh.log
mkdir -p "$(dirname "$LOG")"

echo "== $(date '+%F %T') 대기 시작 — $WAIT_FOR 세션이 사라지면 ja·zh 를 띄운다" >> "$LOG"
while tmux has-session -t "$WAIT_FOR" 2>/dev/null; do sleep 60; done
echo "== $(date '+%F %T') $WAIT_FOR 종료 확인" >> "$LOG"

for l in ja zh; do
  tmux new-session -d -s "judge-$l" -c "$PWD" \
    "SRC=$l RUN=$RUN BUDGET=$BUDGET EXTRA='$EXTRA' bash core/meaning_segmentator/tools/autoseg_x2en/run_judge.sh"
  echo "== $(date '+%F %T') judge-$l 시작 (x2en/${l}-multi/$RUN, budget \$$BUDGET)" >> "$LOG"
  sleep 5
done
echo "== $(date '+%F %T') 큐 끝 — 로그는 ${RUN}_ja.log / ${RUN}_zh.log" >> "$LOG"
