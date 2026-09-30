#!/bin/bash
# judge02 세 언어를 띄운다 — de 를 먼저 단독으로, 성하면 ja·zh 를 동시에.
#
#   tmux new-session -d -s judge02-queue -c <저장소> \
#       "bash core/meaning_segmentator/tools/autoseg_x2en/judge02/queue.sh"
#
# 왜 이 순서인가
#   GPU 가 4090 한 장(24GB)이고 런 하나가 madlad-3b + CometKiwi 로 7~9.5GB 를 쓴다. 셋이 겹치면
#   터진다 (2026-09-16 22:21 de OOM: 9.39 + 6.97 + 6.97GB). 그래서 최대 둘이다.
#   de 를 먼저 단독으로 돌린다. 이 구성에서 처음 하는 일이 --generate-v0 인데, Writer 가
#   severity 역할이 먹을 "질문 + 정도 절" 줄 모양을 못 내면 그 역할이 이터마다 논다. de 하나로
#   그것을 먼저 보고, 성하면 ja·zh 를 동시에 띄워 벽시계 시간을 줄인다.
#
# de 가 0 이 아닌 코드로 끝나면 ja·zh 를 띄우지 않는다. 골격 검사 실패나 예산 초과는 사람이
# 보고 정해야 하는 일이고, 그대로 두면 같은 문제를 두 언어가 $115 어치 반복한다.
#
# WAIT_FOR 에 tmux 세션 이름을 주면 그 세션이 사라질 때까지 기다렸다 시작한다(GPU 를 쓰는 다른
# 작업 뒤에 붙일 때). 대기는 세션 유무로 본다 — PID 로 기다리면 세션이 죽을 때 의미가 없어진다.
#
# 비용 집계는 런이 끝난 뒤
#   .venv-autoseg/bin/python -m core.meaning_segmentator.autoseg.cost_report \
#       --run-id x2en/<L>-multi/judge02 --budget 100
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
WAIT_FOR=${WAIT_FOR:-}
RUN=${RUN:-judge02}
BUDGET=${BUDGET:-100}
ITER=${ITER:-2}
LOG=core/meaning_segmentator/experiment/artifacts/x2en/logs/${RUN}_queue.log
mkdir -p "$(dirname "$LOG")"
say () { echo "== $(date '+%F %T') $*" >> "$LOG"; }

if [ -n "$WAIT_FOR" ]; then
  say "대기 시작 — $WAIT_FOR 세션이 사라지면 시작한다"
  while tmux has-session -t "$WAIT_FOR" 2>/dev/null; do sleep 60; done
  say "$WAIT_FOR 종료 확인"
fi

# 앞 작업이 서버를 남기고 죽었을 수 있다. 메모리가 실제로 비는 것을 보고 시작한다.
for _ in $(seq 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${used:-99999}" -lt 6000 ] && break
  sleep 60
done
say "GPU 여유 확인 — 사용 중 ${used}MiB"

say "de 시작 (x2en/de-multi/$RUN, 이터 $ITER, 예산 \$$BUDGET)"
SRC=de RUN=$RUN BUDGET=$BUDGET ITER=$ITER \
    bash core/meaning_segmentator/tools/autoseg_x2en/judge02/run.sh
rc=$?
say "de 종료 exit=$rc"
if [ "$rc" -ne 0 ]; then
  say "de 가 0 이 아닌 코드로 끝났다 — ja·zh 는 띄우지 않는다. ${RUN}_de.log 를 보고 정한다"
  exit "$rc"
fi

for l in ja zh; do
  tmux new-session -d -s "$RUN-$l" -c "$PWD" \
    "SRC=$l RUN=$RUN BUDGET=$BUDGET ITER=$ITER bash core/meaning_segmentator/tools/autoseg_x2en/judge02/run.sh"
  say "$RUN-$l 시작 (x2en/${l}-multi/$RUN)"
  sleep 5
done
say "큐 끝 — 로그는 ${RUN}_ja.log / ${RUN}_zh.log"
