#!/bin/bash
# sol 정답 생성. GPU 를 쓰지 않는다.
#   tmux new-session -d -s lcmt-ref -c "$PWD" "bash evaluation/LongContextMT/scripts/run_ref_sol.sh"
set -uo pipefail
root=evaluation/LongContextMT
# core 를 다른 사본에서 읽어야 할 때(작업 트리가 merge 도중 등) CORE_PATH 로 넘긴다.
export PYTHONPATH=${CORE_PATH:-$PWD}
python -u $root/scripts/build_ref_sol.py "$@" >> $root/logs/ref_sol.log 2>&1 \
  && touch $root/logs/markers/ref_sol.done || echo "build_ref_sol FAILED" >> $root/logs/ref_sol.log
