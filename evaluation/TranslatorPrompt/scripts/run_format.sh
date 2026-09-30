#!/bin/bash
# 2.4 번역을 모델 하나에 대해 돌린다. tmux 세션 하나에 모델 하나.
#   tmux new-session -d -s fmt-api   -c "$PWD" "bash evaluation/TranslatorPrompt/scripts/run_format.sh gpt-6-luna"
#   tmux new-session -d -s fmt-local -c "$PWD" "bash evaluation/TranslatorPrompt/scripts/run_format.sh qwen3.5-4b-bf16"
set -uo pipefail
model=$1
root=evaluation/TranslatorPrompt
export PYTHONPATH=$PWD
export PYTORCH_ALLOC_CONF=expandable_segments:True
python -u $root/scripts/format_run.py --model "$model" >> "$root/logs/format_$model.log" 2>&1
grep -q "\[$model\] finished" "$root/logs/format_$model.log" && touch "$root/logs/markers/format_$model.done"
