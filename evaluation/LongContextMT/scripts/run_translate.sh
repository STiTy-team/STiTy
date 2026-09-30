#!/bin/bash
# 모델 하나의 1.1 번역을 끝까지 돌리고 완료 표시를 남긴다. tmux 세션 하나에 모델 하나.
#   tmux new-session -d -s lcmt-api   -c "$PWD" "bash evaluation/LongContextMT/scripts/run_translate.sh gpt-6-luna --parallel-chains"
#   tmux new-session -d -s lcmt-local -c "$PWD" "bash evaluation/LongContextMT/scripts/run_translate.sh qwen3.5-4b-bf16"
set -euo pipefail
model=$1
shift
root=evaluation/LongContextMT
marker=$root/logs/markers/translate_$model.done
[ -f "$marker" ] && { echo "already done: $marker"; exit 0; }
export PYTHONPATH=$PWD
export PYTORCH_ALLOC_CONF=expandable_segments:True
if [ "$model" = qwen3.5-4b-bf16 ]; then
  SESSION=lcmt-local bash evaluation/DialogueContext/scripts/gpu_sampler.sh $root/logs/gpu_samples.csv &
fi
python -u $root/scripts/run_translate.py --model "$model" "$@" >> "$root/logs/translate_$model.log" 2>&1 \
  || echo "run_translate exited $? — checking completeness" >> "$root/logs/translate_$model.log"
# 종료 코드가 아니라 결과가 다 찼는지로 완료를 판단한다. 로컬 실행은 번역을 다 마치고도 종료 코드가 0 이 아니었다.
python $root/scripts/check_complete.py "$model" && touch "$marker"
