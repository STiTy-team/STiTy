#!/bin/bash
# 실험 중 GPU 를 같이 쓰는 다른 프로세스를 5초마다 기록한다. 로컬 모델 지연 오염 구간을 가리는 데 쓴다.
out=${1:-evaluation/DialogueContext/logs/gpu_samples.csv}
echo "time,pid,used_mib,gpu_util,cmd" >> "$out"
while tmux has-session -t "=${SESSION:-dctx}" 2>/dev/null; do
  t=$(date +%FT%T); u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | while IFS=, read p m; do
    echo "$t,$p,$m,$u,\"$(ps -o cmd= -p $p 2>/dev/null | cut -c1-80)\"" >> "$out"; done
  sleep 5
done
