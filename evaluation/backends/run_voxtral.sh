#!/bin/bash
# 이 박스에는 nvcc 가 없다. flashinfer 의 top-k/top-p 샘플링은 JIT 로 CUDA 커널을
# 빌드하려 해서 "Could not find nvcc" 로 EngineCore 가 죽는다(2026-09-09).
# 어텐션/컴파일 문제가 아니라 샘플러 문제이므로 샘플러만 끈다.
export VLLM_USE_FLASHINFER_SAMPLER=0
export LD_LIBRARY_PATH=/home/skkai/miniforge3/envs/asr-voxtral/lib
exec /home/skkai/miniforge3/bin/conda run --no-capture-output -n asr-voxtral \
  vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 \
  --tokenizer-mode mistral \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --compilation_config '{"cudagraph_mode":"PIECEWISE"}' \
  --port 8010
