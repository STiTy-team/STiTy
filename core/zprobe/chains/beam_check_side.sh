#!/bin/bash
# Side chain for a machine that lacks the n-best file and hit the missing nbest/ folder in beam_check.sh step 1:
# vLLM smoke + full beam 4, and the HF greedy / beam 8 offline decode (nbest.py, no sampling), while beam_check.sh
# runs the streaming step. Afterwards re-run only the COMET + report steps of beam_check.sh.
#   tmux new-session -d -s beamside -c <repo> "PY=... bash core/zprobe/chains/beam_check_side.sh > core/zprobe/chains/beam_check_side.log 2>&1"
cd "$(dirname "$0")/../../.."
PY=${PY:-.venv/bin/python}
BASE=models/Qwen3-ASR-1.7B-koen-seg-mix-merged; MERGED=models/zprobe/merged/segconn2
D=models/zprobe/data; V=$D/nbest; LOG=models/zprobe/logs; IDS=$D/ev_big2_en-ko.jsonl
mark() { echo "$(date +%T) MARK $1"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
waitgpu() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$1" ]; do sleep 60; done; }
export PYTHONPATH=Qwen3-ASR
mkdir -p $V
N=$(lines $IDS)
mark "1 vllm smoke"
waitgpu 10000
$PY -u core/zprobe/vllm_beam.py --model $MERGED --ids $IDS --out $V/vllm_smoke.jsonl --beams 4 --limit 5 > $LOG/vllm_beam_smoke.log 2>&1
if grep -q VLLM_BEAM_DONE $LOG/vllm_beam_smoke.log; then
  mark "2 vllm full"
  [ "$(lines $V/vllm_beam4_en-ko.jsonl)" -eq "$N" ] || $PY -u core/zprobe/vllm_beam.py --model $MERGED --ids $IDS --out $V/vllm_beam4_en-ko.jsonl --beams 4 > $LOG/vllm_beam.log 2>&1
  tail -3 $LOG/vllm_beam.log
else
  echo "vllm smoke failed"; grep -E "Error|Traceback" -A3 $LOG/vllm_beam_smoke.log | tail -20
fi
mark "3 hf greedy + beam 8 (no sampling)"
waitgpu 8000
[ "$(lines $V/segconn2_en-ko.jsonl)" -eq "$N" ] || PROBE_MODEL=$BASE PROBE_ADAPTER=models/zprobe/adapters/segconn2 $PY -u core/zprobe/nbest.py \
  --ids $IDS --tgt ko --beams 8 --samples 0 --out $V/segconn2_en-ko.jsonl 2>&1 | grep -v -i -E "warn|pad_token|generation flags" | tail -3
mark SIDE_DONE
