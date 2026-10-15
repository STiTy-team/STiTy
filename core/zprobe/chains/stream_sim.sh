#!/bin/bash
# Streaming re-translation stability of big2: transcription (en-en) vs same-order (en-zh) vs different-order (en-ko, en-ja,
# ko-en) targets, 100 FLEURS test utterances each, 1 s audio steps. Starts after bigcv2's SFT is done.
#   tmux new-session -d -s stream-sim -c /home/mobility/STiTy "bash core/zprobe/chains/stream_sim.sh > core/zprobe/chains/stream_sim.log 2>&1"
cd /home/mobility/STiTy
export PYTHONPATH=Qwen3-ASR
PY=.venv/bin/python; D=models/zprobe/data; S=$D/stream; mkdir -p $S
until grep -q "SFT_EXIT" core/zprobe/chains/bigcv2.log; do sleep 60; done
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 8000 ]; do sleep 60; done
echo "$(date +%T) MARK decode"
for p in en-en en-zh en-ko en-ja ko-en; do
  ids=$D/ev_big2_$p.jsonl
  [ -f $S/big2_$p.jsonl ] && [ "$(wc -l < $S/big2_$p.jsonl)" -eq 100 ] && continue
  PROBE_ADAPTER=models/zprobe/adapters/big2 $PY -u core/zprobe/stream_sim.py decode --pair $p --ids $ids --limit 100 --out $S/big2_$p.jsonl 2>&1 | grep -v -i -E "warn|pad_token"
done
echo "$(date +%T) MARK score"
$PY core/zprobe/stream_sim.py score --files $(for p in en-en en-zh en-ko en-ja ko-en; do echo -n "$p=$S/big2_$p.jsonl "; done) | tee models/zprobe/logs/stream_sim_big2.md
echo "$(date +%T) MARK CHAIN_DONE"
