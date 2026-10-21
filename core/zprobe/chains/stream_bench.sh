#!/bin/bash
# segconn2 direct speech translation on bench's FLEURS en-ko vol23 audio (the 5th-round streaming table), real time:
# own-<SEG> commits with agree 2, greedy and beam 4. Scored with bench's YAAL and COMET code, like bench/seamless.
#   tmux new-session -d -s streambench -c <repo> "bash core/zprobe/chains/stream_bench.sh > core/zprobe/chains/stream_bench.log 2>&1"
# Needs the STiTy-bench worktree (BENCH) and its export of fleurs_en-ko_vol23 (bench/seamless/export_items.py,
# STITY_DATA_ROOT pointing at the bench-format datasets).
cd /home/mobility/STiTy
PY=${PY:-.venv/bin/python}
BENCH=${BENCH:-/home/mobility/STiTy-bench}; W=${W:-/home/mobility/zprobe_bench}; X=$W/export/en-ko
export PYTHONPATH=Qwen3-ASR PROBE_MODEL=models/Qwen3-ASR-1.7B-koen-seg-mix-merged PROBE_ADAPTER=models/zprobe/adapters/segconn2
mark() { echo "$(date +%T) MARK $1"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
[ -f $X/manifest.jsonl ] || (cd $BENCH && STITY_DATA_ROOT=${STITY_DATA_ROOT:-/home/mobility/stity-datasets} PYTHONPATH=. \
  bench/.venv/bin/python bench/seamless/export_items.py --dataset fleurs_en-ko_vol23 --out $X)
N=$(lines $X/manifest.jsonl)
for B in 1 4; do
  NAME=zprobe-segconn2-agree2-beam$B; F=$W/out/$NAME.jsonl; mkdir -p $W/out
  mark "decode $NAME"
  # real-time run: needs the GPU to itself, or the latency is not comparable
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -le 2000 ]; do sleep 60; done
  [ "$(lines $F)" -eq "$N" ] || $PY -u core/zprobe/stream_bench.py --manifest $X/manifest.jsonl --tgt ko --agree 2 --beams $B --out $F \
    2>&1 | grep -v -i -E "warn|pad_token|generation flags"
  mark "score $NAME"
  (cd $BENCH && PYTHONPATH=.:bench/seamless bench/.venv/bin/python /home/mobility/STiTy/core/zprobe/stream_bench_score.py \
     --export $X --run $F --name fleurs_en-ko_vol23/$NAME --run-dir $W/runs/$NAME && \
   PATH=$HOME/.local/bin:$PATH uv run --project bench/metrics/comet python -m bench.metrics.comet --run-dir $W/runs/$NAME 2>&1 | tail -2)
  python3 -c "import json; m=json.load(open('$W/runs/$NAME/summary.json'))['metrics']; print({k: m.get(k) for k in ('comet','yaal_ms','yaal_ca_ms','segments_per_item','decodes_per_item')})"
done
mark CHAIN_DONE
