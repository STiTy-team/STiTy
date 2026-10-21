#!/bin/bash
# Step 1 of "make the translation itself stream": no retraining, AlignAtt stopping on segconnx (stream_bench.py
# --mode alignatt): tokens are shown as they are decoded and never revised; decoding stops before a token that attends to
# the newest --att_sec of audio. Starts after chains/prefix.sh finishes (one GPU job at a time).
#   tmux new-session -d -s alignatt -c /home/mobility/STiTy "bash core/zprobe/chains/alignatt.sh >> core/zprobe/chains/alignatt.log 2>&1"
cd /home/mobility/STiTy
PY=${PY:-.venv/bin/python}
BENCH=${BENCH:-/home/mobility/STiTy-bench}; W=${W:-/home/mobility/zprobe_bench}
export PYTHONPATH=Qwen3-ASR PROBE_MODEL=models/Qwen3-ASR-1.7B-koen-seg-mix-merged PROBE_ATTN=eager
A=models/zprobe/adapters; M=${M:-segconnx}
mark() { echo "$(date '+%F %T') MARK $1"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
idle() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -le 2000 ]; do sleep 60; done; }
f() { grep --line-buffered -v -i -E "warn|pad_token|generation flags"; }
mark "wait for prefix chain"
until grep -q CHAIN_DONE core/zprobe/chains/prefix.log 2>/dev/null; do sleep 120; done
mark "smoke"
idle
PROBE_ADAPTER=$A/$M $PY -u core/zprobe/stream_bench.py --mode alignatt --manifest $W/export/en-ko/manifest.jsonl --tgt ko --limit 3 \
  --out $W/out/alignatt_smoke.jsonl 2>&1 | f | tail -3
python3 -c "
import json
for l in open('$W/out/alignatt_smoke.jsonl'): r=json.loads(l); print(r['duration_sec'], r['decodes'], [(s[0][:30], s[1]) for s in r['segments']])"
rm -f $W/out/alignatt_smoke.jsonl
for S in 0.5 1.0; do for P in en-ko ko-en; do
  X=$W/export/$P; N=zprobe-$M-alignatt${S}_$P; F=$W/out/$N.jsonl
  [ "$(lines $F)" -eq "$(lines $X/manifest.jsonl)" ] || { idle; mark "stream $N"; PROBE_ADAPTER=$A/$M $PY -u core/zprobe/stream_bench.py \
    --mode alignatt --att_sec $S --manifest $X/manifest.jsonl --tgt ${P#*-} --out $F 2>&1 | f | grep --line-buffered -E "DONE|Error|Traceback"; }
  grep -q '"comet"' $W/runs/$N/summary.json 2>/dev/null || (cd $BENCH && PYTHONPATH=.:bench/seamless bench/.venv/bin/python \
    /home/mobility/STiTy/core/zprobe/stream_bench_score.py --export $X --run $F --name fleurs_$P/$N --run-dir $W/runs/$N > /dev/null && \
    PATH=$HOME/.local/bin:$PATH uv run --project bench/metrics/comet python -m bench.metrics.comet --run-dir $W/runs/$N 2>&1 | tail -1)
done; done
mark "report"
{
echo "| run | COMET | YAAL CA ms | LAAL CA ms | updates / item | decodes / item | YAAL items |"; echo "|---|---|---|---|---|---|---|"
for N in $(ls $W/runs | grep -E "^zprobe-(prefix1|segconnx)-" | grep -E "en-ko|ko-en" | sort); do
  python3 -c "import json; s=json.load(open('$W/runs/$N/summary.json')); m=s['metrics']; c=s['counts']; g=lambda k: m.get(k) or 0; print(f\"| $N | {g('comet'):.3f} | {g('yaal_ca_ms'):.0f} | {g('laal_ca_ms'):.0f} | {g('segments_per_item'):.2f} | {g('decodes_per_item'):.1f} | {c['yaal_items']}/{c['items']} |\")"
done
} | tee models/zprobe/logs/alignatt_report.md
mark CHAIN_DONE
