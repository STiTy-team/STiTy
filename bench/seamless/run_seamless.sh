#!/bin/bash
# SeamlessStreaming S2S + ASR on the same volume-normalized FLEURS audio as the bench
# runs, scored with bench's metric code. See README.md next to this file.
#   tmux new-session -d -s seamless -c <repo> "bash bench/seamless/run_seamless.sh > logs/chain_seamless.log 2>&1"
export PYTHONUNBUFFERED=1
export PATH=$HOME/.local/bin:$PATH
S=$(cd "$(dirname "$0")" && pwd)
BENCH=$(cd "$S/../.." && pwd)
W=${SEAMLESS_WORK:-$BENCH/bench/seamless/work}
mkdir -p $W
say() { echo "[$(date '+%F %T')] $*"; }

say "start; GPU used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

source ~/miniforge3/etc/profile.d/conda.sh
seam() { (conda activate asr-seamless && python -u $S/seamless_run.py "$@"); }

check() {  # check <dir> <task>: all ok and text present in the first-10 run
  python3 - "$1/$2.jsonl" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
bad = [r["utt_id"] for r in rows if not r.get("ok") or not (r.get("text") or "").strip()]
print(f"CHECK {sys.argv[1]}: {len(rows)} rows, {len(bad)} failed or empty {bad[:3]}")
for r in rows[:2]:
    print("   ", (r.get("text") or r.get("error") or "")[:120])
sys.exit(1 if len(bad) > len(rows) // 2 else 0)
EOF
}

for P in en-ko ko-en; do
  M=$W/export/$P/manifest.jsonl
  [[ -f $M ]] || (cd $BENCH && set -a && . ./.env && set +a && \
    PYTHONPATH=. bench/.venv/bin/python $S/export_items.py --dataset fleurs_${P}_vol23 --out $W/export/$P)
  for T in s2s asr; do
    say "RUN $P $T first 10"
    seam --manifest $M --out-dir $W/check10/$P --task $T --limit 10 > $W/check10.$P.$T.log 2>&1
    if ! check $W/check10/$P $T; then
      say "SKIP $P $T: 10-item check failed"; tail -20 $W/check10.$P.$T.log
      touch $W/$P.$T.check-failed; continue
    fi
    say "RUN $P $T full"
    seam --manifest $M --out-dir $W/out/$P --task $T > $W/out.$P.$T.log 2>&1
    say "END $P $T rc=$? rows=$(wc -l < $W/out/$P/$T.jsonl)"
  done
  if [[ -f $W/out/$P/s2s.jsonl && -f $W/out/$P/asr.jsonl ]]; then
    say "SCORE $P"
    (cd $BENCH && set -a && . ./.env && set +a && \
      PYTHONPATH=. bench/.venv/bin/python $S/seamless_score.py --export $W/export/$P \
        --seamless $W/out/$P --run-dir $W/runs/$P > $W/score.$P.log 2>&1 && \
      uv run --project bench/metrics/comet python -m bench.metrics.comet --run-dir $W/runs/$P >> $W/score.$P.log 2>&1)
    say "SCORED $P rc=$?"; tail -3 $W/score.$P.log
    mkdir -p $S/results && cp $W/runs/$P/summary.json $S/results/$P.json
  fi
done
say "GPU used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
say "CHAIN_DONE"
