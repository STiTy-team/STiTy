#!/bin/bash
# One live brig run on northstar (ko dataset), then both targets scored from its records.
#   tmux new-session -d -s brigbench -c /home/mobility/STiTy "bash scripts/brig_bench.sh >> /tmp/brig_bench.log 2>&1"
cd /home/mobility/STiTy
trap 'echo "BRIG_EXIT $(date "+%F %T")"' EXIT
export STITY_DATA_ROOT=/home/mobility/stity-datasets
export PATH=$HOME/.local/bin:$PATH
CFG=brig.koen+grpoceren
KO=bench/runs/northstar_ko+en-ko@v2/$CFG
EN=bench/runs/northstar_ko+en-en@v2/$CFG
echo "$(date '+%F %T') MARK bench ko"
[ -f $KO/summary.json ] || uv run --project bench python -m bench --config $CFG --dataset northstar_ko+en-ko || { echo BENCH_FAIL; exit 1; }
echo "$(date '+%F %T') MARK rescore en"
[ -f $EN/summary.json ] || uv run --project bench python scripts/brig_rescore.py \
  --pipeline $CFG --src-dataset northstar_ko+en-ko --dst-dataset northstar_ko+en-en || echo RESCORE_FAIL
echo "$(date '+%F %T') MARK comet"
for R in $KO $EN; do
  [ -f $R/summary.json ] && { [ -f $R/comet_scores.jsonl ] || uv run --project bench/metrics/comet python -m bench.metrics.comet --run-dir $R 2>&1 | tail -1; }
done
echo "$(date '+%F %T') MARK report"
python3 - <<'PYEOF'
import json
def row(label, path):
    try: s = json.load(open(path + "/summary.json"))
    except FileNotFoundError: return
    m = s.get("metrics", s); g = lambda k: m.get(k)
    f = lambda k: ("-" if g(k) is None else (f"{g(k):.3f}" if k == "comet" else f"{g(k):.0f}"))
    print(f"| {label} | {f('comet')} | {f('longyaal_ms')} | {f('longyaal_ca_ms')} |")
print("| run | COMET | LongYAAL | CA |")
row("brig ko", "bench/runs/northstar_ko+en-ko@v2/brig.koen+grpoceren")
row("brig en", "bench/runs/northstar_ko+en-en@v2/brig.koen+grpoceren")
print("stall 수정 전(hybrid): ko 0.744/2370/2595 · 연구 브랜치: ko 0.751/2026, en 0.663/1402")
PYEOF
echo "BRIG_DONE $(date '+%F %T')"
