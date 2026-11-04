#!/bin/bash
# First live run of the mainline hybrid pipeline (koen ASR + grpoceren adapter + fold records)
# on northstar, then COMET, then a comparison against the research-branch numbers.
#   tmux new-session -d -s hybmain -c /home/mobility/STiTy "bash scripts/hybrid_main_smoke.sh >> /tmp/hybrid_main_smoke.log 2>&1"
cd /home/mobility/STiTy
trap 'echo "HYBMAIN_EXIT $(date "+%F %T")"' EXIT
export STITY_DATA_ROOT=/home/mobility/stity-datasets
export PATH=$HOME/.local/bin:$PATH
CFG=hybrid.koen+grpoceren
R=bench/runs/northstar_ko+en-ko@v2/$CFG
echo "$(date '+%F %T') MARK bench"
[ -f $R/summary.json ] || uv run --project bench python -m bench --config $CFG --dataset northstar_ko+en-ko || { echo BENCH_FAIL; exit 1; }
echo "$(date '+%F %T') MARK comet"
[ -f $R/comet_scores.jsonl ] || uv run --project bench/metrics/comet python -m bench.metrics.comet --run-dir $R 2>&1 | tail -1
echo "$(date '+%F %T') MARK report"
python3 - <<'PYEOF'
import json
R = "bench/runs/northstar_ko+en-ko@v2/hybrid.koen+grpoceren"
s = json.load(open(f"{R}/summary.json")); m = s.get("metrics", s)
g = lambda k: m.get(k)
counts = {"translated": {"ko": 0, "en": 0}, "folded": {"ko": 0, "en": 0}, "transcribed": 0}
for line in open(f"{R}/items.jsonl"):
    for r in json.loads(line).get("records") or []:
        t = r.get("type")
        if t == "transcribed": counts["transcribed"] += 1
        elif t in ("translated", "folded"): counts[t][r.get("target_lang") or "?"] = counts[t].get(r.get("target_lang") or "?", 0) + 1
print("| run | COMET | LongYAAL | CA | WER |")
print(f"| hybrid main (ko 채점) | {g('comet')} | {g('longyaal_ms')} | {g('longyaal_ca_ms')} | {g('wer')} |")
print("counts:", counts)
print("연구 브랜치 기준: ceren+fold ko 0.751/2026ms (en 쪽은 en 데이터셋 채점에서 0.663/1402)")
PYEOF
echo "HYBMAIN_DONE $(date '+%F %T')"
