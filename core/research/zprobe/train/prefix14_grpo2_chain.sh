#!/bin/bash
# Follow-on to prefix14_chain.sh: after P14_DONE, a second GRPO on the same prefix14 SFT model with the schedule
# terms doubled (--sigma_s 0.4 --sigma_l 0.2, latency floor on) -> merge -> bench (manager queue).
# Markers P14G2 MARK/FAIL/DONE/EXIT. Idempotent by file existence.
cd "$(git rev-parse --show-toplevel)" || exit 1
set -o pipefail
trap 'echo "P14G2_EXIT $(date "+%F %T")"' EXIT
set -a; . /home/skkai/STiTy/.env; set +a
export PATH=$HOME/.local/bin:$PATH
export STITY_DATA_ROOT=/home/skkai/STiTy-team/datasets
PY=.venv/bin/python; CPY=.venv-tpm/bin/python
OUT=models/zprobe/data/prefix14; LOG=models/zprobe/logs; mkdir -p $LOG $OUT
GATED=$OUT/schedule_${LABEL_MODEL:-gpt-6-luna}.gated.jsonl
roomy() { local ok=0; while [ $ok -lt 3 ]; do
  if [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$1" ]; then ok=$((ok + 1)); else ok=0; fi
  [ $ok -lt 3 ] && sleep 60; done; }
stage() { echo "$(date '+%F %T') P14G2 MARK $*"; }

stage "wait for P14_DONE"
# the chain log is appended across relaunches: judge only the lines after the latest launch, and keep waiting
# through a failure (a human may override and rerun the chain) - a note once an hour says why we are waiting
n=0
until grep -q "^P14_DONE" /tmp/p14_chain.log 2>/dev/null; do
  if awk '/^P14L launch/{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' /tmp/p14_chain.log 2>/dev/null | grep -q "^P14_FAIL"; then
    [ $((n % 12)) -eq 0 ] && echo "$(date '+%F %T') P14G2 prefix14 chain shows P14_FAIL since its last launch; waiting for a rerun"
  fi
  n=$((n + 1)); sleep 300
done
[ -d models/zprobe/merged/prefix14_full ] || { echo P14G2_FAIL sft-model-missing; exit 1; }
[ -f $GATED ] || { echo P14G2_FAIL gated-schedule-missing; exit 1; }

stage "grpo2 400 (sigma_s 0.4 / sigma_l 0.2, lat floor, en-ko/ko-en, no tag)"
GOUT=models/zprobe/grpo/prefix14_grpo2_cerenlat; A=models/zprobe/adapters/prefix14_grpo2_cerenlat; mkdir -p $GOUT
if [ ! -d "$A" ]; then
  for TRY in 1 2 3; do
    roomy 20000
    $CPY -u core/research/zprobe/train/comet_server.py --port 8777 > $LOG/comet_server_p14g2.log 2>&1 & CS=$!
    until grep -q "ready" $LOG/comet_server_p14g2.log || ! kill -0 $CS 2>/dev/null; do sleep 5; done
    kill -0 $CS 2>/dev/null || { echo P14G2_FAIL comet_server; exit 1; }
    PYTHONPATH=Qwen3-ASR $PY -u core/research/zprobe/train/grpo_prefix.py \
      --policy_model models/zprobe/merged/prefix14_full --out $GOUT --source_tag 0 \
      --steps 400 --lam 0.1 --sil_frac 0.25 --mu 0.25 --anchor 1 --hard_frac 0.5 --nu 0.3 --rho 0.2 \
      --nli vicgalle/xlm-roberta-large-xnli-anli --pairs en-ko ko-en --utts 400 \
      --schedule $GATED --sentences $OUT/sentences.jsonl --word_times $OUT/word_times.jsonl \
      --sigma_s 0.4 --sigma_l 0.2 --lat_floor 1 \
      > $LOG/grpo_prefix14_grpo2_cerenlat.log 2>&1
    kill $CS 2>/dev/null; wait $CS 2>/dev/null
    last=$(ls -d $GOUT/adapter_step* 2>/dev/null | sort -V | tail -1)
    grep -q GRPO_DONE $LOG/grpo_prefix14_grpo2_cerenlat.log 2>/dev/null && { cp -r "$last" "$A"; break; }
    echo "GRPO try $TRY failed"; tail -3 $LOG/grpo_prefix14_grpo2_cerenlat.log; sleep 30
  done
fi
[ -d "$A" ] || { echo P14G2_FAIL grpo; exit 1; }

stage "merge grpo2"
[ -d models/zprobe/merged/prefix14_grpo2_cerenlat_full ] || PYTHONPATH=Qwen3-ASR $PY core/research/zprobe/train/merge_adapter.py \
  --base models/zprobe/merged/prefix14_full --adapter "$A" --out models/zprobe/merged/prefix14_grpo2_cerenlat_full \
  || { echo P14G2_FAIL merge; exit 1; }

stage "bench grpo2 (manager queue)"
P=brig.koen+prefix14-grpo2-lidreset
if [ -f $OUT/bench_grpo2.job ]; then JOB=$(cat $OUT/bench_grpo2.job); echo "  reuse bench job $JOB"
else
  JOB=$(CFG=$P DSN=northstar_ko+en-ko python3 - <<'PYEOF' | curl -s -XPOST localhost:29130/api/machines/4090/jobs -H 'Content-Type: application/json' -d @- | python3 -c "import json,sys; d=json.load(sys.stdin); x=d.get('data') or {}; print(x.get('id') or ''); sys.stderr.write('P14G2 JOB %s %s %s\n' % (d.get('ok'), x.get('id'), d.get('error','')[:200]))"
import json, os, pathlib
c, d = os.environ["CFG"], os.environ["DSN"]
print(json.dumps({"branch": "main", "pipeline": {"name": c, "yaml": pathlib.Path(f"configs/pipelines/{c}.yml").read_text()},
                  "dataset": {"name": d, "yaml": pathlib.Path(f"configs/datasets/{d}.yml").read_text()}}))
PYEOF
)
  [ -n "$JOB" ] || { echo P14G2_FAIL bench-submit; exit 1; }
  echo "$JOB" > $OUT/bench_grpo2.job
  curl -s -XPOST localhost:29130/api/machines/4090/jobs/$JOB/run-now -H 'Content-Type: application/json' -d '{}' >/dev/null
fi
n=0
while :; do
  curl -s --max-time 20 localhost:29130/api/machines | JID=$JOB python3 -c "
import json, os, sys
try:
    d = json.load(sys.stdin)
    jobs = [x for m in d['machines'] if m['host'] == '4090' for x in m['jobs']]
    sys.exit(0 if any(str(x['id']) == os.environ['JID'] for x in jobs) else 1)
except Exception:
    sys.exit(2)
"
  rc=$?; [ $rc -eq 1 ] && break
  n=$((n + 1)); [ $((n % 10)) -eq 0 ] && echo "$(date '+%F %T') P14G2 waiting for bench job $JOB (rc=$rc)"; sleep 60
done
echo "P14G2_DONE $(date '+%F %T')"
