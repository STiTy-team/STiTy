#!/bin/bash
# Smoke of grpo_prefix.py's schedule flags (--schedule/--sigma_s/--sigma_l/--lat_floor) on a 6-sentence pool built
# from prefix13 aligned data, so the prefix14 chain fails here - before labels are paid for and SFT runs - if the
# GRPO recipe is broken. Pass = GRPO_DONE in the log and emit=/wait= columns in the step line. ~5 min on a free GPU.
# usage: bash core/research/zprobe/train/prefix14_grpo_smoke.sh   (run from the repo root; exit 0 = SMOKE_OK)
cd "$(git rev-parse --show-toplevel)" || exit 1
set -o pipefail
PY=.venv/bin/python; CPY=.venv-tpm/bin/python
S=models/zprobe/data/prefix14/smoke; mkdir -p $S; rm -rf $S/grpo
POLICY=${POLICY:-models/zprobe/merged/prefix13tr_full}
$PY - "$S" <<'PYEOF' || exit 1
import json, sys
S = sys.argv[1]
wt = {}
for l in open("models/zprobe/data/prefix12/word_times.jsonl", encoding="utf-8"):
    r = json.loads(l); wt[r["audio"]] = r
n = {"en-ko": 0, "ko-en": 0}
fs = open(f"{S}/sentences.jsonl", "w", encoding="utf-8"); fc = open(f"{S}/schedule.jsonl", "w", encoding="utf-8")
fw = open(f"{S}/word_times.jsonl", "w", encoding="utf-8")
for l in open("models/zprobe/data/prefix13/aligned.jsonl", encoding="utf-8"):
    r = json.loads(l); pair = f"{r['lang']}-{r['tgt']}"
    if pair not in n or n[pair] >= 3 or r["audio"] not in wt or r.get("dur", 99) > 12:
        continue
    adds = [x for x in r["adds"] if x.strip()]
    if not adds:
        continue
    K = len(r["src"].split())
    sch = [[max(1, -(-K * (i + 1) // len(adds))), t] for i, t in enumerate(adds)]  # spread over the source words
    fs.write(json.dumps({"sid": r["sid"], "lang": r["lang"], "tgt": r["tgt"], "src": r["src"], "ref": " ".join(adds),
                         "source": "smoke", "recs": [{"audio": r["audio"]}]}, ensure_ascii=False) + "\n")
    fc.write(json.dumps({"sid": r["sid"], "lang": r["lang"], "tgt": r["tgt"], "schedule": sch}, ensure_ascii=False) + "\n")
    fw.write(json.dumps(wt[r["audio"]], ensure_ascii=False) + "\n"); n[pair] += 1
print("smoke pool", n)
assert all(v == 3 for v in n.values()), n
PYEOF
$CPY -u core/research/zprobe/train/comet_server.py --port 8778 > $S/comet_server.log 2>&1 & CS=$!
until grep -q "ready" $S/comet_server.log || ! kill -0 $CS 2>/dev/null; do sleep 5; done
kill -0 $CS 2>/dev/null || { echo SMOKE_FAIL comet_server; exit 1; }
PYTHONPATH=Qwen3-ASR $PY -u core/research/zprobe/train/grpo_prefix.py \
  --policy_model $POLICY --out $S/grpo --source_tag 0 --comet http://127.0.0.1:8778/score \
  --steps 2 --utts 4 --G 2 --lam 0.1 --sil_frac 0.5 --mu 0.25 --anchor 1 --nu 0.3 --rho 0.2 \
  --nli vicgalle/xlm-roberta-large-xnli-anli --pairs en-ko ko-en \
  --schedule $S/schedule.jsonl --sentences $S/sentences.jsonl --word_times $S/word_times.jsonl \
  --sigma_s 0.2 --sigma_l 0.1 --lat_floor 1 2>&1 | tee $S/grpo.log | grep -E "^step|utterances|Traceback|Error"
rc=$?
kill $CS 2>/dev/null; wait $CS 2>/dev/null
grep -q GRPO_DONE $S/grpo.log && grep -qE "^step 2 .*emit=[0-9.]+ wait=[0-9.]+" $S/grpo.log || { echo SMOKE_FAIL grpo; tail -5 $S/grpo.log; exit 1; }
echo "SMOKE_OK $(date '+%F %T')"
