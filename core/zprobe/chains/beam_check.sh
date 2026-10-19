#!/bin/bash
# Is beam search usable? (1) vLLM 0.14 beam search on the merged segconn2 (the production server's beam flag silently
# falls back to greedy), (2) beam search inside the true-streaming commit loop on the transformers backend (agree2, beam 4).
#   tmux new-session -d -s beamcheck -c <repo> "PY=<env with peft+qwen_asr+vllm> bash core/zprobe/chains/beam_check.sh > core/zprobe/chains/beam_check.log 2>&1"
cd "$(dirname "$0")/../../.."
PY=${PY:-.venv/bin/python}; CPY=${CPY:-.venv/bin/python}
BASE=models/Qwen3-ASR-1.7B-koen-seg-mix-merged; MERGED=models/zprobe/merged/segconn2
D=models/zprobe/data; H=$D/hy; U=$D/segunits; V=$D/nbest; LOG=models/zprobe/logs; IDS=$D/ev_big2_en-ko.jsonl
mark() { echo "$(date +%T) MARK $1"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
waitgpu() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$1" ]; do sleep 60; done; }
export PYTHONPATH=Qwen3-ASR
N=$(lines $IDS)

mark "1 vllm smoke"
waitgpu 12000
$PY -u core/zprobe/vllm_beam.py --model $MERGED --ids $IDS --out $V/vllm_smoke.jsonl --beams 4 --limit 5 > $LOG/vllm_beam_smoke.log 2>&1
if grep -q VLLM_BEAM_DONE $LOG/vllm_beam_smoke.log; then
  mark "2 vllm full"
  waitgpu 12000
  [ "$(lines $V/vllm_beam4_en-ko.jsonl)" -eq "$N" ] || $PY -u core/zprobe/vllm_beam.py --model $MERGED --ids $IDS --out $V/vllm_beam4_en-ko.jsonl --beams 4 > $LOG/vllm_beam.log 2>&1
  tail -3 $LOG/vllm_beam.log
else
  echo "vllm smoke failed"; grep -E "Error|Traceback" -A3 $LOG/vllm_beam_smoke.log | tail -20
fi

mark "3 transformers streaming agree2 beam4"
waitgpu 12000
F=$U/sc_agree2_beam4_en-ko.jsonl
[ "$(lines $F)" -eq "$N" ] || PROBE_MODEL=$BASE PROBE_ADAPTER=models/zprobe/adapters/segconn2 $PY -u core/zprobe/stream_commit.py \
  --ids $IDS --tgt ko --agree 2 --beams 4 --out $F 2>&1 | grep -v -i -E "warn|pad_token|generation flags"

mark "4 COMET"
waitgpu 8000
$CPY - <<'PY'
import json, os
V = "models/zprobe/data/nbest"; os.makedirs(V + "/hyp", exist_ok=True)
def dump(src, key, label):
    if not os.path.exists(src): return
    with open(f"{V}/hyp/{label}_en-ko.jsonl", "w", encoding="utf-8") as f:
        for l in open(src, encoding="utf-8"):
            r = json.loads(l); f.write(json.dumps({"id": r["id"], "hyp": key(r), "ref_src": r["ref_src"], "ref_tgt": r["ref_tgt"]}, ensure_ascii=False) + "\n")
dump(f"{V}/segconn2_en-ko.jsonl", lambda r: r["greedy"], "hf_greedy")
dump(f"{V}/segconn2_en-ko.jsonl", lambda r: r["beam"][0], "hf_beam8")
dump(f"{V}/vllm_beam4_en-ko.jsonl", lambda r: r["greedy"], "vllm_greedy")
dump(f"{V}/vllm_beam4_en-ko.jsonl", lambda r: r["beam"], "vllm_beam4")
PY
ITEMS="hf_greedy_en-ko=$V/hyp/hf_greedy_en-ko.jsonl hf_beam8_en-ko=$V/hyp/hf_beam8_en-ko.jsonl sc_agree2_beam4_en-ko=$U/sc_agree2_beam4_en-ko.jsonl"
[ -f $V/hyp/vllm_beam4_en-ko.jsonl ] && ITEMS="$ITEMS vllm_greedy_en-ko=$V/hyp/vllm_greedy_en-ko.jsonl vllm_beam4_en-ko=$V/hyp/vllm_beam4_en-ko.jsonl"
$CPY -u core/zprobe/comet_eval.py --n 0 --strip_seg --per_sent $H/scores $ITEMS 2>&1 | grep -v -i warn | tail -8

mark "5 report"
{
echo "## offline: beam vs greedy"
$CPY core/zprobe/paired_delta.py --scores $H/scores --a hf_greedy --b hf_beam8 --pairs en-ko
if [ -f $H/scores/vllm_beam4_en-ko.jsonl ]; then
  $CPY core/zprobe/paired_delta.py --scores $H/scores --a vllm_greedy --b vllm_beam4 --pairs en-ko
  $CPY core/zprobe/paired_delta.py --scores $H/scores --a hf_greedy --b vllm_greedy --pairs en-ko
  $CPY - <<'PY'
import json, statistics as st
R = [json.loads(l) for l in open("models/zprobe/data/nbest/vllm_beam4_en-ko.jsonl")]
for k in ("ms_greedy", "ms_beam"):
    v = sorted(r[k] for r in R[5:]); print(f"vllm {k}: median {st.median(v):.0f} ms, p90 {v[int(0.9 * len(v))]:.0f} ms")
print("same text greedy vs beam:", sum(r["greedy"] == r["beam"] for r in R), "/", len(R))
PY
fi
echo "## streaming agree2: beam4 vs greedy (segconn2)"
$CPY core/zprobe/paired_delta.py --scores $H/scores --a sc_agree2 --b sc_agree2_beam4 --pairs en-ko
echo "## latency"
$PY core/zprobe/seg_units.py score --files sc_agree2=$U/sc_agree2_en-ko.jsonl sc_agree2_beam4=$U/sc_agree2_beam4_en-ko.jsonl
} | tee $LOG/beam_check_report.md
mark CHAIN_DONE
