#!/bin/bash
# True streaming: segconn2 commits wherever it writes <SEG> on growing audio (0.5 s steps). agree1 = commit at once,
# agree2 = same piece at two consecutive steps. Compared with segconn2 offline and commit at SEG-model units (su03).
#   tmux new-session -d -s stream_commit -c /home/mobility/STiTy "bash core/zprobe/chains/stream_commit.sh > core/zprobe/chains/stream_commit.log 2>&1"
cd /home/mobility/STiTy
PY=.venv/bin/python; CPY=.venv-tpm/bin/python
D=models/zprobe/data; H=$D/hy; U=$D/segunits; LOG=models/zprobe/logs
export PYTHONPATH=Qwen3-ASR PROBE_MODEL=models/Qwen3-ASR-1.7B-koen-seg-mix-merged PROBE_ADAPTER=models/zprobe/adapters/segconn2
mark() { echo "$(date +%T) MARK $1"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
N=$(lines $D/ev_big2_en-ko.jsonl)
for AG in 1 2; do
  F=$U/sc_agree${AG}_en-ko.jsonl
  mark "decode agree$AG"
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 7000 ]; do sleep 60; done
  [ "$(lines $F)" -eq "$N" ] || $PY -u core/zprobe/stream_commit.py --ids $D/ev_big2_en-ko.jsonl --tgt ko --agree $AG --out $F 2>&1 | grep -v -i -E "warn|pad_token"
done
mark "COMET"
$CPY -u core/zprobe/comet_eval.py --n 0 --strip_seg --per_sent $H/scores sc_agree1_en-ko=$U/sc_agree1_en-ko.jsonl \
  sc_agree2_en-ko=$U/sc_agree2_en-ko.jsonl 2>&1 | grep -v -i warn | tail -4
mark "report"
{
echo "## streaming commit vs segconn2 offline"
for AG in 1 2; do $CPY core/zprobe/paired_delta.py --scores $H/scores --a segconn2 --b sc_agree$AG --pairs en-ko; done
echo "## streaming commit vs commit at SEG-model units (+0.3 s)"
for AG in 1 2; do $CPY core/zprobe/paired_delta.py --scores $H/scores --a su03_segconn2 --b sc_agree$AG --pairs en-ko; done
echo "## latency"
$PY core/zprobe/seg_units.py score --files su03_segconn2=$U/segconn2_pad03_en-ko.jsonl sc_agree1=$U/sc_agree1_en-ko.jsonl sc_agree2=$U/sc_agree2_en-ko.jsonl
} | tee $LOG/stream_commit_report.md
mark CHAIN_DONE
