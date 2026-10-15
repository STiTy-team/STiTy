#!/bin/bash
# Does cutting the audio exactly at the aligner's word end hurt commit-and-continue? Same units, +0.3 s of audio per cut.
cd /home/mobility/STiTy
PY=.venv/bin/python; CPY=.venv-tpm/bin/python; D=models/zprobe/data; U=$D/segunits; H=$D/hy
BASE=models/Qwen3-ASR-1.7B-koen-seg-mix-merged
for ad in segconn segft9_mix; do
  PYTHONPATH=Qwen3-ASR PROBE_MODEL=$BASE PROBE_ADAPTER=models/zprobe/adapters/$ad $PY -u core/zprobe/seg_units.py adapter --no_repeat 4 --pad 0.3 \
    --units $U/units_en.jsonl --ids $D/ev_big2_en-ko.jsonl --tgt ko --out $U/${ad}_pad03_en-ko.jsonl 2>&1 | grep -i -E "error|trace"
done
$CPY -u core/zprobe/comet_eval.py --n 0 --strip_seg --per_sent $H/scores su03_segconn_en-ko=$U/segconn_pad03_en-ko.jsonl su03_segmix_en-ko=$U/segft9_mix_pad03_en-ko.jsonl 2>&1 | grep -v -i warn | tail -3
{ echo "## +0.3s: segconn vs own offline"; $CPY core/zprobe/paired_delta.py --scores $H/scores --a segconn --b su03_segconn --pairs en-ko
  echo "## +0.3s: segmix vs own offline"; $CPY core/zprobe/paired_delta.py --scores $H/scores --a segmix --b su03_segmix --pairs en-ko
  echo "## latency"; $PY core/zprobe/seg_units.py score --files segconn_pad0=$U/segconn_en-ko.jsonl segconn_pad03=$U/segconn_pad03_en-ko.jsonl; } | tee models/zprobe/logs/pad_test.md
echo "$(date +%T) MARK CHAIN_DONE"
