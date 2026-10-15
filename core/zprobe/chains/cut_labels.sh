#!/bin/bash
# Per-cut COMET labels for a cheap O/X cut classifier: segconn (+0.3 s) translates every single cut of each sentence,
# FLEURS dev (129) + test en-ko (~267, SEG-model cut tagged). Whole-sentence translation is row k=0.
#   tmux new-session -d -s cut-labels -c /home/mobility/STiTy "bash core/zprobe/chains/cut_labels.sh > core/zprobe/chains/cut_labels.log 2>&1"
cd /home/mobility/STiTy
PY=.venv/bin/python; CPY=.venv-tpm/bin/python; D=models/zprobe/data; C=$D/cutlabels; mkdir -p $C/scores
export PYTHONPATH=Qwen3-ASR PROBE_MODEL=models/Qwen3-ASR-1.7B-koen-seg-mix-merged PROBE_ADAPTER=models/zprobe/adapters/segconn
echo "$(date +%T) MARK decode dev"
[ -f $C/dev.done ] || { $PY -u core/zprobe/cut_search.py decode --split dev --ids $D/ids_dev_en-ko.jsonl --tgt ko --limit 1000 --pad 0.3 \
  --out $C/dev.jsonl 2>&1 | grep -v -i -E "warn|pad_token|Loading" && touch $C/dev.done; }
echo "$(date +%T) MARK decode test"
[ -f $C/test.done ] || { $PY -u core/zprobe/cut_search.py decode --split test --ids $D/ev_big2_en-ko.jsonl --units $D/segunits/units_en.jsonl \
  --tgt ko --limit 1000 --pad 0.3 --out $C/test.jsonl 2>&1 | grep -v -i -E "warn|pad_token|Loading" && touch $C/test.done; }
echo "$(date +%T) MARK COMET"
$CPY -u core/zprobe/comet_eval.py --n 0 --strip_seg --per_sent $C/scores dev=$C/dev.jsonl test=$C/test.jsonl 2>&1 | grep -v -i warn | tail -3
for s in dev test; do echo "## $s"; $PY core/zprobe/cut_search.py summary --rows $C/$s.jsonl --scores $C/scores/$s.jsonl; done | tee models/zprobe/logs/cut_labels.md
echo "$(date +%T) MARK CHAIN_DONE"
