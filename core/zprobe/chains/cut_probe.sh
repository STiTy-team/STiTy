#!/bin/bash
# After cut_labels.sh: hidden-state features for every labelled cut, then a linear O/X / ridge cut chooser trained on dev +
# the first 150 test sentences and evaluated on the remaining test sentences against the SEG model's cut.
cd /home/mobility/STiTy
PY=.venv/bin/python; D=models/zprobe/data; C=$D/cutlabels
until grep -q CHAIN_DONE core/zprobe/chains/cut_labels.log; do sleep 60; done
export PYTHONPATH=Qwen3-ASR PROBE_MODEL=models/Qwen3-ASR-1.7B-koen-seg-mix-merged
echo "$(date +%T) MARK features"
for s in dev test; do [ -f $C/feat_$s.npz ] || $PY -u core/zprobe/cut_probe.py features --rows $C/$s.jsonl --split $s --out $C/feat_$s.npz 2>&1 | grep -v -i -E "warn|Loading|flags"; done
python3 -c "
import json
ids=[]; [ids.append(json.loads(l)['id']) for l in open('$C/test.jsonl') if json.loads(l)['id'] not in ids]
open('$C/test_eval_ids.txt','w').write('\n'.join(ids[150:]))
print('eval sentences', len(ids[150:]))"
echo "$(date +%T) MARK fit"
$PY core/zprobe/cut_probe.py fit --data dev:$C/feat_dev.npz:$C/dev.jsonl:$C/scores/dev.jsonl test:$C/feat_test.npz:$C/test.jsonl:$C/scores/test.jsonl \
  --test_ids $C/test_eval_ids.txt | tee models/zprobe/logs/cut_probe.md
echo "$(date +%T) MARK CHAIN_DONE"
