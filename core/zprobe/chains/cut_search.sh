#!/bin/bash
# Cut position (SEG vs random vs oracle) x continuation (raw vs period stripped) for big2 commit-and-continue, en-ko / en-zh.
#   tmux new-session -d -s cut-search -c /home/mobility/STiTy "bash core/zprobe/chains/cut_search.sh > core/zprobe/chains/cut_search.log 2>&1"
cd /home/mobility/STiTy
PY=.venv/bin/python; CPY=.venv-tpm/bin/python
D=models/zprobe/data; C=$D/cutsearch; LOG=models/zprobe/logs; mkdir -p $C/scores
mark() { echo "$(date +%T) MARK $1"; }
mark "1 decode"
for t in ko zh; do for v in raw strip; do
  f=$C/${v}_en-$t.jsonl; [ -f $f.done ] && continue
  flag=""; [ $v = strip ] && flag="--strip"
  PYTHONPATH=Qwen3-ASR PROBE_ADAPTER=models/zprobe/adapters/big2 $PY -u core/zprobe/cut_search.py decode --ids $D/ev_big2_en-$t.jsonl \
    --units $D/segunits/units_en.jsonl --tgt $t --limit 80 $flag --out $f 2>&1 | grep -v -i -E "warn|pad_token|Loading" && touch $f.done
done; done
mark "2 COMET"
items=""; for t in ko zh; do for v in raw strip; do items="$items ${v}_en-$t=$C/${v}_en-$t.jsonl"; done; done
$CPY -u core/zprobe/comet_eval.py --n 0 --per_sent $C/scores $items 2>&1 | grep -v -i warn | tail -6
mark "3 summary"
for t in ko zh; do for v in raw strip; do
  echo "## en-$t $v"; $PY core/zprobe/cut_search.py summary --rows $C/${v}_en-$t.jsonl --scores $C/scores/${v}_en-$t.jsonl --whole $D/hy/scores/big2_en-$t.jsonl
done; done | tee $LOG/cut_search.md
mark CHAIN_DONE
