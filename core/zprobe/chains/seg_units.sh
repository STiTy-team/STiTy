#!/bin/bash
# SEG-unit commit translation: koen-seg-mix decides units on the full English audio, units are moved onto the gold
# transcript (post-correction), then translated once each by Hy (with / without previous units as background) and by the
# big2 adapter (audio prefix + committed translation forced). Compared with offline whole-sentence translation.
#   tmux new-session -d -s seg-units -c /home/mobility/STiTy "bash core/zprobe/chains/seg_units.sh > core/zprobe/chains/seg_units.log 2>&1"
cd /home/mobility/STiTy
PY=.venv/bin/python; CPY=.venv-tpm/bin/python
D=models/zprobe/data; H=$D/hy; U=$D/segunits; LOG=models/zprobe/logs; mkdir -p $U
T="ko ja zh"
mark() { echo "$(date +%T) MARK $1"; }
mark "1 label"
[ -f $U/units_en.jsonl ] || PYTHONPATH=Qwen3-ASR PROBE_MODEL=models/Qwen3-ASR-1.7B-koen-seg-mix-merged $PY -u core/zprobe/seg_units.py label \
  --ids $D/ev_big2_en-ko.jsonl $D/ev_big2_en-ja.jsonl $D/ev_big2_en-zh.jsonl --out $U/units_en.jsonl 2>&1 | grep -v -i -E "warn|pad_token|Loading"
python3 -c "import json; r=[json.loads(l) for l in open('$U/units_en.jsonl')]; print('aligned', sum(x['aligned'] for x in r), 'of', len(r))"
mark "2 translate"
for t in $T; do
  [ -f $U/hyctx_en-$t.jsonl ] || $PY -u core/zprobe/seg_units.py hy --units $U/units_en.jsonl --ids $D/ev_big2_en-$t.jsonl --tgt $t --out $U/hyctx_en-$t.jsonl 2>&1 | grep -i -E "error|trace"
  [ -f $U/hynoctx_en-$t.jsonl ] || $PY -u core/zprobe/seg_units.py hy --no_context --units $U/units_en.jsonl --ids $D/ev_big2_en-$t.jsonl --tgt $t --out $U/hynoctx_en-$t.jsonl 2>&1 | grep -i -E "error|trace"
  [ -f $U/big2_en-$t.jsonl ] || PYTHONPATH=Qwen3-ASR PROBE_ADAPTER=models/zprobe/adapters/big2 $PY -u core/zprobe/seg_units.py adapter --no_repeat 4 \
     --units $U/units_en.jsonl --ids $D/ev_big2_en-$t.jsonl --tgt $t --out $U/big2_en-$t.jsonl 2>&1 | grep -i -E "error|trace"
done
mark "3 COMET"
items=""; for t in $T; do for k in hyctx hynoctx big2; do items="$items su_${k}_en-$t=$U/${k}_en-$t.jsonl"; done; done
$CPY -u core/zprobe/comet_eval.py --n 0 --per_sent $H/scores $items 2>&1 | grep -v -i warn | tail -11
mark "4 report"
{
echo "## Hy, gold source: SEG units (with background) vs whole sentence"; $CPY core/zprobe/paired_delta.py --scores $H/scores --a gold --b su_hyctx --pairs en-ko en-ja en-zh
echo "## Hy, gold source: SEG units (no background) vs whole sentence"; $CPY core/zprobe/paired_delta.py --scores $H/scores --a gold --b su_hynoctx --pairs en-ko en-ja en-zh
echo "## big2 adapter: SEG units vs whole sentence"; $CPY core/zprobe/paired_delta.py --scores $H/scores --a big2 --b su_big2 --pairs en-ko en-ja en-zh
echo "## latency"; $PY core/zprobe/seg_units.py score --files $(for t in $T; do for k in hyctx big2; do echo -n "${k}_en-$t=$U/${k}_en-$t.jsonl "; done; done)
} | tee $LOG/seg_units.md
mark CHAIN_DONE
