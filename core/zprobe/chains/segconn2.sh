#!/bin/bash
# segconn with <SEG> between pieces in every row (--seg_full): does the model stop leaving the continuation empty?
#   tmux new-session -d -s segconn2 -c /home/mobility/STiTy "bash core/zprobe/chains/segconn2.sh > core/zprobe/chains/segconn2.log 2>&1"
cd /home/mobility/STiTy
PY=.venv/bin/python; CPY=.venv-tpm/bin/python
BASE=models/Qwen3-ASR-1.7B-koen-seg-mix-merged
D=models/zprobe/data; K=$D/connective; H=$D/hy; U=$D/segunits; C=$D/cutlabels; A=models/zprobe/adapters; LOG=models/zprobe/logs; OUT=models/zprobe/grpo
NAME=segconn2
mark() { echo "$(date +%T) MARK $1"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
export PYTHONPATH=Qwen3-ASR
mark "1 build"
[ -f $D/train_$NAME.jsonl ] || $PY core/zprobe/build_connective.py build --seg_full --pieces $K/pieces.jsonl --base $D/train_seg9_mix.jsonl \
  --out $D/train_$NAME.jsonl --wav_dir $K/cuts
grep -m2 '<SEG>.*<SEG>' $D/train_$NAME.jsonl | grep en-ko | cut -c1-300
mark "2 SFT"
if [ ! -d $A/$NAME ]; then
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 18000 ]; do sleep 60; done
  $PY -u Qwen3-ASR/finetuning/qwen3_asr_sft.py --model_path $BASE --train_file $D/train_$NAME.jsonl --eval_file $D/val_seg9_mix_clean.jsonl \
    --output_dir $OUT/sft_$NAME --batch_size 2 --grad_acc 8 --lr 1e-4 --epochs 1 --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
    --no_seg 1 --lora_targets q_proj,k_proj,v_proj,o_proj --save_steps 200 --group_by_length 0 > $LOG/train_$NAME.log 2>&1; echo "SFT_EXIT=$?"
  [ -d $OUT/sft_$NAME/final ] || { echo "no final adapter"; tail -20 $LOG/train_$NAME.log; exit 1; }
  cp -r $OUT/sft_$NAME/final $A/$NAME
fi
export PROBE_MODEL=$BASE PROBE_ADAPTER=$A/$NAME
mark "3 offline"
for p in en-ko en-ja en-zh; do
  [ "$(lines $H/${NAME}_$p.jsonl)" -eq "$(lines $D/ev_big2_$p.jsonl)" ] || $PY -u core/zprobe/probe_z.py tagswap --pair $p --out $H/${NAME}_$p.jsonl 2>&1 | grep -v -i warn | tail -1
done
mark "4 commit at SEG units (+0.3 s)"
[ -f $U/${NAME}_pad03_en-ko.jsonl ] || $PY -u core/zprobe/seg_units.py adapter --seg_prefix --pad 0.3 --no_repeat 4 --units $U/units_en.jsonl \
  --ids $D/ev_big2_en-ko.jsonl --tgt ko --out $U/${NAME}_pad03_en-ko.jsonl 2>&1 | grep -i -E "error|trace"
mark "5 every cut, first 120 test sentences"
[ -f $C/${NAME}_test120.done ] || { $PY -u core/zprobe/cut_search.py decode --seg_prefix --split test --ids $D/ev_big2_en-ko.jsonl \
  --units $U/units_en.jsonl --tgt ko --limit 120 --pad 0.3 --out $C/${NAME}_test120.jsonl 2>&1 | grep -v -i -E "warn|pad_token|Loading" && touch $C/${NAME}_test120.done; }
mark "6 COMET"
$CPY -u core/zprobe/comet_eval.py --n 0 --strip_seg --per_sent $H/scores $(for p in en-ko en-ja en-zh; do echo -n "${NAME}_$p=$H/${NAME}_$p.jsonl "; done) \
  su03_${NAME}_en-ko=$U/${NAME}_pad03_en-ko.jsonl 2>&1 | grep -v -i warn | tail -5
$CPY -u core/zprobe/comet_eval.py --n 0 --strip_seg --per_sent $C/scores ${NAME}_test120=$C/${NAME}_test120.jsonl 2>&1 | grep -v -i warn | tail -2
mark "7 report"
{
echo "## offline: $NAME vs segconn"; $CPY core/zprobe/paired_delta.py --scores $H/scores --a segconn --b $NAME --pairs en-ko en-ja en-zh
echo "## commit at SEG units (+0.3 s) vs own offline: segconn"; $CPY core/zprobe/paired_delta.py --scores $H/scores --a segconn --b su03_segconn --pairs en-ko
echo "## commit at SEG units (+0.3 s) vs own offline: $NAME"; $CPY core/zprobe/paired_delta.py --scores $H/scores --a $NAME --b su03_$NAME --pairs en-ko
echo "## every cut, first 120 test sentences: $NAME"; $PY core/zprobe/cut_search.py summary --rows $C/${NAME}_test120.jsonl --scores $C/scores/${NAME}_test120.jsonl
NAME=$NAME python3 - <<'PYEOF'
import json, collections, os
C = "models/zprobe/data/cutlabels"; n = os.environ["NAME"]
def table(rows_f, sc_f, ids=None):
    rows = [json.loads(l) for l in open(rows_f)]; sc = [json.loads(l)["comet"] for l in open(sc_f)]
    whole = {r["id"]: s for r, s in zip(rows, sc) if r["k"] == 0}
    b = collections.defaultdict(lambda: [0, 0, []])
    for r, s in zip(rows, sc):
        if not r["k"] or (ids and r["id"] not in ids): continue
        q = min(4, int(r["k"] / len(r["ref_src"].split()) * 5)); b[q][0] += 1; b[q][1] += not r["units_tgt"][1].strip(); b[q][2].append(s - whole[r["id"]])
    return {q: (v[1] / v[0], sum(v[2]) / len(v[2])) for q, v in b.items()}
new = table(f"{C}/{n}_test120.jsonl", f"{C}/scores/{n}_test120.jsonl")
ids = {json.loads(l)["id"] for l in open(f"{C}/{n}_test120.jsonl")}
old = table(f"{C}/test.jsonl", f"{C}/scores/test.jsonl", ids)
print("\n| cut position | segconn empty / loss | %s empty / loss |\n|---|---|---|" % n)
for q in sorted(new):
    print(f"| {q*20}-{q*20+20}% | {old[q][0]:.2f} / {old[q][1]:+.3f} | {new[q][0]:.2f} / {new[q][1]:+.3f} |")
PYEOF
} | tee $LOG/${NAME}_report.md
mark CHAIN_DONE
