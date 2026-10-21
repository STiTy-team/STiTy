#!/bin/bash
# segconnx: segconn2 + English-target connective pieces (build_connective_en.py: ko/ja/zh FLEURS train speech -> open
# English pieces), one adapter for both targets. Then offline and real-time streaming checks for X->en and a regression
# check on X->ko. Runs in tmux so it survives the editor closing.
#   tmux new-session -d -s segconnx -c /home/mobility/STiTy "bash core/zprobe/chains/segconnx.sh >> core/zprobe/chains/segconnx.log 2>&1"
cd /home/mobility/STiTy
PY=${PY:-.venv/bin/python}; CPY=${CPY:-.venv-tpm/bin/python}
BASE=models/Qwen3-ASR-1.7B-koen-seg-mix-merged
D=models/zprobe/data; K=$D/connective_en; H=$D/hy; A=models/zprobe/adapters; LOG=models/zprobe/logs; OUT=models/zprobe/grpo
BENCH=${BENCH:-/home/mobility/STiTy-bench}; W=${W:-/home/mobility/zprobe_bench}
NAME=segconnx
mark() { echo "$(date +%T) MARK $1"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
freegpu() { until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$1" ]; do sleep 60; done; }
idle() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -le 2000 ]; do sleep 60; done; }
f() { grep -v -i -E "warn|pad_token|generation flags"; }

mark "1 labels"
[ -f $K/manifest.jsonl ] || $PY core/zprobe/build_connective_en.py manifest --out $K/manifest.jsonl
freegpu 20000
[ -f $K/pieces1.done ] || { $PY -u core/zprobe/build_connective_en.py pieces --manifest $K/manifest.jsonl --out $K/pieces1.jsonl 2>&1 | f && touch $K/pieces1.done; }
[ -f $K/pieces1_hy.jsonl ] || PYTHONPATH=. $PY -u core/zprobe/build_connective_en.py hyprefix --pieces $K/pieces1.jsonl --out $K/pieces1_hy.jsonl 2>&1 | f | tail -1
[ -f $K/pieces.done ] || { $PY -u core/zprobe/build_connective_en.py pieces --hy $K/pieces1_hy.jsonl --out $K/pieces.jsonl 2>&1 | f | tail -2 && touch $K/pieces.done; }
python3 -c "
import json, collections
rs=[json.loads(l) for l in open('$K/pieces.jsonl')]; c=collections.Counter(r['fail'].split(':')[0] or 'ok' for r in rs)
ok=[r for r in rs if not r['fail']]; print('labels', dict(c), 'pieces/sentence', round(sum(len(r['pieces']) for r in ok)/len(ok),2))
for r in ok[1000:1003]: print('  ', ' / '.join(r['chunks']), '=>', ' / '.join(r['pieces']))"
[ -f $K/aligned.done ] || { PYTHONPATH=Qwen3-ASR $PY -u core/zprobe/build_connective_en.py align --manifest $K/manifest.jsonl --pieces $K/pieces.jsonl --out $K/aligned.jsonl 2>&1 | f | tail -1 && touch $K/aligned.done; }
[ -f $D/train_$NAME.jsonl ] || $PY core/zprobe/build_connective_en.py build --aligned $K/aligned.jsonl --base $D/train_segconn2.jsonl --wav_dir $K/cuts --out $D/train_$NAME.jsonl

mark "2 SFT"
export PYTHONPATH=Qwen3-ASR
if [ ! -d $A/$NAME ]; then
  for TRY in 1 2 3 4 5 6; do
    [ -d $OUT/sft_$NAME/final ] && break
    freegpu 18000; echo "SFT try $TRY"
    $PY -u Qwen3-ASR/finetuning/qwen3_asr_sft.py --model_path $BASE --train_file $D/train_$NAME.jsonl --eval_file $D/val_seg9_mix_clean.jsonl \
      --output_dir $OUT/sft_$NAME --batch_size 2 --grad_acc 8 --lr 1e-4 --epochs 1 --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 \
      --no_seg 1 --lora_targets q_proj,k_proj,v_proj,o_proj --save_steps 200 --group_by_length 0 > $LOG/train_$NAME.log 2>&1; echo "SFT_EXIT=$?"
  done
  [ -d $OUT/sft_$NAME/final ] || { echo "no final adapter"; tail -20 $LOG/train_$NAME.log; exit 1; }
  cp -r $OUT/sft_$NAME/final $A/$NAME
fi

mark "3 offline"
export PROBE_MODEL=$BASE
for M in $NAME segconn2; do for P in ko-en ja-en zh-en en-ko zh-ko ja-ko; do
  O=$H/${M}_$P.jsonl
  [ "$(lines $O)" -eq "$(lines $D/ev_big2_$P.jsonl)" ] && continue
  freegpu 8000
  PROBE_ADAPTER=$A/$M $PY -u core/zprobe/probe_z.py tagswap --pair $P --out $O 2>&1 | f | tail -1
done; done
ITEMS=""; for M in $NAME segconn2; do for P in ko-en ja-en zh-en en-ko zh-ko ja-ko; do
  [ -f $H/scores/${M}_$P.jsonl ] || ITEMS="$ITEMS ${M}_$P=$H/${M}_$P.jsonl"; done; done
[ -n "$ITEMS" ] && $CPY -u core/zprobe/comet_eval.py --n 0 --strip_seg --per_sent $H/scores $ITEMS 2>&1 | grep -v -i warn | tail -14

mark "4 streaming"
[ -f $W/export/ko-en/manifest.jsonl ] || (cd $BENCH && STITY_DATA_ROOT=${STITY_DATA_ROOT:-/home/mobility/stity-datasets} PYTHONPATH=. \
  bench/.venv/bin/python bench/seamless/export_items.py --dataset fleurs_ko-en_vol23 --out $W/export/ko-en)
for P in ja-en zh-en; do
  X=$W/export/$P; mkdir -p $X
  [ -f $X/manifest.jsonl ] || PYTHONPATH=core/zprobe $PY - $P $X <<'PYEOF'
import json, sys, soundfile as sf
import probe_z as pz
pair, out = sys.argv[1], sys.argv[2]; src, tgt = pair.split("-")
rows = pz.read_rows(src)
with open(f"{out}/manifest.jsonl", "w", encoding="utf-8") as f:
    for r in map(json.loads, open(f"models/zprobe/data/ev_big2_{pair}.jsonl", encoding="utf-8")):
        wav = str(pz.FLEURS / pz.LANGS[src][0] / "audio" / "test" / rows[r["id"]][1])
        f.write(json.dumps({"utt_id": f"{src}_{r['id']}", "wav": wav, "src_lang": src, "tgt_lang": tgt, "src_text": r["ref_src"],
                            "tgt_text": r["ref_tgt"], "duration_sec": round(sf.info(wav).duration, 3)}, ensure_ascii=False) + "\n")
PYEOF
done
run() {  # run <adapter> <pair> <beams>
  local M=$1 P=$2 B=$3; local TGT=${P#*-}; local X=$W/export/$P
  local N=zprobe-$M-agree2-beam${B}_$P; local F=$W/out/$N.jsonl
  [ "$(lines $F)" -eq "$(lines $X/manifest.jsonl)" ] || { idle; mark "stream $N"; PROBE_ADAPTER=$A/$M $PY -u core/zprobe/stream_bench.py \
     --manifest $X/manifest.jsonl --tgt $TGT --agree 2 --beams $B --out $F 2>&1 | f | grep --line-buffered -E "DONE|Error|Traceback"; }
  [ -f $W/runs/$N/summary.json ] && grep -q '"comet"' $W/runs/$N/summary.json && return
  (cd $BENCH && PYTHONPATH=.:bench/seamless bench/.venv/bin/python /home/mobility/STiTy/core/zprobe/stream_bench_score.py \
     --export $X --run $F --name fleurs_$P/$N --run-dir $W/runs/$N > /dev/null && \
   PATH=$HOME/.local/bin:$PATH uv run --project bench/metrics/comet python -m bench.metrics.comet --run-dir $W/runs/$N 2>&1 | tail -1)
}
for P in ko-en ja-en zh-en; do for B in 4 1; do run $NAME $P $B; done; done
run segconn2 ko-en 4
run $NAME en-ko 4

mark "5 report"
{
echo "## offline X->en: $NAME vs segconn2, big2, ASR->Hy"
$CPY core/zprobe/paired_delta.py --scores $H/scores --a segconn2 --b $NAME --pairs ko-en ja-en zh-en
$CPY core/zprobe/paired_delta.py --scores $H/scores --a big2 --b $NAME --pairs ko-en ja-en zh-en
$CPY core/zprobe/paired_delta.py --scores $H/scores --a asr --b $NAME --pairs ko-en ja-en zh-en
echo "## offline X->ko regression: $NAME vs segconn2"
$CPY core/zprobe/paired_delta.py --scores $H/scores --a segconn2 --b $NAME --pairs en-ko zh-ko ja-ko
echo "## streaming (real time, own <SEG>, agree 2)"
echo "| run | COMET | YAAL ms | YAAL CA ms | pieces / item | decodes / item | YAAL items |"; echo "|---|---|---|---|---|---|---|"
for N in $(ls $W/runs | grep -E "^zprobe-($NAME|segconn2)-agree2" | sort); do
  python3 -c "import json; s=json.load(open('$W/runs/$N/summary.json')); m=s['metrics']; c=s['counts']; print(f\"| $N | {m.get('comet',0):.3f} | {m['yaal_ms']:.0f} | {m['yaal_ca_ms']:.0f} | {m['segments_per_item']:.2f} | {m['decodes_per_item']:.1f} | {c['yaal_items']}/{c['items']} |\")"
done
} | tee $LOG/${NAME}_report.md
mark CHAIN_DONE
