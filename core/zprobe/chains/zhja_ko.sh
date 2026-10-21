#!/bin/bash
# segconn2 on Chinese and Japanese speech -> Korean: offline (tagswap, same ids as the big2 / ASR->Hy comparison) and
# real-time streaming with its own <SEG> (stream_bench.py, agree 2, greedy and beam 4), scored with bench's YAAL code.
# Raw FLEURS test audio (bench has no zh/ja export, so no -23 dB normalization here).
#   tmux new-session -d -s zhjako -c <repo> "bash core/zprobe/chains/zhja_ko.sh > core/zprobe/chains/zhja_ko.log 2>&1"
cd /home/mobility/STiTy
PY=${PY:-.venv/bin/python}; CPY=${CPY:-.venv-tpm/bin/python}
BENCH=${BENCH:-/home/mobility/STiTy-bench}; W=${W:-/home/mobility/zprobe_bench}
D=models/zprobe/data; H=$D/hy
export PYTHONPATH=Qwen3-ASR PROBE_MODEL=models/Qwen3-ASR-1.7B-koen-seg-mix-merged PROBE_ADAPTER=models/zprobe/adapters/segconn2
mark() { echo "$(date +%T) MARK $1"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
idle() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -le 2000 ]; do sleep 60; done; }
for P in zh-ko ja-ko; do
  mark "offline $P"
  idle
  [ "$(lines $H/segconn2_$P.jsonl)" -eq "$(lines $D/ev_big2_$P.jsonl)" ] || $PY -u core/zprobe/probe_z.py tagswap --pair $P --out $H/segconn2_$P.jsonl 2>&1 | grep -v -i warn | tail -1
  X=$W/export/$P; mkdir -p $X
  [ -f $X/manifest.jsonl ] || PYTHONPATH=core/zprobe $PY - $P $X <<'PYEOF'
import json, sys, soundfile as sf
import probe_z as pz
pair, out = sys.argv[1], sys.argv[2]; src = pair.split("-")[0]
rows = pz.read_rows(src)
with open(f"{out}/manifest.jsonl", "w", encoding="utf-8") as f:
    for r in map(json.loads, open(f"models/zprobe/data/ev_big2_{pair}.jsonl", encoding="utf-8")):
        wav = str(pz.FLEURS / pz.LANGS[src][0] / "audio" / "test" / rows[r["id"]][1])
        f.write(json.dumps({"utt_id": f"{src}_{r['id']}", "wav": wav, "src_lang": src, "tgt_lang": "ko", "src_text": r["ref_src"],
                            "tgt_text": r["ref_tgt"], "duration_sec": round(sf.info(wav).duration, 3)}, ensure_ascii=False) + "\n")
PYEOF
  for B in 1 4; do
    NAME=zprobe-segconn2-agree2-beam${B}_$P; F=$W/out/$NAME.jsonl
    mark "stream $NAME"
    idle
    [ "$(lines $F)" -eq "$(lines $X/manifest.jsonl)" ] || $PY -u core/zprobe/stream_bench.py --manifest $X/manifest.jsonl --tgt ko --agree 2 --beams $B --out $F \
      2>&1 | grep --line-buffered -v -i -E "warn|pad_token|generation flags"
    (cd $BENCH && PYTHONPATH=.:bench/seamless bench/.venv/bin/python /home/mobility/STiTy/core/zprobe/stream_bench_score.py \
       --export $X --run $F --name fleurs_$P/$NAME --run-dir $W/runs/$NAME > /dev/null && \
     PATH=$HOME/.local/bin:$PATH uv run --project bench/metrics/comet python -m bench.metrics.comet --run-dir $W/runs/$NAME 2>&1 | tail -1)
  done
done
mark "offline COMET"
$CPY -u core/zprobe/comet_eval.py --n 0 --strip_seg --per_sent $H/scores segconn2_zh-ko=$H/segconn2_zh-ko.jsonl segconn2_ja-ko=$H/segconn2_ja-ko.jsonl 2>&1 | grep -v -i warn | tail -3
mark "report"
{
echo "## offline: segconn2 vs big2 and ASR->Hy"
$CPY core/zprobe/paired_delta.py --scores $H/scores --a big2 --b segconn2 --pairs zh-ko ja-ko
$CPY core/zprobe/paired_delta.py --scores $H/scores --a asr --b segconn2 --pairs zh-ko ja-ko
echo "## streaming (real time, own <SEG>, agree 2)"
echo "| run | COMET | YAAL ms | YAAL CA ms | pieces / item | decodes / item | YAAL items |"; echo "|---|---|---|---|---|---|---|"
for P in en-ko zh-ko ja-ko; do for B in 1 4; do
  N=zprobe-segconn2-agree2-beam$B; [ $P = en-ko ] || N=${N}_$P
  python3 -c "import json; s=json.load(open('$W/runs/$N/summary.json')); m=s['metrics']; c=s['counts']; print(f\"| $P beam$B | {m['comet']:.3f} | {m['yaal_ms']:.0f} | {m['yaal_ca_ms']:.0f} | {m['segments_per_item']:.2f} | {m['decodes_per_item']:.1f} | {c['yaal_items']}/{c['items']} |\")"
done; done
} | tee models/zprobe/logs/zhja_ko_report.md
mark CHAIN_DONE
