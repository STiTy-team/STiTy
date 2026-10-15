#!/bin/bash
# In-domain check of the distillation: big2 vs bigcv (and the Hy cascade) on CoVoST2 en TEST clips (never trained on;
# sentences that also occur in the distillation train shard are excluded). 600 clips, same clips for every target.
# en-ja / en-zh: COMET-DA against CoVoST2 gold. en-ko has no reference: CometKiwi (reference-free) for all three.
#   tmux new-session -d -s cv-test -c /home/mobility/STiTy "bash core/zprobe/chains/cv_test_eval.sh > core/zprobe/chains/cv_test_eval.log 2>&1"
cd /home/mobility/STiTy
set -a; . ./.env; set +a
export PYTHONPATH=Qwen3-ASR
PY=.venv/bin/python; CPY=.venv-tpm/bin/python
A=models/zprobe/adapters; LOG=models/zprobe/logs
C=models/zprobe/data/cvtest; mkdir -p $C/da $C/kiwi
mark() { echo "$(date +%T) MARK $1"; }
lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
quiet() { grep -v -E "pad_token_id|Warning|warn" ; }

mark "1 manifest"
[ -f $C/man_en-ko.jsonl ] || $PY - <<'EOF'
import glob, json, random
import pyarrow.parquet as pq
C = "models/zprobe/data/cvtest"; W = "/home/mobility/datasets/covost2_single/en_test"
load = lambda d: {r["id"]: r for f in sorted(glob.glob(f"/home/mobility/datasets/covost2/{d}/test-*.parquet"))
                  for r in pq.read_table(f, columns=["id", "sentence", "translation"]).to_pylist()}
ja, zh = load("en_ja"), load("en_zh-CN")
clean = lambda s: s.strip().strip('"').strip()
train = {json.loads(l)["src"].lower() for l in open("/home/mobility/datasets/covost2_train_en/meta.jsonl")}
ids = sorted(i for i in set(ja) & set(zh) if clean(ja[i]["sentence"]).lower() not in train)
pick = random.Random(0).sample(ids, 600)
for tgt, src in [("ja", ja), ("zh", zh), ("ko", None)]:
    with open(f"{C}/man_en-{tgt}.jsonl", "w", encoding="utf-8") as f:
        for i in pick:
            f.write(json.dumps({"id": i, "audio": f"{W}/{i}.wav", "ref_src": clean(ja[i]["sentence"]),
                                "ref_tgt": src[i]["translation"].strip() if src else ""}, ensure_ascii=False) + "\n")
print("manifest", len(pick), "of", len(ids))
EOF

mark "2 adapters"
for ad in big2 bigcv; do for t in ja zh ko; do
  [ "$(lines $C/${ad}_en-$t.jsonl)" -eq 600 ] && continue
  PROBE_ADAPTER=$A/$ad $PY -u core/zprobe/eval_manifest.py --manifest $C/man_en-$t.jsonl --tgt $t --out $C/${ad}_en-$t.jsonl 2>&1 | quiet | tail -1
done; done

mark "3 cascade"
[ "$(lines $C/asr_base_en.jsonl)" -eq 600 ] || $PY -u core/zprobe/eval_manifest.py --manifest $C/man_en-ja.jsonl --tgt en --out $C/asr_base_en.jsonl 2>&1 | quiet | tail -1
$PY -u core/zprobe/cascade_hy.py --pair en-ja en-zh en-ko --ids "$C/man_{pair}.jsonl" --asr "$C/asr_base_{src}.jsonl" --out "$C/asr_{pair}.jsonl" 2>&1 | quiet | tail -2
$PY -u core/zprobe/cascade_hy.py --pair en-ja en-zh en-ko --ids "$C/man_{pair}.jsonl" --gold --out "$C/gold_{pair}.jsonl" 2>&1 | quiet | tail -2

mark "4 COMET"
da=""; kiwi=""
for k in big2 bigcv asr gold; do
  for t in ja zh; do da="$da ${k}_en-$t=$C/${k}_en-$t.jsonl"; done
  for t in ja zh ko; do kiwi="$kiwi ${k}_en-$t=$C/${k}_en-$t.jsonl"; done
done
$CPY -u core/zprobe/comet_eval.py --n 0 --per_sent $C/da $da 2>&1 | quiet | tail -10
$CPY -u core/zprobe/comet_eval.py --n 0 --kiwi --per_sent $C/kiwi $kiwi 2>&1 | quiet | tail -14

mark "5 paired"
for m in da kiwi; do
  P="en-ja en-zh"; [ $m = kiwi ] && P="en-ja en-zh en-ko"
  echo "## $m: bigcv vs big2"; $CPY core/zprobe/paired_delta.py --scores $C/$m --a big2 --b bigcv --pairs $P
  echo "## $m: bigcv vs ASR->Hy"; $CPY core/zprobe/paired_delta.py --scores $C/$m --a asr --b bigcv --pairs $P
done | tee $LOG/cvtest_bigcv.md
mark CHAIN_DONE
