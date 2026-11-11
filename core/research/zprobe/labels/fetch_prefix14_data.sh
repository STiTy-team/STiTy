#!/bin/bash
# prefix14 data: (1) AI Hub 71693 Korean talks (ko->en, ~5,000 sentences) and English talks (en->ko extra,
# ~2,500 sentences) -> datasets/aihub71693/{ko,en}/manifest.jsonl - the zips are already local, convert pulls
# only the chosen talks' wavs; (2) Zeroth-Korean train parquet into our own folder; (3) kosp2e
# zeroth_train.xlsx (English side) joined with Zeroth audio+transcript -> 3,600 wavs + manifest.
# Nothing here touches the shared bench manifests (kosp2e/zeroth test). Markers P14D MARK/FAIL/DONE.
# Talks listed in $OUT/aihub_skip.txt are left out (audio convert.py cannot read, mixed directions).
cd "$(git rev-parse --show-toplevel)" || exit 1
REPO=$(pwd)
set -o pipefail
trap 'echo "P14D_EXIT $(date "+%F %T")"' EXIT
set -a; . /home/skkai/STiTy/.env; set +a
export PATH=$HOME/.local/bin:$PATH
export AIHUB_APIKEY="${AIHUB_API_KEY:-}"
DS=/home/skkai/STiTy-team/datasets
OUT=models/zprobe/data/prefix14
mkdir -p $OUT
rows() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
touch $OUT/aihub_skip.txt

pick_talks() {  # $1 lang (ko|en)  $2 per-domain count  -> comma list on stdout
  .venv/bin/python - "$DS" "$1" "$2" "$OUT/aihub_skip.txt" <<'PYEOF'
import sys, collections
from pathlib import Path
ds, lang, per, skipf = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
sys.path.insert(0, ds + "/aihub71693")
import convert
tgt = "en" if lang == "ko" else "ko"
skip = {l.strip() for l in open(skipf) if l.strip()}
talks = convert.read_labels(Path(ds + "/aihub71693/data"), lang)
by = collections.defaultdict(list)
for t, rows in talks.items():
    if t in skip:
        continue
    if any(r.get("source_language") != lang or r.get("target_language") != tgt for r in rows):
        continue                                   # convert.py refuses talks with other directions
    n = len(rows)
    if 80 <= n <= 320:                             # mid-sized talks, 4 domains evenly
        by[rows[0]["domain"]].append((n, t))
pick = []
for dom, v in sorted(by.items()):
    v.sort(reverse=True)
    pick += [t for _, t in v[:per]]
if lang == "ko":
    for t in ("K_MA_50095", "K_EA_10085", "K_SA_10208"):   # the three already converted
        if t not in pick and t not in skip:
            pick.append(t)
print(",".join(pick))
PYEOF
}

echo "$(date '+%F %T') P14D MARK aihub ko talks"
if [ "$(rows $DS/aihub71693/ko/manifest.jsonl)" -ge 11000 ]; then
  echo "[skip] aihub ko manifest already has $(rows $DS/aihub71693/ko/manifest.jsonl) rows"
else
  TALKS_KO=$(pick_talks ko 12) || { echo P14D_FAIL aihub-ko-select; exit 1; }
  echo "ko talks: $(echo "$TALKS_KO" | tr ',' '\n' | wc -l)"
  ( cd $DS && uv run python aihub71693/convert.py --lang ko --talks "$TALKS_KO" ) || { echo P14D_FAIL aihub-ko-convert; exit 1; }
  echo "aihub ko manifest rows: $(wc -l < $DS/aihub71693/ko/manifest.jsonl)"
fi

echo "$(date '+%F %T') P14D MARK aihub en talks (en->ko extra, Korean references)"
if [ "$(rows $DS/aihub71693/en/manifest.jsonl)" -ge 2000 ]; then
  echo "[skip] aihub en manifest already has $(rows $DS/aihub71693/en/manifest.jsonl) rows"
else
  # the English audio zips (VS_en_EA/MA/NA/SA = file keys 546225..546228, ~35 GB) were never downloaded
  ( cd $DS/aihub71693/data && need=
    for z in VS_en_EA VS_en_MA VS_en_NA VS_en_SA; do find . -name "$z.zip" -print -quit | grep -q . || need=1; done
    if [ -n "$need" ]; then
      [ -x ../aihubshell ] || { curl -fsSL -o ../aihubshell https://api.aihub.or.kr/api/aihubshell.do && chmod +x ../aihubshell; }
      ../aihubshell -mode d -datasetkey 71693 -filekey 546225,546226,546227,546228 -aihubapikey "$AIHUB_APIKEY" || true
      for z in VS_en_EA VS_en_MA VS_en_NA VS_en_SA; do find . -name "$z.zip" -print -quit | grep -q . || { echo "missing $z.zip after download"; exit 1; }; done
    fi ) || { echo P14D_FAIL aihub-en-download; exit 1; }
  TALKS_EN=$(pick_talks en 5) || { echo P14D_FAIL aihub-en-select; exit 1; }
  echo "en talks: $(echo "$TALKS_EN" | tr ',' '\n' | wc -l)"
  ( cd $DS && uv run python aihub71693/convert.py --lang en --talks "$TALKS_EN" ) || { echo P14D_FAIL aihub-en-convert; exit 1; }
  echo "aihub en manifest rows: $(wc -l < $DS/aihub71693/en/manifest.jsonl)"
fi

echo "$(date '+%F %T') P14D MARK zeroth train parquet"
mkdir -p $OUT/zeroth_train
if ! ls $OUT/zeroth_train/data/train-*.parquet >/dev/null 2>&1; then
  ( cd $DS && uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('kresnik/zeroth_korean', repo_type='dataset', allow_patterns=['data/train-*.parquet'], local_dir='$REPO/$OUT/zeroth_train')" ) \
    || { echo P14D_FAIL zeroth-download; exit 1; }
fi
ls $OUT/zeroth_train/data/ | head -3

echo "$(date '+%F %T') P14D MARK kosp2e zeroth join"
XLSX=$OUT/kosp2e_zeroth_train.xlsx
[ -f $XLSX ] || curl -fsSL -o $XLSX https://raw.githubusercontent.com/warnikchow/kosp2e/main/split/zeroth_train.xlsx || { echo P14D_FAIL xlsx; exit 1; }
.venv/bin/python -u core/research/zprobe/labels/fetch_prefix14_zeroth.py \
  --parquet-dir $OUT/zeroth_train/data --xlsx $XLSX --n 3600 --out-dir $OUT/kosp2e_zeroth \
  || { echo P14D_FAIL zeroth-join; exit 1; }
df -h / | tail -1
echo "P14D_DONE $(date '+%F %T')"
