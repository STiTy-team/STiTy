set -u
cd /home/mobility/STiTy || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LANGSMITH_TRACING=0
L=core/meaning_segmentator/experiment/artifacts/en2x/covost2/full_judge44/logs/hset.log
mkdir -p "$(dirname "$L")"
.venv-autoseg/bin/python -u core/meaning_segmentator/tools/covost2_chain/hset_conditions.py \
    --run full_j44v0 --run full_j44best >> "$L" 2>&1
rc=$?
echo "== $(date '+%F %T') hset exit=$rc" >> "$L"
