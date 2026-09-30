#!/usr/bin/env bash
# Run the plan rows of the given tiers one after another (one GPU job at a time).
#
#   tmux new-session -d -s upgrade -c "$PWD" "bash experiments/upgrade-260930/run.sh 0 1"
#
# Each row is `make bench CONFIG=<pipeline> DATASET=<dataset>` (bench, then COMET).
# A row that finished has logs/upgrade-260930/status/<dataset>__<pipeline>.ok and is
# skipped on the next start; a failed row gets .failed and the chain moves on.
# Progress: logs/upgrade-260930/status/chain.log. Per run: logs/upgrade-260930/<row>.log.
set -u
cd "$(dirname "$0")/../.."
set -a; [ -f .env ] && . ./.env; set +a
PLAN=experiments/upgrade-260930/plan.tsv
LOGS=logs/upgrade-260930
mkdir -p "$LOGS/status"
TIERS=" $* "
tail -n +2 "$PLAN" | while IFS=$'\t' read -r tier id pipeline dataset base watch; do
  [[ "$TIERS" == *" $tier "* ]] || continue
  key="${dataset}__${pipeline}"
  [ -f "$LOGS/status/$key.ok" ] && continue
  echo "$(date -Is) start tier=$tier $id $pipeline x $dataset" >> "$LOGS/status/chain.log"
  if make bench CONFIG="$pipeline" DATASET="$dataset" > "$LOGS/$key.log" 2>&1 < /dev/null; then
    touch "$LOGS/status/$key.ok"; rm -f "$LOGS/status/$key.failed"
    echo "$(date -Is) ok    $id" >> "$LOGS/status/chain.log"
  else
    touch "$LOGS/status/$key.failed"
    echo "$(date -Is) FAIL  $id (see $LOGS/$key.log)" >> "$LOGS/status/chain.log"
  fi
done
echo "$(date -Is) done tiers:$TIERS" >> "$LOGS/status/chain.log"
