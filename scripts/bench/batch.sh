#!/usr/bin/env bash
set -u

list=${1:?usage: make bench-batch LIST=<file with one "pipeline dataset" pair per line>}
failed=()

while read -r config dataset _ || [[ -n ${config:-} ]]; do
  [[ -z ${config:-} || $config == \#* ]] && continue
  if [[ -z ${dataset:-} ]]; then
    echo "=== $(date '+%F %T') $config has no dataset, skipped"
    failed+=("$config × (no dataset)")
    continue
  fi
  echo "=== $(date '+%F %T') $config × $dataset"
  if ! make bench CONFIG="$config" DATASET="$dataset" </dev/null; then
    failed+=("$config × $dataset")
  fi
done <"$list"

echo "=== $(date '+%F %T') finished, ${#failed[@]} failed"
for run in "${failed[@]}"; do echo "  failed: $run"; done
[[ ${#failed[@]} -eq 0 ]]
