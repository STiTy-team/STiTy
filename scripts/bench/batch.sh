#!/usr/bin/env bash
set -u

list=${1:?usage: make batch LIST=<file with one "pipeline dataset" pair per line>}
failed=()

while read -r config dataset _; do
  [[ -z ${config:-} || $config == \#* ]] && continue
  echo "=== $(date '+%F %T') $config × $dataset"
  if ! make bench CONFIG="$config" DATASET="$dataset" </dev/null; then
    failed+=("$config × $dataset")
  fi
done <"$list"

echo "=== $(date '+%F %T') finished, ${#failed[@]} failed"
for run in "${failed[@]}"; do echo "  failed: $run"; done
[[ ${#failed[@]} -eq 0 ]]
