#!/usr/bin/env bash
set -euo pipefail

config=${1:?usage: make bench CONFIG=<pipeline> DATASET=<dataset>}
dataset=${2:?usage: make bench CONFIG=<pipeline> DATASET=<dataset>}

run_dir=$(uv run --project bench python -m bench --config "$config" --dataset "$dataset" --print-run-dir)
uv run --project bench python -m bench --config "$config" --dataset "$dataset"
uv run --project bench/metrics/comet python -m bench.metrics.comet --run-dir "$run_dir"
