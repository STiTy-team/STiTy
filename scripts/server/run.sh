#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ -f .env ]]; then
  while IFS='=' read -r key value; do
    [[ $key =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [[ -n ${!key:-} ]] || export "$key=$value"
  done <.env
fi

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH

pipeline=${PIPELINE:-}
host=${HOST:-}
port=${PORT:-}
stage=${STAGE:-dev}

if [[ -n $pipeline && -n ${STITY_S3_BUCKET:-} ]]; then
  uv run --project bench python -m bench.shared_configs pull "pipeline/$pipeline"
fi

if [[ -n $pipeline ]]; then export STITY_STITY__PIPELINE=$pipeline; fi
if [[ -n $host ]]; then export STITY_SERVER__HOST=$host; fi
if [[ -n $port ]]; then export STITY_SERVER__PORT=$port; fi

if [[ $stage == dev ]]; then
  export STITY_RECORDING__ALLOWED=${STITY_RECORDING__ALLOWED:-true}
  trap 'kill 0' EXIT
  bash scripts/server/tunnel.sh "${port:-8765}" &
fi

PYTHONPATH=server uv run --project server python -m app
