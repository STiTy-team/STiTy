#!/usr/bin/env bash
set -euo pipefail

top_k=${TOPK:-}

if [[ ! -d bench/manager/node_modules ]]; then
  (cd bench/manager && npm ci)
fi

trap 'kill 0' EXIT
uv run --project bench python -m bench.manager.server ${top_k:+--top-k "$top_k"} &
cd bench/manager && npm run dev
