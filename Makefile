.PHONY: bench bench-batch bench-manager configs-pull configs-push configs-list server bench-worker fmt env

-include .env
export CUDA_HOME ?= /usr/local/cuda
export PATH      := $(CUDA_HOME)/bin:$(PATH)

bench:
	run_dir="$$(uv run --project bench python -m bench --config $(CONFIG) --dataset $(DATASET) --print-run-dir)" \
	  && uv run --project bench python -m bench --config $(CONFIG) --dataset $(DATASET) \
	  && uv run --project bench/metrics/comet python -m bench.metrics.comet --run-dir "$$run_dir"

bench-batch:
	bash scripts/bench/batch.sh $(LIST)

bench-manager:
	cd bench/manager && { [ -d node_modules ] || npm ci; }
	trap 'kill 0' EXIT; uv run --project bench python -m bench.manager.server $(TOPK:%=--top-k %) & cd bench/manager && npm run dev

configs-pull:
	uv run --project bench python -m bench.shared_configs pull $(ONLY)

configs-push:
	uv run --project bench python -m bench.shared_configs push

configs-list:
	uv run --project bench python -m bench.shared_configs list

server:
	$(if $(and $(PIPELINE),$(STITY_S3_BUCKET)),uv run --project bench python -m bench.shared_configs pull pipeline/$(PIPELINE))
	$(PIPELINE:%=STITY_STITY__PIPELINE=%) $(HOST:%=STITY_SERVER__HOST=%) $(PORT:%=STITY_SERVER__PORT=%) PYTHONPATH=server uv run --project server python -m app

bench-worker:
	uv run $(patsubst %,--env-file %,$(wildcard .env)) --project bench/worker python -m bench.worker

fmt:
	uvx ruff@0.16.9 format bench server

env:
	@echo CUDA_HOME=$(CUDA_HOME)
	@echo STITY_DATA_ROOT=$(or $(STITY_DATA_ROOT),(not set - required))
