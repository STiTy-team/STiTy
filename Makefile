-include .env

export CUDA_HOME ?= /usr/local/cuda
export PATH := $(CUDA_HOME)/bin:$(PATH)

.PHONY: server bench bench-batch bench-manager bench-worker configs-pull configs-push configs-list fmt env

server:
	bash scripts/server/run.sh

bench:
	bash scripts/bench/run.sh $(CONFIG) $(DATASET)

bench-batch:
	bash scripts/bench/batch.sh $(LIST)

bench-manager:
	TOPK=$(TOPK) bash scripts/bench/manager.sh

bench-worker:
	uv run $(if $(wildcard .env),--env-file .env) --project bench/worker python -m bench.worker

configs-pull:
	uv run --project bench python -m bench.shared_configs pull $(ONLY)

configs-push:
	uv run --project bench python -m bench.shared_configs push

configs-list:
	uv run --project bench python -m bench.shared_configs list

fmt:
	uvx ruff@0.16.9 format .
	cd mobile && npx --yes prettier@3.9.10 --write .
	cd bench/manager && npx --yes prettier@3.9.10 --write src

env:
	@echo CUDA_HOME=$(CUDA_HOME)
	@echo STITY_DATA_ROOT=$(or $(STITY_DATA_ROOT),(not set - required))
