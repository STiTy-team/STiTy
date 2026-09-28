.PHONY: bench replay server fmt

bench:
	uv run --project bench python -m bench --config $(CONFIG) --dataset $(DATASET)
	uv run --project bench/metrics/comet python -m bench.metrics.comet --run-dir "$$(uv run --project bench python -m bench --config $(CONFIG) --dataset $(DATASET) --print-run-dir)"

replay:
	uv run --project bench python -m bench.replay $(RUN:%=--run-dir %) $(TOPK:%=--top-k %)

server:
	$(PIPELINE:%=STITY_STITY__PIPELINE=%) $(HOST:%=STITY_SERVER__HOST=%) $(PORT:%=STITY_SERVER__PORT=%) PYTHONPATH=server uv run --project server python -m app

fmt:
	uvx ruff@0.16.9 format bench server
