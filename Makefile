.PHONY: bench replay serve fmt

bench:
	uv run --project bench python -m bench --config $(CONFIG) --dataset $(DATASET)

replay:
	uv run --project bench python -m bench.replay $(RUN:%=--run-dir %) $(TOPK:%=--top-k %)

serve:
	uv run --project server python -m server --pipeline $(PIPELINE) $(HOST:%=--host %) $(PORT:%=--port %)

fmt:
	uvx ruff@0.16.9 format bench
