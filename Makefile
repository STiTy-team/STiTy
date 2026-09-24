.PHONY: bench replay

bench:
	uv run --project bench python -m bench --config $(CONFIG) --dataset $(DATASET)
	uv run --project bench/comet python -m bench.comet bench/runs/$(CONFIG)-$(DATASET)

replay:
	uv run --project bench python -m bench.replay $(RUN) $(TOPK:%=--top-k %)
