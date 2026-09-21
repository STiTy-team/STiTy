PYTHON ?= python3

.PHONY: bench replay

bench:
	$(PYTHON) -m bench --config $(CONFIG) --dataset $(DATASET)

replay:
	$(PYTHON) -m bench.replay $(RUN) $(TOPK:%=--top-k %)
