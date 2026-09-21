PYTHON ?= python3

.PHONY: bench replay

bench:
	$(PYTHON) -m bench $(CONFIG)

replay:
	$(PYTHON) -m bench.replay $(RUN)
