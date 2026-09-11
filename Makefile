# Thin aliases over `python -m bench`. The knobs live in the YAML config, not here.
#
# Note: make takes variables as NAME=value, not --name=value -- `make bench --config=x`
# makes make itself reject "--config" as an unknown option.

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
CONFIG ?= bench/configs/smoke-fleurs.yml
DATA   ?= fleurs
ARGS   ?=

# Datasets live in their own repo (github.com/STiTy-team/datasets). Check it out
# wherever you like and point STITY_DATA_ROOT at it -- the checkout root IS the
# data root. DATASETS_REPO is only used by `make dataset`, which is a convenience
# wrapper around that repo's install scripts.
DATASETS_REPO ?= $(if $(STITY_DATA_ROOT),$(STITY_DATA_ROOT),../datasets)

.PHONY: bench dry bench-smoke dataset validate test help

help:
	@echo "make bench CONFIG=bench/configs/<n>.yml   run a benchmark"
	@echo "make dry   CONFIG=...                     resolve + validate, no model"
	@echo "make bench-smoke                          5-item FLEURS smoke run (free)"
	@echo "make eval  DATA=fleurs                   run bench/configs/<DATA>.yml"
	@echo "make dataset DATA=fleurs [SRC=...]       install + convert (in the datasets repo)"
	@echo "make validate DATA=fleurs                validate a converted dataset"
	@echo "make test                                 unit tests (no GPU)"

bench:
	$(PYTHON) -m bench --config $(CONFIG) $(ARGS)

dry:
	$(PYTHON) -m bench --config $(CONFIG) --dry-run $(ARGS)

bench-smoke:
	$(PYTHON) -m bench --config bench/configs/smoke-fleurs.yml $(ARGS)

eval:
	$(PYTHON) -m bench --config bench/configs/$(DATA).yml $(ARGS)

dataset:
	@test -d "$(DATASETS_REPO)/$(DATA)" || { \
		echo "no $(DATA)/ under $(DATASETS_REPO)."; \
		echo "Clone the dataset repo and point STITY_DATA_ROOT (or DATASETS_REPO) at it:"; \
		echo "  git clone git@github.com:STiTy-team/datasets.git ~/datasets"; \
		echo "  export STITY_DATA_ROOT=~/datasets"; exit 1; }
	bash $(DATASETS_REPO)/$(DATA)/install.sh $(ARGS)

validate:
	$(PYTHON) -m bench.data.manifest --validate $(DATA)

test:
	@fail=0; for t in bench/tests/test_*.py; do \
		printf '%-34s' "$$(basename $$t)"; \
		if $(PYTHON) $$t >/tmp/bench_test_out 2>&1; then \
			grep -E '^(OK|Ran )' /tmp/bench_test_out | tr '\n' ' '; echo; \
		else \
			echo FAILED; cat /tmp/bench_test_out; fail=1; \
		fi; \
	done; exit $$fail
