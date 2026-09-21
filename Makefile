PYTHON ?= python3

.PHONY: bench replay

bench:
	$(PYTHON) -m bench $(CONFIG)

replay:
	$(PYTHON) -m bench.replay $(RUN)

.PHONY: retranslate annotate-quality asr-text-robustness

retranslate:
	$(PYTHON) -m bench.retranslate $(SOURCE) $(CONFIG)

annotate-quality:
	$(PYTHON) -m bench.annotate_quality $(SOURCE) $(OUTPUT)

asr-text-robustness:
	$(PYTHON) -m bench.asr_text_robustness $(SOURCE) $(CONFIG) $(OUTPUT)
