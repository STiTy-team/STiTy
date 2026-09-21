PYTHON ?= python3

.PHONY: bench replay

bench:
	$(PYTHON) -m bench $(CONFIG)

replay:
	$(PYTHON) -m bench.replay $(RUN)

.PHONY: retranslate annotate-quality asr-text-robustness

retranslate:
	$(PYTHON) -m core.utils.metrics.retranslate $(SOURCE) $(CONFIG)

annotate-quality:
	$(PYTHON) -m core.utils.metrics.annotate_quality $(SOURCE) $(OUTPUT)

asr-text-robustness:
	$(PYTHON) -m core.utils.metrics.asr_text_robustness $(SOURCE) $(CONFIG) $(OUTPUT)
