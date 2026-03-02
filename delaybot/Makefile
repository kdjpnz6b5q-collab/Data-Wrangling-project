PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python

.PHONY: setup scrape extract chunk tag all ask clean

setup:
	bash setup_local.sh

scrape:
	$(PY) src/scrape/scrape_pages.py

extract:
	$(PY) src/process/extract_text.py

chunk:
	$(PY) src/process/chunk_policy_text.py

tag:
	$(PY) src/analysis/tag_policy_chunks.py

all: scrape extract chunk tag

ask:
	@if [ -z "$(Q)" ]; then \
		echo "Usage: make ask Q='your question'"; \
		exit 1; \
	fi
	$(PY) src/agent/ask_policy.py "$(Q)"

clean:
	rm -rf data/raw/html/*.html data/processed/*.csv
