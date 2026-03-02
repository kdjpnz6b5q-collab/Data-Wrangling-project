PYTHON ?= python3
VENV ?= .venv
PY := $(VENV)/bin/python

.PHONY: setup scrape extract chunk tag all ask recommend ui clean

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
	@QUESTION_INPUT="$(Q)"; \
	if [ -z "$$QUESTION_INPUT" ]; then QUESTION_INPUT="$(QUESTION)"; fi; \
	if [ -z "$$QUESTION_INPUT" ]; then \
		echo "Usage: make ask Q='your question'"; \
		echo "   or: make ask QUESTION='your question'"; \
		exit 1; \
	fi; \
	$(PY) src/agent/ask_policy.py "$$QUESTION_INPUT"

recommend:
	@if [ -z "$(FLIGHT)" ] || [ -z "$(ORIGIN)" ] || [ -z "$(DEST)" ] || [ -z "$(DEPART)" ]; then \
		echo "Usage: make recommend FLIGHT='AA123' ORIGIN='JFK' DEST='LHR' DEPART='2026-03-10T14:30'"; \
		exit 1; \
	fi
	$(PY) src/agent/recommend_alternatives.py \
		--flight-number "$(FLIGHT)" \
		--origin "$(ORIGIN)" \
		--destination "$(DEST)" \
		--departure-time "$(DEPART)"

ui:
	$(PY) -m streamlit run src/agent/ask_policy_ui.py

clean:
	rm -rf data/raw/html/*.html data/processed/*.csv
