# DelayBot

DelayBot is a small local QA pipeline for airline delay/cancellation policy questions.

## Quick start

```bash
cd delaybot
bash setup_local.sh
make all
make ask Q="If my American Airlines flight is canceled because of weather, do they owe me a hotel?"
```

## What it does

1. `src/scrape/scrape_pages.py`
   - Tries direct page fetch.
   - Falls back to `r.jina.ai` for blocked pages.
   - Falls back to local seed policy text if network fetch fails.
2. `src/process/extract_text.py`
   - Extracts text from raw HTML.
3. `src/process/chunk_policy_text.py`
   - Chunks long policy text into retrieval-sized segments.
4. `src/analysis/tag_policy_chunks.py`
   - Applies keyword-based policy tags.
5. `src/agent/ask_policy.py`
   - Retrieves relevant chunks and returns an evidence-backed answer.

## Data outputs

- `data/raw/html/*.html`
- `data/processed/policy_texts.csv`
- `data/processed/policy_chunks.csv`
- `data/processed/policy_chunks_tagged.csv`
