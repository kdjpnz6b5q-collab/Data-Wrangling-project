# DelayBot

DelayBot is a local QA pipeline for airline delay and cancellation policy questions.

## Quick start

```bash
cd delaybot
bash setup_local.sh
make all
make ask Q="If my American Airlines flight is canceled because of weather, do they owe me a hotel?"
```

You can also use `QUESTION` instead of `Q`:

```bash
make ask QUESTION="my plane got delayed because of weather"
```

For alliance-based alternative flights:

```bash
make recommend FLIGHT="AA123" ORIGIN="JFK" DEST="LHR" DEPART="2026-03-10T14:30"
```

## Click-box UI (guided follow-up)

If the question is missing details (like airline or disruption type), use the UI boxes:

```bash
make ui
```

Then open the local Streamlit URL (usually `http://localhost:8501`).

The UI now includes:
- Policy Q&A with missing-field dropdowns (airline/disruption/event type)
- Optional compensation inputs (delay hours and cancellation notice timing)
- Expected compensation guidance (non-guaranteed) and draft refund/compensation email text
- Alternative Flight Finder with:
  - flight number
  - depart airport code (3 letters)
  - destination airport code (3 letters)
  - departure time
  - alliance-aware recommendations + Google Flights links + airline contact links

## Scraping architecture

Each airline has its own scraping script, named after the airline:

- `src/scrape/scrape_dot.py`
- `src/scrape/scrape_american.py`
- `src/scrape/scrape_delta.py`
- `src/scrape/scrape_united.py`
- `src/scrape/scrape_southwest.py`
- `src/scrape/scrape_jetblue.py`
- `src/scrape/scrape_alaska.py`
- `src/scrape/scrape_frontier.py`
- `src/scrape/scrape_spirit.py`
- `src/scrape/scrape_hawaiian.py`
- `src/scrape/scrape_allegiant.py`
- `src/scrape/scrape_avelo.py`
- `src/scrape/scrape_breeze.py`
- `src/scrape/scrape_sun_country.py`
- `src/scrape/scrape_lufthansa.py`
- `src/scrape/scrape_ryanair.py`
- `src/scrape/scrape_easyjet.py`
- `src/scrape/scrape_air_france.py`
- `src/scrape/scrape_british_airways.py`
- `src/scrape/scrape_emirates.py`
- `src/scrape/scrape_qatar_airways.py`
- `src/scrape/scrape_singapore_airlines.py`
- `src/scrape/scrape_turkish_airlines.py`
- `src/scrape/scrape_air_canada.py`
- `src/scrape/scrape_klm.py`
- `src/scrape/scrape_iberia.py`
- `src/scrape/scrape_latam.py`
- `src/scrape/scrape_avianca.py`
- `src/scrape/scrape_etihad.py`
- `src/scrape/scrape_virgin_atlantic.py`
- `src/scrape/scrape_ana.py`
- `src/scrape/scrape_japan_airlines.py`
- `src/scrape/scrape_china_eastern.py`
- `src/scrape/scrape_china_southern.py`
- `src/scrape/scrape_air_china.py`
- `src/scrape/scrape_indigo.py`
- `src/scrape/scrape_qantas.py`
- `src/scrape/scrape_saudia.py`
- `src/scrape/scrape_swiss.py`

`src/scrape/scrape_pages.py` runs all of them in sequence.

Current coverage:
- U.S.-based airlines: American, Delta, United, Southwest, JetBlue, Alaska, Frontier, Spirit, Hawaiian, Allegiant, Avelo, Breeze, Sun Country
- Europe: Lufthansa, Ryanair, easyJet, Air France, British Airways, KLM, Iberia, SWISS, Virgin Atlantic
- Middle East: Emirates, Qatar Airways, Etihad, Saudia, Turkish Airlines
- Asia-Pacific: Singapore Airlines, ANA, Japan Airlines, IndiGo, Qantas
- Americas (additional): Air Canada, LATAM, Avianca
- China: Air China, China Eastern, China Southern

## Pipeline

1. `src/scrape/scrape_pages.py`
   - Runs all per-airline scripts.
   - Each script uses direct fetch, then `r.jina.ai` fallback, then local seed fallback.
2. `src/process/extract_text.py`
   - Extracts text from raw HTML.
3. `src/process/chunk_policy_text.py`
   - Chunks policy text for retrieval.
4. `src/analysis/tag_policy_chunks.py`
   - Adds tags (weather, ATC/NAS, mechanical, late inbound aircraft, strike/labor, hotel, meal, refund, compensation, controllable/uncontrollable, etc.).
5. `src/agent/ask_policy.py`
   - Answers in CLI.
   - Prompts for missing airline/disruption info if needed.
   - Supports optional event type and timing overrides.
   - Adds post-answer contact instruction, expected compensation guidance, and draft refund email output.
6. `src/agent/ask_policy_ui.py`
   - Streamlit UI with clickable dropdown boxes for missing fields.
   - Adds event type, delay-hours, and cancellation-notice inputs.
   - Shows contact guidance, expected compensation section, and draft email text.
7. `src/agent/flight_recommender.py`
   - Alliance-aware alternative flight recommendations (Phase 1, no live fare feed).
8. `src/agent/recommend_alternatives.py`
   - CLI wrapper for alternative-flight recommendations.

## Live integration note

DelayBot does not scrape Google Flights directly. It generates Google Flights search links and alliance-based options.
DelayBot can now use live offers from Amadeus when credentials are set.

Set credentials before running (three supported options):

Option 1 (shell env vars):
```bash
export AMADEUS_CLIENT_ID="your_client_id"
export AMADEUS_CLIENT_SECRET="your_client_secret"
# optional, defaults to test endpoint:
export AMADEUS_BASE_URL="https://test.api.amadeus.com"
```

Option 2 (`delaybot/.env`):
```bash
AMADEUS_CLIENT_ID=your_client_id
AMADEUS_CLIENT_SECRET=your_client_secret
AMADEUS_BASE_URL=https://test.api.amadeus.com
```

Option 3 (Streamlit secrets in `.streamlit/secrets.toml`):
```toml
AMADEUS_CLIENT_ID = "your_client_id"
AMADEUS_CLIENT_SECRET = "your_client_secret"
AMADEUS_BASE_URL = "https://test.api.amadeus.com"
```

If credentials are missing or API calls fail, DelayBot automatically falls back to alliance-based recommendations.

## Data outputs

- `data/raw/html/*.html`
- `data/processed/policy_texts.csv`
- `data/processed/policy_chunks.csv`
- `data/processed/policy_chunks_tagged.csv`
