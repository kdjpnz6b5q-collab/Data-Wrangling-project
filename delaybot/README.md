# DelayBot

DelayBot is a Streamlit app for airline disruption support.
It helps travelers understand likely rights for delays/cancellations and suggests alternative flights.

## Live app

- Streamlit: https://kdjpnz6b5q-collab-data-wra-delaybotsrcagentask-policy-ui-fpq4m6.streamlit.app/

## What it does

- Answers delay/cancellation policy questions with airline-aware guidance
- Prompts for missing context (airline, disruption type, event type)
- Generates expected compensation guidance (non-guaranteed)
- Generates a draft refund/compensation email
- Recommends alternative flights (alliance + regional fallback)
- Supports optional live fare offers from Amadeus (with credentials)

## Project structure

- `src/agent/ask_policy_ui.py`: Streamlit app
- `src/agent/policy_engine.py`: policy query + answer logic
- `src/agent/flight_recommender.py`: alternative flight logic + Amadeus integration
- `src/agent/ask_policy.py`: CLI Q&A
- `src/agent/recommend_alternatives.py`: CLI alternatives tool
- `src/scrape/`: per-airline scrapers
- `data/seeds/fallback_policies.json`: built-in fallback policy data
- `requirements.txt`: Streamlit Cloud runtime dependencies

## Local setup

```bash
cd delaybot
bash setup_local.sh
```

Run CLI policy Q&A:

```bash
make ask Q="My Delta flight is delayed 4 hours because of weather. What can I expect?"
```

Run CLI alternatives:

```bash
make recommend FLIGHT="AA123" ORIGIN="JFK" DEST="LHR" DEPART="2026-03-10T14:30"
```

Run Streamlit UI:

```bash
make ui
```

## Streamlit Cloud deploy

Use these settings in Streamlit Community Cloud:

- Repository: `kdjpnz6b5q-collab/Data-Wrangling-project`
- Branch: `main`
- App file: `delaybot/src/agent/ask_policy_ui.py`

### Optional secrets (for live Amadeus fares)

Add in Streamlit app secrets:

```toml
AMADEUS_CLIENT_ID = "your_client_id"
AMADEUS_CLIENT_SECRET = "your_client_secret"
AMADEUS_BASE_URL = "https://test.api.amadeus.com"
```

If secrets are missing, DelayBot automatically falls back to alliance-based recommendations.

## Data behavior

DelayBot checks data in this order:

1. `data/processed/policy_chunks_tagged.csv`
2. `data/processed/policy_chunks.csv`
3. `data/seeds/fallback_policies.json` (automatic fallback for cloud deploys)

To rebuild full processed data locally:

```bash
make all
```

## Notes

- Policy/compensation output is guidance, not legal advice.
- Always confirm final eligibility directly with the operating airline.
