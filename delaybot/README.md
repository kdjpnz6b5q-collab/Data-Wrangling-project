# DelayBot

I made an app on Streamlit called Delaybot. We all know how frustrating it is when your flight gets delayed or sometimes even cancelled. What people often don't realize is that they have the right to those delays and cancellations. This app helps travelers understand those often long and difficult to understand airline policies a lot easier! We also offer alternative flights.


## Live app

the link to the app is: https://kdjpnz6b5q-collab-data-wra-delaybotsrcagentask-policy-ui-fpq4m6.streamlit.app/

**What is the app acutally doing: **
First, it answers all your questions about delays and cancellation policies for 25+ airlines. 
It recognises different types of delays, based on which it will give you guidance on your expected compensation from the airline
As well it could draft you a refund/compensation email for the involved airline.
The second part of the app offers you alternative flight options based on the alliance of the cancelled flight

**Amadeus credentials **
-  I don't want those public, will send them in an email to you:)

## Project structure

- `src/agent/ask_policy_ui.py`: Streamlit app
- `src/agent/policy_engine.py`: policy query + answer logic
- `src/agent/flight_recommender.py`: alternative flight logic + Amadeus integration
- `src/agent/ask_policy.py`: CLI Q&A
- `src/agent/recommend_alternatives.py`: CLI alternatives tool
- `src/scrape/`: per-airline scrapers
- `data/seeds/fallback_policies.json`: built-in fallback policy data
- `requirements.txt`: Streamlit Cloud runtime dependencies

## unstructured data part
The airline policies are very complicated, and I saw many that run more than 100 pages. Personally, I don't really want to go over all those pages to find out my rights, but some compensation for my cancelled flight would be nice sometimes. That's why I scraped the policies of more than 25+ different airlines with Beautifullsoup. When the client in the app asks a question, the app will recognize which airline they fly with and match it to the right policy we have scraped. In the future i would like to include most airline companies to the app.



