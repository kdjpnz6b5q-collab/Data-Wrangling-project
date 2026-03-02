#!/usr/bin/env python3
from __future__ import annotations

import re
from datetime import datetime

import streamlit as st

from policy_engine import (
    AIRLINE_LABELS,
    DISRUPTION_LABELS,
    query_policy,
)
from flight_recommender import recommend_alternative_flights

st.set_page_config(page_title="DelayBot", page_icon="✈", layout="wide")
st.title("DelayBot: Airline Policy QA")
st.caption(
    "Ask a delay/cancellation question. If key details are missing, use the dropdown boxes to fill them in."
)

question = st.text_input(
    "Your question",
    placeholder="Example: My Delta flight was delayed because of weather. Do they owe me a hotel?",
)

selected_airline = ""
selected_disruption = ""

if question.strip():
    base = query_policy(question, top_k=3)

    cols = st.columns(2)
    with cols[0]:
        airline_options = [""] + base.get("airline_options", [])
        format_airline = lambda key: "Select airline..." if key == "" else AIRLINE_LABELS.get(key, key)
        inferred_airline = base.get("airline") or ""
        default_airline_index = airline_options.index(inferred_airline) if inferred_airline in airline_options else 0
        selected_airline = st.selectbox(
            "Airline",
            options=airline_options,
            index=default_airline_index,
            format_func=format_airline,
            help="Use this when the question text does not clearly include the airline.",
        )

    with cols[1]:
        disruption_options = [""] + base.get("disruption_options", [])
        format_disruption = lambda key: "Select disruption type..." if key == "" else DISRUPTION_LABELS.get(key, key)
        inferred_disruption = base.get("disruption") or ""
        default_disruption_index = (
            disruption_options.index(inferred_disruption) if inferred_disruption in disruption_options else 0
        )
        selected_disruption = st.selectbox(
            "Delay/disruption type",
            options=disruption_options,
            index=default_disruption_index,
            format_func=format_disruption,
            help="Examples: weather, mechanical, crew, air traffic control.",
        )

    if base.get("missing_fields"):
        st.warning(base.get("follow_up_prompt", "Please fill missing fields."))

    if st.button("Get answer", type="primary"):
        result = query_policy(
            question,
            airline_override=selected_airline or None,
            disruption_override=selected_disruption or None,
        )

        if not result.get("ok"):
            st.error(result.get("error", "Unknown error."))
        else:
            airline = result.get("airline")
            disruption = result.get("disruption")

            chips = []
            if airline:
                chips.append(f"Airline: {AIRLINE_LABELS.get(airline, airline)}")
            if disruption:
                chips.append(f"Type: {DISRUPTION_LABELS.get(disruption, disruption)}")
            if chips:
                st.info(" | ".join(chips))

            st.subheader("Answer")
            st.write(result.get("answer", "No answer available."))

            st.subheader("Contact the airline")
            st.write(result.get("contact_message", ""))
            contact_url = result.get("contact_url", "")
            if contact_url:
                st.markdown(f"Official contact page: [{contact_url}]({contact_url})")

            evidence = result.get("evidence", [])
            if evidence:
                st.subheader("Evidence")
                shown = set()
                for row in evidence:
                    key = (row.get("title", ""), row.get("chunk_text", ""))
                    if key in shown:
                        continue
                    shown.add(key)

                    snippet = row.get("chunk_text", "")
                    snippet = re.sub(r"\s+", " ", snippet).strip()
                    if len(snippet) > 420:
                        snippet = snippet[:420].rstrip() + "..."

                    title = row.get("title", "Unknown")
                    url = row.get("url", "")
                    with st.expander(title):
                        if url:
                            st.markdown(f"Source: [{url}]({url})")
                        st.write(snippet)
else:
    st.write("Enter a question to start.")

st.divider()
st.header("Alternative Flight Finder")
st.caption(
    "Enter your original flight details and get alliance-aware fallback options with contact links."
)

with st.form("alternative_flights_form", clear_on_submit=False):
    now = datetime.now().replace(second=0, microsecond=0)
    c1, c2, c3 = st.columns(3)
    with c1:
        alt_flight_number = st.text_input("Flight number", placeholder="AA123")
        alt_origin = st.text_input("Depart airport code", placeholder="JFK")
    with c2:
        alt_destination = st.text_input("Destination airport code", placeholder="LHR")
        alt_departure_date = st.date_input("Departure date", value=now.date())
    with c3:
        alt_departure_clock = st.time_input("Departure time", value=now.time(), step=900)

    alt_departure_time = f"{alt_departure_date.isoformat()}T{alt_departure_clock.strftime('%H:%M')}"
    st.caption(f"Selected departure: {alt_departure_time}")

    alt_max_results = st.slider("How many alternatives", min_value=3, max_value=8, value=5)
    alt_submit = st.form_submit_button("Recommend alternatives", type="primary")

if alt_submit:
    alt_result = recommend_alternative_flights(
        flight_number=alt_flight_number,
        origin=alt_origin,
        destination=alt_destination,
        departure_time=alt_departure_time,
        max_results=alt_max_results,
    )

    if not alt_result.get("ok"):
        for err in alt_result.get("errors", []):
            st.error(err)
    else:
        st.success(
            f"Detected airline: {alt_result['source_airline_label']} | "
            f"Alliance: {alt_result['source_alliance_label']}"
        )

        st.write(alt_result.get("contact_message", ""))
        if alt_result.get("contact_url"):
            st.markdown(
                f"Primary contact page: [{alt_result['contact_url']}]({alt_result['contact_url']})"
            )
        st.caption(alt_result.get("live_data_note", ""))
        data_source = alt_result.get("data_source", "")
        if data_source == "amadeus_live":
            st.success("Using live flight offers from Amadeus.")
        else:
            st.info("Using alliance fallback recommendations.")

        st.subheader("Recommended alternatives")
        for i, rec in enumerate(alt_result.get("recommendations", []), start=1):
            with st.expander(f"{i}. {rec['airline_label']}"):
                code = rec.get("airline_code") or ""
                if code:
                    st.write(f"Carrier code: {code}")
                st.write(f"Why: {rec['reason']}")
                if rec.get("live_offer"):
                    cols = st.columns(3)
                    with cols[0]:
                        if rec.get("price"):
                            st.write(f"Price: {rec['price']}")
                        if rec.get("stops") is not None:
                            st.write(f"Stops: {rec['stops']}")
                    with cols[1]:
                        if rec.get("departure_at"):
                            st.write(f"Departure: {rec['departure_at']}")
                        if rec.get("arrival_at"):
                            st.write(f"Arrival: {rec['arrival_at']}")
                    with cols[2]:
                        if rec.get("duration"):
                            st.write(f"Duration: {rec['duration']}")
                if rec.get("contact_url"):
                    st.markdown(f"Contact: [{rec['contact_url']}]({rec['contact_url']})")
                st.markdown(
                    f"Google Flights search: [{rec['google_flights_url']}]({rec['google_flights_url']})"
                )
