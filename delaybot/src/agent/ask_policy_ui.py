#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

import streamlit as st

from policy_engine import (
    AIRLINE_LABELS,
    DISRUPTION_LABELS,
    EVENT_TYPE_LABELS,
    query_policy,
)
from flight_recommender import (
    get_amadeus_credential,
    has_amadeus_credentials,
    recommend_alternative_flights,
)


def inject_notre_dame_theme() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@700&family=Work+Sans:wght@400;500;600;700&display=swap');

          :root {
            --nd-navy: #0C2340;
            --nd-navy-deep: #081729;
            --nd-gold: #C99700;
            --nd-gold-soft: #E6C86A;
            --nd-cream: #F7F3E9;
            --nd-cream-soft: #EDE8DA;
            --nd-muted: rgba(247, 243, 233, 0.84);
            --nd-panel: rgba(12, 35, 64, 0.74);
            --nd-panel-2: rgba(8, 23, 41, 0.72);
            --nd-border: rgba(201, 151, 0, 0.35);
          }

          .stApp {
            font-family: "Work Sans", sans-serif;
            color: var(--nd-cream);
            background:
              radial-gradient(circle at 12% 8%, rgba(201,151,0,0.16), transparent 34%),
              radial-gradient(circle at 88% 15%, rgba(201,151,0,0.10), transparent 26%),
              linear-gradient(165deg, var(--nd-navy-deep) 0%, var(--nd-navy) 52%, #102d53 100%);
          }

          [data-testid="stHeader"] {
            background: transparent;
          }

          .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--nd-gold-soft) 0%, var(--nd-gold) 50%, var(--nd-gold-soft) 100%);
            z-index: 1000;
          }

          h1, h2, h3 {
            font-family: "Libre Baskerville", serif;
            letter-spacing: 0.01em;
            color: var(--nd-cream);
          }

          h1 {
            text-shadow: 0 2px 14px rgba(0, 0, 0, 0.42);
          }

          p, label, .stCaption, .stMarkdown, .stText {
            font-family: "Work Sans", sans-serif;
            color: var(--nd-cream-soft);
          }

          .stCaption,
          [data-testid="stCaptionContainer"],
          .stApp small {
            color: var(--nd-muted) !important;
          }

          [data-testid="stWidgetLabel"] p,
          [data-testid="stWidgetLabel"] label,
          [data-testid="stWidgetLabel"] span {
            color: var(--nd-cream) !important;
            font-weight: 600;
            letter-spacing: 0.01em;
          }

          .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2rem;
          }

          [data-testid="stForm"] {
            border: 1px solid var(--nd-border);
            border-radius: 14px;
            background: linear-gradient(180deg, var(--nd-panel) 0%, var(--nd-panel-2) 100%);
            padding: 1rem 1rem 0.35rem 1rem;
          }

          .stTextInput > div > div > input,
          .stTextArea textarea,
          .stNumberInput input,
          .stDateInput input,
          .stTimeInput input,
          div[data-baseweb="input"] input,
          div[data-baseweb="base-input"] input,
          div[data-baseweb="select"] input {
            background: rgba(2, 12, 24, 0.70);
            color: var(--nd-cream) !important;
            border: 1px solid rgba(201,151,0,0.28);
            border-radius: 10px;
            caret-color: var(--nd-cream);
          }

          .stTextInput > div > div > input:focus,
          .stTextArea textarea:focus,
          .stNumberInput input:focus,
          .stDateInput input:focus,
          .stTimeInput input:focus {
            border-color: var(--nd-gold);
            box-shadow: 0 0 0 1px var(--nd-gold);
          }

          div[data-baseweb="select"] > div {
            background: rgba(2, 12, 24, 0.70);
            border: 1px solid rgba(201,151,0,0.28);
            border-radius: 10px;
            color: var(--nd-cream) !important;
          }

          div[data-baseweb="select"] * {
            color: var(--nd-cream) !important;
          }

          .stTextInput input::placeholder,
          .stTextArea textarea::placeholder,
          .stNumberInput input::placeholder,
          .stDateInput input::placeholder,
          .stTimeInput input::placeholder {
            color: rgba(247, 243, 233, 0.68) !important;
          }

          .stButton > button,
          .stFormSubmitButton > button {
            border-radius: 11px;
            border: 1px solid #7a5e0b;
            background: linear-gradient(180deg, #d8b85b 0%, var(--nd-gold) 56%, #9d7600 100%);
            color: #081729;
            font-weight: 700;
            letter-spacing: 0.01em;
            box-shadow: 0 8px 18px rgba(0, 0, 0, 0.35);
            transition: transform 0.16s ease, box-shadow 0.16s ease;
          }

          .stButton > button:hover,
          .stFormSubmitButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 22px rgba(0, 0, 0, 0.42);
          }

          .stButton > button:active,
          .stFormSubmitButton > button:active {
            transform: translateY(0);
          }

          [data-testid="stExpander"] details {
            background: linear-gradient(180deg, rgba(8, 23, 41, 0.78), rgba(10, 26, 47, 0.72));
            border: 1px solid rgba(201,151,0,0.25);
            border-radius: 12px;
          }

          [data-testid="stAlert"] {
            border-radius: 11px;
            border: 1px solid rgba(201,151,0,0.28);
            background: rgba(8, 23, 41, 0.60);
          }

          hr {
            border-top: 1px solid rgba(201,151,0,0.25);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ND_LOGO_PATH = PROJECT_ROOT / "assets" / "notre_dame_university_logo.svg"


def render_header() -> None:
    title_col, logo_col = st.columns([9, 2])

    with title_col:
        st.title("DelayBot: Airline Policy QA")
        st.caption(
            "Ask a delay/cancellation question. If key details are missing, use the dropdown boxes to fill them in."
        )

    with logo_col:
        logo_path = os.getenv("DELAYBOT_LOGO_PATH", "").strip()
        logo_url = os.getenv("DELAYBOT_LOGO_URL", "").strip()
        logo_source = logo_path or (str(DEFAULT_ND_LOGO_PATH) if DEFAULT_ND_LOGO_PATH.exists() else logo_url)

        rendered = False
        if logo_source:
            try:
                st.image(logo_source, width=96)
                rendered = True
            except Exception:
                rendered = False

        if not rendered:
            st.markdown(
                (
                    "<div style='margin-top:0.6rem;text-align:center;font-family:\"Libre Baskerville\",serif;"
                    "font-size:1.45rem;color:#C99700;'>University of Notre Dame</div>"
                ),
                unsafe_allow_html=True,
            )


st.set_page_config(page_title="DelayBot", page_icon="✈", layout="wide")
inject_notre_dame_theme()
render_header()

question = st.text_input(
    "Your question",
    placeholder="Example: My Delta flight was delayed because of weather. Do they owe me a hotel?",
)

selected_airline = ""
selected_disruption = ""
selected_event_type = ""
delay_hours_override: float | None = None
notice_hours_override: float | None = None

if "policy_result" not in st.session_state:
    st.session_state["policy_result"] = None
if "show_email_draft" not in st.session_state:
    st.session_state["show_email_draft"] = False
if "alt_result" not in st.session_state:
    st.session_state["alt_result"] = None


def render_policy_result(result: dict) -> None:
    airline = result.get("airline")
    disruption = result.get("disruption")
    event_type = result.get("event_type")

    chips = []
    if airline:
        chips.append(f"Airline: {AIRLINE_LABELS.get(airline, airline)}")
    if disruption:
        chips.append(f"Type: {DISRUPTION_LABELS.get(disruption, disruption)}")
    if event_type:
        chips.append(f"Event: {EVENT_TYPE_LABELS.get(event_type, event_type)}")
    if chips:
        st.info(" | ".join(chips))

    st.subheader("Answer")
    st.markdown(result.get("answer", "No answer available."))

    st.subheader("Contact the airline")
    st.write(result.get("contact_message", ""))
    contact_url = result.get("contact_url", "")
    if contact_url:
        st.markdown(f"Official contact page: [{contact_url}]({contact_url})")

    expected_comp = result.get("expected_compensation")
    comp_notes = result.get("compensation_notes", [])
    if expected_comp or comp_notes:
        st.subheader("Expected compensation (not guaranteed)")
        if expected_comp:
            st.write(expected_comp)
        if comp_notes:
            st.markdown("\n".join(f"- {line}" for line in comp_notes))

    email_subject = result.get("refund_email_subject")
    email_body = result.get("refund_email_body")
    if email_subject and email_body:
        if st.button("Generate draft email", key="generate_refund_email_btn"):
            st.session_state["show_email_draft"] = True
        if st.session_state.get("show_email_draft"):
            st.subheader("Draft refund/compensation email")
            st.write(f"Subject: {email_subject}")
            st.code(email_body, language="text")

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


if question.strip():
    stored = st.session_state.get("policy_result")
    if isinstance(stored, dict) and stored.get("question") != question:
        st.session_state["policy_result"] = None
        st.session_state["show_email_draft"] = False

    base = query_policy(question, top_k=3)

    cols = st.columns(3)
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
            help="Examples: weather, mechanical, crew, air traffic control, security/geopolitical.",
        )

    with cols[2]:
        event_type_options = [""] + base.get("event_type_options", [])
        format_event = lambda key: "Select event type..." if key == "" else EVENT_TYPE_LABELS.get(key, key)
        inferred_event_type = base.get("event_type") or ""
        default_event_idx = (
            event_type_options.index(inferred_event_type) if inferred_event_type in event_type_options else 0
        )
        selected_event_type = st.selectbox(
            "Event type",
            options=event_type_options,
            index=default_event_idx,
            format_func=format_event,
            help="Cancellation, delay, denied boarding, or general disruption.",
        )

    c_delay, c_notice = st.columns(2)
    with c_delay:
        delay_val = st.number_input(
            "Delay duration (hours, optional)",
            min_value=0.0,
            max_value=72.0,
            value=float(base.get("delay_hours") or 0.0),
            step=0.5,
        )
        delay_hours_override = delay_val if delay_val > 0 else None
    with c_notice:
        notice_val = st.number_input(
            "Cancellation notice before departure (hours, optional)",
            min_value=0.0,
            max_value=720.0,
            value=float(base.get("notice_hours") or 0.0),
            step=1.0,
        )
        notice_hours_override = notice_val if notice_val > 0 else None

    missing_after_selection: list[str] = []
    if not (selected_airline or base.get("airline")):
        missing_after_selection.append("airline")
    if not (selected_disruption or base.get("disruption")):
        missing_after_selection.append("delay/disruption type")
    if missing_after_selection:
        st.warning(
            "I need a bit more information before giving a precise answer: "
            + ", ".join(missing_after_selection)
            + "."
        )

    if st.button("Get answer", type="primary"):
        result = query_policy(
            question,
            airline_override=selected_airline or None,
            disruption_override=selected_disruption or None,
            event_type_override=selected_event_type or None,
            delay_hours_override=delay_hours_override,
            notice_hours_override=notice_hours_override,
        )

        if not result.get("ok"):
            st.error(result.get("error", "Unknown error."))
        else:
            st.session_state["policy_result"] = result
            st.session_state["show_email_draft"] = False

    final_result = st.session_state.get("policy_result")
    if isinstance(final_result, dict) and final_result.get("ok"):
        render_policy_result(final_result)
else:
    st.session_state["policy_result"] = None
    st.session_state["show_email_draft"] = False
    st.write("Enter a question to start.")

st.divider()
st.header("Alternative Flight Finder")
st.caption(
    "Enter your original flight details and get alliance-aware fallback options with contact links."
)

if "amadeus_client_id" not in st.session_state:
    st.session_state["amadeus_client_id"] = get_amadeus_credential("AMADEUS_CLIENT_ID")
if "amadeus_client_secret" not in st.session_state:
    st.session_state["amadeus_client_secret"] = get_amadeus_credential("AMADEUS_CLIENT_SECRET")
if "amadeus_base_url" not in st.session_state:
    st.session_state["amadeus_base_url"] = (
        get_amadeus_credential("AMADEUS_BASE_URL") or "https://test.api.amadeus.com"
    )


def render_alt_result(alt_result: dict) -> None:
    if not alt_result.get("ok"):
        for err in alt_result.get("errors", []):
            st.error(err)
        return

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

    recs = alt_result.get("recommendations", [])
    st.subheader(f"Recommended alternatives ({len(recs)})")
    if not recs:
        st.warning("No alternatives returned for this route/date. Try changing time or date.")
        return

    for i, rec in enumerate(recs, start=1):
        summary_bits: list[str] = []
        if rec.get("price"):
            summary_bits.append(str(rec["price"]))
        if rec.get("departure_at"):
            summary_bits.append(f"Dep {rec['departure_at']}")
        if rec.get("stops") is not None:
            summary_bits.append(f"Stops {rec['stops']}")
        summary = " | ".join(summary_bits)
        st.markdown(
            f"**{i}. {rec['airline_label']}**" + (f" - {summary}" if summary else "")
        )

        with st.expander(f"Details for option {i}", expanded=(i == 1)):
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

with st.expander("Live fare setup (Amadeus)", expanded=not has_amadeus_credentials()):
    st.caption(
        "Paste Amadeus credentials here for this Streamlit session. "
        "They are not written to disk unless you add them to delaybot/.env."
    )
    st.text_input("AMADEUS_CLIENT_ID", key="amadeus_client_id")
    st.text_input("AMADEUS_CLIENT_SECRET", type="password", key="amadeus_client_secret")
    st.text_input("AMADEUS_BASE_URL", key="amadeus_base_url")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Use these credentials", type="primary"):
            client_id = st.session_state.get("amadeus_client_id", "").strip()
            client_secret = st.session_state.get("amadeus_client_secret", "").strip()
            base_url = st.session_state.get("amadeus_base_url", "").strip()

            if client_id and client_secret:
                os.environ["AMADEUS_CLIENT_ID"] = client_id
                os.environ["AMADEUS_CLIENT_SECRET"] = client_secret
                if base_url:
                    os.environ["AMADEUS_BASE_URL"] = base_url
                st.success("Live Amadeus credentials loaded for this app session.")
                st.rerun()
            else:
                st.error("Please provide both AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET.")

    with b2:
        if st.button("Clear credentials"):
            os.environ.pop("AMADEUS_CLIENT_ID", None)
            os.environ.pop("AMADEUS_CLIENT_SECRET", None)
            os.environ.pop("AMADEUS_BASE_URL", None)
            st.session_state["amadeus_client_id"] = ""
            st.session_state["amadeus_client_secret"] = ""
            st.session_state["amadeus_base_url"] = "https://test.api.amadeus.com"
            st.info("Live credentials cleared for this session.")
            st.rerun()

    if has_amadeus_credentials():
        st.success("Live fare credentials detected.")
    else:
        st.warning(
            "Live fare credentials not detected yet. "
            "You can still use alliance fallback recommendations."
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
    st.session_state["alt_result"] = recommend_alternative_flights(
        flight_number=alt_flight_number,
        origin=alt_origin,
        destination=alt_destination,
        departure_time=alt_departure_time,
        max_results=alt_max_results,
    )

final_alt_result = st.session_state.get("alt_result")
if isinstance(final_alt_result, dict):
    render_alt_result(final_alt_result)
