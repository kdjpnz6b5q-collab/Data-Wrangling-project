#!/usr/bin/env python3
from __future__ import annotations

import csv
import difflib
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TAGGED_CSV = PROJECT_ROOT / "data" / "processed" / "policy_chunks_tagged.csv"
FALLBACK_CHUNKS = PROJECT_ROOT / "data" / "processed" / "policy_chunks.csv"

AIRLINE_LABELS = {
    "american": "American Airlines",
    "delta": "Delta Air Lines",
    "united": "United Airlines",
    "southwest": "Southwest Airlines",
    "jetblue": "JetBlue",
    "alaska": "Alaska Airlines",
    "frontier": "Frontier Airlines",
    "spirit": "Spirit Airlines",
    "hawaiian": "Hawaiian Airlines",
    "allegiant": "Allegiant Air",
    "avelo": "Avelo Airlines",
    "breeze": "Breeze Airways",
    "sun_country": "Sun Country Airlines",
    "lufthansa": "Lufthansa",
    "ryanair": "Ryanair",
    "easyjet": "easyJet",
    "air_france": "Air France",
    "british_airways": "British Airways",
}
AIRLINE_OPTIONS = list(AIRLINE_LABELS.keys())

AIRLINE_CONTACT_URLS = {
    "american": "https://www.aa.com/i18n/customer-service/contact-american/american-customer-service.jsp",
    "delta": "https://www.delta.com/us/en/need-help/overview",
    "united": "https://www.united.com/en/us/fly/help-center.html",
    "southwest": "https://support.southwest.com/helpcenter/s/",
    "jetblue": "https://www.jetblue.com/contact-us",
    "alaska": "https://www.alaskaair.com/content/about-us/help-contact",
    "frontier": "https://www.flyfrontier.com/customer-support/",
    "spirit": "https://customersupport.spirit.com/en-us",
    "hawaiian": "https://www.hawaiianairlines.com/contact-us",
    "allegiant": "https://www.allegiantair.com/contactus",
    "avelo": "https://www.aveloair.com/contact-us",
    "breeze": "https://www.flybreeze.com/support",
    "sun_country": "https://www.suncountry.com/contact-us",
    "lufthansa": "https://www.lufthansa.com/us/en/help-and-contact",
    "ryanair": "https://help.ryanair.com/hc/en-gb",
    "easyjet": "https://www.easyjet.com/en/help/contact",
    "air_france": "https://wwws.airfrance.us/information/aide-contact",
    "british_airways": "https://www.britishairways.com/en-us/information/help-and-contacts/contact-us",
}

DISRUPTION_LABELS = {
    "weather": "Weather",
    "air traffic control": "Air traffic control",
    "mechanical": "Mechanical or maintenance issue",
    "crew": "Crew or staffing issue",
    "other uncontrollable": "Other uncontrollable event",
    "unknown": "Not sure yet",
}
DISRUPTION_OPTIONS = list(DISRUPTION_LABELS.keys())

AIRLINE_ALIASES = {
    "american": ["american", "aa", "american airlines", "amercian"],
    "delta": ["delta", "detla"],
    "united": ["united", "ua", "united airlines"],
    "southwest": ["southwest", "south west", "wn"],
    "jetblue": ["jetblue", "jet blue", "b6"],
    "alaska": ["alaska", "alaska air"],
    "frontier": ["frontier", "f9"],
    "spirit": ["spirit", "nk"],
    "hawaiian": ["hawaiian", "hawaiian airlines", "ha"],
    "allegiant": ["allegiant", "allegiant air", "g4"],
    "avelo": ["avelo", "avelo airlines", "xp"],
    "breeze": ["breeze", "breeze airways", "mx"],
    "sun_country": ["sun country", "sun country airlines", "sy"],
    "lufthansa": ["lufthansa", "lh"],
    "ryanair": ["ryanair", "ryan air", "fr"],
    "easyjet": ["easyjet", "easy jet", "u2"],
    "air_france": ["air france", "airfrance", "af"],
    "british_airways": ["british airways", "ba"],
}

DISRUPTION_ALIASES = {
    "weather": ["weather", "storm", "snow", "wind", "hurricane", "fog"],
    "air traffic control": ["air traffic", "atc", "faa ground stop", "ground stop"],
    "mechanical": ["mechanical", "maintenance", "defect", "aircraft issue", "plane issue"],
    "crew": ["crew", "staffing", "pilot", "flight attendant", "crew timeout"],
    "other uncontrollable": ["uncontrollable", "act of god", "airport closure", "security event"],
}

UNCONTROLLABLE_TYPES = {"weather", "air traffic control", "other uncontrollable"}
CONTROLLABLE_TYPES = {"mechanical", "crew"}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "they",
    "to",
    "we",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "you",
    "your",
}

EXTRA_WEIGHTS = {
    "weather": 2.5,
    "hotel": 2.5,
    "meal": 1.8,
    "refund": 2.2,
    "rebook": 1.8,
    "rebooking": 1.8,
    "canceled": 2.0,
    "cancelled": 2.0,
    "delay": 1.2,
    "delayed": 1.2,
}


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return {w for w in words if len(w) > 1 and w not in STOPWORDS}


def normalize_airline(value: str | None) -> str | None:
    if not value:
        return None
    val = value.strip().lower()
    if val in AIRLINE_OPTIONS:
        return val

    for airline, aliases in AIRLINE_ALIASES.items():
        for alias in aliases:
            if val == alias:
                return airline

    choices = [alias for aliases in AIRLINE_ALIASES.values() for alias in aliases]
    match = difflib.get_close_matches(val, choices, n=1, cutoff=0.8)
    if match:
        m = match[0]
        for airline, aliases in AIRLINE_ALIASES.items():
            if m in aliases:
                return airline

    return None


def normalize_disruption(value: str | None) -> str | None:
    if not value:
        return None
    val = value.strip().lower()
    if val in DISRUPTION_OPTIONS:
        return val

    for disruption, aliases in DISRUPTION_ALIASES.items():
        for alias in aliases:
            if val == alias:
                return disruption

    choices = [alias for aliases in DISRUPTION_ALIASES.values() for alias in aliases]
    match = difflib.get_close_matches(val, choices, n=1, cutoff=0.8)
    if match:
        m = match[0]
        for disruption, aliases in DISRUPTION_ALIASES.items():
            if m in aliases:
                return disruption

    return None


def detect_airline(question: str) -> str | None:
    q = question.lower()
    for airline, aliases in AIRLINE_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", q):
                return airline

    tokens = re.findall(r"[a-zA-Z]+", q)
    alias_to_airline = {
        alias: airline for airline, aliases in AIRLINE_ALIASES.items() for alias in aliases
    }
    for token in tokens:
        match = difflib.get_close_matches(token, list(alias_to_airline.keys()), n=1, cutoff=0.84)
        if match:
            return alias_to_airline[match[0]]
    return None


def detect_disruption(question: str) -> str | None:
    q = question.lower()
    for disruption, aliases in DISRUPTION_ALIASES.items():
        for alias in aliases:
            if alias in q:
                return disruption
    return None


def build_contact_guidance(airline: str | None) -> tuple[str, str]:
    if airline:
        label = AIRLINE_LABELS.get(airline, airline)
        contact_url = AIRLINE_CONTACT_URLS.get(airline, "")
        message = f'Please reach out to your airline "{label}" - Go Irisch.'
        return message, contact_url
    return (
        'Please reach out to your airline support team - Go Irisch. '
        "Select the airline to get a direct contact link.",
        "",
    )


def load_rows() -> list[dict[str, str]]:
    source = TAGGED_CSV if TAGGED_CSV.exists() else FALLBACK_CHUNKS
    if not source.exists():
        return []

    with source.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if not row.get("airline"):
            doc_id = row.get("doc_id", "")
            row["airline"] = doc_id.split("_")[0] if doc_id else "unknown"

    return rows


def score_row(
    question: str,
    query_tokens: set[str],
    row: dict[str, str],
    airline: str | None,
    disruption: str | None,
) -> float:
    text = row.get("chunk_text", "")
    tokens = tokenize(text)
    overlap = query_tokens.intersection(tokens)
    score = float(len(overlap))

    text_l = text.lower()
    for term, weight in EXTRA_WEIGHTS.items():
        if term in query_tokens and term in text_l:
            score += weight

    row_airline = row.get("airline", "").lower()
    if airline:
        if row_airline == airline:
            score += 4.0
        elif row_airline == "dot":
            score += 1.5
        else:
            score -= 2.0

    if disruption:
        tags = (row.get("tags") or "").split("|")
        if disruption in tags:
            score += 2.5

        if disruption == "weather" and "weather" in text_l:
            score += 1.5
        if disruption == "mechanical" and any(k in text_l for k in ["mechanical", "maintenance", "aircraft"]):
            score += 1.5
        if disruption == "crew" and any(k in text_l for k in ["crew", "staffing", "pilot"]):
            score += 1.5

    if any(k in question.lower() for k in ["hotel", "meal", "voucher", "accommodation"]):
        if any(k in text_l for k in ["hotel", "meal", "voucher", "accommodation", "lodging"]):
            score += 1.2

    return score


def synthesize_answer(
    question: str,
    airline: str | None,
    disruption: str | None,
    top_rows: list[dict[str, str]],
) -> str:
    if not top_rows:
        return "No strong policy match found yet."

    q = question.lower()
    asks_hotel_meal = any(k in q for k in ["hotel", "meal", "voucher", "accommodation", "lodging"])
    asks_refund = "refund" in q

    airline_label = AIRLINE_LABELS.get(airline or "", "this airline")

    if asks_refund:
        return (
            "DOT baseline: when a qualifying significant cancellation or delay occurs and you choose not to travel, "
            "a refund should be returned to the original form of payment."
        )

    if airline and disruption and asks_hotel_meal:
        if disruption in UNCONTROLLABLE_TYPES:
            return (
                f"For {airline_label}, {DISRUPTION_LABELS[disruption].lower()} is usually treated as outside airline control. "
                "In those cases, rebooking is typically offered, while hotel and meal costs are often not guaranteed."
            )
        if disruption in CONTROLLABLE_TYPES:
            return (
                f"For {airline_label}, {DISRUPTION_LABELS[disruption].lower()} is usually treated as controllable. "
                "Policies often provide meal and possible hotel support for long or overnight disruptions."
            )

    if airline and disruption:
        if disruption in UNCONTROLLABLE_TYPES:
            return (
                f"For {airline_label}, {DISRUPTION_LABELS[disruption].lower()} is usually treated as outside airline control. "
                "Rebooking is typically prioritized, while compensation or overnight coverage is often limited."
            )
        if disruption in CONTROLLABLE_TYPES:
            return (
                f"For {airline_label}, {DISRUPTION_LABELS[disruption].lower()} is usually treated as controllable. "
                "Operationally caused disruptions may qualify for additional care depending on delay length and overnight impact."
            )
        return (
            f"Based on the retrieved policy text for {airline_label} and {DISRUPTION_LABELS.get(disruption, disruption).lower()}, "
            "see the evidence below for the exact conditions and thresholds."
        )

    return "Best answer from available policy text is shown below with supporting snippets."


def query_policy(
    question: str,
    airline_override: str | None = None,
    disruption_override: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    rows = load_rows()
    if not rows:
        return {
            "ok": False,
            "error": "No policy data available. Run: make all",
            "missing_fields": [],
        }

    parsed_airline = detect_airline(question)
    parsed_disruption = detect_disruption(question)

    airline = normalize_airline(airline_override) or parsed_airline
    disruption = normalize_disruption(disruption_override) or parsed_disruption

    missing = []
    if not airline:
        missing.append("airline")
    if not disruption:
        missing.append("disruption")

    # When airline is known, keep retrieval focused on that airline + DOT baseline.
    candidate_rows = rows
    if airline:
        focused = [r for r in rows if r.get("airline", "").lower() in {airline, "dot"}]
        if focused:
            candidate_rows = focused

    if disruption:
        target_tags = {disruption}
        if disruption in UNCONTROLLABLE_TYPES:
            target_tags.add("uncontrollable")
        if disruption in CONTROLLABLE_TYPES:
            target_tags.add("controllable")

        disruption_focused = []
        for row in candidate_rows:
            row_tags = set((row.get("tags") or "").split("|"))
            if target_tags.intersection(row_tags):
                disruption_focused.append(row)
        if disruption_focused:
            candidate_rows = disruption_focused

    query_tokens = tokenize(question)
    scored = []
    for row in candidate_rows:
        s = score_row(question, query_tokens, row, airline, disruption)
        if s > 0:
            scored.append((s, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_rows = [r for _, r in scored[:top_k]]

    answer = synthesize_answer(question, airline, disruption, top_rows)

    follow_up_prompt = None
    if missing:
        missing_labels = []
        if "airline" in missing:
            missing_labels.append("airline")
        if "disruption" in missing:
            missing_labels.append("delay/disruption type")
        follow_up_prompt = (
            "I need a bit more information before giving a precise answer: "
            + ", ".join(missing_labels)
            + "."
        )

    contact_message, contact_url = build_contact_guidance(airline)

    return {
        "ok": True,
        "question": question,
        "airline": airline,
        "disruption": disruption,
        "missing_fields": missing,
        "follow_up_prompt": follow_up_prompt,
        "answer": answer,
        "contact_message": contact_message,
        "contact_url": contact_url,
        "evidence": top_rows,
        "airline_options": AIRLINE_OPTIONS,
        "disruption_options": DISRUPTION_OPTIONS,
    }
