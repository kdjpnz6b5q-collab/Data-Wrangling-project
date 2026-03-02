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
    "security_geopolitical": "Security, war, or geopolitical event",
    "airport operations": "Airport closure or operations issue",
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
    "weather": [
        "weather",
        "storm",
        "snow",
        "wind",
        "hurricane",
        "fog",
        "thunderstorm",
        "lightning",
        "blizzard",
        "ice",
        "rain",
    ],
    "air traffic control": [
        "air traffic",
        "atc",
        "faa ground stop",
        "ground stop",
        "flow control",
        "airspace congestion",
        "slot restriction",
    ],
    "mechanical": [
        "mechanical",
        "maintenance",
        "defect",
        "aircraft issue",
        "plane issue",
        "technical issue",
        "technical fault",
        "equipment issue",
        "broken",
        "broken window",
        "engine issue",
        "hydraulic issue",
    ],
    "crew": [
        "crew",
        "staffing",
        "pilot",
        "flight attendant",
        "crew timeout",
        "crew rest",
        "crew legal",
        "no pilot",
    ],
    "security_geopolitical": [
        "war",
        "conflict",
        "military conflict",
        "geopolitical",
        "terror",
        "terrorism",
        "security threat",
        "security event",
        "bomb threat",
        "missile",
        "drone attack",
        "airspace closure",
        "no-fly zone",
        "civil unrest",
        "political unrest",
        "middle east",
    ],
    "airport operations": [
        "airport closure",
        "runway closure",
        "airport strike",
        "ground handling issue",
        "airport operations",
        "terminal closure",
        "power outage",
    ],
    "other uncontrollable": [
        "uncontrollable",
        "act of god",
        "volcanic ash",
        "earthquake",
        "government restriction",
        "government order",
    ],
}

UNCONTROLLABLE_TYPES = {
    "weather",
    "air traffic control",
    "security_geopolitical",
    "airport operations",
    "other uncontrollable",
}
CONTROLLABLE_TYPES = {"mechanical", "crew"}

DISRUPTION_TAG_MAP = {
    "weather": {"weather"},
    "air traffic control": {"air_traffic_control", "air traffic control"},
    "mechanical": {"mechanical"},
    "crew": {"crew"},
    "security_geopolitical": {"security_geopolitical", "security", "geopolitical"},
    "airport operations": {"airport_operations"},
    "other uncontrollable": {"uncontrollable"},
}

EUROPEAN_AIRLINES = {"lufthansa", "ryanair", "easyjet", "air_france", "british_airways"}

NUMBER_WORDS = {
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
}

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
    "compensation": 2.2,
    "reimburse": 2.0,
    "reimbursement": 2.0,
    "voucher": 1.7,
    "rights": 1.4,
    "cancel": 2.0,
    "rebook": 1.8,
    "rebooking": 1.8,
    "canceled": 2.0,
    "cancelled": 2.0,
    "delay": 1.2,
    "delayed": 1.2,
    "war": 2.2,
    "conflict": 2.2,
    "security": 2.0,
    "terrorism": 2.0,
    "mechanical": 1.8,
    "maintenance": 1.8,
    "broken": 1.5,
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
            if re.search(rf"\b{re.escape(alias)}\b", q):
                return disruption
    return None


def disruption_tags(disruption: str | None) -> set[str]:
    if not disruption:
        return set()
    tags = {disruption}
    tags.update(DISRUPTION_TAG_MAP.get(disruption, set()))
    if disruption in UNCONTROLLABLE_TYPES:
        tags.add("uncontrollable")
    if disruption in CONTROLLABLE_TYPES:
        tags.add("controllable")
    return tags


def extract_delay_hours(question: str) -> float | None:
    q = question.lower()
    digit_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|hr|h)\b", q)
    if digit_match:
        return float(digit_match.group(1))

    word_match = re.search(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:hours?|hrs?|hr)\b",
        q,
    )
    if word_match:
        return NUMBER_WORDS.get(word_match.group(1))
    return None


def detect_event_type(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["denied boarding", "bumped", "overbook"]):
        return "denied_boarding"
    if any(k in q for k in ["cancelled", "canceled", "cancellation", "cancel"]):
        return "cancellation"
    if any(k in q for k in ["delay", "delayed"]):
        return "delay"
    return "disruption"


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
        tags = set((row.get("tags") or "").split("|"))
        if disruption_tags(disruption).intersection(tags):
            score += 2.5

        if disruption == "weather" and "weather" in text_l:
            score += 1.5
        if disruption == "mechanical" and any(k in text_l for k in ["mechanical", "maintenance", "aircraft"]):
            score += 1.5
        if disruption == "crew" and any(k in text_l for k in ["crew", "staffing", "pilot"]):
            score += 1.5
        if disruption == "security_geopolitical" and any(
            k in text_l
            for k in ["war", "conflict", "security", "terror", "airspace", "military", "geopolitical"]
        ):
            score += 1.6

    if any(k in question.lower() for k in ["hotel", "meal", "voucher", "accommodation"]):
        if any(k in text_l for k in ["hotel", "meal", "voucher", "accommodation", "lodging"]):
            score += 1.2

    if any(k in question.lower() for k in ["compensation", "rights", "refund", "reimburse"]):
        if any(k in text_l for k in ["refund", "compensation", "voucher", "reimburse", "credit"]):
            score += 1.4

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
    asks_hotel_meal = any(
        k in q for k in ["hotel", "meal", "voucher", "accommodation", "lodging", "food"]
    )
    asks_refund = any(k in q for k in ["refund", "money back", "original form of payment"])
    asks_compensation = any(
        k in q
        for k in [
            "compensation",
            "rights",
            "reimburse",
            "reimbursement",
            "voucher",
            "cash",
            "claim",
        ]
    )
    event_type = detect_event_type(question)
    delay_hours = extract_delay_hours(question)
    airline_label = AIRLINE_LABELS.get(airline or "", "this airline")
    disruption_label = (
        DISRUPTION_LABELS.get(disruption, "the disruption type you selected") if disruption else "this disruption"
    )

    lines = [f"Likely rights summary for {airline_label}:"]

    if disruption:
        if disruption in CONTROLLABLE_TYPES:
            lines.append(
                f"- {disruption_label} is usually treated as within airline control, so care support is more likely."
            )
            lines.append(
                "- Rebooking is typically offered, and for long/overnight disruptions you should ask for meal and hotel support."
            )
        elif disruption in UNCONTROLLABLE_TYPES:
            lines.append(
                f"- {disruption_label} is usually treated as outside airline control (similar to weather/ATC cases)."
            )
            lines.append(
                "- Rebooking is typically prioritized, but hotel/meal compensation is often limited or not guaranteed."
            )
        else:
            lines.append(
                f"- {disruption_label} may be treated differently by carrier; ask the airline to classify it as controllable vs uncontrollable."
            )
    else:
        lines.append(
            "- Rights depend heavily on the disruption cause. Select a disruption type for a more specific answer."
        )

    if event_type in {"cancellation", "delay"} or asks_refund or asks_compensation:
        lines.append(
            "- DOT baseline: if there is a qualifying cancellation or significant delay/change and you choose not to travel, ask for a refund to the original form of payment."
        )
    if event_type == "denied_boarding":
        lines.append(
            "- For involuntary denied boarding (bumped flights), ask for written denied-boarding rights and compensation details."
        )

    if asks_hotel_meal:
        lines.append(
            "- Ask specifically whether meals, hotel, and ground transport are covered for your disruption category and delay length."
        )

    if delay_hours is not None:
        if delay_hours >= 3:
            lines.append(
                f"- You reported about {delay_hours:g} hours of delay. Ask whether meal vouchers or reimbursement trigger at this threshold."
            )
        else:
            lines.append(
                f"- You reported about {delay_hours:g} hours of delay. Hotel coverage is less common unless it becomes overnight."
            )

    if airline in EUROPEAN_AIRLINES:
        lines.append(
            "- EU/UK route note: EC261/UK261 protections may apply depending on route, airline, notice period, and disruption cause."
        )

    lines.append("- Keep receipts and screenshots; ask the agent to note the disruption cause in writing.")
    lines.append("- If denied by the airline, escalate through the airline complaint channel and then DOT/regulator complaint process.")
    lines.append("- The evidence snippets below are the best matches from your loaded policy documents.")

    return "\n".join(lines)


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
        target_tags = disruption_tags(disruption)

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
            missing_labels.append(
                "delay/disruption type (weather, mechanical, crew, security/geopolitical, etc.)"
            )
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
