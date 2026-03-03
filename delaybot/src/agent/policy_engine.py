#!/usr/bin/env python3
from __future__ import annotations

import csv
import difflib
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TAGGED_CSV = PROJECT_ROOT / "data" / "processed" / "policy_chunks_tagged.csv"
FALLBACK_CHUNKS = PROJECT_ROOT / "data" / "processed" / "policy_chunks.csv"
FALLBACK_SEED_JSON = PROJECT_ROOT / "data" / "seeds" / "fallback_policies.json"

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
    "emirates": "Emirates",
    "qatar_airways": "Qatar Airways",
    "singapore_airlines": "Singapore Airlines",
    "turkish_airlines": "Turkish Airlines",
    "air_canada": "Air Canada",
    "klm": "KLM",
    "iberia": "Iberia",
    "latam": "LATAM",
    "avianca": "Avianca",
    "etihad": "Etihad Airways",
    "virgin_atlantic": "Virgin Atlantic",
    "ana": "ANA",
    "japan_airlines": "Japan Airlines",
    "china_eastern": "China Eastern",
    "china_southern": "China Southern",
    "air_china": "Air China",
    "indigo": "IndiGo",
    "qantas": "Qantas",
    "saudia": "Saudia",
    "swiss": "SWISS",
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
    "emirates": "https://www.emirates.com/us/english/help/",
    "qatar_airways": "https://www.qatarairways.com/en/help.html",
    "singapore_airlines": "https://www.singaporeair.com/en_UK/us/help/",
    "turkish_airlines": "https://www.turkishairlines.com/en-int/any-questions/",
    "air_canada": "https://www.aircanada.com/contact",
    "klm": "https://www.klm.com/contact",
    "iberia": "https://www.iberia.com/contact/",
    "latam": "https://www.latamairlines.com/us/en/help-center",
    "avianca": "https://www.avianca.com/us/en/help/",
    "etihad": "https://www.etihad.com/en-us/help",
    "virgin_atlantic": "https://help.virginatlantic.com/",
    "ana": "https://www.ana.co.jp/en/us/contact/",
    "japan_airlines": "https://www.jal.co.jp/jp/en/inter/contact/",
    "china_eastern": "https://www.ceair.com/global/en_static/Announcement/",
    "china_southern": "https://www.csair.com/en/",
    "air_china": "https://www.airchina.com.cn/en/",
    "indigo": "https://www.goindigo.in/contact-us.html",
    "qantas": "https://www.qantas.com/us/en/support/contact-us.html",
    "saudia": "https://www.saudia.com/pages/help",
    "swiss": "https://www.swiss.com/us/en/customer-support",
}

DISRUPTION_LABELS = {
    "weather": "Weather",
    "air traffic control": "Air traffic control / NAS congestion",
    "mechanical": "Mechanical or maintenance issue",
    "crew": "Crew or staffing issue",
    "late inbound aircraft": "Late inbound aircraft",
    "security_geopolitical": "Security, war, or geopolitical event",
    "airport operations": "Airport closure or operations issue",
    "strike_or_labor": "Strike or labor action",
    "other uncontrollable": "Other uncontrollable event",
    "unknown": "Not sure yet",
}
DISRUPTION_OPTIONS = list(DISRUPTION_LABELS.keys())

EVENT_TYPE_LABELS = {
    "cancellation": "Cancellation",
    "delay": "Delay",
    "denied_boarding": "Denied boarding",
    "disruption": "General disruption",
}
EVENT_TYPE_OPTIONS = list(EVENT_TYPE_LABELS.keys())

AIRLINE_ALIASES = {
    "american": ["american", "aa", "american airlines", "amercian"],
    "delta": ["delta", "detla", "dl"],
    "united": ["united", "ua", "united airlines"],
    "southwest": ["southwest", "south west", "wn"],
    "jetblue": ["jetblue", "jet blue", "b6"],
    "alaska": ["alaska", "alaska air", "as"],
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
    "emirates": ["emirates", "ek"],
    "qatar_airways": ["qatar", "qatar airways", "qr"],
    "singapore_airlines": ["singapore airlines", "sia", "sq"],
    "turkish_airlines": ["turkish", "turkish airlines", "tk"],
    "air_canada": ["air canada", "ac"],
    "klm": ["klm", "royal dutch", "kl"],
    "iberia": ["iberia", "ib"],
    "latam": ["latam", "la"],
    "avianca": ["avianca", "av"],
    "etihad": ["etihad", "ey"],
    "virgin_atlantic": ["virgin atlantic", "virgin", "vs"],
    "ana": ["ana", "all nippon", "nh"],
    "japan_airlines": ["japan airlines", "jal", "jl"],
    "china_eastern": ["china eastern", "mu"],
    "china_southern": ["china southern", "cz"],
    "air_china": ["air china", "ca"],
    "indigo": ["indigo", "6e"],
    "qantas": ["qantas", "qf"],
    "saudia": ["saudia", "saudi arabian", "sv"],
    "swiss": ["swiss", "swiss international", "lx"],
}

DISRUPTION_ALIASES = {
    "weather": [
        "weather",
        "storm",
        "winter storm",
        "winterstorm",
        "snow",
        "wind",
        "hurricane",
        "fog",
        "thunderstorm",
        "lightning",
        "blizzard",
        "ice",
        "rain",
        "hail",
        "typhoon",
    ],
    "air traffic control": [
        "air traffic",
        "air traffic control",
        "atc",
        "nas",
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
    "late inbound aircraft": [
        "late inbound aircraft",
        "late arriving aircraft",
        "late aircraft",
        "incoming aircraft",
        "aircraft rotation",
        "knock-on delay",
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
        "de-icing",
    ],
    "strike_or_labor": [
        "strike",
        "labor action",
        "industrial action",
        "union action",
        "walkout",
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
CONTROLLABLE_TYPES = {"mechanical", "crew", "late inbound aircraft"}

DISRUPTION_TAG_MAP = {
    "weather": {"weather"},
    "air traffic control": {"air_traffic_control", "air traffic control"},
    "mechanical": {"mechanical"},
    "crew": {"crew"},
    "late inbound aircraft": {"late_inbound_aircraft"},
    "security_geopolitical": {"security_geopolitical", "security", "geopolitical"},
    "airport operations": {"airport_operations"},
    "strike_or_labor": {"strike_or_labor", "labor"},
    "other uncontrollable": {"uncontrollable"},
}

US_AIRLINES = {
    "american",
    "delta",
    "united",
    "southwest",
    "jetblue",
    "alaska",
    "frontier",
    "spirit",
    "hawaiian",
    "allegiant",
    "avelo",
    "breeze",
    "sun_country",
}

EU_UK_AIRLINES = {
    "lufthansa",
    "ryanair",
    "easyjet",
    "air_france",
    "british_airways",
    "klm",
    "iberia",
    "swiss",
    "virgin_atlantic",
}

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
    "thirteen": 13.0,
    "fourteen": 14.0,
    "fifteen": 15.0,
    "sixteen": 16.0,
    "seventeen": 17.0,
    "eighteen": 18.0,
    "nineteen": 19.0,
    "twenty": 20.0,
    "twenty four": 24.0,
    "forty eight": 48.0,
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
    "strike": 1.4,
    "late": 1.2,
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


def normalize_event_type(value: str | None) -> str | None:
    if not value:
        return None
    val = value.strip().lower()
    if val in EVENT_TYPE_OPTIONS:
        return val

    aliases = {
        "canceled": "cancellation",
        "cancelled": "cancellation",
        "cancel": "cancellation",
        "delay": "delay",
        "delayed": "delay",
        "bumped": "denied_boarding",
        "denied boarding": "denied_boarding",
        "overbooked": "denied_boarding",
    }
    return aliases.get(val)


def detect_airline(question: str) -> str | None:
    q = question.lower()
    for airline, aliases in AIRLINE_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", q):
                return airline

    tokens = re.findall(r"[a-zA-Z0-9]+", q)
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


def detect_event_type(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["denied boarding", "bumped", "overbook"]):
        return "denied_boarding"
    if any(k in q for k in ["cancelled", "canceled", "cancellation", "cancel"]):
        return "cancellation"
    if any(k in q for k in ["delay", "delayed", "late"]):
        return "delay"
    return "disruption"


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


def _number_from_match(raw: str) -> float | None:
    val = raw.strip().lower()
    if val in NUMBER_WORDS:
        return NUMBER_WORDS[val]
    try:
        return float(val)
    except ValueError:
        return None


def _unit_to_hours(amount: float, unit: str) -> float:
    unit_l = unit.lower()
    if unit_l.startswith("day") or unit_l == "d":
        return amount * 24.0
    return amount


def extract_delay_hours(question: str) -> float | None:
    q = question.lower()

    digit_patterns = [
        r"(?:delayed?|delay(?:ed)? by|late by)\s*(\d+(?:\.\d+)?)\s*(hours?|hrs?|hr|h)",
        r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|hr|h)\s*(?:delay|late)",
    ]
    for pattern in digit_patterns:
        m = re.search(pattern, q)
        if m:
            amount = _number_from_match(m.group(1))
            if amount is not None:
                return _unit_to_hours(amount, m.group(2))

    word_number = "|".join(re.escape(k) for k in NUMBER_WORDS.keys())
    word_patterns = [
        rf"(?:delayed?|delay(?:ed)? by|late by)\s*({word_number})\s*(hours?|hrs?|hr)",
        rf"({word_number})\s*(hours?|hrs?|hr)\s*(?:delay|late)",
    ]
    for pattern in word_patterns:
        m = re.search(pattern, q)
        if m:
            amount = _number_from_match(m.group(1))
            if amount is not None:
                return _unit_to_hours(amount, m.group(2))

    return None


def extract_cancellation_notice_hours(question: str) -> float | None:
    q = question.lower()

    digit_pattern = r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|hr|h|days?|d)\s*before\s*(?:take\s*off|takeoff|departure|flight)?"
    m = re.search(digit_pattern, q)
    if m:
        amount = _number_from_match(m.group(1))
        if amount is not None:
            return _unit_to_hours(amount, m.group(2))

    word_number = "|".join(re.escape(k) for k in NUMBER_WORDS.keys())
    word_pattern = rf"({word_number})\s*(hours?|hrs?|hr|days?|d)\s*before\s*(?:take\s*off|takeoff|departure|flight)?"
    m = re.search(word_pattern, q)
    if m:
        amount = _number_from_match(m.group(1))
        if amount is not None:
            return _unit_to_hours(amount, m.group(2))

    if "less than 14 days" in q:
        return 335.0

    return None


def is_eu_uk_context(airline: str | None, question: str) -> bool:
    if airline in EU_UK_AIRLINES:
        return True
    q = question.lower()
    markers = ["ec261", "uk261", "europe", "european union", "eu flight", "uk flight", "united kingdom"]
    return any(m in q for m in markers)


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
    if source.exists():
        with source.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            if not row.get("airline"):
                doc_id = row.get("doc_id", "")
                row["airline"] = doc_id.split("_")[0] if doc_id else "unknown"

        return rows

    if not FALLBACK_SEED_JSON.exists():
        return []

    try:
        payload = json.loads(FALLBACK_SEED_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(payload, list):
        return []

    rows: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        doc_id = str(item.get("doc_id") or "").strip()
        airline = str(item.get("airline") or "").strip().lower()
        if not airline and doc_id:
            airline = doc_id.split("_")[0]
        rows.append(
            {
                "doc_id": doc_id,
                "airline": airline or "unknown",
                "title": str(item.get("title") or ""),
                "url": str(item.get("url") or ""),
                "chunk_text": text,
                "tags": "",
            }
        )

    return rows


def score_row(
    question: str,
    query_tokens: set[str],
    row: dict[str, str],
    airline: str | None,
    disruption: str | None,
    event_type: str,
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
            score += 2.6

    if event_type == "cancellation" and any(k in text_l for k in ["cancel", "refund", "rebook"]):
        score += 1.1
    if event_type == "delay" and any(k in text_l for k in ["delay", "late", "voucher", "meal"]):
        score += 1.1
    if event_type == "denied_boarding" and any(k in text_l for k in ["denied boarding", "overbook", "bumped"]):
        score += 1.2

    if any(k in question.lower() for k in ["hotel", "meal", "voucher", "accommodation"]):
        if any(k in text_l for k in ["hotel", "meal", "voucher", "accommodation", "lodging"]):
            score += 1.2

    if any(k in question.lower() for k in ["compensation", "rights", "refund", "reimburse"]):
        if any(k in text_l for k in ["refund", "compensation", "voucher", "reimburse", "credit"]):
            score += 1.4

    return score


def build_compensation_summary(
    question: str,
    airline: str | None,
    disruption: str | None,
    event_type: str,
    delay_hours: float | None,
    notice_hours: float | None,
) -> dict[str, Any]:
    eu_uk = is_eu_uk_context(airline, question)
    notes: list[str] = []
    expected: str | None = None

    if eu_uk:
        notes.append("EU/UK-style rights may apply based on route, operating carrier, and legal scope.")
        notes.append("Compensation is usually distance-based (typically EUR 250 / 400 / 600 brackets).")

        if disruption in UNCONTROLLABLE_TYPES:
            notes.append(
                "If the airline proves extraordinary circumstances (for example severe weather or ATC restrictions), cash compensation may be denied even when care/rebooking is still owed."
            )

        if event_type == "delay":
            if delay_hours is None:
                expected = "Potentially EUR 250 to EUR 600 if arrival delay was at least 3 hours and disruption was airline-responsible."
                notes.append("For EU/UK claims, the common threshold is around 3+ hours arrival delay.")
            elif delay_hours >= 3:
                expected = "Potentially EUR 250 to EUR 600 (distance-based), if airline-responsible and no extraordinary circumstances."
                notes.append(f"You reported about {delay_hours:g} hours of delay, which may cross EC261/UK261 timing thresholds.")
            else:
                expected = "Likely below EC261/UK261 delay-compensation threshold unless other claim conditions apply."

        elif event_type == "cancellation":
            if notice_hours is None:
                expected = "Potentially EUR 250 to EUR 600 if cancellation notice was under 14 days and disruption was airline-responsible."
                notes.append("Cancellation compensation is often tied to notice timing (under 14 days) and rerouting details.")
            elif notice_hours < 336:
                expected = "Potentially EUR 250 to EUR 600 if cancellation was airline-responsible and rerouting did not meet exemption windows."
                notes.append(f"You reported about {notice_hours:g} hours notice before departure (under 14 days).")
            else:
                expected = "Cash compensation is less likely when cancellation notice is 14+ days before departure."

        elif event_type == "denied_boarding":
            expected = "Potentially EUR 250 to EUR 600 equivalent (distance-based), depending on rerouting and delay at destination."
        else:
            expected = "Compensation depends on event type, route distance, and responsibility classification."

    else:
        if airline in US_AIRLINES:
            notes.append("U.S. baseline: there is generally no automatic cash compensation for ordinary delays/cancellations.")
            notes.append("If you choose not to travel after a significant cancellation/schedule change, request a refund to original payment method.")

            if disruption in CONTROLLABLE_TYPES:
                notes.append("For airline-caused disruptions, ask for meal/hotel vouchers or reimbursement of reasonable documented expenses.")

            if event_type == "denied_boarding":
                expected = "Denied-boarding compensation may apply if you were involuntarily bumped and meet DOT conditions."
            else:
                expected = "Case-by-case; refund rights are usually stronger than cash-compensation rights for most U.S. delay/cancellation cases."
        else:
            notes.append("Compensation outside EU/UK depends on country-specific passenger-rights law, route, and airline contract terms.")
            notes.append("Request written cause classification and ask the airline which legal regime applies to your ticket.")
            notes.append("If you decline travel after a major cancellation/change, ask for available refund options to original payment method.")

            if disruption in CONTROLLABLE_TYPES:
                notes.append("For airline-caused disruptions, ask for meal/hotel support or reimbursement of documented costs.")

            expected = "Jurisdiction-specific; potential compensation/refund depends on local regulation and airline policy."

    return {
        "expected_compensation": expected,
        "notes": notes,
        "eu_uk_context": eu_uk,
    }


def build_refund_email_draft(
    airline: str | None,
    event_type: str,
    disruption: str | None,
    delay_hours: float | None,
    notice_hours: float | None,
    compensation: dict[str, Any],
) -> tuple[str | None, str | None]:
    if not airline:
        return None, None

    airline_label = AIRLINE_LABELS.get(airline, airline)
    disruption_label = DISRUPTION_LABELS.get(disruption or "", "a travel disruption")
    event_label = EVENT_TYPE_LABELS.get(event_type, "disruption")

    subject = f"Request for refund/compensation - {airline_label} {event_label}"

    lines = [
        f"Dear {airline_label} Customer Relations Team,",
        "",
        f"I am writing regarding a {event_label.lower()} on my itinerary.",
        "",
        "Booking reference: [ADD RECORD LOCATOR]",
        "Passenger name: [ADD FULL NAME]",
        "Flight number: [ADD FLIGHT NUMBER]",
        "Travel date: [ADD DATE]",
        "Route: [ADD ORIGIN] -> [ADD DESTINATION]",
        "",
        f"Disruption reason provided: {disruption_label}.",
    ]

    if delay_hours is not None:
        lines.append(f"Reported delay duration: approximately {delay_hours:g} hours.")
    if notice_hours is not None:
        lines.append(f"Cancellation notice received approximately {notice_hours:g} hours before departure.")

    lines.extend(
        [
            "",
            "Please review this case under your policy and applicable passenger-rights regulations.",
            "I request the following where applicable:",
            "1. Refund to original form of payment (if eligible and not traveling).",
            "2. Compensation assessment under applicable rules.",
            "3. Reimbursement review for documented expenses (meals/hotel/transport), if eligible.",
            "",
        ]
    )

    expected = compensation.get("expected_compensation")
    if expected:
        lines.append(f"Expected compensation guidance (non-guaranteed): {expected}")
        lines.append("")

    lines.extend(
        [
            "Attached documents: receipts, boarding pass/ticket, and relevant screenshots.",
            "Please respond with a written eligibility decision and timeline for payment/refund.",
            "",
            "Thank you,",
            "[YOUR NAME]",
            "[YOUR EMAIL]",
            "[YOUR PHONE]",
        ]
    )

    return subject, "\n".join(lines)


def synthesize_answer(
    question: str,
    airline: str | None,
    disruption: str | None,
    event_type: str,
    delay_hours: float | None,
    notice_hours: float | None,
    compensation: dict[str, Any],
    top_rows: list[dict[str, str]],
) -> str:
    if not top_rows:
        return "No strong policy match found yet."

    airline_label = AIRLINE_LABELS.get(airline or "", "this airline")
    event_label = EVENT_TYPE_LABELS.get(event_type, "disruption")
    disruption_label = (
        DISRUPTION_LABELS.get(disruption, "the disruption type you selected") if disruption else "this disruption"
    )

    lines = [f"Likely rights summary for {airline_label}:"]
    lines.append(f"- Event type: {event_label}.")

    if disruption:
        if disruption in CONTROLLABLE_TYPES:
            lines.append(f"- {disruption_label} is usually treated as within airline control.")
            lines.append("- Rebooking is typically offered; for long/overnight events, ask for meal and hotel support.")
        elif disruption in UNCONTROLLABLE_TYPES:
            lines.append(f"- {disruption_label} is usually treated as outside airline control.")
            lines.append("- Rebooking is typically prioritized; hotel/meal compensation is often limited.")
        elif disruption == "strike_or_labor":
            lines.append("- Strike/labor events are case-specific: eligibility may depend on whether the strike was within the operating carrier's control.")
        else:
            lines.append(f"- {disruption_label} may be handled differently by carrier; request written cause classification.")
    else:
        lines.append("- Rights depend heavily on disruption cause. Select a disruption type for precision.")

    if event_type in {"cancellation", "delay"}:
        lines.append("- If you choose not to travel after a qualifying significant change, request refund to original payment form.")
    if event_type == "denied_boarding":
        lines.append("- For involuntary denied boarding, ask for written denied-boarding rights and compensation explanation.")

    if delay_hours is not None:
        lines.append(f"- Reported delay duration: about {delay_hours:g} hours.")
    if notice_hours is not None:
        lines.append(f"- Reported cancellation notice: about {notice_hours:g} hours before departure.")

    expected = compensation.get("expected_compensation")
    if expected:
        lines.append(f"- Expected compensation (non-guaranteed): {expected}")

    for note in compensation.get("notes", []):
        lines.append(f"- {note}")

    lines.append("- Keep receipts/screenshots and ask the airline to document disruption cause in writing.")
    lines.append("- The evidence snippets below are the best matches from your loaded policy documents.")
    return "\n".join(lines)


def query_policy(
    question: str,
    airline_override: str | None = None,
    disruption_override: str | None = None,
    top_k: int = 5,
    event_type_override: str | None = None,
    delay_hours_override: float | None = None,
    notice_hours_override: float | None = None,
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
    parsed_event_type = detect_event_type(question)
    parsed_delay_hours = extract_delay_hours(question)
    parsed_notice_hours = extract_cancellation_notice_hours(question)

    airline = normalize_airline(airline_override) or parsed_airline
    disruption = normalize_disruption(disruption_override) or parsed_disruption
    event_type = normalize_event_type(event_type_override) or parsed_event_type

    delay_hours = delay_hours_override if delay_hours_override is not None else parsed_delay_hours
    notice_hours = notice_hours_override if notice_hours_override is not None else parsed_notice_hours

    missing = []
    if not airline:
        missing.append("airline")
    if not disruption:
        missing.append("disruption")

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
        s = score_row(question, query_tokens, row, airline, disruption, event_type)
        if s > 0:
            scored.append((s, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_rows: list[dict[str, str]] = []
    seen_docs: set[tuple[str, str, str]] = set()
    for _, row in scored:
        doc_key = (
            row.get("doc_id", ""),
            row.get("title", ""),
            row.get("url", ""),
        )
        if doc_key in seen_docs:
            continue
        seen_docs.add(doc_key)
        top_rows.append(row)
        if len(top_rows) >= top_k:
            break

    compensation = build_compensation_summary(
        question=question,
        airline=airline,
        disruption=disruption,
        event_type=event_type,
        delay_hours=delay_hours,
        notice_hours=notice_hours,
    )

    answer = synthesize_answer(
        question=question,
        airline=airline,
        disruption=disruption,
        event_type=event_type,
        delay_hours=delay_hours,
        notice_hours=notice_hours,
        compensation=compensation,
        top_rows=top_rows,
    )

    email_subject, email_body = build_refund_email_draft(
        airline=airline,
        event_type=event_type,
        disruption=disruption,
        delay_hours=delay_hours,
        notice_hours=notice_hours,
        compensation=compensation,
    )

    follow_up_prompt = None
    if missing:
        missing_labels = []
        if "airline" in missing:
            missing_labels.append("airline")
        if "disruption" in missing:
            missing_labels.append(
                "delay/disruption type (weather, mechanical, crew, late inbound aircraft, security/geopolitical, etc.)"
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
        "event_type": event_type,
        "delay_hours": delay_hours,
        "notice_hours": notice_hours,
        "missing_fields": missing,
        "follow_up_prompt": follow_up_prompt,
        "answer": answer,
        "expected_compensation": compensation.get("expected_compensation"),
        "compensation_notes": compensation.get("notes", []),
        "eu_uk_context": compensation.get("eu_uk_context", False),
        "refund_email_subject": email_subject,
        "refund_email_body": email_body,
        "contact_message": contact_message,
        "contact_url": contact_url,
        "evidence": top_rows,
        "airline_options": AIRLINE_OPTIONS,
        "disruption_options": DISRUPTION_OPTIONS,
        "event_type_options": EVENT_TYPE_OPTIONS,
        "event_type_labels": EVENT_TYPE_LABELS,
    }
