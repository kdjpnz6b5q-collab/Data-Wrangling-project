#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus

from policy_engine import AIRLINE_CONTACT_URLS, AIRLINE_LABELS

IATA_TO_AIRLINE = {
    "AA": "american",
    "DL": "delta",
    "UA": "united",
    "WN": "southwest",
    "B6": "jetblue",
    "AS": "alaska",
    "F9": "frontier",
    "NK": "spirit",
    "HA": "hawaiian",
    "G4": "allegiant",
    "XP": "avelo",
    "MX": "breeze",
    "SY": "sun_country",
    "LH": "lufthansa",
    "FR": "ryanair",
    "U2": "easyjet",
    "AF": "air_france",
    "BA": "british_airways",
}

AIRLINE_TO_IATA = {airline: code for code, airline in IATA_TO_AIRLINE.items()}

ALLIANCE_OF = {
    "american": "oneworld",
    "alaska": "oneworld",
    "british_airways": "oneworld",
    "delta": "skyteam",
    "air_france": "skyteam",
    "united": "star_alliance",
    "lufthansa": "star_alliance",
}

ALLIANCE_LABELS = {
    "oneworld": "oneworld",
    "skyteam": "SkyTeam",
    "star_alliance": "Star Alliance",
}

ALLIANCE_MEMBERS = {
    "oneworld": ["american", "alaska", "british_airways"],
    "skyteam": ["delta", "air_france"],
    "star_alliance": ["united", "lufthansa"],
}

AIRLINE_REGION = {
    "american": "us",
    "delta": "us",
    "united": "us",
    "southwest": "us",
    "jetblue": "us",
    "alaska": "us",
    "frontier": "us",
    "spirit": "us",
    "hawaiian": "us",
    "allegiant": "us",
    "avelo": "us",
    "breeze": "us",
    "sun_country": "us",
    "lufthansa": "eu",
    "ryanair": "eu",
    "easyjet": "eu",
    "air_france": "eu",
    "british_airways": "eu",
}

REGIONAL_FALLBACKS = {
    "us": ["american", "delta", "united", "southwest", "jetblue", "alaska"],
    "eu": ["lufthansa", "ryanair", "easyjet", "air_france", "british_airways"],
    "global": ["american", "delta", "united", "lufthansa", "air_france", "british_airways"],
}


def normalize_airport(code: str) -> str:
    return code.strip().upper()


def validate_airport(code: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{3}", code))


def parse_departure_time(raw: str) -> datetime | None:
    text = raw.strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    if " " in text and "T" not in text:
        text = text.replace(" ", "T", 1)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def parse_flight_number(flight_number: str) -> tuple[str | None, str | None, str]:
    clean = re.sub(r"\s+", "", flight_number.upper())
    m = re.match(r"^([A-Z]{2,3})(\d{1,4}[A-Z]?)$", clean)
    if not m:
        return None, None, clean

    raw_code = m.group(1)
    code = raw_code[:2] if raw_code[:2] in IATA_TO_AIRLINE else raw_code
    airline = IATA_TO_AIRLINE.get(code)
    return code, airline, clean


def build_google_flights_url(origin: str, destination: str, departure_date: str, airline_label: str) -> str:
    query = f"flights from {origin} to {destination} on {departure_date} {airline_label}"
    return f"https://www.google.com/travel/flights?q={quote_plus(query)}"


def has_amadeus_credentials() -> bool:
    return bool(os.getenv("AMADEUS_CLIENT_ID", "") and os.getenv("AMADEUS_CLIENT_SECRET", ""))


def build_candidate_airlines(source_airline: str) -> list[str]:
    ordered: list[str] = []

    def add(airline_key: str) -> None:
        if airline_key in AIRLINE_LABELS and airline_key not in ordered:
            ordered.append(airline_key)

    add(source_airline)

    alliance = ALLIANCE_OF.get(source_airline)
    if alliance:
        for member in ALLIANCE_MEMBERS.get(alliance, []):
            add(member)

    region = AIRLINE_REGION.get(source_airline, "global")
    for carrier in REGIONAL_FALLBACKS.get(region, []):
        add(carrier)

    for carrier in REGIONAL_FALLBACKS["global"]:
        add(carrier)

    return ordered


def build_reason(source_airline: str, candidate: str) -> str:
    if candidate == source_airline:
        return "Same airline (best chance of direct re-accommodation)."

    source_alliance = ALLIANCE_OF.get(source_airline)
    if source_alliance and candidate in ALLIANCE_MEMBERS.get(source_alliance, []):
        return f"Alliance partner ({ALLIANCE_LABELS[source_alliance]})."

    return "Fallback major carrier option for the route/date search."


def _alliance_recommendations(
    source_airline: str,
    origin_code: str,
    destination_code: str,
    departure_date: str,
    max_results: int,
) -> list[dict[str, Any]]:
    candidates = build_candidate_airlines(source_airline)
    recs: list[dict[str, Any]] = []

    for candidate in candidates[: max(1, max_results)]:
        label = AIRLINE_LABELS.get(candidate, candidate)
        recs.append(
            {
                "airline": candidate,
                "airline_label": label,
                "airline_code": AIRLINE_TO_IATA.get(candidate, ""),
                "reason": build_reason(source_airline, candidate),
                "contact_url": AIRLINE_CONTACT_URLS.get(candidate, ""),
                "google_flights_url": build_google_flights_url(
                    origin_code,
                    destination_code,
                    departure_date,
                    label,
                ),
                "live_offer": False,
            }
        )

    return recs


def _amadeus_get_access_token(base_url: str, client_id: str, client_secret: str, timeout: int = 15) -> str:
    import requests

    token_url = f"{base_url}/v1/security/oauth2/token"
    try:
        resp = requests.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"amadeus_token_request_error: {exc}") from exc

    if resp.status_code >= 400:
        detail = ""
        try:
            payload = resp.json()
            detail = payload.get("error_description") or payload.get("error") or ""
        except Exception:
            detail = (resp.text or "").strip()[:220]
        raise RuntimeError(f"amadeus_token_http_{resp.status_code}: {detail}")

    body = resp.json()
    token = body.get("access_token", "")
    if not token:
        raise RuntimeError("amadeus_token_missing")
    return token


def _amadeus_fetch_live_offers(
    origin_code: str,
    destination_code: str,
    departure_date: str,
    candidate_airlines: list[str],
    max_results: int,
) -> tuple[list[dict[str, Any]], str, str]:
    """
    Returns: (recommendations, data_source, note)
    """
    if not has_amadeus_credentials():
        return [], "alliance_fallback", "Live fare data is not enabled yet (no Amadeus credentials found)."

    client_id = os.getenv("AMADEUS_CLIENT_ID", "")
    client_secret = os.getenv("AMADEUS_CLIENT_SECRET", "")
    base_url = os.getenv("AMADEUS_BASE_URL", "https://test.api.amadeus.com").rstrip("/")

    try:
        import requests
    except Exception:
        return [], "alliance_fallback", "Live fare data disabled because requests library is unavailable."

    try:
        token = _amadeus_get_access_token(base_url, client_id, client_secret)
    except Exception as exc:
        return [], "alliance_fallback", f"Amadeus auth failed: {exc}. Using alliance fallback."

    airline_codes = [AIRLINE_TO_IATA[a] for a in candidate_airlines if AIRLINE_TO_IATA.get(a)]
    params = {
        "originLocationCode": origin_code,
        "destinationLocationCode": destination_code,
        "departureDate": departure_date,
        "adults": "1",
        "max": str(max(5, min(20, max_results * 3))),
        "nonStop": "false",
        "currencyCode": "USD",
    }
    if airline_codes:
        params["includedAirlineCodes"] = ",".join(sorted(set(airline_codes)))

    try:
        resp = requests.get(
            f"{base_url}/v2/shopping/flight-offers",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
            timeout=20,
        )
        if resp.status_code >= 400:
            detail = ""
            try:
                payload = resp.json()
                detail = payload.get("errors", [{}])[0].get("detail", "")
            except Exception:
                detail = (resp.text or "").strip()[:220]
            raise RuntimeError(f"amadeus_search_http_{resp.status_code}: {detail}")
        payload = resp.json()
    except Exception as exc:
        return [], "alliance_fallback", f"Amadeus shopping call failed: {exc}. Using alliance fallback."

    offers = payload.get("data") or []
    if not offers:
        return [], "alliance_fallback", "No live offers returned by Amadeus; using alliance fallback."

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for offer in offers:
        validating = offer.get("validatingAirlineCodes") or []
        primary_code = (validating[0] if validating else "").upper()
        airline_key = IATA_TO_AIRLINE.get(primary_code)
        airline_label = AIRLINE_LABELS.get(airline_key or "", primary_code or "Unknown carrier")

        itineraries = offer.get("itineraries") or []
        if not itineraries:
            continue
        first_it = itineraries[0]
        segments = first_it.get("segments") or []
        if not segments:
            continue

        dep_at = segments[0].get("departure", {}).get("at", "")
        arr_at = segments[-1].get("arrival", {}).get("at", "")
        stops = max(0, len(segments) - 1)
        duration = first_it.get("duration", "")

        price = offer.get("price", {})
        total = price.get("grandTotal", "")
        currency = price.get("currency", "")
        price_text = f"{total} {currency}".strip()

        key = (primary_code, dep_at, arr_at, price_text)
        if key in seen:
            continue
        seen.add(key)

        label_for_query = airline_label if airline_label != "Unknown carrier" else primary_code

        out.append(
            {
                "airline": airline_key or primary_code.lower() or "unknown",
                "airline_label": airline_label,
                "airline_code": primary_code,
                "reason": "Live offer from Amadeus for your route/date.",
                "contact_url": AIRLINE_CONTACT_URLS.get(airline_key or "", ""),
                "google_flights_url": build_google_flights_url(
                    origin_code,
                    destination_code,
                    departure_date,
                    label_for_query,
                ),
                "live_offer": True,
                "price": price_text,
                "departure_at": dep_at,
                "arrival_at": arr_at,
                "stops": stops,
                "duration": duration,
            }
        )

        if len(out) >= max_results:
            break

    if not out:
        return [], "alliance_fallback", "No parsable live offers returned; using alliance fallback."

    return out, "amadeus_live", "Live offers fetched from Amadeus API."


def recommend_alternative_flights(
    flight_number: str,
    origin: str,
    destination: str,
    departure_time: str,
    max_results: int = 5,
) -> dict[str, Any]:
    errors: list[str] = []

    origin_code = normalize_airport(origin)
    destination_code = normalize_airport(destination)

    if not validate_airport(origin_code):
        errors.append("Origin airport must be a 3-letter IATA code (example: JFK).")
    if not validate_airport(destination_code):
        errors.append("Destination airport must be a 3-letter IATA code (example: LHR).")

    dep_dt = parse_departure_time(departure_time)
    if dep_dt is None:
        errors.append("Departure time must be ISO format (example: 2026-03-10T14:30).")

    iata_code, source_airline, normalized_flight = parse_flight_number(flight_number)
    if source_airline is None:
        errors.append(
            "Could not detect airline from flight number. Use IATA style (example: AA123, DL456, BA98)."
        )

    if errors:
        return {
            "ok": False,
            "errors": errors,
        }

    source_airline = str(source_airline)
    source_airline_label = AIRLINE_LABELS.get(source_airline, source_airline)
    source_alliance = ALLIANCE_OF.get(source_airline)

    departure_date = dep_dt.date().isoformat() if dep_dt else ""

    candidate_airlines = build_candidate_airlines(source_airline)

    live_recs, data_source, live_note = _amadeus_fetch_live_offers(
        origin_code=origin_code,
        destination_code=destination_code,
        departure_date=departure_date,
        candidate_airlines=candidate_airlines,
        max_results=max_results,
    )

    if live_recs:
        recs = live_recs
    else:
        recs = _alliance_recommendations(
            source_airline=source_airline,
            origin_code=origin_code,
            destination_code=destination_code,
            departure_date=departure_date,
            max_results=max_results,
        )

    source_contact_url = AIRLINE_CONTACT_URLS.get(source_airline, "")

    return {
        "ok": True,
        "flight_number": normalized_flight,
        "origin": origin_code,
        "destination": destination_code,
        "departure_time": dep_dt.isoformat() if dep_dt else departure_time,
        "source_airline": source_airline,
        "source_airline_label": source_airline_label,
        "source_iata_code": iata_code or "",
        "source_alliance": source_alliance,
        "source_alliance_label": ALLIANCE_LABELS.get(source_alliance or "", "None"),
        "contact_message": f'Please reach out to your airline "{source_airline_label}" - Go Irisch.',
        "contact_url": source_contact_url,
        "live_data_enabled": has_amadeus_credentials(),
        "data_source": data_source,
        "live_data_note": live_note,
        "recommendations": recs,
    }


if __name__ == "__main__":
    print("This module defines helper functions.")
    print("Run the CLI wrapper instead:")
    print(
        "  python src/agent/recommend_alternatives.py "
        "--flight-number AA123 --origin JFK --destination LHR --departure-time 2026-03-10T14:30"
    )
