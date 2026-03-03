#!/usr/bin/env python3
from __future__ import annotations

import argparse

from flight_recommender import recommend_alternative_flights


def main() -> int:
    parser = argparse.ArgumentParser(description="Recommend alternative flights using alliance logic")
    parser.add_argument("--flight-number", required=True, help="Original flight number, e.g. AA123")
    parser.add_argument("--origin", required=True, help="3-letter origin airport code, e.g. JFK")
    parser.add_argument("--destination", required=True, help="3-letter destination airport code, e.g. LHR")
    parser.add_argument(
        "--departure-time",
        required=True,
        help="Original departure datetime (ISO), e.g. 2026-03-10T14:30",
    )
    parser.add_argument("--max-results", type=int, default=5, help="Number of alternatives to return")
    args = parser.parse_args()

    result = recommend_alternative_flights(
        flight_number=args.flight_number,
        origin=args.origin,
        destination=args.destination,
        departure_time=args.departure_time,
        max_results=args.max_results,
    )

    if not result.get("ok"):
        print("Could not generate recommendations:")
        for err in result.get("errors", []):
            print(f"- {err}")
        return 1

    print("Alternative Flight Recommendations\n")
    print(f"Original flight: {result['flight_number']}")
    print(f"Route: {result['origin']} -> {result['destination']}")
    print(f"Departure time: {result['departure_time']}")
    print(f"Detected airline: {result['source_airline_label']} ({result['source_iata_code']})")
    print(f"Alliance: {result['source_alliance_label']}")
    print(f"Recommendation source: {result.get('data_source', 'unknown')}")
    print(f"\n{result['contact_message']}")
    if result.get("contact_url"):
        print(f"Primary contact page: {result['contact_url']}")

    print(f"\nNote: {result['live_data_note']}")

    print("\nRecommended alternatives:")
    for i, rec in enumerate(result.get("recommendations", []), start=1):
        code = rec.get("airline_code", "")
        code_text = f" ({code})" if code else ""
        print(f"{i}. {rec['airline_label']}{code_text}")
        reason = str(rec.get("reason") or "").strip()
        if reason:
            print(f"   Why: {reason}")
        if rec.get("live_offer"):
            if rec.get("price"):
                print(f"   Price: {rec['price']}")
            if rec.get("departure_at"):
                print(f"   Departure: {rec['departure_at']}")
            if rec.get("arrival_at"):
                print(f"   Arrival: {rec['arrival_at']}")
            if rec.get("duration"):
                print(f"   Duration: {rec['duration']}")
            if rec.get("stops") is not None:
                print(f"   Stops: {rec['stops']}")
        if rec.get("contact_url"):
            print(f"   Contact: {rec['contact_url']}")
        print(f"   Search: {rec['google_flights_url']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
