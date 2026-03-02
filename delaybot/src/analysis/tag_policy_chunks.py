#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_CSV = PROJECT_ROOT / "data" / "processed" / "policy_chunks.csv"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "policy_chunks_tagged.csv"

TAG_RULES = {
    "weather": [
        "weather",
        "storm",
        "snow",
        "hurricane",
        "thunderstorm",
        "lightning",
        "ice",
        "fog",
        "winter storm",
        "blizzard",
        "typhoon",
    ],
    "air_traffic_control": [
        "air traffic control",
        "atc",
        "ground stop",
        "flow control",
        "airspace congestion",
    ],
    "hotel": ["hotel", "overnight", "accommodation", "lodging"],
    "meal": ["meal", "voucher", "food"],
    "refund": ["refund", "original form of payment", "credit card"],
    "compensation": ["compensation", "cash payment", "cash compensation", "claim"],
    "reimbursement": ["reimburse", "reimbursement", "reasonable costs"],
    "rebooking": ["rebook", "next available flight", "re-accommodate"],
    "late_inbound_aircraft": [
        "late inbound aircraft",
        "late arriving aircraft",
        "incoming aircraft",
        "aircraft rotation",
    ],
    "controllable": ["within the airline", "controllable", "caused by the airline", "within our control"],
    "uncontrollable": ["uncontrollable", "outside our control", "acts of god"],
    "strike_or_labor": ["strike", "labor action", "industrial action", "union action", "walkout"],
    "mechanical": [
        "mechanical",
        "maintenance",
        "aircraft",
        "defect",
        "technical issue",
        "broken",
        "engine issue",
    ],
    "crew": ["crew", "staffing", "pilot", "flight attendant", "crew rest", "crew legal"],
    "security_geopolitical": [
        "war",
        "conflict",
        "geopolitical",
        "security threat",
        "terror",
        "airspace closure",
        "no-fly zone",
        "military",
        "civil unrest",
        "middle east",
    ],
    "airport_operations": ["airport closure", "runway closure", "airport strike", "terminal closure"],
    "denied_boarding": ["denied boarding", "involuntary denied boarding", "overbook", "bumped"],
}


def infer_tags(text: str) -> str:
    t = text.lower()
    tags = [name for name, terms in TAG_RULES.items() if any(term in t for term in terms)]
    return "|".join(sorted(set(tags)))


def main() -> int:
    if not IN_CSV.exists():
        print(f"No chunk file found: {IN_CSV}")
        return 0

    with IN_CSV.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("No chunks to process; writing empty tagged file.")
        with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["chunk_id", "doc_id", "airline", "title", "url", "chunk_text", "tags"],
            )
            writer.writeheader()
        return 0

    tagged_rows = []
    for row in rows:
        tagged = dict(row)
        tagged["tags"] = infer_tags(row.get("chunk_text", ""))
        tagged_rows.append(tagged)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["chunk_id", "doc_id", "airline", "title", "url", "chunk_text", "tags"],
        )
        writer.writeheader()
        writer.writerows(tagged_rows)

    print(f"Tagged {len(tagged_rows)} chunks -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
