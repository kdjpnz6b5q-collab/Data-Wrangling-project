#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_CSV = PROJECT_ROOT / "data" / "processed" / "policy_chunks.csv"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "policy_chunks_tagged.csv"

TAG_RULES = {
    "weather": ["weather", "uncontrollable", "acts of god", "air traffic control"],
    "hotel": ["hotel", "overnight", "accommodation"],
    "meal": ["meal", "voucher", "food"],
    "refund": ["refund", "credit card", "original form of payment"],
    "rebooking": ["rebook", "next available flight"],
    "controllable": ["within the airline's control", "controllable", "caused by the airline"],
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

    if not rows:
        print("No chunks to process; writing empty tagged file.")
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["chunk_id", "doc_id", "title", "url", "chunk_text", "tags"],
            )
            writer.writeheader()
        return 0

    tagged_rows = []
    for row in rows:
        tagged = dict(row)
        tagged["tags"] = infer_tags(row.get("chunk_text", ""))
        tagged_rows.append(tagged)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["chunk_id", "doc_id", "title", "url", "chunk_text", "tags"],
        )
        writer.writeheader()
        writer.writerows(tagged_rows)

    print(f"Tagged {len(tagged_rows)} chunks -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
