#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TAGGED_CSV = PROJECT_ROOT / "data" / "processed" / "policy_chunks_tagged.csv"
FALLBACK_CHUNKS = PROJECT_ROOT / "data" / "processed" / "policy_chunks.csv"

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
    "american": 2.8,
    "delta": 2.0,
}


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    return {w for w in words if len(w) > 1 and w not in STOPWORDS}


def load_rows() -> list[dict[str, str]]:
    source = TAGGED_CSV if TAGGED_CSV.exists() else FALLBACK_CHUNKS
    if not source.exists():
        return []
    with source.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_row(query_tokens: set[str], row: dict[str, str]) -> float:
    text = row.get("chunk_text", "")
    tokens = tokenize(text)
    overlap = query_tokens.intersection(tokens)
    score = float(len(overlap))

    text_l = text.lower()
    for term, weight in EXTRA_WEIGHTS.items():
        if term in query_tokens and term in text_l:
            score += weight

    # Bias for matching airline name in title.
    title_l = row.get("title", "").lower()
    if "american" in query_tokens and "american" in title_l:
        score += 2.0
    if "delta" in query_tokens and "delta" in title_l:
        score += 2.0

    return score


def answer_for_query(question: str, top_rows: list[dict[str, str]]) -> str:
    q = question.lower()
    joined = " ".join(r.get("chunk_text", "") for r in top_rows).lower()

    if "american" in q and "weather" in q and ("hotel" in q or "accommodation" in q):
        return (
            "Short answer: usually no. If an American Airlines cancellation is caused by weather "
            "(an uncontrollable event), hotel and meal costs are typically your responsibility; "
            "American usually rebooks you on the next available flight."
        )

    if "delta" in q and "weather" in q and ("hotel" in q or "accommodation" in q):
        return (
            "For Delta, hotel/meal support is generally tied to controllable disruptions. "
            "If weather caused the disruption, those benefits usually do not apply."
        )

    if "refund" in q and ("cancel" in q or "delay" in q):
        return (
            "DOT baseline: for qualifying significant delays/cancellations, refunds should be automatic "
            "to the original payment method when you choose not to travel."
        )

    if "responsible for their own hotel" in joined or "customers are responsible for their own hotel" in joined:
        return (
            "Based on the retrieved policy text, weather-related disruptions are treated as uncontrollable, "
            "and hotel/meal costs are usually not covered by the airline."
        )

    if top_rows:
        return "Best answer from available policy text is shown below with supporting snippets."

    return "No policy data available. Run: make all"


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask DelayBot a policy question")
    parser.add_argument("question", nargs="+", help="Question to ask")
    args = parser.parse_args()
    question = " ".join(args.question).strip()

    rows = load_rows()
    if not rows:
        print("No policy data available. Run: make all")
        return 1

    query_tokens = tokenize(question)
    scored = []
    for row in rows:
        s = score_row(query_tokens, row)
        if s > 0:
            scored.append((s, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [r for _, r in scored[:5]]

    if not top:
        print("No strong matches found in the current policy data.")
        return 0

    print(f"Question: {question}\n")
    print("Answer:")
    print(answer_for_query(question, top))
    print("\nEvidence:")

    shown = set()
    for row in top:
        key = (row.get("title", ""), row.get("chunk_text", ""))
        if key in shown:
            continue
        shown.add(key)
        snippet = row.get("chunk_text", "").strip()
        snippet = re.sub(r"\s+", " ", snippet)
        if len(snippet) > 240:
            snippet = snippet[:240].rstrip() + "..."
        title = row.get("title", "Unknown")
        url = row.get("url", "")
        print(f"- {title}")
        if url:
            print(f"  {url}")
        print(f"  {snippet}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
