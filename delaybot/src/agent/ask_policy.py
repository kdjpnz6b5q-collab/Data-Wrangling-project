#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys

from policy_engine import (
    AIRLINE_LABELS,
    DISRUPTION_LABELS,
    query_policy,
)


def prompt_choice(field_name: str, options: list[str], labels: dict[str, str]) -> str | None:
    print(f"\nSelect {field_name}:")
    for i, option in enumerate(options, start=1):
        print(f"  {i}. {labels.get(option, option)}")

    while True:
        raw = input(f"Enter number (1-{len(options)}), or press Enter to skip: ").strip()
        if raw == "":
            return None
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        print("Invalid choice. Try again.")


def print_result(result: dict) -> None:
    print(f"Question: {result['question']}\n")

    airline = result.get("airline")
    disruption = result.get("disruption")

    if airline:
        print(f"Airline: {AIRLINE_LABELS.get(airline, airline)}")
    if disruption:
        print(f"Disruption type: {DISRUPTION_LABELS.get(disruption, disruption)}")

    if result.get("follow_up_prompt"):
        print(f"\nFollow-up needed:\n{result['follow_up_prompt']}")

    print("\nAnswer:")
    print(result.get("answer", "No answer available."))

    contact_message = result.get("contact_message", "")
    contact_url = result.get("contact_url", "")
    if contact_message:
        print("\nNext step:")
        print(contact_message)
        if contact_url:
            print(f"Contact page: {contact_url}")

    evidence = result.get("evidence", [])
    if evidence:
        print("\nEvidence:")
        shown = set()
        for row in evidence:
            key = (row.get("title", ""), row.get("chunk_text", ""))
            if key in shown:
                continue
            shown.add(key)

            snippet = row.get("chunk_text", "").strip()
            snippet = re.sub(r"\s+", " ", snippet)
            if len(snippet) > 260:
                snippet = snippet[:260].rstrip() + "..."

            title = row.get("title", "Unknown")
            url = row.get("url", "")
            print(f"- {title}")
            if url:
                print(f"  {url}")
            print(f"  {snippet}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask DelayBot a policy question")
    parser.add_argument("question", nargs="+", help="Question to ask")
    parser.add_argument("--airline", default="", help="Optional airline override")
    parser.add_argument("--disruption", default="", help="Optional disruption type override")
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Do not prompt for missing fields; only print follow-up guidance.",
    )
    args = parser.parse_args()

    question = " ".join(args.question).strip()
    result = query_policy(question, airline_override=args.airline, disruption_override=args.disruption)

    if not result.get("ok"):
        print(result.get("error", "Unknown error"))
        return 1

    missing = result.get("missing_fields", [])
    if missing and not args.no_interactive and sys.stdin.isatty():
        airline_override = args.airline or None
        disruption_override = args.disruption or None

        if "airline" in missing and not airline_override:
            airline_options = result.get("airline_options", [])
            pick = prompt_choice("airline", airline_options, AIRLINE_LABELS)
            if pick:
                airline_override = pick

        if "disruption" in missing and not disruption_override:
            disruption_options = result.get("disruption_options", [])
            pick = prompt_choice("delay/disruption type", disruption_options, DISRUPTION_LABELS)
            if pick:
                disruption_override = pick

        result = query_policy(
            question,
            airline_override=airline_override,
            disruption_override=disruption_override,
        )

    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
