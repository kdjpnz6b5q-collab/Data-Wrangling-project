#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from html import unescape
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "html"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "policy_texts.csv"

DOC_INFO = {
    "dot_refunds": {
        "title": "DOT Automatic Refund Rule",
        "url": "https://www.transportation.gov/briefing-room/what-airline-passengers-need-know-about-dots-automatic-refund-rule",
    },
    "aa_customer_service_plan": {
        "title": "American Airlines Customer Service Plan",
        "url": "https://www.aa.com/i18n/customer-service/support/customer-service-plan.jsp",
    },
    "delta_customer_commitment": {
        "title": "Delta Customer Commitment",
        "url": "https://www.delta.com/us/en/legal/customer-commitment",
    },
}


def html_to_text(raw_html: str) -> str:
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(raw_html, "html.parser")
        text = soup.get_text("\n")
    except Exception:
        text = re.sub(r"<script.*?</script>", " ", raw_html, flags=re.I | re.S)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)

    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> int:
    files = sorted(RAW_DIR.glob("*.html"))
    if not files:
        print(f"No raw HTML files found in {RAW_DIR}")
        return 0

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for path in files:
        doc_id = path.stem
        info = DOC_INFO.get(doc_id, {"title": doc_id, "url": ""})
        raw = path.read_text(encoding="utf-8", errors="replace")
        text = html_to_text(raw)
        if not text:
            continue
        rows.append(
            {
                "doc_id": doc_id,
                "title": info["title"],
                "url": info["url"],
                "text": text,
            }
        )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "title", "url", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Extracted {len(rows)} documents -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
