#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="united_customer_commitment",
    title="United Customer Commitment",
    url="https://www.united.com/en/us/fly/customer-commitment.html",
    airline="united",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
