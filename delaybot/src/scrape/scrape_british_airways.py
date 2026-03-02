#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="british_airways_customer_commitment",
    title="British Airways Customer Commitment",
    url="https://www.britishairways.com/en-us/information/help-and-contacts/customer-commitment",
    airline="british_airways",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
