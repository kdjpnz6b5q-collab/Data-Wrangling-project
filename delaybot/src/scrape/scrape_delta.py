#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="delta_customer_commitment",
    title="Delta Customer Commitment",
    url="https://www.delta.com/us/en/legal/customer-commitment",
    airline="delta",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
