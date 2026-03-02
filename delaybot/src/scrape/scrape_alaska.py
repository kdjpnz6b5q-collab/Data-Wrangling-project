#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="alaska_customer_commitment",
    title="Alaska Customer Commitment",
    url="https://www.alaskaair.com/content/about-us/customer-commitment/customer-commitment-overview",
    airline="alaska",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
