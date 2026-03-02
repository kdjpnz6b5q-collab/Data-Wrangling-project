#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="air_france_customer_commitments",
    title="Air France Customer Commitments",
    url="https://wwws.airfrance.us/information/legal/edito-cgvu",
    airline="air_france",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
