#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="turkish_airlines_passenger_rights",
    title="Turkish Airlines Passenger Rights",
    url="https://www.turkishairlines.com/en-int/legal-notice/other-regulations/",
    airline="turkish_airlines",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
