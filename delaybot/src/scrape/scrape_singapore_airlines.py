#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="singapore_airlines_passenger_rights",
    title="Singapore Airlines Passenger Rights",
    url="https://www.singaporeair.com/en_UK/us/travel-info/customer-commitment/pax-rights-regulations-uk/",
    airline="singapore_airlines",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
