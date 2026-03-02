#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="lufthansa_passenger_rights",
    title="Lufthansa Passenger Rights",
    url="https://www.lufthansa.com/us/en/passenger-rights",
    airline="lufthansa",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
