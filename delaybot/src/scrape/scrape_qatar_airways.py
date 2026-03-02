#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="qatar_airways_passenger_rights",
    title="Qatar Airways EU Air Passenger Rights",
    url="https://www.qatarairways.com/en-na/legal/eu-air-passenger-rights.html",
    airline="qatar_airways",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
