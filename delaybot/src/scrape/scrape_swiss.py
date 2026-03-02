#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="swiss_flight_irregularities",
    title="SWISS Flight Irregularities",
    url="https://www.swiss.com/us/en/fly/flight-information/flight-irregularities",
    airline="swiss",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
