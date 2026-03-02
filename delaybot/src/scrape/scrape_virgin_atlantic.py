#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="virgin_atlantic_ec261",
    title="Virgin Atlantic EC261 Passenger Rights",
    url="https://help.virginatlantic.com/fr/en/cancelled-delayed-or-disrupted-flights/regulation-ec-no2612004.html",
    airline="virgin_atlantic",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
