#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="iberia_passenger_rights",
    title="Iberia Passenger Rights",
    url="https://www.iberia.com/gb/passengers-rights/",
    airline="iberia",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
