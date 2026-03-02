#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="klm_passenger_rights",
    title="KLM Passenger Rights",
    url="https://www.klm.com/information/legal/passenger-rights",
    airline="klm",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
