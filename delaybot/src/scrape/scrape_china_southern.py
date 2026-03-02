#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="china_southern_passenger_rights",
    title="China Southern Passenger Rights",
    url="https://www.csair.com/en/tourguide/faq/airport/hbywhqx.shtml",
    airline="china_southern",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
