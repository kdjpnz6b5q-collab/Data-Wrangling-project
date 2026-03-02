#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="air_canada_conditions_of_carriage",
    title="Air Canada Conditions of Carriage",
    url="https://www.aircanada.com/conditionsofcarriage",
    airline="air_canada",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
