#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="indigo_conditions_of_carriage",
    title="IndiGo Conditions of Carriage",
    url="https://www.goindigo.in/information/conditions-of-carriage.html",
    airline="indigo",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
