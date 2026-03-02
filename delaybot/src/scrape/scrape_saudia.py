#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="saudia_conditions_of_carriage",
    title="Saudia Conditions of Carriage",
    url="https://www.saudia.com/pages/help/useful-links/legal-and-terms-and-conditions/general-conditions-of-carriage",
    airline="saudia",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
