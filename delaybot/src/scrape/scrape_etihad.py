#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="etihad_rules_and_notices",
    title="Etihad Rules and Notices",
    url="https://www.etihad.com/en-us/legal/rules-and-notices",
    airline="etihad",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
