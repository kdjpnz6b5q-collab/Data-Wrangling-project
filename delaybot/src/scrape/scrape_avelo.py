#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="avelo_contract_of_carriage",
    title="Avelo Contract of Carriage",
    url="https://www.aveloair.com/contract-of-carriage",
    airline="avelo",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
