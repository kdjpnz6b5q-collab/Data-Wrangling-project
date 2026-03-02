#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="ryanair_customer_service_charter",
    title="Ryanair Customer Service Charter",
    url="https://www.ryanair.com/gb/en/useful-info/help-centre/terms-and-conditions/customer-service-charter",
    airline="ryanair",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
