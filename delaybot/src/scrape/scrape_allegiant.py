#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="allegiant_customer_service_plan",
    title="Allegiant Customer Service Plan",
    url="https://www.allegiantair.com/customer-service-plan",
    airline="allegiant",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
