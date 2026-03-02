#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="breeze_customer_service_plan",
    title="Breeze Customer Service Plan",
    url="https://www.flybreeze.com/support/customer-service-plan",
    airline="breeze",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
