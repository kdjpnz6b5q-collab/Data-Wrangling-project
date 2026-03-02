#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="frontier_customer_service_plan",
    title="Frontier Customer Service Plan",
    url="https://www.flyfrontier.com/legal/customer-service-plan",
    airline="frontier",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
