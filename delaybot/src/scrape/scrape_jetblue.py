#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="jetblue_customer_service_plan",
    title="JetBlue Customer Service Plan",
    url="https://www.jetblue.com/customer-assurance/customer-service-plan",
    airline="jetblue",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
