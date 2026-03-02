#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="hawaiian_customer_service_plan",
    title="Hawaiian Airlines Customer Service Plan",
    url="https://www2.hawaiianairlines.com/legal/customer-service-plan",
    airline="hawaiian",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
