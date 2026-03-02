#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="american_customer_service_plan",
    title="American Airlines Customer Service Plan",
    url="https://www.aa.com/i18n/customer-service/support/customer-service-plan.jsp",
    airline="american",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
