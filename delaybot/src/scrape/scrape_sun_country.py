#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="sun_country_customer_service_plan",
    title="Sun Country Customer Service Plan",
    url="https://www.suncountry.com/customer-service-plan",
    airline="sun_country",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
