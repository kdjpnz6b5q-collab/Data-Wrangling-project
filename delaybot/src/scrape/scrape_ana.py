#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="ana_customer_service_plan",
    title="ANA Customer Service Plan",
    url="https://www.ana.co.jp/en/jp/guide/flight_service_info/assist/customer-service-plan/",
    airline="ana",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
