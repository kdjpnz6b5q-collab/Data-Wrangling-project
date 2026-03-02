#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="spirit_customer_service_plan",
    title="Spirit Customer Service Plan",
    url="https://customersupport.spirit.com/en-us/category/article/KA-01182",
    airline="spirit",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
