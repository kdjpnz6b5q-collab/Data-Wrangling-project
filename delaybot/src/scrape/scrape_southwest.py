#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="southwest_customer_service_commitment",
    title="Southwest Customer Service Commitment",
    url="https://www.southwest.com/about-southwest/customer-service-commitment/",
    airline="southwest",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
