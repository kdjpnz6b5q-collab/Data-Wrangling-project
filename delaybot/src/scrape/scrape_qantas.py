#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="qantas_compensation_and_refunds",
    title="Qantas Compensation and Refunds Policy",
    url="https://www.qantas.com/us/en/book-a-trip/flights/compensation-and-refunds-policy.html",
    airline="qantas",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
