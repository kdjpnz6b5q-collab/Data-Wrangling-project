#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="easyjet_delays_and_cancellations",
    title="easyJet Delays and Cancellations Policy",
    url="https://www.easyjet.com/en/help/boarding-and-flying/delays-and-cancellations",
    airline="easyjet",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
