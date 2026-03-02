#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="dot_refunds",
    title="DOT Automatic Refund Rule",
    url="https://www.transportation.gov/briefing-room/what-airline-passengers-need-know-about-dots-automatic-refund-rule",
    airline="dot",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
