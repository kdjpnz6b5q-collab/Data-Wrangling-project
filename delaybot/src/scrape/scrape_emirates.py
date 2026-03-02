#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="emirates_delay_notice",
    title="Emirates Delay Notice",
    url="https://www.emirates.com/us/english/before-you-fly/travel/rules-and-notices/delay-notice/",
    airline="emirates",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
