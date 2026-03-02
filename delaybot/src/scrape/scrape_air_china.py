#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="air_china_rules_and_notices",
    title="Air China Rules and Notices",
    url="https://ru.airchina.com/ES/GB/info/notice/",
    airline="air_china",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
