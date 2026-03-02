#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="china_eastern_passenger_notice",
    title="China Eastern Passenger Rights Notice",
    url="https://www.ceair.com/global/en_static/Announcement/AnnouncementMessage/202507/t20250710_28581.html",
    airline="china_eastern",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
