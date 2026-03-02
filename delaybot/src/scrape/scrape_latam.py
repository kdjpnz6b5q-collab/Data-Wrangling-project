#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="latam_delays_and_cancellations",
    title="LATAM Delays and Cancellations",
    url="https://www.latamairlines.com/es/en/help-center/faq/changes/tickets/cancellation-or-rescheduling",
    airline="latam",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
