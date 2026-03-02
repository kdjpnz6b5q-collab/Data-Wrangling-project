#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="avianca_delays_and_cancellations",
    title="Avianca Delays and Cancellations",
    url="https://ayuda.avianca.com/hc/en-us/articles/29522455248795-What-happens-if-my-flight-is-delayed-or-was-canceled",
    airline="avianca",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
