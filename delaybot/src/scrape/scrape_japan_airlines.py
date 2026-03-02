#!/usr/bin/env python3
from __future__ import annotations

from scrape_common import ScrapeTarget, run_single_target

TARGET = ScrapeTarget(
    doc_id="japan_airlines_ec261",
    title="Japan Airlines EC261 Rights",
    url="https://www.jal.co.jp/fr/en/info/ec261_04_eu/",
    airline="japan_airlines",
)


if __name__ == "__main__":
    raise SystemExit(run_single_target(TARGET))
