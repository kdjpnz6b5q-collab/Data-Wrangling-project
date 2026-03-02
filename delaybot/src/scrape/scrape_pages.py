#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_ORDER = [
    "scrape_dot.py",
    "scrape_american.py",
    "scrape_delta.py",
    "scrape_united.py",
    "scrape_southwest.py",
    "scrape_jetblue.py",
    "scrape_alaska.py",
    "scrape_frontier.py",
    "scrape_spirit.py",
    "scrape_hawaiian.py",
    "scrape_allegiant.py",
    "scrape_avelo.py",
    "scrape_breeze.py",
    "scrape_sun_country.py",
    "scrape_lufthansa.py",
    "scrape_ryanair.py",
    "scrape_easyjet.py",
    "scrape_air_france.py",
    "scrape_british_airways.py",
]


def main() -> int:
    scrape_dir = Path(__file__).resolve().parent
    total = len(SCRIPT_ORDER)
    ok = 0

    print(f"Scraping {total} policy pages via per-airline scripts...")
    for script_name in SCRIPT_ORDER:
        script_path = scrape_dir / script_name
        if not script_path.exists():
            print(f"[FAIL] missing script: {script_path}")
            continue

        print(f"\n==> Running {script_name}")
        proc = subprocess.run([sys.executable, str(script_path)], check=False)
        if proc.returncode == 0:
            ok += 1

    print(f"\nDone. Successful scrapes: {ok}/{total}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
