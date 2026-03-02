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
    "scrape_emirates.py",
    "scrape_qatar_airways.py",
    "scrape_singapore_airlines.py",
    "scrape_turkish_airlines.py",
    "scrape_air_canada.py",
    "scrape_klm.py",
    "scrape_iberia.py",
    "scrape_latam.py",
    "scrape_avianca.py",
    "scrape_etihad.py",
    "scrape_virgin_atlantic.py",
    "scrape_ana.py",
    "scrape_japan_airlines.py",
    "scrape_china_eastern.py",
    "scrape_china_southern.py",
    "scrape_air_china.py",
    "scrape_indigo.py",
    "scrape_qantas.py",
    "scrape_saudia.py",
    "scrape_swiss.py",
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
