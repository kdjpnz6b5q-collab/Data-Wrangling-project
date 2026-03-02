#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from html import escape
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "html"
SEED_FILE = PROJECT_ROOT / "data" / "seeds" / "fallback_policies.json"

TARGETS = [
    {
        "doc_id": "dot_refunds",
        "title": "DOT Automatic Refund Rule",
        "url": "https://www.transportation.gov/briefing-room/what-airline-passengers-need-know-about-dots-automatic-refund-rule",
    },
    {
        "doc_id": "aa_customer_service_plan",
        "title": "American Airlines Customer Service Plan",
        "url": "https://www.aa.com/i18n/customer-service/support/customer-service-plan.jsp",
    },
    {
        "doc_id": "delta_customer_commitment",
        "title": "Delta Customer Commitment",
        "url": "https://www.delta.com/us/en/legal/customer-commitment",
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_url(url: str, timeout: int = 25) -> Tuple[int, str]:
    req = Request(url, headers=HEADERS)
    with urlopen(req, timeout=timeout) as resp:
        status = int(getattr(resp, "status", 200) or 200)
        body = resp.read().decode("utf-8", errors="replace")
        return status, body


def looks_blocked(text: str) -> bool:
    t = text.lower()
    blocked_markers = [
        "access denied",
        "request unsuccessful",
        "forbidden",
        "attention required",
        "are you a human",
        "captcha",
    ]
    return any(marker in t for marker in blocked_markers)


def fetch_with_fallback(url: str) -> Tuple[str | None, str]:
    try:
        status, text = fetch_url(url)
        if status == 200 and not looks_blocked(text):
            return text, "direct"
    except (HTTPError, URLError, TimeoutError, OSError):
        pass

    jina_candidates = [f"https://r.jina.ai/{url}"]
    if url.startswith("https://"):
        jina_candidates.append(f"https://r.jina.ai/http://{url[len('https://'):]}")

    for candidate in jina_candidates:
        try:
            status, text = fetch_url(candidate)
            if status == 200 and len(text.strip()) > 80:
                # r.jina.ai usually returns markdown/plain text; convert to simple HTML for downstream parsing.
                wrapped = (
                    "<html><body><article>"
                    f"<pre>{escape(text)}</pre>"
                    "</article></body></html>"
                )
                return wrapped, "jina"
        except (HTTPError, URLError, TimeoutError, OSError):
            continue

    return None, "none"


def load_seed_map() -> Dict[str, dict]:
    with SEED_FILE.open("r", encoding="utf-8") as f:
        docs = json.load(f)
    return {d["doc_id"]: d for d in docs}


def to_seed_html(doc: dict) -> str:
    text = escape(doc["text"])
    title = escape(doc["title"])
    url = escape(doc["url"])
    return (
        "<html><body>"
        f"<h1>{title}</h1>"
        f"<p><strong>Source URL:</strong> {url}</p>"
        f"<article><p>{text}</p></article>"
        "</body></html>"
    )


def save_html(doc_id: str, html: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{doc_id}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def main() -> int:
    seed_map = load_seed_map()
    total = len(TARGETS)
    saved = 0

    print(f"Scraping {total} policy pages...")

    for target in TARGETS:
        doc_id = target["doc_id"]
        title = target["title"]
        url = target["url"]

        html, method = fetch_with_fallback(url)
        if html is None:
            seed = seed_map.get(doc_id)
            if not seed:
                print(f"[FAIL] {doc_id}: no network data and no seed fallback")
                continue
            html = to_seed_html(seed)
            method = "seed"

        out_path = save_html(doc_id, html)
        saved += 1
        print(f"[OK] {title} -> {out_path} (method={method})")

    print(f"Done. Saved {saved}/{total} pages to {RAW_DIR}")
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
