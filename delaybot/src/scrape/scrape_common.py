#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "html"
SEED_FILE = PROJECT_ROOT / "data" / "seeds" / "fallback_policies.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass(frozen=True)
class ScrapeTarget:
    doc_id: str
    title: str
    url: str
    airline: str


def fetch_url(url: str, timeout: int = 30) -> Tuple[int, str]:
    req = Request(url, headers=HEADERS)
    with urlopen(req, timeout=timeout) as resp:
        status = int(getattr(resp, "status", 200) or 200)
        body = resp.read().decode("utf-8", errors="replace")
    return status, body


def looks_blocked(text: str) -> bool:
    blocked_markers = [
        "access denied",
        "request unsuccessful",
        "forbidden",
        "attention required",
        "are you a human",
        "captcha",
        "security challenge",
    ]
    lower = text.lower()
    return any(marker in lower for marker in blocked_markers)


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
            if status == 200 and len(text.strip()) > 120:
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
    if not SEED_FILE.exists():
        return {}
    with SEED_FILE.open("r", encoding="utf-8") as f:
        docs = json.load(f)
    return {d["doc_id"]: d for d in docs}


def to_seed_html(doc: dict) -> str:
    text = escape(doc.get("text", ""))
    title = escape(doc.get("title", doc.get("doc_id", "")))
    url = escape(doc.get("url", ""))
    airline = escape(doc.get("airline", "unknown"))
    return (
        "<html><body>"
        f"<h1>{title}</h1>"
        f"<p><strong>Airline:</strong> {airline}</p>"
        f"<p><strong>Source URL:</strong> {url}</p>"
        f"<article><p>{text}</p></article>"
        "</body></html>"
    )


def save_html(doc_id: str, html: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{doc_id}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def run_single_target(target: ScrapeTarget) -> int:
    seeds = load_seed_map()

    html, method = fetch_with_fallback(target.url)
    if html is None:
        seed_doc = seeds.get(target.doc_id)
        if seed_doc is None:
            print(f"[FAIL] {target.title}: network blocked and no seed fallback")
            return 1
        html = to_seed_html(seed_doc)
        method = "seed"

    out_path = save_html(target.doc_id, html)
    print(f"[OK] {target.title} -> {out_path} (method={method})")
    return 0
