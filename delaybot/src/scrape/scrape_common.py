#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, Tuple
from urllib.parse import urljoin, urlparse, urlunparse
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
    enable_discovery: bool = True
    max_discovered_pages: int = 3
    discovery_keywords: tuple[str, ...] = (
        "customer-service",
        "customer commitment",
        "conditions",
        "contract-of-carriage",
        "carriage",
        "delay",
        "cancellation",
        "cancelled",
        "refund",
        "compensation",
        "disruption",
        "denied-boarding",
        "denied boarding",
        "involuntary",
        "baggage",
        "claim",
        "voucher",
        "meal",
        "hotel",
        "reimburse",
        "irregular-operations",
        "policy",
        "rights",
        "passenger",
        "eu261",
        "uk261",
    )


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


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    cleaned = parsed._replace(fragment="", query=parsed.query.strip())
    return urlunparse(cleaned)


def _url_domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def _same_domain(a: str, b: str) -> bool:
    da = _url_domain(a)
    db = _url_domain(b)
    return da == db or da.endswith(f".{db}") or db.endswith(f".{da}")


def _policy_keyword_hit(url: str, keywords: tuple[str, ...]) -> bool:
    hay = url.lower()
    return any(k in hay for k in keywords)


def _extract_links(html: str, base_url: str) -> list[str]:
    urls: list[str] = []
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if not href:
                continue
            urls.append(urljoin(base_url, href))
    except Exception:
        pass

    # HTML href fallback.
    for m in re.finditer(r"""href=["']([^"'#]+)""", html, flags=re.I):
        href = m.group(1).strip()
        if not href:
            continue
        urls.append(urljoin(base_url, href))

    # Markdown links (common in r.jina.ai responses).
    for m in re.finditer(r"""\[[^\]]+\]\((https?://[^)\s]+)\)""", html, flags=re.I):
        urls.append(m.group(1).strip())

    # Raw absolute URLs.
    for m in re.finditer(r"""https?://[^\s"'<>]+""", html, flags=re.I):
        urls.append(m.group(0).strip())
    out: list[str] = []
    seen: set[str] = set()
    for url in urls:
        try:
            norm = _normalize_url(url)
        except ValueError:
            continue
        if norm.startswith("http") and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def discover_policy_links(target: ScrapeTarget, base_html: str) -> list[str]:
    links = _extract_links(base_html, target.url)
    out: list[str] = []
    for url in links:
        if not _same_domain(url, target.url):
            continue
        if not _policy_keyword_hit(url, target.discovery_keywords):
            continue
        out.append(url)
    return out


def _html_to_text(raw_html: str) -> str:
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(
            [
                "script",
                "style",
                "noscript",
                "svg",
                "form",
                "button",
                "header",
                "footer",
                "nav",
                "aside",
            ]
        ):
            tag.decompose()
        root = soup.find("main") or soup.find("article") or soup.body or soup
        lines = [ln.strip() for ln in root.get_text("\n").splitlines()]
        cleaned: list[str] = []
        seen: set[str] = set()
        for ln in lines:
            if len(ln) < 2:
                continue
            low = ln.lower()
            if any(
                marker in low
                for marker in [
                    "cookie",
                    "privacy policy",
                    "accept all",
                    "sign in",
                    "subscribe",
                    "skip to content",
                    "javascript",
                ]
            ):
                continue
            if low in seen:
                continue
            seen.add(low)
            cleaned.append(ln)
        text = " ".join(cleaned)
    except Exception:
        text = re.sub(r"<script.*?</script>", " ", raw_html, flags=re.I | re.S)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<(nav|header|footer|aside).*?</\\1>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_pages_html(target: ScrapeTarget, pages: list[tuple[str, str, str]]) -> str:
    sections: list[str] = []
    for i, (url, html, method) in enumerate(pages, start=1):
        text = _html_to_text(html)
        if not text:
            continue
        sections.append(
            "<section>"
            f"<h2>Source {i}</h2>"
            f"<p><strong>URL:</strong> {escape(url)}</p>"
            f"<p><strong>Fetch method:</strong> {escape(method)}</p>"
            f"<article><p>{escape(text)}</p></article>"
            "</section>"
        )

    return (
        "<html><body>"
        f"<h1>{escape(target.title)}</h1>"
        f"<p><strong>Airline:</strong> {escape(target.airline)}</p>"
        + "".join(sections)
        + "</body></html>"
    )


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

    pages: list[tuple[str, str, str]] = [(target.url, html, method)]
    if method != "seed" and target.enable_discovery and target.max_discovered_pages > 0:
        discovered = discover_policy_links(target, html)
        for link in discovered:
            if len(pages) >= target.max_discovered_pages + 1:
                break
            link_html, link_method = fetch_with_fallback(link)
            if link_html is None:
                continue
            pages.append((link, link_html, link_method))

    merged_html = combine_pages_html(target, pages)
    out_path = save_html(target.doc_id, merged_html)
    print(
        f"[OK] {target.title} -> {out_path} "
        f"(base_method={method}, pages={len(pages)})"
    )
    return 0
