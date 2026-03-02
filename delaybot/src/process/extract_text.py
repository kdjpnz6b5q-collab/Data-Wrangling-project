#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from html import unescape
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "html"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "policy_texts.csv"
SEED_FILE = PROJECT_ROOT / "data" / "seeds" / "fallback_policies.json"


def load_doc_info() -> dict[str, dict[str, str]]:
    if not SEED_FILE.exists():
        return {}
    with SEED_FILE.open("r", encoding="utf-8") as f:
        docs = json.load(f)
    out: dict[str, dict[str, str]] = {}
    for doc in docs:
        out[doc["doc_id"]] = {
            "title": doc.get("title", doc["doc_id"]),
            "url": doc.get("url", ""),
            "airline": doc.get("airline", "unknown"),
        }
    return out


def html_to_text(raw_html: str) -> str:
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
        cleaned_lines: list[str] = []
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
            cleaned_lines.append(ln)
        text = " ".join(cleaned_lines)
    except Exception:
        text = re.sub(r"<script.*?</script>", " ", raw_html, flags=re.I | re.S)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<(nav|header|footer|aside).*?</\\1>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)

    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main() -> int:
    files = sorted(RAW_DIR.glob("*.html"))
    if not files:
        print(f"No raw HTML files found in {RAW_DIR}")
        return 0

    info_map = load_doc_info()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for path in files:
        doc_id = path.stem
        info = info_map.get(
            doc_id,
            {
                "title": doc_id,
                "url": "",
                "airline": "unknown",
            },
        )
        raw = path.read_text(encoding="utf-8", errors="replace")
        text = html_to_text(raw)
        if not text:
            continue
        rows.append(
            {
                "doc_id": doc_id,
                "airline": info["airline"],
                "title": info["title"],
                "url": info["url"],
                "text": text,
            }
        )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "airline", "title", "url", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Extracted {len(rows)} documents -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
