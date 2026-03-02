#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_CSV = PROJECT_ROOT / "data" / "processed" / "policy_texts.csv"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "policy_chunks.csv"

MAX_CHARS = 700


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    buff: list[str] = []
    size = 0

    for sentence in sentences:
        if len(sentence) > max_chars:
            if buff:
                chunks.append(" ".join(buff).strip())
                buff = []
                size = 0
            for i in range(0, len(sentence), max_chars):
                chunks.append(sentence[i : i + max_chars])
            continue

        projected = size + len(sentence) + (1 if buff else 0)
        if projected <= max_chars:
            buff.append(sentence)
            size = projected
        else:
            chunks.append(" ".join(buff).strip())
            buff = [sentence]
            size = len(sentence)

    if buff:
        chunks.append(" ".join(buff).strip())

    return chunks


def main() -> int:
    if not IN_CSV.exists():
        print(f"Input not found: {IN_CSV}")
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["chunk_id", "doc_id", "title", "url", "chunk_text"],
            )
            writer.writeheader()
        print(f"Wrote empty chunk file -> {OUT_CSV}")
        return 0

    with IN_CSV.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    chunks_out = []
    for row in rows:
        doc_chunks = chunk_text(row.get("text", ""))
        for i, ck in enumerate(doc_chunks, start=1):
            chunks_out.append(
                {
                    "chunk_id": f"{row['doc_id']}_{i:03d}",
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "url": row["url"],
                    "chunk_text": ck,
                }
            )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["chunk_id", "doc_id", "title", "url", "chunk_text"],
        )
        writer.writeheader()
        writer.writerows(chunks_out)

    print(f"Chunked {len(rows)} documents into {len(chunks_out)} chunks -> {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
