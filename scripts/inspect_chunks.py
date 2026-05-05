"""
Inspect the chunked output of a 10-K's Risk Factors section.

Run:
    python -m scripts.inspect_chunks data/10ks/apple-10k-fy25.pdf

What it does:
  1. Parses the PDF with Unstructured.io
  2. Extracts the Risk Factors (Item 1A) section
  3. Cleans + chunks the section into ~1024-char retrievable units
  4. Prints stats: chunk count, char distribution, page coverage
  5. Prints the first and last chunk so you can sanity-check quality
  6. Dumps all chunks to data/processed/<basename>.chunks.json

This is a one-shot exploration — not part of the production pipeline.
The point is to verify cleaning + chunking quality before we send chunks
through the embedder or the LLM extractor.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from app.config import settings
from ingestion.chunkers.section_chunker import chunk_elements
from ingestion.parsers.pdf_parser import extract_section, parse_pdf

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("inspect_chunks")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python -m scripts.inspect_chunks <path-to-pdf>")
        return 2
    pdf_path = Path(argv[1])
    doc_id = pdf_path.stem  # e.g. "apple-10k-fy25"

    # ---- Parse + extract ----
    try:
        elements = parse_pdf(pdf_path)
    except Exception as e:
        log.exception("PDF parsing failed: %s", e)
        return 1

    risk_elements = extract_section(elements, start_marker="item 1a. risk factors")
    if not risk_elements:
        log.error("Risk Factors section not found.")
        return 1
    log.info("Extracted %d Risk Factors elements", len(risk_elements))

    # ---- Clean + chunk ----
    chunks = chunk_elements(
        elements=risk_elements,
        doc_id=doc_id,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        log.error("Chunking produced no chunks.")
        return 1

    # ---- Stats ----
    sizes = [c.char_count for c in chunks]
    pages = sorted({c.page_start for c in chunks if c.page_start is not None})
    log.info("==== Chunk statistics ====")
    log.info("Total chunks: %d", len(chunks))
    log.info("Char count: min=%d, max=%d, avg=%d, total=%d",
             min(sizes), max(sizes), sum(sizes) // len(sizes), sum(sizes))
    log.info("Page coverage: %s", pages)

    # ---- Show first and last chunk ----
    log.info("==== First chunk ====")
    first = chunks[0]
    log.info("id=%s pages=%s..%s chars=%d",
             first.chunk_id, first.page_start, first.page_end, first.char_count)
    log.info("text: %r", first.text[:300])

    log.info("==== Last chunk ====")
    last = chunks[-1]
    log.info("id=%s pages=%s..%s chars=%d",
             last.chunk_id, last.page_start, last.page_end, last.char_count)
    log.info("text: %r", last.text[:300])

    # ---- Dump ----
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{doc_id}.chunks.json"
    out_path.write_text(
        json.dumps([c.to_dict() for c in chunks], indent=2),
        encoding="utf-8",
    )
    log.info("Wrote %d chunks to %s (%d KB)",
             len(chunks), out_path, out_path.stat().st_size // 1024)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
