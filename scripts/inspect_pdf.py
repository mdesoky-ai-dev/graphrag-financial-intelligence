"""
Inspect a parsed 10-K PDF.

Run:
    python -m scripts.inspect_pdf data/10ks/apple-10k-fy25.pdf

What it does:
  1. Parses the full PDF with Unstructured.io ('fast' strategy, no OCR)
  2. Prints summary stats (total elements, breakdown by type, page range)
  3. Locates and extracts the 'Risk Factors' section
  4. Prints the first 3 elements of the section so you can eyeball quality
  5. Dumps the section to data/processed/<basename>.risks.json for inspection

This is a one-shot exploration tool — not part of the production pipeline.
The point is to stare at the JSON and decide chunking strategy from real data.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from app.config import settings
from ingestion.parsers.pdf_parser import (
    extract_section,
    parse_pdf,
    summarize_elements,
)

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("inspect_pdf")


def truncate(text: str, n: int = 200) -> str:
    """One-line preview of an element's text."""
    text = (text or "").replace("\n", " ").strip()
    return text[:n] + ("..." if len(text) > n else "")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python -m scripts.inspect_pdf <path-to-pdf>")
        return 2
    pdf_path = Path(argv[1])

    # ---- 1. Parse the full document ----
    try:
        elements = parse_pdf(pdf_path)
    except Exception as e:
        log.exception("PDF parsing failed: %s", e)
        return 1

    # ---- 2. Summary statistics ----
    summary = summarize_elements(elements)
    log.info("==== Document summary ====")
    log.info("Total elements: %d", summary["total_elements"])
    log.info("Page range: %s..%s (%d pages)",
             summary["first_page"], summary["last_page"], summary["page_count"])
    log.info("Elements by type:")
    for t, count in summary["by_type"].items():
        log.info("  %-25s %d", t, count)

    # ---- 3. Extract Risk Factors ----
    # Apple (and many SEC filings parsed with 'fast' strategy) glues section
    # headers onto paragraph bodies. So we search for the longer prefix
    # 'item 1a. risk factors' which is distinctive enough to skip TOC noise.
    log.info("==== Extracting 'Risk Factors' (Item 1A) ====")
    risk_elements = extract_section(elements, start_marker="item 1a. risk factors")

    if not risk_elements:
        log.error(
            "Could not locate Risk Factors section. "
            "Try opening the JSON dump of the full document to inspect headers."
        )
        # Still dump the full element list so we can debug header detection
        out_dir = Path("data/processed")
        out_dir.mkdir(parents=True, exist_ok=True)
        full_path = out_dir / f"{pdf_path.stem}.full.json"
        full_path.write_text(json.dumps(elements, indent=2), encoding="utf-8")
        log.info("Dumped all %d elements to %s for inspection", len(elements), full_path)
        return 1

    risk_summary = summarize_elements(risk_elements)
    log.info("Risk Factors section: %d elements across pages %s..%s",
             risk_summary["total_elements"],
             risk_summary["first_page"],
             risk_summary["last_page"])

    # ---- 4. Show the first few elements so we can sanity-check parsing quality ----
    log.info("==== First 3 elements of Risk Factors ====")
    for i, el in enumerate(risk_elements[:3]):
        log.info("[%d] type=%s page=%s text=%r", i, el["type"], el["page"], truncate(el["text"]))

    # ---- 5. Dump to JSON for offline review ----
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdf_path.stem}.risks.json"
    out_path.write_text(json.dumps(risk_elements, indent=2), encoding="utf-8")
    log.info("Wrote Risk Factors section to %s", out_path)
    log.info("File size: %d KB", out_path.stat().st_size // 1024)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
