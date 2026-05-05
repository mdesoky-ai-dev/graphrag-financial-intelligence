"""
PDF parser for SEC 10-K filings.

Wraps Unstructured.io with sensible defaults for financial documents:
  - 'fast' strategy (no Tesseract / Poppler required, works on Windows)
  - returns a list of structured Element objects with type, text, page, and metadata
  - includes a section extractor that pulls out the Risk Factors block specifically

Usage:
    from ingestion.parsers.pdf_parser import parse_pdf, extract_section

    elements = parse_pdf("data/10ks/apple-10k-fy25.pdf")
    risk_elements = extract_section(elements, start_marker="risk factors")
    print(f"Found {len(risk_elements)} elements in the Risk Factors section")

Design notes:
  - 10-Ks are text-based PDFs; OCR is unnecessary and slow. We disable it.
  - Section extraction defaults to occurrence=2 because the first match is
    always the TOC entry; the actual section starts at the second match.
  - Unstructured sometimes glues page numbers onto headers ('Risk Factors5').
    We strip trailing digits from header text before matching.
  - We deliberately keep the data model small: a list of dicts with the fields
    we actually use downstream (text, type, page, depth). Unstructured returns
    much richer objects but we don't need them yet.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from unstructured.partition.pdf import partition_pdf

logger = logging.getLogger(__name__)


# Element types Unstructured.io produces. We care most about Title (headers),
# NarrativeText (paragraphs), Table, and ListItem.
HEADER_TYPES = {"Title", "Header"}

# Trailing page-number pattern: 'Risk Factors5' -> 'Risk Factors'.
# Match 1-3 digits at the end, optionally preceded by whitespace.
_TRAILING_PAGE_NUM = re.compile(r"\s*\d{1,3}\s*$")


def parse_pdf(path: str | Path, strategy: str = "fast") -> list[dict[str, Any]]:
    """Parse a PDF into a list of structured elements.

    Args:
        path: Path to the PDF file.
        strategy: Unstructured strategy. 'fast' avoids OCR and is sufficient
            for text-based PDFs like SEC filings. Use 'hi_res' if you ever need
            to handle scanned documents (requires Tesseract + Poppler).

    Returns:
        List of dicts, one per parsed element, with keys:
          - text: the element's text content
          - type: the element's type (Title, NarrativeText, Table, ListItem, etc.)
          - page: page number (1-indexed)
          - category_depth: heading level if known (None for non-headers)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found at {path}")

    logger.info("Parsing %s with strategy=%s ...", path, strategy)
    elements = partition_pdf(filename=str(path), strategy=strategy)
    logger.info("Parsed %d elements from %s", len(elements), path.name)

    return [
        {
            "text": el.text or "",
            "type": el.category if hasattr(el, "category") else type(el).__name__,
            "page": getattr(el.metadata, "page_number", None),
            "category_depth": getattr(el.metadata, "category_depth", None),
        }
        for el in elements
    ]


def summarize_elements(elements: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a quick statistical summary: counts by type, page range, etc."""
    by_type: dict[str, int] = {}
    pages: set[int] = set()
    for el in elements:
        t = el["type"]
        by_type[t] = by_type.get(t, 0) + 1
        if el["page"] is not None:
            pages.add(el["page"])
    return {
        "total_elements": len(elements),
        "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
        "page_count": len(pages),
        "first_page": min(pages) if pages else None,
        "last_page": max(pages) if pages else None,
    }


def extract_section(
    elements: list[dict[str, Any]],
    start_marker: str,
    end_markers: list[str] | None = None,
    occurrence: int = 1,
) -> list[dict[str, Any]]:
    """Extract a contiguous run of elements that belong to a named section.

    Apple's 10-K (and many SEC filings parsed with Unstructured's 'fast'
    strategy) glue section headers onto paragraph bodies as NarrativeText
    rather than emitting them as Title elements. So we cannot rely on
    element type to find boundaries. Instead, we search element TEXT for
    the start_marker as a prefix, regardless of element type.

    Strategy:
      1. Find the Nth element whose text *starts with* `start_marker`
         (case-insensitive). Default occurrence=1: first content match
         is usually the real section, since glued-in section text is
         distinctive enough to skip TOC noise. (TOC entries often just
         say 'Risk Factors' alone, while the real section opens with
         'Item 1A. Risk Factors' followed by body text — different prefix.)
      2. Collect every subsequent element until we hit an element whose
         text starts with any of `end_markers` (e.g. 'Item 3.'), at which
         point we stop *before* including it.

    Args:
        elements: parsed output from parse_pdf
        start_marker: text prefix to find (e.g. 'item 1a' or 'risk factors')
        end_markers: text prefixes that signal the next section.
            If None, defaults to 10-K Item 1A end markers.
        occurrence: which match to return (1=first).

    Returns:
        Subset of elements belonging to the target section. Empty list if
        not found.
    """
    if end_markers is None:
        end_markers = [
            "item 1b.",
            "item 1c.",
            "item 2.",
            "item 3.",
            "item 4.",
            "part ii",
        ]

    start_marker = start_marker.lower().strip()
    end_markers = [m.lower().strip() for m in end_markers]

    def text_prefix(el: dict[str, Any]) -> str:
        """Cleaned, lowercased prefix of an element's text."""
        return (el["text"] or "").strip().lower()

    # 1. Find the Nth content match
    matches: list[int] = []
    for i, el in enumerate(elements):
        if text_prefix(el).startswith(start_marker):
            matches.append(i)
            logger.info(
                "Match %d for start %r at element %d, page %s, type=%s, text=%r",
                len(matches), start_marker, i, el["page"], el["type"],
                el["text"][:80],
            )
            if len(matches) >= occurrence:
                break

    if len(matches) < occurrence:
        logger.warning(
            "Section %r: found %d matches, needed %d.",
            start_marker, len(matches), occurrence,
        )
        return []

    start_idx = matches[occurrence - 1]

    # 2. Walk forward until an end marker
    section: list[dict[str, Any]] = [elements[start_idx]]
    for el in elements[start_idx + 1 :]:
        prefix = text_prefix(el)
        if any(prefix.startswith(m) for m in end_markers):
            logger.info(
                "Section ended at element with end-marker prefix: %r",
                el["text"][:100],
            )
            break
        section.append(el)

    logger.info("Extracted section: %d elements", len(section))
    return section
