"""
Index a 10-K's chunks into Pinecone.

Run:
    python -m scripts.index_pinecone data/10ks/apple-10k-fy25.pdf

What it does:
  1. Re-runs the parse + section-extract + chunk pipeline (in memory)
  2. Embeds each chunk via Bedrock Titan (1024-dim)
  3. Upserts vectors to Pinecone with chunk_id as the vector ID
  4. Prints index stats so we can see the new vector count

Cost guideline:
  ~102 Bedrock embedding calls × $0.0001 ≈ $0.01-$0.05 per document.

Re-running this script for the same document is safe — Pinecone's upsert
is idempotent on vector ID. Vectors are overwritten, not duplicated.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from app.config import settings
from ingestion.chunkers.section_chunker import chunk_elements
from ingestion.parsers.pdf_parser import extract_section, parse_pdf
from ingestion.writers.pinecone_writer import PineconeWriter

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("index_pinecone")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: python -m scripts.index_pinecone <path-to-pdf>")
        return 2
    pdf_path = Path(argv[1])
    doc_id = pdf_path.stem

    # ---- Pipeline up to chunks ----
    elements = parse_pdf(pdf_path)
    risk_elements = extract_section(elements, start_marker="item 1a. risk factors")
    if not risk_elements:
        log.error("Risk Factors section not found.")
        return 1
    chunks = chunk_elements(risk_elements, doc_id=doc_id,
                            chunk_size=settings.chunk_size,
                            chunk_overlap=settings.chunk_overlap)
    log.info("Pipeline produced %d chunks for %s", len(chunks), doc_id)

    # ---- Index ----
    writer = PineconeWriter()
    log.info("Index stats BEFORE upsert: %s", writer.stats())
    stats = writer.upsert_chunks(chunks, doc_id=doc_id)
    log.info("Upsert stats: %s", stats)
    log.info("Index stats AFTER upsert:  %s", writer.stats())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
