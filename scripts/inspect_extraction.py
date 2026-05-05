"""
Inspect the LLM extraction on the first N chunks of a 10-K.

Run:
    python -m scripts.inspect_extraction data/10ks/apple-10k-fy25.pdf
    python -m scripts.inspect_extraction data/10ks/apple-10k-fy25.pdf --n 5
    python -m scripts.inspect_extraction data/10ks/apple-10k-fy25.pdf --n 20

What it does:
  1. Runs the full pipeline (parse + extract section + chunk) in memory
  2. Sends the first N chunks to Claude on Bedrock for entity extraction
  3. Prints per-chunk extraction results so you can eyeball quality
  4. Aggregates entity counts by type across all extracted chunks
  5. Dumps the raw extractions to data/processed/<doc>.extraction.json

Cost guideline (Claude Sonnet on Bedrock, ~3K in / 800 out per chunk):
    5 chunks  ≈ $0.10
    20 chunks ≈ $0.40
   100 chunks ≈ $2.00

This is a one-shot exploration tool. The production pipeline (next step)
will wire this into a batched runner with caching and progress tracking.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from app.config import settings
from ingestion.chunkers.section_chunker import chunk_elements
from ingestion.extractors.llm_extractor import (
    PROMPT_VERSION,
    DocContext,
    LLMExtractor,
)
from ingestion.parsers.pdf_parser import extract_section, parse_pdf

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("inspect_extraction")


# Doc context registry. Stable, hand-curated metadata for each company in
# our corpus. Will move to a config file once we ingest more than a couple
# of documents.
DOC_CONTEXT_REGISTRY: dict[str, DocContext] = {
    "apple": DocContext(company_name="Apple Inc.", ticker="AAPL", fiscal_year=""),
    "microsoft": DocContext(company_name="Microsoft Corporation", ticker="MSFT", fiscal_year=""),
    "alphabet": DocContext(company_name="Alphabet Inc.", ticker="GOOGL", fiscal_year=""),
    "meta": DocContext(company_name="Meta Platforms, Inc.", ticker="META", fiscal_year=""),
    "amazon": DocContext(company_name="Amazon.com, Inc.", ticker="AMZN", fiscal_year=""),
}


def doc_context_from_filename(doc_id: str) -> DocContext:
    """Resolve doc_id like 'apple-10k-fy25' into a DocContext.

    Filename convention: <ticker-or-company>-10k-fy<year>.pdf
    """
    parts = doc_id.split("-")
    company_key = parts[0].lower()
    fy = next((p for p in parts if p.startswith("fy")), "").upper()  # 'FY25'
    template = DOC_CONTEXT_REGISTRY.get(company_key)
    if template is None:
        raise ValueError(
            f"No DocContext registered for company key '{company_key}'. "
            f"Add it to DOC_CONTEXT_REGISTRY in inspect_extraction.py."
        )
    return DocContext(
        company_name=template.company_name,
        ticker=template.ticker,
        fiscal_year=fy or "unknown",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path, help="Path to the 10-K PDF")
    parser.add_argument("--n", type=int, default=5, help="How many chunks to extract")
    args = parser.parse_args()

    pdf_path: Path = args.pdf
    doc_id = pdf_path.stem
    n = args.n

    # Resolve document context for prompt grounding (so 'the Company' -> Apple Inc.)
    try:
        doc_context = doc_context_from_filename(doc_id)
    except ValueError as e:
        log.error("%s", e)
        return 1
    log.info("Doc context: company=%s ticker=%s fy=%s",
             doc_context.company_name, doc_context.ticker, doc_context.fiscal_year)

    # ---- Pipeline up to chunks ----
    elements = parse_pdf(pdf_path)
    risk_elements = extract_section(elements, start_marker="item 1a. risk factors")
    if not risk_elements:
        log.error("Risk Factors section not found.")
        return 1
    chunks = chunk_elements(risk_elements, doc_id=doc_id,
                            chunk_size=settings.chunk_size,
                            chunk_overlap=settings.chunk_overlap)
    log.info("Pipeline produced %d chunks; running extraction on first %d", len(chunks), n)
    target_chunks = chunks[:n]

    # ---- Extraction ----
    extractor = LLMExtractor()
    log.info("Using prompt_version=%s, model=%s", PROMPT_VERSION, settings.bedrock_llm_model_id)

    results = []
    total_entities = 0
    total_rels = 0
    entity_counts_by_type: dict[str, int] = {}

    for i, chunk in enumerate(target_chunks):
        log.info("---- Chunk %d/%d (id=%s, %d chars, pages %s..%s) ----",
                 i + 1, len(target_chunks), chunk.chunk_id, chunk.char_count,
                 chunk.page_start, chunk.page_end)
        t0 = time.time()
        response = extractor.extract(chunk.text, doc_context=doc_context)
        elapsed = time.time() - t0
        log.info("Extracted in %.1fs: %s", elapsed, response.summary())

        # Show every entity inline
        for e in response.entities:
            log.info("  • [%s] %s %s", e.type.value, e.name,
                     dict(e.properties) if e.properties else "")
            entity_counts_by_type[e.type.value] = entity_counts_by_type.get(e.type.value, 0) + 1
        for r in response.relationships:
            log.info("  → %s -[%s]-> %s", r.source, r.predicate.value, r.target)

        total_entities += len(response.entities)
        total_rels += len(response.relationships)

        results.append({
            "chunk_id": chunk.chunk_id,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "char_count": chunk.char_count,
            "elapsed_seconds": round(elapsed, 2),
            "entities": [e.model_dump() for e in response.entities],
            "relationships": [r.model_dump() for r in response.relationships],
        })

    # ---- Aggregate stats ----
    log.info("==== Extraction summary ====")
    log.info("Chunks processed: %d", len(target_chunks))
    log.info("Total entities: %d", total_entities)
    log.info("Total relationships: %d", total_rels)
    log.info("Entities by type:")
    for t, c in sorted(entity_counts_by_type.items(), key=lambda kv: -kv[1]):
        log.info("  %-20s %d", t, c)

    # ---- Dump ----
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{doc_id}.extraction.json"
    out_path.write_text(json.dumps({
        "doc_id": doc_id,
        "doc_context": {
            "company_name": doc_context.company_name,
            "ticker": doc_context.ticker,
            "fiscal_year": doc_context.fiscal_year,
        },
        "prompt_version": PROMPT_VERSION,
        "model_id": settings.bedrock_llm_model_id,
        "chunks_processed": len(target_chunks),
        "total_entities": total_entities,
        "total_relationships": total_rels,
        "entity_counts_by_type": entity_counts_by_type,
        "results": results,
    }, indent=2), encoding="utf-8")
    log.info("Wrote extraction results to %s (%d KB)",
             out_path, out_path.stat().st_size // 1024)
    return 0


if __name__ == "__main__":
    sys.exit(main())
