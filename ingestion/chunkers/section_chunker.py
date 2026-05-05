"""
Section chunker for 10-K text.

Two responsibilities:
  1. Clean parsed PDF elements (strip page footers, fix glued words).
  2. Chunk the cleaned text into ~1000-char retrievable units with overlap,
     respecting sentence boundaries and tracking page provenance.

The output is a list of Chunk records. Each chunk has a stable ID that
serves as the bridge key between Neo4j (graph) and Pinecone (vectors).

Usage:
    from ingestion.chunkers.section_chunker import chunk_elements

    chunks = chunk_elements(
        elements=risk_elements,
        doc_id="apple-10k-fy25",
        chunk_size=1024,
        chunk_overlap=200,
    )
    for c in chunks[:3]:
        print(c.chunk_id, c.page_start, c.char_count, c.text[:80])
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Cleaning regexes
# ----------------------------------------------------------------------------

# Apple page footer: "Apple Inc. | 2025 Form 10-K | 6"
# Generalized to handle any 4-digit year and any company prefix that follows
# the same "<Co> | YYYY Form 10-K | N" pattern.
_PAGE_FOOTER = re.compile(
    r"\s*[A-Z][A-Za-z.\s]+(?:Inc|Corp|Corporation|Co|Company)\.?\s*\|\s*\d{4}\s+Form\s+10-K\s*\|\s*\d+\s*"
)

# lowercase letter immediately followed by uppercase letter — the classic
# line-wrap glue artifact ('Companymust' -> 'Company must').
# Side effect: splits CamelCase product names ('iPhone' -> 'i Phone').
# Acceptable trade-off for embedding quality; can be whitelisted later.
_GLUED_CASE_BOUNDARY = re.compile(r"([a-z])([A-Z])")

# Sentence-ending punctuation followed directly by an uppercase letter:
# 'services.The Company' -> 'services. The Company'
_GLUED_SENTENCE = re.compile(r"([.!?])([A-Z])")

# Repeated whitespace -> single space
_WHITESPACE = re.compile(r"\s+")


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------


@dataclass
class Chunk:
    """A single retrievable unit of text with provenance.

    chunk_id is the bridge key between Neo4j and Pinecone: the same id is
    used as the Pinecone vector id and as a property on the Neo4j Chunk node.
    """

    chunk_id: str
    text: str
    doc_id: str
    page_start: int | None
    page_end: int | None
    char_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Apply all cleaning rules to a single piece of text.

    Order matters:
      1. Strip page footers FIRST (they often sit between glued sentences).
      2. Insert spaces at glued-sentence boundaries ('.X' -> '. X').
      3. Insert spaces at lowercase-uppercase boundaries ('aB' -> 'a B').
      4. Collapse repeated whitespace.
    """
    if not text:
        return ""
    text = _PAGE_FOOTER.sub(" ", text)
    text = _GLUED_SENTENCE.sub(r"\1 \2", text)
    text = _GLUED_CASE_BOUNDARY.sub(r"\1 \2", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def clean_elements(
    elements: list[dict[str, Any]],
    min_chars: int = 100,
) -> list[dict[str, Any]]:
    """Clean each element's text and drop any that fall below min_chars.

    Most sub-100-char elements are page-footer fragments left behind after
    stripping. Real risk-factor paragraphs are always longer.
    """
    cleaned = []
    dropped = 0
    for el in elements:
        new_text = clean_text(el.get("text") or "")
        if len(new_text) < min_chars:
            dropped += 1
            continue
        cleaned.append({**el, "text": new_text})
    logger.info("Cleaning: kept %d, dropped %d (under %d chars)", len(cleaned), dropped, min_chars)
    return cleaned


# ----------------------------------------------------------------------------
# Sentence splitting
# ----------------------------------------------------------------------------

# Split on sentence endings — a period/exclamation/question mark followed by
# whitespace and an uppercase letter. Naive but robust for prose.
# Note: this is intentionally simple. We keep the punctuation with the
# preceding sentence by using a lookbehind-style regex.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences. Keeps trailing punctuation."""
    sentences = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in sentences if s.strip()]


# ----------------------------------------------------------------------------
# Chunking
# ----------------------------------------------------------------------------


def chunk_elements(
    elements: list[dict[str, Any]],
    doc_id: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Clean and chunk a list of parsed elements into retrievable units.

    Strategy:
      1. Clean every element's text.
      2. Concatenate the cleaned text in order, preserving (offset, page)
         markers so we can recover which page each chunk came from.
      3. Walk the concatenated text, packing sentences into chunks until we
         exceed chunk_size. Add chunk_overlap chars of the previous chunk's
         tail to the start of each new chunk.

    Args:
        elements: Output of parse_pdf -> extract_section.
        doc_id: stable identifier for this document, used in chunk IDs.
        chunk_size: target maximum characters per chunk.
        chunk_overlap: characters of trailing context carried into the next
            chunk to preserve continuity for embeddings and retrieval.

    Returns:
        List of Chunk dataclasses, one per chunk, in document order.
    """
    cleaned = clean_elements(elements)
    if not cleaned:
        logger.warning("No elements survived cleaning; nothing to chunk.")
        return []

    # Build a single text stream and a list of (offset, page) markers so we
    # can derive page_start / page_end for each chunk later.
    parts: list[str] = []
    page_markers: list[tuple[int, int | None]] = []  # (char_offset, page)
    cursor = 0
    for el in cleaned:
        page_markers.append((cursor, el.get("page")))
        parts.append(el["text"])
        cursor += len(el["text"]) + 1  # +1 for the space we'll join with

    full_text = " ".join(parts)
    logger.info("Concatenated text: %d chars from %d elements", len(full_text), len(cleaned))

    def page_for_offset(offset: int) -> int | None:
        """Find the page of the element that contains this character offset."""
        page = None
        for marker_offset, marker_page in page_markers:
            if marker_offset > offset:
                break
            page = marker_page
        return page

    # Sentence-aware chunking: greedily pack sentences into chunks.
    sentences = split_sentences(full_text)
    chunks: list[Chunk] = []
    current_text = ""
    current_start_offset = 0  # char offset of where current chunk starts in full_text

    # We need to know each sentence's offset in full_text. Recompute by walking.
    sentence_offsets: list[int] = []
    cursor = 0
    for s in sentences:
        idx = full_text.find(s, cursor)
        if idx == -1:
            idx = cursor  # fallback (shouldn't happen with our split)
        sentence_offsets.append(idx)
        cursor = idx + len(s)

    def emit_chunk(text: str, start_offset: int) -> None:
        """Build and append a Chunk for the given text + start offset."""
        end_offset = start_offset + len(text)
        page_start = page_for_offset(start_offset)
        page_end = page_for_offset(end_offset - 1)
        chunk_id = f"{doc_id}_chunk_{len(chunks):04d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=text.strip(),
                doc_id=doc_id,
                page_start=page_start,
                page_end=page_end,
                char_count=len(text.strip()),
            )
        )

    for sentence, offset in zip(sentences, sentence_offsets, strict=True):
        # If this sentence alone exceeds chunk_size, hard-split it. Rare
        # for prose but possible for tables or run-on legal sentences.
        if len(sentence) > chunk_size:
            if current_text:
                emit_chunk(current_text, current_start_offset)
                current_text = ""
            for i in range(0, len(sentence), chunk_size - chunk_overlap):
                emit_chunk(sentence[i : i + chunk_size], offset + i)
            current_start_offset = offset + len(sentence)
            continue

        # Would adding this sentence exceed chunk_size?
        candidate = (current_text + " " + sentence).strip() if current_text else sentence
        if len(candidate) > chunk_size and current_text:
            # Emit the current chunk and start a new one with overlap.
            emit_chunk(current_text, current_start_offset)
            # Overlap: take the last `chunk_overlap` chars from the chunk we
            # just emitted, then prepend to the new chunk.
            tail = current_text[-chunk_overlap:] if chunk_overlap > 0 else ""
            current_text = (tail + " " + sentence).strip()
            current_start_offset = offset - len(tail) - 1 if tail else offset
        else:
            if not current_text:
                current_start_offset = offset
            current_text = candidate

    if current_text:
        emit_chunk(current_text, current_start_offset)

    logger.info("Produced %d chunks (target size=%d, overlap=%d)", len(chunks), chunk_size, chunk_overlap)
    return chunks
