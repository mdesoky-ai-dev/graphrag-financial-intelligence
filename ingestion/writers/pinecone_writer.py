"""
Pinecone writer.

Takes a list of Chunks, embeds each one with Bedrock Titan, and upserts
them to the configured Pinecone index. The chunk_id is used as the vector
id in Pinecone — the same ID is referenced by Neo4j relationships, which
is what makes the bridge between the graph and the source text work.

Metadata stored alongside each vector:
  - doc_id        e.g. "apple-10k-fy25"
  - page_start
  - page_end
  - char_count
  - text_excerpt  (first 500 chars, for fast preview without re-fetching)

The full text is NOT stored in Pinecone metadata — Pinecone limits metadata
to ~40KB per vector and we want the index light. Full text is reconstructed
from the source chunks JSON when needed for grounding answers.

Usage:
    from ingestion.writers.pinecone_writer import PineconeWriter

    writer = PineconeWriter()
    writer.upsert_chunks(chunks, doc_id="apple-10k-fy25")
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from pinecone import Pinecone

from app.config import settings
from app.embeddings import BedrockEmbedder
from ingestion.chunkers.section_chunker import Chunk

logger = logging.getLogger(__name__)


# Pinecone serverless allows up to 100 vectors per upsert request. Keep the
# batch a bit smaller to leave headroom for metadata size.
DEFAULT_BATCH_SIZE = 50
TEXT_EXCERPT_LIMIT = 500   # chars stored in metadata for fast preview


class PineconeWriter:
    """Wraps a Pinecone Index, batches embeddings + upserts."""

    def __init__(
        self,
        index_name: str | None = None,
        embedder: BedrockEmbedder | None = None,
    ) -> None:
        index_name = index_name or settings.pinecone_index_name
        self._pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
        self._index = self._pc.Index(index_name)
        self._embedder = embedder or BedrockEmbedder()
        logger.info("PineconeWriter ready: index=%s", index_name)

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        doc_id: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> dict[str, int]:
        """Embed and upsert a list of chunks.

        Returns a small stats dict: {embedded: N, upserted: N, batches: B}.
        """
        if not chunks:
            logger.warning("No chunks to upsert.")
            return {"embedded": 0, "upserted": 0, "batches": 0}

        # 1. Embed (sequential — Titan has no native batch endpoint)
        logger.info("Embedding %d chunks via Bedrock Titan ...", len(chunks))
        vectors_payload: list[dict[str, Any]] = []
        for i, c in enumerate(chunks):
            embedding = self._embedder.embed(c.text)
            vectors_payload.append({
                "id": c.chunk_id,
                "values": embedding,
                "metadata": {
                    "doc_id": doc_id,
                    "page_start": c.page_start if c.page_start is not None else -1,
                    "page_end": c.page_end if c.page_end is not None else -1,
                    "char_count": c.char_count,
                    "text_excerpt": c.text[:TEXT_EXCERPT_LIMIT],
                },
            })
            if (i + 1) % 25 == 0:
                logger.info("  embedded %d/%d", i + 1, len(chunks))

        # 2. Upsert in batches
        n_batches = 0
        for i in range(0, len(vectors_payload), batch_size):
            batch = vectors_payload[i : i + batch_size]
            self._index.upsert(vectors=batch)
            n_batches += 1
            logger.info("Upserted batch %d (%d vectors)", n_batches, len(batch))

        logger.info(
            "Upsert complete: %d vectors across %d batches",
            len(vectors_payload), n_batches,
        )
        return {
            "embedded": len(vectors_payload),
            "upserted": len(vectors_payload),
            "batches": n_batches,
        }

    def stats(self) -> dict[str, Any]:
        """Pinecone index stats — total vector count, namespaces, etc."""
        return self._index.describe_index_stats()
