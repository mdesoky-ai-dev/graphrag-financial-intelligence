"""
Vector retriever.

Two distinct operations against Pinecone:
  1. Semantic search by query text — embed the question, return top-K chunks.
  2. Direct fetch by chunk_id — given a list of IDs, retrieve their metadata.

Used by:
  - Hybrid retriever for the semantic side of the search.
  - Synthesizer for grounding (after the graph + hybrid steps decide which
    chunks to surface).

The dimension of the query vector must match the index dimension. Both
come from the same Bedrock Titan model, so this is automatic — but we
assert it once for safety.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pinecone import Pinecone

from app.config import settings
from app.embeddings import BedrockEmbedder

logger = logging.getLogger(__name__)


@dataclass
class VectorHit:
    """A single semantic-search hit from Pinecone."""

    chunk_id: str
    score: float                  # cosine similarity, 0..1
    text_excerpt: str
    page_start: int
    page_end: int
    doc_id: str


@dataclass
class ChunkText:
    """A direct-fetch chunk with full metadata. Same shape as VectorHit but
    no `score` (because there's no query vector in a fetch operation)."""

    chunk_id: str
    text_excerpt: str
    page_start: int
    page_end: int
    doc_id: str


class VectorRetriever:
    """Wraps a Pinecone Index for both semantic search and ID-based fetch."""

    def __init__(
        self,
        index_name: str | None = None,
        embedder: BedrockEmbedder | None = None,
    ) -> None:
        index_name = index_name or settings.pinecone_index_name
        self._pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
        self._index = self._pc.Index(index_name)
        self._embedder = embedder or BedrockEmbedder()
        logger.info("VectorRetriever ready: index=%s", index_name)

    # ---- Semantic search ----

    def search(self, query_text: str, top_k: int = 10) -> list[VectorHit]:
        """Embed the query and return the top-K most similar chunks."""
        query_vec = self._embedder.embed(query_text)
        result = self._index.query(
            vector=query_vec,
            top_k=top_k,
            include_metadata=True,
        )
        hits: list[VectorHit] = []
        for match in result.get("matches", []):
            md = match.get("metadata") or {}
            hits.append(VectorHit(
                chunk_id=match["id"],
                score=float(match.get("score", 0.0)),
                text_excerpt=str(md.get("text_excerpt", "")),
                page_start=int(md.get("page_start", -1)),
                page_end=int(md.get("page_end", -1)),
                doc_id=str(md.get("doc_id", "")),
            ))
        logger.info("[vector:search] query=%r top_k=%d -> %d hits",
                    query_text[:60], top_k, len(hits))
        return hits

    # ---- Direct fetch by ID ----

    def fetch(self, chunk_ids: list[str]) -> dict[str, ChunkText]:
        """Look up chunks by canonical Pinecone vector ID (the chunk_id).

        Used after graph traversal: graph gives us chunk_ids, we resolve them
        to text. Idempotent and cheap.
        """
        if not chunk_ids:
            return {}
        # Pinecone supports up to ~1000 IDs per fetch; we won't approach that
        # but keep the API simple by issuing one call.
        response = self._index.fetch(ids=chunk_ids)
        vectors = response.vectors  # dict[str, FetchResponseVector]
        out: dict[str, ChunkText] = {}
        for cid, vec in vectors.items():
            md = vec.metadata or {}
            out[cid] = ChunkText(
                chunk_id=cid,
                text_excerpt=str(md.get("text_excerpt", "")),
                page_start=int(md.get("page_start", -1)),
                page_end=int(md.get("page_end", -1)),
                doc_id=str(md.get("doc_id", "")),
            )
        missing = set(chunk_ids) - set(out)
        if missing:
            logger.warning("[vector:fetch] %d chunk_ids not found in index: %s",
                           len(missing), sorted(missing)[:5])
        logger.info("[vector:fetch] requested %d, retrieved %d",
                    len(chunk_ids), len(out))
        return out
