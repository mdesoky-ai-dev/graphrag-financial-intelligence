"""
Hybrid retriever.

Combines the two retrieval signals — semantic (vector) and structural
(graph) — into a single ranked list of chunk_ids, using Reciprocal Rank
Fusion (RRF). The output drives the synthesizer's grounding pass.

Why RRF:
  - Graph and vector retrievers produce scores on incompatible scales
    (cosine similarity vs. cite-count), so summing raw scores is meaningless.
  - RRF discards the scores and keeps only the *ranks* — strictly comparable
    across retrievers.
  - It's parameter-free in practice (default k=60 from the original 2009
    paper has been re-validated repeatedly since).

Reference:
  Cormack, Clarke, Buettcher (2009), "Reciprocal rank fusion outperforms
  Condorcet and individual rank learning methods."

Formula:
  rrf(chunk) = sum over retrievers r of:    1 / (k + rank_in_r(chunk))
  k=60, rank starts at 1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from retrieval.graph_retriever import GraphHit, collect_chunk_ids
from retrieval.vector_retriever import VectorHit

logger = logging.getLogger(__name__)


# Default smoothing constant. k=60 is the standard from the original RRF
# paper. Higher k -> ranks matter less, fusion is more permissive. Lower k
# -> top ranks matter much more. 60 is the de-facto default.
DEFAULT_RRF_K = 60


@dataclass
class FusedHit:
    """A chunk_id with its fused score and the retrievers that surfaced it."""

    chunk_id: str
    rrf_score: float
    sources: list[str]            # which retrievers contributed (e.g. ['graph', 'vector'])
    graph_rank: int | None = None
    vector_rank: int | None = None


def _rank_chunks(hits: Iterable, get_chunk_ids) -> dict[str, int]:
    """Walk hits in their incoming order and produce {chunk_id: rank}.

    Each chunk_id gets its FIRST observed rank — if a chunk_id appears in
    multiple hits, only the first (best) appearance counts. This keeps
    fusion intuitive: 'rank 1 is the best mention.'
    """
    ranks: dict[str, int] = {}
    cursor = 1
    for hit in hits:
        for cid in get_chunk_ids(hit):
            if cid not in ranks:
                ranks[cid] = cursor
                cursor += 1
    return ranks


def fuse(
    graph_hits: list[GraphHit],
    vector_hits: list[VectorHit],
    k: int = DEFAULT_RRF_K,
    top_n: int | None = None,
) -> list[FusedHit]:
    """Reciprocal Rank Fusion of graph + vector results.

    Args:
        graph_hits:   from any GraphRetriever method (in retrieval order)
        vector_hits:  from VectorRetriever.search (already in score order)
        k:            RRF smoothing constant (default 60)
        top_n:        optionally truncate the fused list to this many results

    Returns:
        FusedHits sorted by rrf_score desc.
    """
    # Build per-retriever rank tables, keyed by chunk_id.
    graph_ranks = _rank_chunks(graph_hits, lambda h: h.chunk_ids)
    vector_ranks = _rank_chunks(vector_hits, lambda h: [h.chunk_id])

    all_chunk_ids = set(graph_ranks) | set(vector_ranks)
    if not all_chunk_ids:
        return []

    fused: list[FusedHit] = []
    for cid in all_chunk_ids:
        score = 0.0
        sources: list[str] = []
        g_rank = graph_ranks.get(cid)
        v_rank = vector_ranks.get(cid)

        if g_rank is not None:
            score += 1.0 / (k + g_rank)
            sources.append("graph")
        if v_rank is not None:
            score += 1.0 / (k + v_rank)
            sources.append("vector")

        fused.append(FusedHit(
            chunk_id=cid,
            rrf_score=score,
            sources=sources,
            graph_rank=g_rank,
            vector_rank=v_rank,
        ))

    fused.sort(key=lambda h: h.rrf_score, reverse=True)
    if top_n is not None:
        fused = fused[:top_n]

    logger.info(
        "[hybrid:fuse] graph=%d hits (%d unique chunks), "
        "vector=%d hits (%d unique chunks), fused=%d (k=%d)",
        len(graph_hits), len(graph_ranks),
        len(vector_hits), len(vector_ranks),
        len(fused), k,
    )
    return fused


def fused_chunk_ids(fused: list[FusedHit]) -> list[str]:
    """Convenience: get the chunk_ids out, in fused-rank order."""
    return [h.chunk_id for h in fused]
