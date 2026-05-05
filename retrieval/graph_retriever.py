"""
Graph retriever.

Exposes a small library of parameterized graph-traversal patterns. Each
returns a list of "hits" — typed records with the chunk_ids that ground
them. The query planner (next sub-step) decides which pattern applies to
a given user question.

Design notes:
  - All Cypher is parameterized. Never string-interpolate user input.
  - Each method returns GraphHit records, uniformly shaped, so the fuser
    downstream doesn't care which pattern produced them.
  - Patterns lean on the unique constraint on canonical_id and the indexes
    on canonical_name and entity_type for cheap lookups.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from graph.client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class GraphHit:
    """A single graph match, with chunks that ground it.

    `pattern` identifies which retrieval pattern produced this hit (useful
    for explaining/debugging the fusion result).
    `entities` is the (label, name) of the matched entity or entities.
    `chunk_ids` are the source_chunks from the matching edge(s).
    `score` is a coarse relevance signal (counts of matching evidence).
    """

    pattern: str
    entity_label: str
    entity_id: str
    entity_name: str
    chunk_ids: list[str]
    score: float = 1.0
    extras: dict[str, Any] = field(default_factory=dict)


# ---- Cypher patterns -------------------------------------------------------

# Pattern A: risks reported by a company, optionally filtered by category.
_CYPHER_RISKS_FOR_COMPANY = """
    MATCH (c:Company)-[r:REPORTS_RISK]->(risk:RiskFactor)
    WHERE c.canonical_name = $company
      AND ($category IS NULL OR risk.category = $category)
    RETURN risk.canonical_name AS name,
           risk.canonical_id   AS id,
           risk.category       AS category,
           r.source_chunks     AS chunks
"""

# Pattern B: risks tied to a specific geography (any company).
_CYPHER_RISKS_IN_GEOGRAPHY = """
    MATCH (risk:RiskFactor)-[r:RISK_IN]->(g:Geography)
    WHERE g.canonical_name = $geography
    RETURN risk.canonical_name AS name,
           risk.canonical_id   AS id,
           risk.category       AS category,
           r.source_chunks     AS chunks
"""

# Pattern C: competitors of a company. (Useful for multi-hop questions later.)
_CYPHER_COMPETITORS_OF = """
    MATCH (c:Company)-[r:COMPETES_WITH]->(comp)
    WHERE c.canonical_name = $company
    RETURN comp.canonical_name AS name,
           comp.canonical_id   AS id,
           labels(comp)        AS labels,
           r.source_chunks     AS chunks
"""

# Pattern D: SHARED RISKS — risks reported by both `company_a` and any of
# `company_b_list`. Lights up once a corpus has ≥2 companies ingested.
_CYPHER_SHARED_RISKS = """
    MATCH (a:Company {canonical_name: $company_a})-[ra:REPORTS_RISK]->(risk:RiskFactor)
          <-[rb:REPORTS_RISK]-(b:Company)
    WHERE b.canonical_name IN $company_b_list
    RETURN risk.canonical_name AS name,
           risk.canonical_id   AS id,
           risk.category       AS category,
           collect(DISTINCT b.canonical_name) AS shared_with,
           ra.source_chunks + rb.source_chunks AS chunks
"""

# Pattern E: any chunk_id citing any of a set of entity ids. Fallback /
# neighborhood expansion: 'give me all the text grounding these entities'.
_CYPHER_CHUNKS_FOR_ENTITIES = """
    MATCH (e:Entity)-[r]-()
    WHERE e.canonical_id IN $entity_ids
    RETURN e.canonical_id AS id,
           collect(DISTINCT chunk_id) AS chunks
"""


# ---- Retriever -------------------------------------------------------------


class GraphRetriever:
    """Library of parameterized graph-traversal patterns."""

    def __init__(self, client: Neo4jClient | None = None) -> None:
        self._client = client or Neo4jClient()

    def close(self) -> None:
        self._client.close()

    # ---- Pattern A ----

    def risks_for_company(
        self,
        company: str,
        category: str | None = None,
    ) -> list[GraphHit]:
        with self._client.session() as session:
            result = session.run(
                _CYPHER_RISKS_FOR_COMPANY, company=company, category=category,
            )
            hits = [
                GraphHit(
                    pattern="risks_for_company",
                    entity_label="RiskFactor",
                    entity_id=rec["id"],
                    entity_name=rec["name"],
                    chunk_ids=list(rec["chunks"] or []),
                    score=float(len(rec["chunks"] or [])),  # more chunks = more grounded
                    extras={"category": rec["category"]},
                )
                for rec in result
            ]
        logger.info("[graph:risks_for_company] %s, category=%s -> %d hits",
                    company, category, len(hits))
        return hits

    # ---- Pattern B ----

    def risks_in_geography(self, geography: str) -> list[GraphHit]:
        with self._client.session() as session:
            result = session.run(_CYPHER_RISKS_IN_GEOGRAPHY, geography=geography)
            hits = [
                GraphHit(
                    pattern="risks_in_geography",
                    entity_label="RiskFactor",
                    entity_id=rec["id"],
                    entity_name=rec["name"],
                    chunk_ids=list(rec["chunks"] or []),
                    score=float(len(rec["chunks"] or [])),
                    extras={"category": rec["category"], "geography": geography},
                )
                for rec in result
            ]
        logger.info("[graph:risks_in_geography] %s -> %d hits", geography, len(hits))
        return hits

    # ---- Pattern C ----

    def competitors_of(self, company: str) -> list[GraphHit]:
        with self._client.session() as session:
            result = session.run(_CYPHER_COMPETITORS_OF, company=company)
            hits = [
                GraphHit(
                    pattern="competitors_of",
                    entity_label=(rec["labels"] or ["Entity"])[-1],
                    entity_id=rec["id"],
                    entity_name=rec["name"],
                    chunk_ids=list(rec["chunks"] or []),
                    score=float(len(rec["chunks"] or [])),
                )
                for rec in result
            ]
        logger.info("[graph:competitors_of] %s -> %d hits", company, len(hits))
        return hits

    # ---- Pattern D ----

    def shared_risks(
        self,
        company_a: str,
        company_b_list: list[str],
    ) -> list[GraphHit]:
        if not company_b_list:
            return []
        with self._client.session() as session:
            result = session.run(
                _CYPHER_SHARED_RISKS,
                company_a=company_a,
                company_b_list=company_b_list,
            )
            hits = [
                GraphHit(
                    pattern="shared_risks",
                    entity_label="RiskFactor",
                    entity_id=rec["id"],
                    entity_name=rec["name"],
                    chunk_ids=list(rec["chunks"] or []),
                    score=float(len(rec["shared_with"])),  # more sharers = stronger signal
                    extras={
                        "category": rec["category"],
                        "shared_with": list(rec["shared_with"]),
                    },
                )
                for rec in result
            ]
        logger.info("[graph:shared_risks] %s vs %s -> %d hits",
                    company_a, company_b_list, len(hits))
        return hits


# ---- Helper: collect deduplicated chunk_ids from a list of GraphHits -------


def collect_chunk_ids(hits: list[GraphHit]) -> list[str]:
    """Flatten + deduplicate every chunk_id across hits, sorted for determinism."""
    return sorted({cid for h in hits for cid in h.chunk_ids})
