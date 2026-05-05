"""
Neo4j writer.

Takes the resolved.json (canonical entities + resolved relationships) and
writes them to Neo4j as nodes and edges. Two passes:

  Pass 1: write all canonical entities as nodes
            (so edges can find their endpoints)

  Pass 2: write all resolved relationships as edges

All operations are MERGE-based and idempotent: running the writer twice
for the same document produces the same graph, never duplicates.

Usage:
    from ingestion.writers.neo4j_writer import Neo4jWriter

    writer = Neo4jWriter()
    writer.bootstrap_schema()                       # once per instance
    writer.write_entities(canonical_entities, doc_id)
    writer.write_relationships(resolved_relationships)
    print(writer.summary())
"""

from __future__ import annotations

import logging
from typing import Any

from graph.client import Neo4jClient
from graph.queries import (
    CONSTRAINTS,
    COUNT_NODES_BY_TYPE,
    COUNT_RELS_BY_TYPE,
    INDEXES,
    upsert_entity,
    upsert_relationship,
)
from graph.schema import ENTITY_TYPE_TO_LABEL, RelType

logger = logging.getLogger(__name__)


# Property keys that are stored as TOP-LEVEL node properties (not inside the
# free-form `properties` map). Type-specific extras like 'ticker', 'category',
# 'kind' get flattened up to the node so they're queryable directly.
TYPE_SPECIFIC_PROP_KEYS = {"ticker", "role", "category", "kind", "value", "period", "jurisdiction"}


class Neo4jWriter:
    """Engine: bootstraps schema and writes resolved entities/relationships."""

    def __init__(self, client: Neo4jClient | None = None) -> None:
        self._client = client or Neo4jClient()
        self._stats: dict[str, int] = {
            "constraints_created": 0,
            "indexes_created": 0,
            "nodes_written": 0,
            "edges_written": 0,
            "edges_skipped": 0,
        }

    # ------------------------------------------------------------------
    # Schema setup — run once per instance
    # ------------------------------------------------------------------

    def bootstrap_schema(self) -> None:
        """Create constraints and indexes. Idempotent."""
        with self._client.session() as session:
            for cypher in CONSTRAINTS:
                session.run(cypher)
                self._stats["constraints_created"] += 1
                logger.info("Constraint applied: %s", cypher.split("FOR")[0].strip())
            for cypher in INDEXES:
                session.run(cypher)
                self._stats["indexes_created"] += 1
                logger.info("Index applied: %s", cypher.split("FOR")[0].strip())

    # ------------------------------------------------------------------
    # Pass 1: write nodes
    # ------------------------------------------------------------------

    def write_entities(
        self,
        canonical_entities: list[dict[str, Any]],
        doc_id: str,
    ) -> None:
        """Upsert each canonical entity as a node.

        canonical_entities is the list straight from resolved.json — each
        item has canonical_id, canonical_name, type, properties, aliases,
        source_chunks.
        """
        logger.info("Writing %d entities to Neo4j ...", len(canonical_entities))
        with self._client.session() as session:
            for ent in canonical_entities:
                node_label = ENTITY_TYPE_TO_LABEL.get(ent["type"])
                if node_label is None:
                    logger.warning("Unknown entity type %r, skipping %s",
                                   ent["type"], ent["canonical_id"])
                    continue

                # Flatten type-specific extras out of `properties` into a dict
                # we'll pass as the $properties parameter. Anything not in our
                # known set still gets stored — Neo4j accepts ad-hoc keys.
                props_map = {k: v for k, v in ent.get("properties", {}).items() if v}

                session.run(
                    upsert_entity(node_label.value),
                    canonical_id=ent["canonical_id"],
                    canonical_name=ent["canonical_name"],
                    entity_type=ent["type"],
                    properties=props_map,
                    aliases=sorted(ent.get("aliases", [])),
                    source_chunks=sorted(ent.get("source_chunks", [])),
                    doc_id=doc_id,
                )
                self._stats["nodes_written"] += 1
                if self._stats["nodes_written"] % 50 == 0:
                    logger.info("  wrote %d/%d nodes",
                                self._stats["nodes_written"], len(canonical_entities))
        logger.info("Wrote %d nodes total.", self._stats["nodes_written"])

    # ------------------------------------------------------------------
    # Pass 2: write edges
    # ------------------------------------------------------------------

    def write_relationships(
        self,
        resolved_relationships: list[dict[str, Any]],
    ) -> None:
        """Upsert each resolved relationship as an edge.

        Skips any relationship whose predicate isn't in our schema (defensive)
        or whose endpoints don't exist in the graph (shouldn't happen if
        Pass 1 ran first).
        """
        logger.info("Writing %d edges to Neo4j ...", len(resolved_relationships))
        valid_predicates = {r.value for r in RelType}

        with self._client.session() as session:
            for rel in resolved_relationships:
                pred = rel["predicate"]
                if pred not in valid_predicates:
                    logger.warning("Unknown predicate %r, skipping edge.", pred)
                    self._stats["edges_skipped"] += 1
                    continue

                result = session.run(
                    upsert_relationship(pred),
                    source_id=rel["source_id"],
                    target_id=rel["target_id"],
                    source_chunks=sorted(rel.get("source_chunks", [])),
                    predicate=pred,
                )
                if result.single() is None:
                    # Endpoint missing — shouldn't happen if Pass 1 succeeded.
                    logger.warning(
                        "Edge endpoints not found: %s -[%s]-> %s",
                        rel["source_id"], pred, rel["target_id"],
                    )
                    self._stats["edges_skipped"] += 1
                    continue
                self._stats["edges_written"] += 1
                if self._stats["edges_written"] % 50 == 0:
                    logger.info("  wrote %d/%d edges",
                                self._stats["edges_written"], len(resolved_relationships))
        logger.info("Wrote %d edges total. Skipped: %d",
                    self._stats["edges_written"], self._stats["edges_skipped"])

    # ------------------------------------------------------------------
    # Validation queries
    # ------------------------------------------------------------------

    def graph_counts(self) -> dict[str, dict[str, int]]:
        """Read back node and edge counts grouped by type. For post-write check."""
        with self._client.session() as session:
            nodes = {r["type"]: r["n"] for r in session.run(COUNT_NODES_BY_TYPE)}
            rels = {r["predicate"]: r["n"] for r in session.run(COUNT_RELS_BY_TYPE)}
        return {"nodes_by_type": nodes, "relationships_by_predicate": rels}

    def summary(self) -> dict[str, int]:
        return dict(self._stats)

    def close(self) -> None:
        self._client.close()
