"""
Index a resolved.json into Neo4j.

Run:
    python -m scripts.index_neo4j data/processed/apple-10k-fy25.resolved.json

What it does:
  1. Loads the resolved JSON (244 canonical entities + 293 edges)
  2. Bootstraps Neo4j schema (constraints + indexes) — idempotent
  3. Pass 1: writes every canonical entity as a node
  4. Pass 2: writes every resolved relationship as an edge
  5. Reads back counts to confirm what landed

Idempotent: re-running for the same document does not duplicate nodes
or edges. MERGE-based throughout.

After this completes, the graph is queryable at https://console.neo4j.io
under your instance's "Query" view.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.config import settings
from ingestion.writers.neo4j_writer import Neo4jWriter

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("index_neo4j")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("resolved", type=Path,
                        help="Path to resolved.json (output of resolve_entities.py)")
    args = parser.parse_args()

    if not args.resolved.exists():
        log.error("File not found: %s", args.resolved)
        return 1

    with args.resolved.open() as f:
        data = json.load(f)
    log.info("Loaded %s: %d canonical entities, %d resolved relationships",
             args.resolved.name,
             len(data["canonical_entities"]),
             len(data["resolved_relationships"]))

    writer = Neo4jWriter()
    try:
        writer.bootstrap_schema()
        writer.write_entities(data["canonical_entities"], doc_id=data["doc_id"])
        writer.write_relationships(data["resolved_relationships"])

        # Read back counts so we see what landed
        counts = writer.graph_counts()
        log.info("==== Graph contents after write ====")
        log.info("Nodes by type:")
        for t, n in counts["nodes_by_type"].items():
            log.info("  %-20s %d", t, n)
        log.info("Edges by predicate:")
        for p, n in counts["relationships_by_predicate"].items():
            log.info("  %-20s %d", p, n)

        log.info("==== Writer summary ====")
        for k, v in writer.summary().items():
            log.info("  %-20s %d", k, v)
    finally:
        writer.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
