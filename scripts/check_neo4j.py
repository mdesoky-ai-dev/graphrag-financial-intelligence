"""
Sanity-check script: prove the Python -> Neo4j path works end-to-end.

Run:
    python -m scripts.check_neo4j

What it does:
  1. Loads .env via app.config (proves config plumbing works)
  2. Opens a Neo4j driver via graph.client (proves credentials work)
  3. Runs a trivial Cypher query (proves we can read from Aura)
  4. Reports node and relationship counts (proves we're hitting our own DB)

Exit code 0 on success, 1 on any failure.
"""

from __future__ import annotations

import logging
import sys

from app.config import settings
from graph.client import Neo4jClient

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("check_neo4j")


def main() -> int:
    log.info("env=%s, neo4j_uri=%s, db=%s", settings.env, settings.neo4j_uri, settings.neo4j_database)

    try:
        client = Neo4jClient()
    except Exception as e:
        log.error("Failed to construct Neo4j driver: %s", e)
        return 1

    try:
        client.verify_connectivity()
        log.info("Driver connectivity verified.")

        with client.session() as session:
            greeting = session.run('RETURN "aura is alive (from python)" AS msg').single()
            assert greeting is not None
            log.info("Greeting: %s", greeting["msg"])

            counts = session.run(
                """
                MATCH (n) WITH count(n) AS nodes
                OPTIONAL MATCH ()-[r]->() RETURN nodes, count(r) AS rels
                """
            ).single()
            assert counts is not None
            log.info("Current graph contents: %d nodes, %d relationships", counts["nodes"], counts["rels"])

    except Exception as e:
        log.exception("Neo4j check failed: %s", e)
        return 1
    finally:
        client.close()

    log.info("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
