"""
Neo4j client wrapper.

Owns the driver lifecycle. Every other module that needs to talk to Neo4j
goes through this client, so connection settings, retry policy, and pool
tuning live in one place.

Usage:
    from graph.client import Neo4jClient

    client = Neo4jClient()
    with client.session() as session:
        result = session.run("RETURN 1 AS n")
        print(result.single()["n"])
    client.close()

Or as a context manager (auto-closes):
    with Neo4jClient() as client:
        ...
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from neo4j import Driver, GraphDatabase, Session

from app.config import settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Thin wrapper around the official neo4j driver.

    Manages a single long-lived driver instance with built-in connection pooling.
    Sessions are short-lived and acquired per-operation via `session()`.
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        self._uri = uri or settings.neo4j_uri
        self._user = user or settings.neo4j_user
        self._password = password or settings.neo4j_password.get_secret_value()
        self._database = database or settings.neo4j_database

        logger.info("Connecting to Neo4j at %s as %s", self._uri, self._user)
        self._driver: Driver = GraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password),
            # Aura free tier disconnects idle sessions; tune liveness checks.
            max_connection_lifetime=30 * 60,  # 30 minutes
            max_connection_pool_size=10,
            connection_acquisition_timeout=30,
        )

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Context manager yielding a Neo4j session bound to the configured database."""
        session = self._driver.session(database=self._database)
        try:
            yield session
        finally:
            session.close()

    def verify_connectivity(self) -> None:
        """Raise if the driver cannot reach the server. Useful for startup checks."""
        self._driver.verify_connectivity()

    def close(self) -> None:
        """Close the driver. Idempotent."""
        if self._driver is not None:
            self._driver.close()
            logger.info("Neo4j driver closed")

    def __enter__(self) -> "Neo4jClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
