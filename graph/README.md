# graph/

Everything Neo4j-specific. Shared by `app/` (query-time reads) and `ingestion/` (batch writes).

- `schema.py` — node types, relationship types, properties as constants
- `client.py` — thin Neo4j driver wrapper with connection pooling
- `queries.py` — parameterized Cypher queries, one function per query pattern
- `bootstrap.py` — create indexes and uniqueness constraints (run once at setup)
