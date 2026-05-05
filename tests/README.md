# tests/

- `unit/` — pure logic tests (entity resolver, query planner, schema constants). Fast.
- `integration/` — spin up local Neo4j container, run real queries. Slow, marked `@pytest.mark.integration`.

Run:
    pytest                              # unit only (default)
    pytest -m integration               # integration only
    pytest -m "not integration"         # explicit unit
