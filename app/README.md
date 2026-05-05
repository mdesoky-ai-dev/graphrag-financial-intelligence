# app/

FastAPI service. Query-time only — reads Neo4j and Pinecone, never writes.

- `main.py` — app factory, middleware, router registration
- `dependencies.py` — Neo4j client, Pinecone client, LLM as FastAPI deps
- `routes/` — endpoint handlers (one file per resource: `query.py`, `health.py`, `documents.py`)
- `schemas/` — Pydantic request/response models

Same pattern as Project 1.
