# ingestion/

Offline document pipeline. Batch, slow, expensive. Runs as a CLI, not as part of the API.

    python -m ingestion.run --docs ./data/10ks/

Subdirectories:
- `parsers/` — Unstructured.io wrappers, one per document type
- `chunkers/` — slice parsed docs into retrievable units (section-aware for 10-Ks)
- `extractors/` — LLM calls that pull entities and relationships from chunks
- `resolvers/` — entity resolution (normalization + embedding + LLM adjudication)
- `writers/` — dual-write to Neo4j and Pinecone with shared chunk IDs
