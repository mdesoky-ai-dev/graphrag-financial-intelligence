# retrieval/

Query-time hybrid retriever. Shared by `app/`.

- `query_planner.py` — decides per-query which retrieval strategies to run
- `vector.py` — Pinecone dense + sparse (BM25) search
- `graph_retriever.py` — graph walks over Neo4j
- `hybrid.py` — runs strategies in parallel and fuses results (reciprocal rank fusion)
- `synthesizer.py` — builds the final prompt from retrieved chunks + calls LLM
