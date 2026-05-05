# graphrag-financial-intelligence

Knowledge-graph-augmented RAG over SEC 10-K filings. Extracts companies, executives, risk factors, subsidiaries, and competitors from financial documents into a Neo4j knowledge graph, and combines graph traversal with hybrid vector search to answer multi-hop questions that standard RAG cannot.

## Status

Under construction. This is Project 2 of a two-project AI engineering portfolio.
Project 1 (multi-agent SMB loan risk assessment, LangGraph + Bedrock): https://smb-risk-agent.netlify.app

## The problem this solves

Standard RAG answers questions by retrieving text that *looks like* the question and asking an LLM to summarize it. That works for single-fact lookups ("what was Apple's FY24 revenue?") but fails on relational questions ("which risks does Apple share with its competitors?") because the relevant text in competitors' 10-Ks doesn't resemble a query about Apple.

This system extracts a structured graph of entities and relationships during ingestion, so the query layer can *plan*: resolve "Apple's competitors" via graph traversal, then fetch the right source chunks for grounding.

## Architecture

Two separable lifecycles:

- **Ingestion** (offline, batch): parse PDF -> chunk -> extract entities + relationships -> dual-write to Neo4j + Pinecone.
- **Query** (online, per-request): plan -> graph walk + vector search -> rerank/fuse -> synthesize grounded answer.

```
PDFs (10-Ks) -> Unstructured.io -> chunker -> LLM extractor -> entity resolver
                                                                    |
                                       +----------------------------+
                                       |                            |
                                       v                            v
                                   Neo4j Aura                   Pinecone
                                (graph + chunk ids)         (vectors + chunk ids)
                                       |                            |
                                       +-------------+--------------+
                                                     v
                                             Hybrid retriever
                                                     v
                                               FastAPI (Render)
                                                     v
                                            React + Vite (Netlify)
```

## Stack

| Layer | Choice | Why |
|---|---|---|
| LLM (extraction + synthesis) | Claude Sonnet on AWS Bedrock | Strong structured output; AWS creds already wired from P1 |
| Embeddings | Amazon Titan Text Embeddings V2 (1024d) | Same provider, no new credentials |
| Graph DB | Neo4j Aura Free | Hosted, 200k nodes / 400k rels is plenty for this corpus |
| Vector DB | Pinecone serverless | Built-in hybrid (dense + sparse); ops-light |
| Document parsing | Unstructured.io | Handles 10-K tables, headers, layout |
| Orchestration | LlamaIndex | Native graph + vector hybrid retrievers |
| Evaluation | Ragas | Faithfulness / answer relevancy / context precision |
| Observability | LangSmith | Same as P1 |
| API | FastAPI on Render | Same pattern as P1 |
| Frontend | React + Vite on Netlify | Simpler than Next.js; static upload + query UI |

## Corpus

Big-tech 10-Ks for last 2 fiscal years: Apple, Microsoft, Alphabet, Meta, Amazon (~10 documents). Same-sector corpus makes `COMPETES_WITH` edges dense and cross-company queries meaningful.

## Evaluation targets (Ragas)

- Faithfulness > 0.85 (answers grounded in sources)
- Answer relevancy > 0.80 (answers address the question)
- Context precision > 0.75 (retrieved chunks are relevant)

## Repository layout

See the `README.md` inside each top-level directory for details.

```
app/            FastAPI service (query-time)
ingestion/      Offline document pipeline (batch)
graph/          Neo4j schema, queries, client
retrieval/      Hybrid retriever (vector + graph + fusion)
evaluation/     Ragas test harness + ground truth
frontend/       React + Vite app (Netlify)
scripts/        Operational CLIs
tests/          Unit + integration tests
docs/           Architecture notes and design decisions
```

## Getting started

Not ready yet. Setup instructions will be added at milestone 1 completion.

## Milestones

- [ ] 1. Scaffold + cloud resources provisioned
- [ ] 2. Ingestion pipeline (parse -> extract -> dual-write)
- [ ] 3. Hybrid retriever (vector + BM25 + graph)
- [ ] 4. Query engine + FastAPI endpoints
- [ ] 5. Ragas evaluation harness
- [ ] 6. Frontend + deploy

## License

MIT
