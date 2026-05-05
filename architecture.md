# Architecture

Date: 2026-04-24
Status: Draft, updated as decisions are made.

## Problem statement

Build a question-answering system over SEC 10-K filings that can handle *relational*
queries — questions whose answers require traversing connections between entities
across multiple documents. Examples:

- "Which risks does Apple share with its top three competitors?"
- "How has the CEO's language around supply chain changed over four earnings calls?"
- "What subsidiaries does Apple own and what risks apply to each?"

Standard RAG (cosine similarity over chunk embeddings + LLM synthesis) cannot answer
these. It retrieves text that *looks like* the question, but relational answers live
in the *connections* between documents, not in any single passage.

## Solution shape

Two-brain architecture:

- **Structured brain (Neo4j).** Stores entities and relationships extracted from the
  corpus. Enables deterministic multi-hop traversal.
- **Fuzzy brain (Pinecone).** Stores chunk embeddings and BM25 sparse vectors.
  Provides grounding — the actual source text that the LLM quotes in answers.

At query time, a planner decides per-question whether the answer needs graph
traversal, vector search, or both. Results are fused (reciprocal rank fusion),
and the LLM synthesizes an answer grounded in the retrieved chunks.

## Two lifecycles

The repo is organized around the hard separation between ingestion and query:

| | Ingestion | Query |
|---|---|---|
| Frequency | Occasional (per doc batch) | Constant (per request) |
| Latency budget | Minutes per document | < 3 seconds per query |
| Cost profile | Expensive (LLM per chunk) | Cheap (mostly DB reads) |
| Writes to DBs | Yes | Never |
| Entry point | `ingestion/run.py` CLI | `app/main.py` FastAPI |

Shared code lives in `graph/` and `retrieval/` and is imported by both.

## Component choices

### LLM: Claude Sonnet on AWS Bedrock
Strong structured output under Pydantic schemas, which is critical for reliable
entity extraction. AWS credentials already wired from Project 1.

### Embeddings: Amazon Titan Text Embeddings V2 (1024-dim)
Same provider as the LLM, no new credentials. Sufficient quality for financial
prose. Alternative: Cohere Embed English V3 (could A/B later).

### Graph DB: Neo4j Aura Free
- Hosted, no ops.
- 200k nodes / 400k relationships on the free tier is plenty for ~10 documents.
- Cypher is mature and well-documented.

### Vector DB: Pinecone serverless
- Built-in hybrid search (dense + sparse via `pinecone-text` BM25 encoder).
- Serverless pricing scales to zero for a portfolio project.
- Alternative considered: Amazon OpenSearch. More capable but significantly more
  ops overhead. Not worth it for this scope.

### Document parser: Unstructured.io
Handles 10-K layout quirks (tables, section headers, footnotes) better than
naive PDF text extraction. Finicky to install but the quality gain is real.

### Orchestration: LlamaIndex
Native support for both Neo4j graph stores and Pinecone vector stores, with a
hybrid retriever abstraction. Cleaner fit for this project than LangChain.

## Corpus

Big-tech 10-Ks for the last 2 fiscal years:
- Apple (AAPL)
- Microsoft (MSFT)
- Alphabet (GOOGL)
- Meta (META)
- Amazon (AMZN)

~10 documents total. Same-sector choice is deliberate: `COMPETES_WITH` edges are
dense, and the cross-company queries in the problem statement actually work.

Earnings call transcripts are out of scope for v1 (adds another parser and a
temporal dimension to the graph). May be added later.

## Graph schema (starting point)

Nodes:
- `Company`: name, ticker, fiscal_year
- `Executive`: name, role (CEO/CFO), tenure_start
- `RiskFactor`: canonical_name, category, first_seen_date
- `Subsidiary`: name, jurisdiction
- `Competitor`: name (may also be a Company node if in corpus)
- `FinancialMetric`: name, value, period
- `Chunk`: chunk_id, doc_id, page, text_excerpt  (bridge to Pinecone)

Relationships:
- `(Company)-[:HAS_EXECUTIVE]->(Executive)`
- `(Company)-[:REPORTS_RISK]->(RiskFactor)`
- `(Company)-[:OWNS]->(Subsidiary)`
- `(Company)-[:COMPETES_WITH]->(Company|Competitor)`
- `(Company)-[:REPORTS_METRIC]->(FinancialMetric)`
- `(RiskFactor)-[:MENTIONED_IN]->(Chunk)`  — critical bridge edge

Every extracted relationship also records the source `Chunk` it came from, so any
answer can be cited back to a specific passage in a specific 10-K.

## Entity resolution policy

The hardest problem in the system. Same concept appears with different wording
across documents ("dependence on Asian component suppliers" vs "reliance on
offshore manufacturing"). We need to collapse these to the same node.

Three-layer cascade:

1. **Normalization.** Lowercase, strip stopwords, canonicalize to a slug.
   Cheap, catches easy cases.
2. **Embedding similarity.** Embed the entity name/description. If cosine
   distance to an existing node is >= `ER_MERGE_THRESHOLD` (default 0.87),
   auto-merge. If < `ER_REJECT_THRESHOLD` (default 0.78), auto-reject.
3. **LLM adjudication.** For the gray zone between thresholds, ask the LLM
   "are these the same underlying business concept?" Slow but catches the
   subtle cases.

Thresholds are tunable in `.env` and will be adjusted based on Ragas eval scores.
The failure modes to watch for:
- **Over-merging**: distinct concepts collapsed (loses information).
- **Under-merging**: same concept as two nodes (loses shared-risk queries).

See `docs/entity-resolution-policy.md` for detailed policy and examples.

## Evaluation

Ragas with a hand-curated ground-truth set (~50 Q/A pairs covering single-doc
lookup, cross-doc comparison, and multi-hop relational queries).

Targets:
- Faithfulness > 0.85
- Answer relevancy > 0.80
- Context precision > 0.75

Regression: scores are checkpointed per commit. Any PR that drops any metric
by > 0.03 requires explicit justification.

## Deployment

- API: Render (matches Project 1 pattern)
- Graph DB: Neo4j Aura Free (cloud-hosted)
- Vector DB: Pinecone serverless
- Frontend: Netlify (React + Vite)
- Observability: LangSmith

## Open questions

- Chunking strategy: semantic vs fixed-size vs section-aware? Likely
  section-aware given 10-K structure, but benchmark before committing.
- Query planner: rule-based classification or LLM-based? Start rule-based,
  upgrade if eval shows benefit.
- Reranking: keep RRF simple, or add a cross-encoder reranker? Defer until
  Ragas baseline is established.
