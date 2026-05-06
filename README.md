# GraphRAG Financial Intelligence

> **Hybrid graph + vector RAG over SEC 10-K filings.** A knowledge graph (Neo4j) captures named entities and relationships from the filing; dense embeddings (Pinecone) capture semantic content. Retrieval fuses both signals via reciprocal rank fusion. Claude Sonnet 4.6 synthesizes grounded, citation-rich answers.

**🌐 Live demo:** [graphrag-financial-intelligence.netlify.app](https://graphrag-financial-intelligence.netlify.app)
**📂 Source:** [github.com/mdesoky-ai-dev/graphrag-financial-intelligence](https://github.com/mdesoky-ai-dev/graphrag-financial-intelligence)
**📊 Evaluation:** Faithfulness **0.957** · Answer Relevancy **0.824** · 10 ground-truth questions · Ragas with Claude judge

---

## What it does

Ask natural-language questions about Apple's 10-K filing and get answers grounded in the document, with every claim citing the specific paragraph that supports it. Click any citation pill in the answer to jump to its source chunk in the diagnostics panel — full transparency between claim and evidence.

```
"What risks does Apple face related to China?"
   │
   ▼
Query planner detects geography=China → graph pattern: risks_in_geography
   │
   ├─► Graph retriever: 3 RiskFactor nodes connected to (:Geography {name:'China'}) via RISK_IN
   ├─► Vector retriever: 10 semantically-similar chunks via Pinecone (Titan embeddings)
   │
   ▼ Reciprocal Rank Fusion (k=60)
   │
13 fused chunks → Claude Sonnet 4.6 synthesizes answer with [chunk_NNNN] citations
```

The interesting bit isn't the LLM — it's the **two-brain retrieval**. Graphs answer "show me all risks tied to China" precisely (structured query). Vectors answer "what's semantically about supply chain disruption" robustly (semantic match). Most questions need both.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| LLM | Claude Sonnet 4.6 (AWS Bedrock) | Strong reasoning + structured output; one provider for LLM + embeddings |
| Embeddings | Amazon Titan v2 (1024-dim, normalized) | Same Bedrock account; matches the LLM stack |
| Knowledge graph | Neo4j Aura (free tier) | Industry standard for property graphs; Cypher is expressive |
| Vector DB | Pinecone serverless | Managed, fast, cosine similarity, free tier |
| Backend | FastAPI + uvicorn | Type-safe, async, automatic OpenAPI docs |
| Frontend | React + Vite + TypeScript + Tailwind | Modern stack; fast dev loop; tiny bundles |
| Evaluation | Ragas 0.2.x with Bedrock judge | Industry-standard RAG metrics |
| Observability | LangSmith (`@traceable` decorators) | Hierarchical trace of plan → retrieve → fuse → synthesize |
| Backend host | Render (free tier) | Auto-deploy on git push; one-click env vars |
| Frontend host | Netlify | Same model; both auto-deploy from `main` |

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │  React frontend (Netlify)                     │
                    │  - QueryBox, AnswerPanel, DiagnosticsPanel    │
                    └────────────────────┬──────────────────────────┘
                                         │ POST /ask
                                         ▼
                    ┌──────────────────────────────────────────────┐
                    │  FastAPI backend (Render)                     │
                    │  Lifespan-managed Synthesizer                 │
                    └──┬─────────────────┬─────────────────┬────────┘
                       │                 │                 │
                       ▼                 ▼                 ▼
              ┌──────────────┐  ┌────────────────┐  ┌──────────────┐
              │ Query planner│  │  Graph (Neo4j) │  │Vector(Pine-  │
              │  — picks     │  │  244 entities  │  │ cone, 1024d) │
              │  pattern +   │  │  293 edges     │  │ 102 chunks   │
              │  params      │  │  7 entity types│  │              │
              └──────┬───────┘  └────────┬───────┘  └──────┬───────┘
                     │                   │                  │
                     │      ┌────────────┴──────────────────┘
                     │      │
                     ▼      ▼
              ┌──────────────────────────────────┐
              │ Reciprocal Rank Fusion (k=60)    │
              │ Merges graph + vector hits       │
              └────────────┬─────────────────────┘
                           │
                           ▼
              ┌──────────────────────────────────┐
              │ Claude Sonnet 4.6 synthesizer    │
              │ Strict citation grounding        │
              │ Returns: answer + diagnostics    │
              └──────────────────────────────────┘
```

**Data flow at ingestion time** (one-time, before the API runs):

1. **Parse** PDF → 600 elements via `unstructured`
2. **Chunk** Item 1A (Risk Factors) → 17 risk chunks; rest → 85 narrative chunks → **102 total**
3. **Extract** entities + relationships from each chunk → Claude Sonnet 4.6 with structured-output prompt → 427 raw entities, 362 raw relationships
4. **Resolve** entities (3-layer cascade: slug match → embedding similarity → LLM adjudication for the murky middle) → 244 canonical entities
5. **Write** to Neo4j (`:Entity:<Type>` double label, canonical_id constraint, `source_chunks` on edges) and Pinecone (1024-dim Titan vectors, cosine similarity)

---

## Evaluation

The system is evaluated on **10 ground-truth questions** (mix of easy/medium/hard) using **Ragas** with **Claude Sonnet 4.6 as the judge** and **Titan v2 for embedding similarity**.

| Metric | Score | Interpretation |
|---|---|---|
| **Faithfulness** | **0.957** ✅ | Of the claims in answers, ~96% are supported by the retrieved context. Hallucinations are rare. |
| **Answer Relevancy** | **0.824** ✅ | Generated answers stay on-topic to the question. |
| **Context Precision** | **0.462** ⚠️ | Of the retrieved chunks, ~46% are directly used. The rest are nearby/related context. |

**Why Context Precision is lower (and why that's fine):** Hybrid retrieval intentionally favors *consensus* over *top-rank precision*. RRF rewards chunks that BOTH retrievers found — even if neither ranked them #1. This boosts recall and cross-validation at the cost of including some adjacent context that doesn't directly contribute to the final answer. Future work: add a cross-encoder re-ranker on the top-N to lift precision without sacrificing recall.

**Question types covered:**

| Type | Example | Pattern used |
|---|---|---|
| Geographic risk | "What risks does Apple face related to China?" | `risks_in_geography` |
| Risk category | "What macroeconomic risks does Apple identify?" | `risks_by_category` |
| Competitive landscape | "What competitive risks does Apple face?" | `competitor_risks` |
| Refusal handling | "Who are the top 5 executives?" | Graph returns no match → system declines rather than confabulating |

The refusal test (q10) is **deliberate** — the system correctly recognizes it can't answer (graph has only 1 named executive) and says so rather than inventing names. Ragas penalizes this in its automated scoring (it expects an answer), so we frame it as a strength: an honest "I don't know" beats a confident hallucination.

Run the evaluation yourself: `python -m scripts.run_eval`

---

## Key engineering decisions

### 1. Two-brain retrieval (graph + vector + RRF)

Most RAG demos use vectors only. Vectors are great at fuzzy semantic match but bad at structural questions:

- *"Which subsidiaries does Apple have in Asia?"* — vectors might find paragraphs that *mention* Asia, not subsidiaries IN Asia.
- The graph answers this exactly via `MATCH (Apple)-[:OWNS]->(s:Subsidiary)-[:OPERATES_IN]->(g:Geography {name:'Asia'})`.

Conversely, vectors handle the unstructured narrative well — "what's the general sentiment about supply chain risk" doesn't have a clean Cypher pattern.

So we run BOTH in parallel and fuse with **reciprocal rank fusion** (`score = 1/(k+rank)`, k=60). This weights consensus highly and still pulls in the long tail.

### 2. Three-layer entity resolution

Naively storing every entity Claude extracts produces duplicates: *"Apple Inc."*, *"Apple"*, *"AAPL"* would each become separate nodes. The resolver runs three layers:

1. **Slug match** — normalize ("apple inc" / "apple, inc." / "Apple Inc") → exact slug match deduplicates trivially
2. **Embedding similarity** — Titan v2 cosine on entity descriptions; ≥ 0.87 → auto-merge, ≤ 0.78 → keep separate
3. **LLM adjudication** — for the murky 0.78–0.87 band, Claude judges "are these the same entity?" with strict YES/NO

This produces 244 clean canonical entities from 427 raw extractions — about 43% deduplication rate, with virtually no false merges.

### 3. Citation-grounded synthesis

The synthesizer prompt requires every factual claim to end with `[chunk_NNNN]`. The frontend regex-parses these tokens at render time into clickable indigo pills. Clicking a pill scrolls to the matching row in the diagnostics table — full transparency between claim and evidence, no opaque "trust me" answers.

### 4. Hierarchical observability

The synthesizer is decorated with `@traceable(name="answer", run_type="chain")`. The graph step is `run_type="retriever"`. The Bedrock call is `run_type="llm"`. LangSmith renders these as a nested timeline you can step through to see exactly which chunk was retrieved when, what the prompt looked like, and what Claude returned.

---

## Local setup

### Prerequisites

- Python 3.11+ with `pip` and `venv`
- Node.js 20+ with `npm`
- AWS credentials with Bedrock access (Claude + Titan)
- Neo4j Aura instance (free tier works)
- Pinecone account (free tier works)
- LangSmith API key (optional but recommended)

### Backend

```bash
git clone https://github.com/mdesoky-ai-dev/graphrag-financial-intelligence.git
cd graphrag-financial-intelligence

python -m venv .venv
.venv/bin/activate       # macOS/Linux
.venv\Scripts\activate   # Windows PowerShell

pip install -e ".[dev]"
cp .env.example .env     # then edit with your credentials
```

To run the API locally:

```bash
python -m scripts.serve  # http://127.0.0.1:8000
```

To re-run the ingestion pipeline (only needed once unless adding documents):

```bash
python -m scripts.inspect_pdf data/10ks/apple-10k-fy25.pdf
python -m scripts.inspect_chunks
python -m scripts.inspect_extraction
python -m scripts.resolve_entities
python -m scripts.index_neo4j
python -m scripts.index_pinecone
```

### Frontend

```bash
cd frontend
npm install
echo 'VITE_API_BASE_URL=http://127.0.0.1:8000' > .env
npm run dev    # http://localhost:5173
```

---

## Project structure

```
graphrag-financial-intelligence/
├── app/                          FastAPI app + config + embeddings client
│   ├── api.py                    POST /ask, GET /health, lifespan-managed Synthesizer
│   ├── config.py                 Pydantic settings (env-driven)
│   ├── embeddings.py             Titan v2 wrapper (single + batch)
│   └── schemas.py                AskRequest / AskResponse / FusedChunkInfo
│
├── ingestion/                    One-time pipeline (parse → chunk → extract → resolve → write)
│   ├── parsers/pdf_parser.py
│   ├── chunkers/section_chunker.py
│   ├── extractors/llm_extractor.py     Claude with structured-output prompt
│   ├── resolvers/entity_resolver.py    3-layer cascade (slug → embed → LLM)
│   └── writers/{neo4j_writer,pinecone_writer}.py
│
├── graph/                        Neo4j helpers
│   ├── client.py                 Connection + session management
│   ├── schema.py                 Constraints + indexes (one-time setup)
│   └── queries.py                The Cypher patterns
│
├── retrieval/                    Query-time pipeline
│   ├── query_planner.py          NL question → (pattern, params)
│   ├── graph_retriever.py        Runs Cypher + returns ranked chunks
│   ├── vector_retriever.py       Pinecone search + returns ranked chunks
│   ├── hybrid.py                 Reciprocal rank fusion
│   └── synthesizer.py            Orchestrates retrieval + Claude synthesis
│
├── evaluation/                   Ragas eval setup
│   ├── test_questions.py         10 ground-truth Q&A pairs
│   └── ragas_runner.py           Wires Bedrock judge + Titan similarity
│
├── frontend/                     React app (Vite + TS + Tailwind)
│   └── src/
│       ├── App.tsx
│       ├── components/{Header,QueryBox,AnswerPanel,DiagnosticsPanel}.tsx
│       ├── api/client.ts          Typed fetch wrapper
│       └── types/api.ts           AskResponse / FusedChunkInfo (mirrors backend Pydantic)
│
├── scripts/                      One-shot CLIs (ingest steps, eval, ad-hoc queries)
└── docs/
    ├── architecture.md
    └── entity-resolution-policy.md
```

---

## What's out of scope

This is a portfolio engineering demonstration, not a product. Productionizing for end-user document upload would require:

- **Async ingestion pipeline** with a job queue (Celery + Redis) — current pipeline takes ~10 min per document, can't block HTTP requests
- **Multi-tenant data isolation** — currently single-tenant (one shared Neo4j graph + one Pinecone index)
- **Cost controls + billing** — extraction costs ~$2/document; would need per-tenant limits
- **File storage** — S3 instead of local disk
- **Auth + authorization** — who owns which document?

The ingestion pipeline IS corpus-agnostic — running it on a different 10-K just requires changing the input PDF and re-running the indexing scripts. But adding multi-document support to the running app (with a proper upload UX, document picker, cross-document queries) is a meaningfully larger build.

Demonstrated on **Apple's 10-K FY25** (105 pages, 102 chunks, 244 entities, 293 relationships).

---

## Notes

- Free tiers used throughout. Render's free tier sleeps after 15 min idle → first request after sleep takes ~30-60s (cold start). Subsequent requests are fast (~17s).
- The corpus is gitignored. The processed JSONs live under `data/processed/` and are regenerated by the ingestion scripts; the source PDF lives under `data/10ks/`.
- LangSmith traces project: `graphrag-financial-intelligence`. Hierarchical view shows planner → retrievers → fuser → synthesizer with exact prompt + response payloads.

---

## License

MIT

---

*Built by [Mohamed Desoky](https://github.com/mdesoky-ai-dev) — full-stack AI engineer with a finance background. Project 2 of a two-project portfolio. Project 1: [smb-risk-agent.netlify.app](https://smb-risk-agent.netlify.app) — a multi-agent SMB loan underwriting system built with LangGraph.*
