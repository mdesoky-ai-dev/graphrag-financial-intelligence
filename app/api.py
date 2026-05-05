"""
FastAPI application — exposes the GraphRAG Synthesizer as a REST API.

Endpoints:
  GET  /health           — liveness check (returns 200 if app is up)
  POST /ask              — accepts a question, returns a grounded answer
  GET  /docs             — auto-generated interactive API explorer (Swagger UI)
  GET  /openapi.json     — machine-readable schema (OpenAPI 3.x)

Design notes:
  - The Synthesizer is constructed ONCE at startup and reused across requests.
    Reconstructing it per-request would re-open Neo4j and re-init Pinecone every
    time, adding ~1-2 seconds per call.
  - CORS is configured to allow the deployed Netlify frontend to call this API.
  - LangSmith @traceable decorators on the Synthesizer fire automatically on
    every request, so the dashboard captures real production traffic.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import (
    AskRequest,
    AskResponse,
    FusedChunkInfo,
    GraphStepInfo,
    HealthResponse,
    PlanInfo,
)
from retrieval.synthesizer import Synthesizer

logger = logging.getLogger(__name__)

# Module-level handle on the synthesizer; populated at startup.
_synthesizer: Synthesizer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Start/stop hook for app-wide resources.

    On startup: build the Synthesizer (opens Neo4j, inits Pinecone, etc.).
    On shutdown: close the Synthesizer (releases Neo4j driver).

    Using FastAPI's modern `lifespan` context manager (replaces the deprecated
    @app.on_event hooks).
    """
    global _synthesizer
    logger.info("Starting up: building Synthesizer ...")
    _synthesizer = Synthesizer()
    logger.info("Synthesizer ready. API is live.")
    yield
    logger.info("Shutting down: closing Synthesizer ...")
    if _synthesizer is not None:
        _synthesizer.close()
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


app = FastAPI(
    title="GraphRAG Financial Intelligence API",
    description=(
        "Hybrid graph + vector retrieval over SEC 10-K filings. "
        "Answers questions with grounded citations to the source document."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow the frontend (and local dev) to call this API from a browser.
# Wildcard `*` would also work for a portfolio piece but is bad practice;
# explicit allow-list is safer.
ALLOWED_ORIGINS = [
    "http://localhost:5173",      # Vite dev server (default)
    "http://localhost:3000",      # alternate dev port
    "http://127.0.0.1:5173",
    # Add the deployed Netlify URL once it exists, e.g.:
    #   "https://graphrag-financial.netlify.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check. Returns 200 if the server is up. Used by Render and uptime monitors."""
    return HealthResponse(
        status="ok",
        service="graphrag-financial-intelligence",
        version=app.version,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """
    Answer a question using hybrid graph + vector retrieval.

    Pipeline (delegated to the Synthesizer):
      1. Plan retrieval (rule-based intent classifier)
      2. Run graph patterns against Neo4j
      3. Run vector search against Pinecone
      4. Fuse with reciprocal rank fusion
      5. Fetch chunk text from Pinecone
      6. Synthesize answer with Claude (Bedrock)

    Returns the answer plus retrieval diagnostics (which makes the demo
    distinctive — most RAG UIs hide what happened internally).
    """
    if _synthesizer is None:
        # Should never happen if startup succeeded, but defensive check.
        raise HTTPException(status_code=503, detail="Synthesizer not initialized")

    logger.info("[POST /ask] question=%r", request.question)
    try:
        result = _synthesizer.answer(request.question)
    except Exception as exc:
        logger.exception("Synthesizer failed for question: %r", request.question)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {exc}") from exc

    # Map our internal Answer dataclass to the API response schema.
    return AskResponse(
        question=result.question,
        answer=result.text,
        cited_chunk_ids=list(result.cited_chunk_ids),
        elapsed_seconds=result.elapsed_seconds,
        plan=_plan_to_schema(result.plan),
        graph_hits_count=len(result.graph_hits),
        vector_hits_count=len(result.vector_hits),
        fused_chunks=[
            FusedChunkInfo(
                chunk_id=h.chunk_id,
                rrf_score=h.rrf_score,
                sources=list(h.sources),
                graph_rank=h.graph_rank,
                vector_rank=h.vector_rank,
            )
            for h in result.fused_chunks
        ],
        chunks_fed_to_synthesis=len(result.fused_chunks),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plan_to_schema(plan: Any) -> PlanInfo:
    """Convert a RetrievalPlan dataclass to the API's PlanInfo schema."""
    return PlanInfo(
        graph_steps=[
            GraphStepInfo(pattern=s.pattern.value, params=dict(s.params))
            for s in plan.graph_steps
        ],
        run_vector=plan.run_vector,
        vector_top_k=plan.vector_top_k,
        notes=list(plan.notes),
    )
