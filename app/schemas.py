"""
API schemas — the JSON contract between the backend and any client.

Pydantic models do double duty here:
  1. Validate incoming requests (reject malformed JSON early with a clear error)
  2. Serialize outgoing responses (consistent JSON shape, auto-documented)

FastAPI inspects these to generate the OpenAPI schema served at /docs, so
the frontend can also use them as a reference when consuming the API.

Design choice: we expose retrieval diagnostics (graph hits, vector hits,
fused chunks, plan) alongside the answer. This is what makes the demo
distinctive — most RAG UIs hide retrieval; ours reveals it.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    """The question payload from the frontend."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural-language question about the ingested filings.",
        examples=["What does Apple say about supply chain risks?"],
    )


# ---------------------------------------------------------------------------
# Response — main pieces
# ---------------------------------------------------------------------------


class GraphStepInfo(BaseModel):
    """One graph retrieval step the planner emitted, surfaced for the UI."""
    pattern: str
    params: dict


class PlanInfo(BaseModel):
    """Compact view of the planner's decisions."""
    graph_steps: list[GraphStepInfo]
    run_vector: bool
    vector_top_k: int
    notes: list[str] = Field(default_factory=list)


class FusedChunkInfo(BaseModel):
    """One fused chunk with its provenance — which retriever(s) found it."""
    chunk_id: str
    rrf_score: float
    sources: list[str]                  # e.g., ["graph", "vector"] or ["graph"]
    graph_rank: Optional[int] = None
    vector_rank: Optional[int] = None


class AskResponse(BaseModel):
    """Full answer payload returned to the frontend."""

    question: str
    answer: str = Field(description="Markdown-formatted answer with [chunk_id] citations.")
    cited_chunk_ids: list[str] = Field(
        description="The chunk_ids referenced in the answer's citations."
    )
    elapsed_seconds: float

    # Retrieval diagnostics — what makes this demo interesting
    plan: PlanInfo
    graph_hits_count: int
    vector_hits_count: int
    fused_chunks: list[FusedChunkInfo]
    chunks_fed_to_synthesis: int


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Liveness/readiness check. Render and Netlify use this to verify deploys."""
    status: str = Field(description="Always 'ok' if the server is up.")
    service: str = Field(description="Service identifier for log correlation.")
    version: str = Field(description="App version (read from pyproject.toml).")
