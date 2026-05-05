"""
Synthesizer.

Top-level orchestrator. Given a user question:
  1. Ask the planner for a RetrievalPlan
  2. Run the graph patterns named in the plan
  3. Run vector search if requested
  4. Fuse graph + vector hits with Reciprocal Rank Fusion
  5. Fetch the chunk text for the top fused chunks from Pinecone
  6. Build a synthesis prompt and call Claude
  7. Return a typed Answer

Single entry point: `Synthesizer.answer(question)`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import boto3
from botocore.config import Config
from langsmith import traceable

from app.config import settings
from retrieval.graph_retriever import GraphHit, GraphRetriever
from retrieval.hybrid import FusedHit, fuse, fused_chunk_ids
from retrieval.query_planner import GraphStep, PatternId, QueryPlanner, RetrievalPlan
from retrieval.vector_retriever import ChunkText, VectorHit, VectorRetriever

logger = logging.getLogger(__name__)


# Cap on chunks fed into the synthesis prompt. Limits cost + keeps the model
# focused. Tunable; 25 felt right with ~1 KB excerpts on Sonnet.
DEFAULT_TOP_N_CHUNKS = 25


@dataclass
class Answer:
    """A synthesized answer with citations and provenance."""

    question: str
    plan: RetrievalPlan
    text: str
    cited_chunk_ids: list[str]
    fused_chunks: list[FusedHit]
    graph_hits: list[GraphHit] = field(default_factory=list)
    vector_hits: list[VectorHit] = field(default_factory=list)
    elapsed_seconds: float = 0.0


SYNTHESIS_PROMPT = """You are a financial analyst answering a question about a public company's 10-K filing.

# Question

{question}

# Source passages

You have been given source passages from the filing, each identified by a chunk_id.
Answer the question using ONLY information present in these passages. Cite passages
inline using their chunk_id in square brackets, e.g. [apple-10k-fy25_chunk_0014].

If the passages don't contain enough information to answer the question, say so directly
and explain what's missing. Do not invent facts not in the source material.

# Passages

{passages}

# Answer requirements

- Be concise and well-organized: 2-4 paragraphs typical.
- Group related points logically; don't just list every passage.
- Every factual claim must have a citation.
- Use plain language; avoid hedging unless the source hedges.

# Answer
"""


def _build_passages_block(chunks: dict[str, ChunkText]) -> str:
    """Format the chunks for inclusion in the synthesis prompt."""
    if not chunks:
        return "(no passages retrieved)"
    blocks = []
    for cid, c in chunks.items():
        blocks.append(
            f"[{cid}] (pages {c.page_start}-{c.page_end})\n{c.text_excerpt}"
        )
    return "\n\n---\n\n".join(blocks)


class Synthesizer:
    """Top-level question-answering pipeline."""

    def __init__(
        self,
        graph: GraphRetriever | None = None,
        vector: VectorRetriever | None = None,
        planner: QueryPlanner | None = None,
        top_n_chunks: int = DEFAULT_TOP_N_CHUNKS,
    ) -> None:
        self._graph = graph or GraphRetriever()
        self._vector = vector or VectorRetriever()
        self._planner = planner or QueryPlanner()
        self._top_n_chunks = top_n_chunks

        boto_config = Config(
            region_name=settings.aws_region,
            retries={"max_attempts": 4, "mode": "standard"},
        )
        self._claude = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=settings.aws_access_key_id.get_secret_value(),
            aws_secret_access_key=settings.aws_secret_access_key.get_secret_value(),
            config=boto_config,
        )
        logger.info("Synthesizer ready (top_n_chunks=%d)", self._top_n_chunks)

    def close(self) -> None:
        self._graph.close()

    # ---- main entrypoint ----

    @traceable(name="synthesizer.answer", run_type="chain")
    def answer(self, question: str) -> Answer:
        import time
        t0 = time.time()

        # 1. Plan
        plan = self._planner.plan(question)

        # 2. Run graph steps
        graph_hits = self._run_graph_steps(plan.graph_steps)

        # 3. Run vector search (always on by default)
        vector_hits: list[VectorHit] = []
        if plan.run_vector:
            vector_hits = self._vector.search(question, top_k=plan.vector_top_k)

        # 4. Fuse
        fused = fuse(graph_hits, vector_hits, top_n=self._top_n_chunks)
        chunk_ids = fused_chunk_ids(fused)
        if not chunk_ids:
            return Answer(
                question=question, plan=plan,
                text="No source passages were found that match this question.",
                cited_chunk_ids=[], fused_chunks=[],
                graph_hits=graph_hits, vector_hits=vector_hits,
                elapsed_seconds=time.time() - t0,
            )

        # 5. Fetch chunk text
        chunks = self._vector.fetch(chunk_ids)
        # Order chunks by fused rank for the prompt (best first)
        ordered_chunks: dict[str, ChunkText] = {
            cid: chunks[cid] for cid in chunk_ids if cid in chunks
        }

        # 6. Synthesize
        text = self._synthesize(question, ordered_chunks)

        return Answer(
            question=question, plan=plan, text=text,
            cited_chunk_ids=list(ordered_chunks.keys()),
            fused_chunks=fused,
            graph_hits=graph_hits, vector_hits=vector_hits,
            elapsed_seconds=time.time() - t0,
        )

    # ---- internals ----

    @traceable(name="synthesizer.run_graph_steps", run_type="retriever")
    def _run_graph_steps(self, steps: list[GraphStep]) -> list[GraphHit]:
        all_hits: list[GraphHit] = []
        for step in steps:
            try:
                if step.pattern == PatternId.RISKS_FOR_COMPANY:
                    hits = self._graph.risks_for_company(
                        company=step.params["company"],
                        category=step.params.get("category"),
                    )
                elif step.pattern == PatternId.RISKS_IN_GEOGRAPHY:
                    hits = self._graph.risks_in_geography(
                        geography=step.params["geography"],
                    )
                elif step.pattern == PatternId.COMPETITORS_OF:
                    hits = self._graph.competitors_of(
                        company=step.params["company"],
                    )
                elif step.pattern == PatternId.SHARED_RISKS:
                    hits = self._graph.shared_risks(
                        company_a=step.params["company_a"],
                        company_b_list=step.params["company_b_list"],
                    )
                else:
                    logger.warning("Unknown pattern in plan: %s", step.pattern)
                    continue
                all_hits.extend(hits)
            except Exception as e:
                logger.exception("Graph step %s failed: %s", step.pattern, e)
        return all_hits

    @traceable(name="synthesizer.claude_synthesis", run_type="llm")
    def _synthesize(self, question: str, chunks: dict[str, ChunkText]) -> str:
        prompt = SYNTHESIS_PROMPT.format(
            question=question,
            passages=_build_passages_block(chunks),
        )
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        })
        response = self._claude.invoke_model(
            modelId=settings.bedrock_llm_model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read())
        return "\n".join(b["text"] for b in payload["content"] if b["type"] == "text").strip()
