"""
End-to-end retrieval demo.

Demonstrates that the two-brain architecture works:
  1. Walk Neo4j to find supply-chain risks Apple reports
  2. Fetch the source chunks from Pinecone using chunk_ids on the edges
  3. Pass chunks to Claude with a grounded-answer prompt
  4. Print the answer with citations

Run:
    python -m scripts.demo_retrieval

This is a one-shot proof. The formal retriever module comes after.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

import boto3
from botocore.config import Config
from pinecone import Pinecone

from app.config import settings
from graph.client import Neo4jClient

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("demo_retrieval")


# ----------------------------------------------------------------------------
# The question we're answering
# ----------------------------------------------------------------------------

QUESTION = "What does Apple say about supply chain risks?"
RISK_CATEGORY = "supply_chain"
COMPANY_NAME = "Apple Inc."


# ----------------------------------------------------------------------------
# Step 1: Graph query
# ----------------------------------------------------------------------------

CYPHER_FETCH_RISKS = """
    MATCH (c:Company {canonical_name: $company})-[r:REPORTS_RISK]->(risk:RiskFactor)
    WHERE risk.category = $category
    RETURN risk.canonical_name AS risk_name,
           risk.canonical_id   AS risk_id,
           r.source_chunks     AS source_chunks
"""


def fetch_risks_from_graph() -> list[dict[str, Any]]:
    """Find supply-chain risks Apple reports, with the chunks that cite them."""
    log.info("[Step 1] Querying Neo4j for %s risks reported by %s ...",
             RISK_CATEGORY, COMPANY_NAME)
    client = Neo4jClient()
    try:
        with client.session() as session:
            result = session.run(
                CYPHER_FETCH_RISKS,
                company=COMPANY_NAME,
                category=RISK_CATEGORY,
            )
            risks = [dict(record) for record in result]
    finally:
        client.close()

    log.info("  found %d %s risks", len(risks), RISK_CATEGORY)
    for r in risks:
        log.info("    - %s  (cited in %d chunk(s))",
                 r["risk_name"][:80], len(r["source_chunks"]))
    return risks


# ----------------------------------------------------------------------------
# Step 2: Pinecone fetch
# ----------------------------------------------------------------------------


def fetch_chunks_from_pinecone(chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch the metadata (incl. text excerpts) for a set of chunk_ids."""
    log.info("[Step 2] Fetching %d chunks from Pinecone ...", len(chunk_ids))
    pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
    index = pc.Index(settings.pinecone_index_name)

    response = index.fetch(ids=chunk_ids)
    vectors = response.vectors  # dict[str, FetchResponseVector]

    chunks: dict[str, dict[str, Any]] = {}
    for chunk_id, vec in vectors.items():
        md = vec.metadata or {}
        chunks[chunk_id] = {
            "chunk_id": chunk_id,
            "text_excerpt": md.get("text_excerpt", ""),
            "page_start": int(md.get("page_start", -1)),
            "page_end": int(md.get("page_end", -1)),
            "doc_id": md.get("doc_id", ""),
        }
    log.info("  retrieved %d/%d chunks", len(chunks), len(chunk_ids))
    return chunks


# ----------------------------------------------------------------------------
# Step 3: Build prompt and call Claude
# ----------------------------------------------------------------------------


SYNTHESIS_PROMPT = """You are a financial analyst answering a question about Apple's 10-K filing.

# Question

{question}

# Source passages

You have been given source passages from Apple's 10-K. Each is identified by a chunk_id.
Answer the question using ONLY information present in these passages. Cite passages
inline using their chunk_id in square brackets, e.g. [apple-10k-fy25_chunk_0014].

If the passages don't contain enough information to answer the question, say so directly.
Do not invent risks or details that aren't in the source material.

{passages}

# Instructions

- Be concise (3-6 sentences).
- Every factual claim must have a citation.
- Group related risks logically; don't just list them.
- Use plain language; avoid hedging unless the source hedges.

# Answer
"""


def build_passages_block(chunks: dict[str, dict[str, Any]]) -> str:
    """Format the chunks for inclusion in the synthesis prompt."""
    lines = []
    for chunk_id, c in chunks.items():
        lines.append(
            f"[{chunk_id}] (pages {c['page_start']}-{c['page_end']})\n{c['text_excerpt']}\n"
        )
    return "\n---\n".join(lines)


def synthesize_answer(question: str, chunks: dict[str, dict[str, Any]]) -> str:
    """Call Claude on Bedrock with the passages and the question."""
    log.info("[Step 3] Sending question + %d passages to Claude ...", len(chunks))
    boto_config = Config(
        region_name=settings.aws_region,
        retries={"max_attempts": 4, "mode": "standard"},
    )
    client = boto3.client(
        "bedrock-runtime",
        aws_access_key_id=settings.aws_access_key_id.get_secret_value(),
        aws_secret_access_key=settings.aws_secret_access_key.get_secret_value(),
        config=boto_config,
    )
    prompt = SYNTHESIS_PROMPT.format(
        question=question,
        passages=build_passages_block(chunks),
    )
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    })
    response = client.invoke_model(
        modelId=settings.bedrock_llm_model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(response["body"].read())
    text = "\n".join(b["text"] for b in payload["content"] if b["type"] == "text")
    return text.strip()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    log.info("==== Question ====")
    log.info("%s", QUESTION)

    # 1. Graph query
    risks = fetch_risks_from_graph()
    if not risks:
        log.error("No risks found. Either the graph isn't populated or the category is wrong.")
        return 1

    # Collect every chunk_id cited by any matching edge (deduplicated)
    chunk_ids = sorted({cid for r in risks for cid in r["source_chunks"]})
    log.info("  unique chunks cited by these risks: %d", len(chunk_ids))

    # 2. Pinecone fetch
    chunks = fetch_chunks_from_pinecone(chunk_ids)
    if not chunks:
        log.error("No chunks retrieved from Pinecone — was the index populated?")
        return 1

    # 3. Synthesis
    answer = synthesize_answer(QUESTION, chunks)

    # 4. Print
    log.info("==== Answer ====\n%s\n", answer)
    log.info("==== Sources cited (the chunk IDs available to Claude) ====")
    for cid in sorted(chunks):
        c = chunks[cid]
        log.info("  %s  pages=%d-%d", cid, c["page_start"], c["page_end"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
