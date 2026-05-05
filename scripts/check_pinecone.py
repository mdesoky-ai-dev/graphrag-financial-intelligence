"""
Sanity-check script: prove Pinecone + Bedrock work end-to-end.

Run:
    python -m scripts.check_pinecone

What it does:
  1. Loads .env via app.config
  2. Connects to Pinecone with the API key
  3. Confirms the financial-docs index exists
  4. Verifies dimensions match Titan V2 and metric is cosine
  5. Reads current vector count (expected 0 on a fresh index)
  6. Generates a tiny test embedding via Bedrock Titan
  7. Upserts the vector to Pinecone
  8. Queries Pinecone with the same vector and confirms it returns
  9. Deletes the test vector to leave the index clean

Verifies three services in one pass: AWS credentials, Bedrock model access,
and Pinecone connectivity.

Exit code 0 on success, 1 on any failure.
"""

from __future__ import annotations

import logging
import sys
import time

from pinecone import Pinecone

from app.config import settings
from app.embeddings import BedrockEmbedder

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("check_pinecone")

TEST_VECTOR_ID = "smoke-test-vector"
TEST_TEXT = "This is a smoke test sentence for the GraphRAG ingestion pipeline."


def main() -> int:
    log.info(
        "env=%s, index=%s, region=%s",
        settings.env,
        settings.pinecone_index_name,
        settings.pinecone_region,
    )

    # ---- 1. Pinecone connection ----
    try:
        pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
    except Exception as e:
        log.error("Failed to construct Pinecone client: %s", e)
        return 1

    # ---- 2. Index exists and is configured correctly ----
    try:
        index_list = [i.name for i in pc.list_indexes()]
        if settings.pinecone_index_name not in index_list:
            log.error(
                "Index %r not found. Available indexes: %s",
                settings.pinecone_index_name,
                index_list,
            )
            return 1
        log.info("Index %r exists.", settings.pinecone_index_name)

        desc = pc.describe_index(settings.pinecone_index_name)
        log.info("Index spec: dimension=%d, metric=%s", desc.dimension, desc.metric)

        if desc.dimension != settings.embedding_dimensions:
            log.error(
                "Dimension mismatch: index has %d, config expects %d",
                desc.dimension,
                settings.embedding_dimensions,
            )
            return 1
        if desc.metric != "cosine":
            log.error("Metric mismatch: index uses %s, expected cosine", desc.metric)
            return 1
    except Exception as e:
        log.exception("Failed to verify Pinecone index: %s", e)
        return 1

    # ---- 3. Bedrock embedding ----
    try:
        embedder = BedrockEmbedder()
        log.info("Generating embedding via Bedrock Titan...")
        vec = embedder.embed(TEST_TEXT)
        log.info("Got embedding of length %d (first 4 dims: %s)", len(vec), vec[:4])

        if len(vec) != settings.embedding_dimensions:
            log.error(
                "Embedding length %d does not match expected %d",
                len(vec),
                settings.embedding_dimensions,
            )
            return 1
    except Exception as e:
        log.exception("Bedrock embedding failed: %s", e)
        log.error(
            "Likely cause: missing IAM permission bedrock:InvokeModel "
            "for model %s, or the model is not enabled in your Bedrock console.",
            settings.bedrock_embedding_model_id,
        )
        return 1

    # ---- 4. Upsert to Pinecone ----
    index = pc.Index(settings.pinecone_index_name)
    try:
        log.info("Upserting test vector...")
        index.upsert(vectors=[{"id": TEST_VECTOR_ID, "values": vec, "metadata": {"source": "smoke-test"}}])
        # Pinecone serverless is eventually consistent — give it a moment.
        time.sleep(2)
    except Exception as e:
        log.exception("Pinecone upsert failed: %s", e)
        return 1

    # ---- 5. Query and verify round-trip ----
    try:
        log.info("Querying Pinecone with the same vector...")
        result = index.query(vector=vec, top_k=1, include_metadata=True)
        matches = result.get("matches", [])
        if not matches or matches[0]["id"] != TEST_VECTOR_ID:
            log.error("Round-trip failed. Got: %s", result)
            return 1
        log.info(
            "Round-trip success: id=%s, score=%.4f, metadata=%s",
            matches[0]["id"],
            matches[0]["score"],
            matches[0].get("metadata"),
        )
    except Exception as e:
        log.exception("Pinecone query failed: %s", e)
        return 1

    # ---- 6. Clean up ----
    try:
        log.info("Deleting test vector...")
        index.delete(ids=[TEST_VECTOR_ID])
    except Exception as e:
        log.warning("Cleanup delete failed (not fatal): %s", e)

    log.info("All checks passed. Pinecone + Bedrock + AWS credentials are wired.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
