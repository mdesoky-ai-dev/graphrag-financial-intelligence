"""
Bedrock Titan embeddings wrapper.

Thin client around AWS Bedrock's embedding endpoint. Used during ingestion to
turn text chunks into 1024-dim vectors for Pinecone, and during entity
resolution to compare candidate names.

Usage:
    from app.embeddings import BedrockEmbedder

    embedder = BedrockEmbedder()
    vec = embedder.embed("dependence on Asian component suppliers")
    assert len(vec) == 1024

    vecs = embedder.embed_batch(["chunk one text", "chunk two text"])
    assert len(vecs) == 2
"""

from __future__ import annotations

import json
import logging
from typing import Sequence

import boto3
from botocore.config import Config

from app.config import settings

logger = logging.getLogger(__name__)


class BedrockEmbedder:
    """Wrapper around Bedrock's embedding model.

    Default model is amazon.titan-embed-text-v2:0 (1024 dims). The model is
    configured via app.config.settings, not hardcoded.

    Boto3 client construction is centralized here so that retry config, region,
    and credential plumbing live in one place.
    """

    def __init__(
        self,
        model_id: str | None = None,
        region: str | None = None,
        max_retries: int = 4,
    ) -> None:
        self._model_id = model_id or settings.bedrock_embedding_model_id
        self._region = region or settings.aws_region

        # Standard retry mode handles Bedrock throttling gracefully.
        boto_config = Config(
            region_name=self._region,
            retries={"max_attempts": max_retries, "mode": "standard"},
        )
        self._client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=settings.aws_access_key_id.get_secret_value(),
            aws_secret_access_key=settings.aws_secret_access_key.get_secret_value(),
            config=boto_config,
        )
        logger.info("BedrockEmbedder initialized: model=%s, region=%s", self._model_id, self._region)

    def embed(self, text: str) -> list[float]:
        """Embed a single string. Returns a 1024-dim float vector for Titan V2."""
        body = json.dumps({"inputText": text})
        response = self._client.invoke_model(
            modelId=self._model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read())
        embedding: list[float] = payload["embedding"]
        return embedding

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed many strings. Titan does not natively batch, so we loop.

        Bedrock's standard retry config handles transient throttling; for very
        large batches we'd want to add explicit rate limiting and backoff.
        Sufficient for the smoke test and for our ~10-document corpus.
        """
        return [self.embed(t) for t in texts]
