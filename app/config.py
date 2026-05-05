"""
Application configuration.

Loads environment variables from .env into a single typed Settings object.
Every other module imports `settings` from here rather than reading os.environ
directly. This keeps config access centralized, type-checked, and testable.

Usage:
    from app.config import settings
    print(settings.neo4j_uri)
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings loaded from .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # tolerate extra env vars without crashing
    )

    # ---- Runtime ----
    env: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"

    # ---- AWS Bedrock ----
    aws_access_key_id: SecretStr
    aws_secret_access_key: SecretStr
    aws_region: str = "us-east-1"
    bedrock_llm_model_id: str
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"

    # ---- Neo4j ----
    neo4j_uri: str
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr
    neo4j_database: str = "neo4j"

    # ---- Pinecone ----
    pinecone_api_key: SecretStr
    pinecone_index_name: str = "financial-docs"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # ---- LangSmith ----
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "graphrag-financial-intelligence"
    langsmith_tracing: bool = True

    # ---- Ingestion / retrieval params ----
    embedding_dimensions: int = 1024
    chunk_size: int = 1024
    chunk_overlap: int = 200

    # Entity resolution thresholds
    er_merge_threshold: float = Field(default=0.87, ge=0.0, le=1.0)
    er_reject_threshold: float = Field(default=0.78, ge=0.0, le=1.0)


@lru_cache
def get_settings() -> Settings:
    """Cached factory. Settings are loaded once per process."""
    return Settings()  # type: ignore[call-arg]


# Convenience module-level singleton.
settings = get_settings()
