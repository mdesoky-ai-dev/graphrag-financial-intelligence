"""
Entity resolution: the three-layer cascade.

Takes a stream of extracted entities (with duplicates) and merges them
into canonical entities, where each canonical entity tracks all the
surface forms it was seen as.

The cascade:

    Layer 1 — Normalization (instant, free)
        Two names that share the same normalized slug are auto-merged.
        Catches case differences, punctuation, articles.

    Layer 2 — Embedding similarity (fast, ~$0.0001 per check)
        For names that don't slug-match, embed them and compare cosine
        distance to existing canonical names.
            sim >= ER_MERGE_THRESHOLD   -> auto-merge
            sim <  ER_REJECT_THRESHOLD  -> auto-reject (new entity)
            else                        -> escalate to layer 3

    Layer 3 — LLM adjudication (slow, ~$0.01 per check)
        For the gray zone, ask Claude:
            "Are these two entities the same underlying business concept?"
        Used sparingly — only on uncertain pairs.

Resolution scope: by entity TYPE. We never compare a Company to a
RiskFactor — they're different types so they can't merge by definition.
This also bounds the search space: comparing 257 risks against each other
is tractable; comparing 427 entities of mixed types pairwise would not be.

Output: a list of CanonicalEntity records, each carrying:
  - canonical_id   (stable slug)
  - canonical_name (the chosen representative surface form)
  - type           (Company, RiskFactor, ...)
  - properties     (merged from all surface forms)
  - aliases        (every surface form ever seen)
  - source_chunks  (every chunk_id that mentioned any surface form)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from app.config import settings
from app.embeddings import BedrockEmbedder
from ingestion.extractors.llm_extractor import LLMExtractor  # for adjudication
from ingestion.extractors.schemas import Entity
from ingestion.resolvers.normalizer import normalize, slug

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------


@dataclass
class CanonicalEntity:
    """A merged entity with all surface forms it was seen as."""

    canonical_id: str
    canonical_name: str
    type: str
    properties: dict[str, str] = field(default_factory=dict)
    aliases: set[str] = field(default_factory=set)
    source_chunks: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "canonical_name": self.canonical_name,
            "type": self.type,
            "properties": dict(self.properties),
            "aliases": sorted(self.aliases),
            "source_chunks": sorted(self.source_chunks),
        }


# ----------------------------------------------------------------------------
# Cosine similarity (numpy, no sklearn needed)
# ----------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0.0:
        return 0.0
    return float(np.dot(av, bv) / denom)


# ----------------------------------------------------------------------------
# LLM adjudication — only used in the gray zone
# ----------------------------------------------------------------------------


_ADJUDICATION_PROMPT = """You are merging entities extracted from financial filings. Decide whether the two entities below describe the same underlying business concept and should be merged into a single graph node.

Return ONLY one of these single tokens, with no other output:
  YES        - same concept, merge them
  NO         - different concepts, keep separate
  RELATED    - related but not identical, keep separate

# Examples for risk factors

  YES   "Dependence on Asian component suppliers" / "Reliance on suppliers in Asia"
  YES   "Cybersecurity threats" / "Information security risk"
  NO    "Cybersecurity risk" / "Data privacy risk"
  NO    "Supply chain disruption" / "Foreign exchange risk"
  RELATED "Currency fluctuations" / "Foreign tax liability"

# Entities to compare

Type: {entity_type}

A: {name_a}
B: {name_b}

Decision (YES, NO, or RELATED):"""


def _adjudicate(
    extractor: LLMExtractor,
    entity_type: str,
    name_a: str,
    name_b: str,
) -> bool:
    """Ask Claude whether two entity names mean the same thing.

    Reuses the LLMExtractor for its Bedrock client; we just substitute a
    different prompt. Returns True for YES, False for NO or RELATED.
    """
    prompt = _ADJUDICATION_PROMPT.format(
        entity_type=entity_type, name_a=name_a, name_b=name_b
    )
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 8,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    })
    response = extractor._client.invoke_model(
        modelId=extractor._model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(response["body"].read())
    text = "".join(b["text"] for b in payload["content"] if b["type"] == "text").strip().upper()
    decision = text.split()[0] if text else "NO"
    is_merge = decision == "YES"
    logger.debug("Adjudicate [%s]: %r vs %r -> %s (merge=%s)",
                 entity_type, name_a, name_b, decision, is_merge)
    return is_merge


# ----------------------------------------------------------------------------
# Main resolver
# ----------------------------------------------------------------------------


class EntityResolver:
    """Three-layer cascade resolver, scoped per entity type.

    Usage:
        resolver = EntityResolver()
        resolver.add_entity(entity, source_chunk_id="apple-10k-fy25_chunk_0042")
        ...
        canonicals = resolver.resolve()  # list[CanonicalEntity]
    """

    def __init__(
        self,
        merge_threshold: float | None = None,
        reject_threshold: float | None = None,
        adjudication_max_calls: int = 100,
    ) -> None:
        self._merge_threshold = merge_threshold or settings.er_merge_threshold
        self._reject_threshold = reject_threshold or settings.er_reject_threshold
        self._adjudication_max_calls = adjudication_max_calls

        self._embedder = BedrockEmbedder()
        self._extractor = LLMExtractor()  # reused for adjudication LLM calls

        # Pending entities, grouped by type. Each entry: (Entity, source_chunk_id)
        self._pending: dict[str, list[tuple[Entity, str]]] = {}

        # Counters (for the post-run summary)
        self._stats = {
            "slug_merges": 0,
            "embedding_merges": 0,
            "embedding_rejects": 0,
            "llm_merges": 0,
            "llm_rejects": 0,
            "adjudication_calls": 0,
        }

    def add_entity(self, entity: Entity, source_chunk_id: str) -> None:
        """Queue an entity for resolution. Call repeatedly across all chunks."""
        self._pending.setdefault(entity.type.value, []).append((entity, source_chunk_id))

    def resolve(self) -> list[CanonicalEntity]:
        """Run the cascade across all pending entities. Returns canonicals."""
        all_canonicals: list[CanonicalEntity] = []
        for entity_type, items in self._pending.items():
            logger.info("Resolving %d %s entities ...", len(items), entity_type)
            canonicals = self._resolve_type(entity_type, items)
            logger.info("  -> %d canonical %s entities", len(canonicals), entity_type)
            all_canonicals.extend(canonicals)
        logger.info("Resolution stats: %s", dict(self._stats))
        return all_canonicals

    # ---- per-type resolution ----

    def _resolve_type(
        self,
        entity_type: str,
        items: list[tuple[Entity, str]],
    ) -> list[CanonicalEntity]:
        """Resolve all entities of a single type."""

        # ---- Layer 1: slug-based grouping ----
        # Group entities by their normalized slug. Same slug = guaranteed merge.
        by_slug: dict[str, list[tuple[Entity, str]]] = {}
        for entity, chunk_id in items:
            key = slug(entity.name)
            by_slug.setdefault(key, []).append((entity, chunk_id))
        # Count layer-1 merges (groups with >1 member each contribute n-1 merges)
        for group in by_slug.values():
            if len(group) > 1:
                self._stats["slug_merges"] += len(group) - 1

        # Build initial canonical list, one per slug group.
        canonicals: list[CanonicalEntity] = []
        canonical_embeddings: list[list[float] | None] = []  # parallel list, lazy-filled

        for slug_key, group in by_slug.items():
            # Pick a representative name: the longest is usually the most
            # informative ("Adverse macroeconomic conditions reducing consumer
            # confidence and product demand" beats "Adverse macroeconomic
            # conditions"). Tie-broken by alphabetical order for determinism.
            best = max(group, key=lambda gi: (len(gi[0].name), gi[0].name))[0]
            merged_props = self._merge_properties(g[0].properties for g in group)
            canonical = CanonicalEntity(
                canonical_id=f"{entity_type.lower()}_{slug_key}" if slug_key else f"{entity_type.lower()}_unknown",
                canonical_name=best.name,
                type=entity_type,
                properties=merged_props,
                aliases={g[0].name for g in group},
                source_chunks={g[1] for g in group},
            )
            canonicals.append(canonical)
            canonical_embeddings.append(None)

        # ---- Layer 2 + 3: pairwise compare canonicals, merge iteratively ----
        # Strategy: for each canonical (in order), compare against earlier
        # canonicals. If we find a match (>= merge threshold via embeddings,
        # or YES via adjudication), merge into the earlier one and remove this.
        #
        # This is O(n^2) pairs but n is per-type and small (Risks max ~250).
        merged_indices: set[int] = set()
        for i in range(len(canonicals)):
            if i in merged_indices:
                continue
            for j in range(i + 1, len(canonicals)):
                if j in merged_indices:
                    continue
                merge = self._should_merge(
                    entity_type=entity_type,
                    a_idx=i, b_idx=j,
                    canonicals=canonicals,
                    embeddings=canonical_embeddings,
                )
                if merge:
                    canonicals[i] = self._merge_into(canonicals[i], canonicals[j])
                    merged_indices.add(j)

        # Drop merged-away entries
        return [c for k, c in enumerate(canonicals) if k not in merged_indices]

    # ---- decision: should canonicals[i] absorb canonicals[j]? ----

    def _should_merge(
        self,
        entity_type: str,
        a_idx: int, b_idx: int,
        canonicals: list[CanonicalEntity],
        embeddings: list[list[float] | None],
    ) -> bool:
        a = canonicals[a_idx]
        b = canonicals[b_idx]

        # Embedding layer
        if embeddings[a_idx] is None:
            embeddings[a_idx] = self._embedder.embed(a.canonical_name)
        if embeddings[b_idx] is None:
            embeddings[b_idx] = self._embedder.embed(b.canonical_name)
        sim = _cosine(embeddings[a_idx], embeddings[b_idx])  # type: ignore[arg-type]

        if sim >= self._merge_threshold:
            self._stats["embedding_merges"] += 1
            logger.debug("EMBED MERGE  [%s] sim=%.3f  %r <- %r",
                         entity_type, sim, a.canonical_name, b.canonical_name)
            return True
        if sim < self._reject_threshold:
            self._stats["embedding_rejects"] += 1
            return False

        # Gray zone: LLM adjudicates
        if self._stats["adjudication_calls"] >= self._adjudication_max_calls:
            logger.warning(
                "Adjudication budget exhausted (%d calls). Defaulting gray-zone pairs to NO MERGE.",
                self._adjudication_max_calls,
            )
            return False
        self._stats["adjudication_calls"] += 1
        decision = _adjudicate(self._extractor, entity_type, a.canonical_name, b.canonical_name)
        if decision:
            self._stats["llm_merges"] += 1
            logger.debug("LLM MERGE    [%s] sim=%.3f  %r <- %r",
                         entity_type, sim, a.canonical_name, b.canonical_name)
        else:
            self._stats["llm_rejects"] += 1
        return decision

    # ---- helpers ----

    @staticmethod
    def _merge_properties(props_iter: Iterable[dict[str, str]]) -> dict[str, str]:
        """Union dictionaries; first non-empty value wins per key."""
        merged: dict[str, str] = {}
        for props in props_iter:
            for k, v in props.items():
                if v and k not in merged:
                    merged[k] = v
        return merged

    @staticmethod
    def _merge_into(target: CanonicalEntity, source: CanonicalEntity) -> CanonicalEntity:
        """Fold `source` into `target`. The longer canonical_name wins."""
        target.aliases |= source.aliases
        target.source_chunks |= source.source_chunks
        for k, v in source.properties.items():
            if v and k not in target.properties:
                target.properties[k] = v
        if len(source.canonical_name) > len(target.canonical_name):
            target.canonical_name = source.canonical_name
        return target
