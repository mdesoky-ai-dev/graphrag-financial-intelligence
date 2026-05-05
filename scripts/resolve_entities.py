"""
Resolve entities from a previous extraction run.

Run:
    python -m scripts.resolve_entities data/processed/apple-10k-fy25.extraction.json

Reads an extraction.json (the output of inspect_extraction.py), runs the
three-layer resolution cascade (normalize -> embed -> LLM adjudicate),
and writes a resolved.json with:

  - canonical_entities: deduplicated entities with merged aliases + chunks
  - resolved_relationships: relationships rewritten to use canonical IDs

Also prints a comparison table so you can see what merged.

Cost guideline (Apple's 102-chunk extraction with 427 entities):
  Layer 1 (slug) merges:           free
  Layer 2 embeddings:              ~$0.02 (one Titan call per unique slug)
  Layer 3 adjudications:           ~$0.20 (capped at 100 calls)
  Total expected:                  < $0.25
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.config import settings
from ingestion.extractors.schemas import Entity
from ingestion.resolvers.entity_resolver import EntityResolver

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("resolve_entities")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("extraction", type=Path,
                        help="Path to extraction.json")
    parser.add_argument("--max-adjudications", type=int, default=100,
                        help="Cap LLM adjudication calls (cost control)")
    args = parser.parse_args()

    # ---- Load extraction.json ----
    if not args.extraction.exists():
        log.error("File not found: %s", args.extraction)
        return 1
    with args.extraction.open() as f:
        data = json.load(f)
    log.info("Loaded %s: %d chunks, %d entities, %d relationships",
             args.extraction.name,
             data["chunks_processed"],
             data["total_entities"],
             data["total_relationships"])

    # ---- Feed entities into the resolver ----
    resolver = EntityResolver(adjudication_max_calls=args.max_adjudications)
    for chunk_result in data["results"]:
        chunk_id = chunk_result["chunk_id"]
        for ent_dict in chunk_result["entities"]:
            try:
                entity = Entity.model_validate(ent_dict)
            except Exception as e:
                log.warning("Skipping bad entity in %s: %s", chunk_id, e)
                continue
            resolver.add_entity(entity, source_chunk_id=chunk_id)

    # ---- Run resolution ----
    canonicals = resolver.resolve()

    # ---- Build name -> canonical_id index for relationship rewriting ----
    # An alias may map to multiple canonicals if the same surface form ever
    # crossed types (rare). We index by (type, name) -> canonical_id.
    alias_index: dict[tuple[str, str], str] = {}
    for c in canonicals:
        for alias in c.aliases:
            alias_index[(c.type, alias)] = c.canonical_id

    # ---- Rewrite relationships to use canonical IDs ----
    resolved_rels: list[dict[str, str]] = []
    unresolved_rel_count = 0
    type_by_name: dict[str, set[str]] = {}
    for c in canonicals:
        for alias in c.aliases:
            type_by_name.setdefault(alias, set()).add(c.type)

    def lookup_canonical_id(name: str) -> str | None:
        # Try a strict (type, name) lookup but we don't know the type from
        # the relationship side. Fall back to scanning by alias.
        types = type_by_name.get(name, set())
        if len(types) == 1:
            return alias_index.get((next(iter(types)), name))
        # If 0 types, the name was never seen as an entity; if >1, ambiguous.
        return None

    for chunk_result in data["results"]:
        for rel in chunk_result["relationships"]:
            src_id = lookup_canonical_id(rel["source"])
            tgt_id = lookup_canonical_id(rel["target"])
            if not src_id or not tgt_id:
                unresolved_rel_count += 1
                continue
            resolved_rels.append({
                "source_id": src_id,
                "predicate": rel["predicate"],
                "target_id": tgt_id,
                "source_chunk": chunk_result["chunk_id"],
            })

    # Deduplicate identical relationships from different chunks but keep
    # source_chunk provenance: we collapse on (source_id, predicate, target_id).
    by_triple: dict[tuple[str, str, str], dict[str, str | list[str]]] = {}
    for r in resolved_rels:
        key = (r["source_id"], r["predicate"], r["target_id"])  # type: ignore[arg-type]
        if key not in by_triple:
            by_triple[key] = {
                "source_id": r["source_id"],
                "predicate": r["predicate"],
                "target_id": r["target_id"],
                "source_chunks": [],
            }
        by_triple[key]["source_chunks"].append(r["source_chunk"])  # type: ignore[union-attr]
    deduped_rels = list(by_triple.values())

    # ---- Summary ----
    log.info("==== Resolution summary ====")
    log.info("Input entities:          %d", data["total_entities"])
    log.info("Canonical entities:      %d", len(canonicals))
    log.info("Reduction:               %.1f%%",
             (1 - len(canonicals) / data["total_entities"]) * 100)
    log.info("Input relationships:     %d", data["total_relationships"])
    log.info("Unresolved (dropped):    %d", unresolved_rel_count)
    log.info("Deduped relationships:   %d", len(deduped_rels))
    by_type: dict[str, int] = {}
    for c in canonicals:
        by_type[c.type] = by_type.get(c.type, 0) + 1
    log.info("Canonicals by type:      %s",
             ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))

    # Sample biggest merges (most aliases) so the user can eyeball quality
    biggest = sorted(canonicals, key=lambda c: -len(c.aliases))[:5]
    log.info("==== Top 5 biggest merges ====")
    for c in biggest:
        if len(c.aliases) > 1:
            log.info("[%s] %s  (%d aliases, %d chunks)",
                     c.type, c.canonical_name, len(c.aliases), len(c.source_chunks))
            for alias in sorted(c.aliases):
                if alias != c.canonical_name:
                    log.info("    alias: %s", alias[:120])

    # ---- Dump ----
    out_path = args.extraction.parent / args.extraction.name.replace(".extraction.", ".resolved.")
    out_path.write_text(json.dumps({
        "doc_id": data["doc_id"],
        "doc_context": data.get("doc_context"),
        "source_extraction": args.extraction.name,
        "prompt_version": data.get("prompt_version"),
        "merge_threshold": settings.er_merge_threshold,
        "reject_threshold": settings.er_reject_threshold,
        "stats": {
            "input_entities": data["total_entities"],
            "canonical_entities": len(canonicals),
            "input_relationships": data["total_relationships"],
            "deduped_relationships": len(deduped_rels),
            "unresolved_relationships": unresolved_rel_count,
        },
        "canonical_entities": [c.to_dict() for c in canonicals],
        "resolved_relationships": deduped_rels,
    }, indent=2), encoding="utf-8")
    log.info("Wrote resolved entities/relationships to %s (%d KB)",
             out_path, out_path.stat().st_size // 1024)
    return 0


if __name__ == "__main__":
    sys.exit(main())
