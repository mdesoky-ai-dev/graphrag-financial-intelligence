"""
Parameterized Cypher queries for the project.

Centralizing queries here keeps them reviewable, testable, and free of
typos that could split a relationship type into two. Every Cypher string
the project sends to Neo4j lives in this file.

All queries use parameters (`$var`) — never string interpolation of
user data — to avoid Cypher injection.
"""

from __future__ import annotations


# ----------------------------------------------------------------------------
# Schema setup
# ----------------------------------------------------------------------------
# Run once after instance creation. Idempotent.

# Each entity has a unique canonical_id; without this constraint, MERGE on
# canonical_id could create duplicates. Constraints also implicitly create
# indexes, so lookups by canonical_id are O(log n).
CONSTRAINTS = [
    "CREATE CONSTRAINT entity_canonical_id_unique IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.canonical_id IS UNIQUE",
]

# Additional indexes for queries we know we'll run.
INDEXES = [
    "CREATE INDEX entity_canonical_name IF NOT EXISTS "
    "FOR (e:Entity) ON (e.canonical_name)",
    "CREATE INDEX entity_type_idx IF NOT EXISTS "
    "FOR (e:Entity) ON (e.entity_type)",
]


# ----------------------------------------------------------------------------
# Writes (used by ingestion)
# ----------------------------------------------------------------------------

# Upsert a node. MERGE finds-or-creates by canonical_id, then SET fills in
# the rest. The double label `Entity:<Type>` is set inline.
#
# Parameters:
#   $canonical_id, $canonical_name, $entity_type, $properties (dict),
#   $aliases (list), $source_chunks (list), $doc_id
#
# Note the dynamic label — we use a label parameter, which Neo4j 5+ supports
# via the `$$label` syntax in some contexts. For safety + simplicity we
# instead format the label into the query at python-call time, since the
# label always comes from a closed enum (NodeLabel) and is never user input.
def upsert_entity(node_label: str) -> str:
    return f"""
        MERGE (e:Entity {{canonical_id: $canonical_id}})
        SET e:{node_label}
        SET e.canonical_name = $canonical_name,
            e.entity_type    = $entity_type,
            e.aliases        = $aliases,
            e.source_chunks  = $source_chunks,
            e.doc_id         = $doc_id
        WITH e, $properties AS props
        SET e += props
        RETURN e.canonical_id AS id
    """


# Upsert an edge. Both endpoints must already exist (Pass 1 created them).
# MERGE on the relationship pattern finds-or-creates the edge.
# We store source_chunks as a property on the edge — that's how an answer
# gets its citations later.
#
# Parameters:
#   $source_id, $target_id, $source_chunks (list)
def upsert_relationship(rel_type: str) -> str:
    return f"""
        MATCH (s:Entity {{canonical_id: $source_id}})
        MATCH (t:Entity {{canonical_id: $target_id}})
        MERGE (s)-[r:{rel_type}]->(t)
        SET r.source_chunks = $source_chunks,
            r.predicate     = $predicate
        RETURN type(r) AS predicate
    """


# ----------------------------------------------------------------------------
# Reads (used later by retrieval; included now for completeness)
# ----------------------------------------------------------------------------

# Count nodes per type. Useful for post-ingestion validation.
COUNT_NODES_BY_TYPE = """
    MATCH (e:Entity)
    RETURN e.entity_type AS type, count(*) AS n
    ORDER BY n DESC
"""

# Count edges per type.
COUNT_RELS_BY_TYPE = """
    MATCH ()-[r]->()
    RETURN type(r) AS predicate, count(*) AS n
    ORDER BY n DESC
"""

# Drop everything. Local development only.
DELETE_ALL = "MATCH (n) DETACH DELETE n"
