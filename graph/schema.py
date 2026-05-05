"""
Neo4j schema constants.

All node labels, relationship types, and property keys used in the graph
are defined here as module-level constants. This is the single source of
truth — every other module imports from here so we never have a typo
splitting `COMPETES_WITH` and `COMPETES-WITH` into two different edge types.

Mirror of the Pydantic enums in ingestion/extractors/schemas.py, but
expressed as strings for use in Cypher queries.

Usage:
    from graph.schema import NodeLabel, RelType, PropKey
    cypher = f"MATCH (c:{NodeLabel.COMPANY}) RETURN c"
"""

from __future__ import annotations

from enum import StrEnum


# ----------------------------------------------------------------------------
# Node labels
# ----------------------------------------------------------------------------
# Every entity in the graph carries TWO labels: a generic `Entity` label
# (lets queries operate over all entities at once) and a type-specific label
# (e.g. Company). This dual labeling makes both broad queries
# (`MATCH (e:Entity)`) and narrow queries (`MATCH (c:Company)`) cheap.


class NodeLabel(StrEnum):
    ENTITY = "Entity"  # universal label
    COMPANY = "Company"
    EXECUTIVE = "Executive"
    RISK_FACTOR = "RiskFactor"
    SUBSIDIARY = "Subsidiary"
    COMPETITOR = "Competitor"
    FINANCIAL_METRIC = "FinancialMetric"
    GEOGRAPHY = "Geography"


# Map from EntityType (the Pydantic enum value) to the matching NodeLabel.
# The keys here are the string values of EntityType members.
ENTITY_TYPE_TO_LABEL: dict[str, NodeLabel] = {
    "Company": NodeLabel.COMPANY,
    "Executive": NodeLabel.EXECUTIVE,
    "RiskFactor": NodeLabel.RISK_FACTOR,
    "Subsidiary": NodeLabel.SUBSIDIARY,
    "Competitor": NodeLabel.COMPETITOR,
    "FinancialMetric": NodeLabel.FINANCIAL_METRIC,
    "Geography": NodeLabel.GEOGRAPHY,
}


# ----------------------------------------------------------------------------
# Relationship types
# ----------------------------------------------------------------------------


class RelType(StrEnum):
    HAS_EXECUTIVE = "HAS_EXECUTIVE"
    REPORTS_RISK = "REPORTS_RISK"
    OWNS = "OWNS"
    COMPETES_WITH = "COMPETES_WITH"
    REPORTS_METRIC = "REPORTS_METRIC"
    OPERATES_IN = "OPERATES_IN"
    RISK_IN = "RISK_IN"


# ----------------------------------------------------------------------------
# Property keys
# ----------------------------------------------------------------------------
# Centralized so refactors stay safe. Any time we set or read a property,
# we use these constants rather than raw strings.


class PropKey(StrEnum):
    # Node properties
    CANONICAL_ID = "canonical_id"          # primary key
    CANONICAL_NAME = "canonical_name"
    ENTITY_TYPE = "entity_type"
    ALIASES = "aliases"
    SOURCE_CHUNKS = "source_chunks"

    # Type-specific (stored as flat properties on the typed nodes)
    TICKER = "ticker"
    ROLE = "role"
    CATEGORY = "category"
    KIND = "kind"
    VALUE = "value"
    PERIOD = "period"
    JURISDICTION = "jurisdiction"

    # Document-level provenance (also on nodes)
    DOC_ID = "doc_id"

    # Edge properties
    PREDICATE = "predicate"                # redundant with rel-type but useful for queries
