"""
Pydantic schema for LLM entity and relationship extraction.

This is the contract Claude must fulfill when extracting structured data
from a chunk. Pydantic validation enforces it: malformed JSON or wrong
field types fail loudly and we retry rather than silently writing junk
to the graph.

Entity types (7 total):
    Company         — public companies, named subsidiaries' parents
    Executive       — CEO, CFO, named officers (with role + employer)
    RiskFactor      — risks, threats, vulnerabilities reported by a company
    Subsidiary      — subsidiaries / business units owned by a company
    Competitor      — companies named as competitors (often also Companies)
    FinancialMetric — revenue, margins, costs (with value + period if given)
    Geography       — countries, regions, jurisdictions (relevant to a risk)

Relationship types:
    HAS_EXECUTIVE     Company -> Executive
    REPORTS_RISK      Company -> RiskFactor
    OWNS              Company -> Subsidiary
    COMPETES_WITH     Company -> Competitor (or Company)
    REPORTS_METRIC    Company -> FinancialMetric
    OPERATES_IN       Company -> Geography
    RISK_IN           RiskFactor -> Geography
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ----------------------------------------------------------------------------
# Enums for type safety
# ----------------------------------------------------------------------------


class EntityType(str, Enum):
    COMPANY = "Company"
    EXECUTIVE = "Executive"
    RISK_FACTOR = "RiskFactor"
    SUBSIDIARY = "Subsidiary"
    COMPETITOR = "Competitor"
    FINANCIAL_METRIC = "FinancialMetric"
    GEOGRAPHY = "Geography"


class RelationshipType(str, Enum):
    HAS_EXECUTIVE = "HAS_EXECUTIVE"
    REPORTS_RISK = "REPORTS_RISK"
    OWNS = "OWNS"
    COMPETES_WITH = "COMPETES_WITH"
    REPORTS_METRIC = "REPORTS_METRIC"
    OPERATES_IN = "OPERATES_IN"
    RISK_IN = "RISK_IN"


# ----------------------------------------------------------------------------
# Entity model
# ----------------------------------------------------------------------------


class Entity(BaseModel):
    """A single named entity extracted from a chunk.

    `name` is the canonical surface form as it appears in the text. Entity
    resolution (next milestone) will normalize across chunks — for now we
    just capture what the LLM saw.

    `properties` is a small free-form dict for type-specific attributes:
        Company:         {ticker?: str}
        Executive:       {role: str}      e.g. "CEO", "CFO"
        RiskFactor:      {category?: str} e.g. "supply_chain", "regulatory"
        FinancialMetric: {value?: str, period?: str}
        Geography:       {kind?: str}     e.g. "country", "region"
    """

    type: EntityType
    name: str = Field(min_length=2, max_length=200)
    properties: dict[str, str] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


# ----------------------------------------------------------------------------
# Relationship model
# ----------------------------------------------------------------------------


class Relationship(BaseModel):
    """A directed relationship from one entity to another.

    `source` and `target` reference entities by their `name` (the same
    surface form used in the entities list of the same response).
    Cross-chunk relationships are formed at graph-write time, not here.
    """

    source: str = Field(min_length=2)
    predicate: RelationshipType
    target: str = Field(min_length=2)


# ----------------------------------------------------------------------------
# Top-level response — what Claude must return
# ----------------------------------------------------------------------------


class ExtractionResponse(BaseModel):
    """The full structured response Claude returns for a single chunk.

    Empty lists are fine — many chunks contain no extractable entities
    (boilerplate, transitional sentences). Better to extract nothing than
    to hallucinate.
    """

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)

    def summary(self) -> str:
        """Short string for logging."""
        by_type: dict[str, int] = {}
        for e in self.entities:
            by_type[e.type.value] = by_type.get(e.type.value, 0) + 1
        type_summary = ", ".join(f"{k}={v}" for k, v in sorted(by_type.items()))
        return f"{len(self.entities)} entities ({type_summary}), {len(self.relationships)} relationships"
