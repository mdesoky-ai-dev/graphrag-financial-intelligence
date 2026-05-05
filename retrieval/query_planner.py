"""
Query planner.

Takes a user question and produces a RetrievalPlan — a structured spec
of which graph patterns to run (with what arguments) and whether to also
run vector search.

This is the *rule-based* planner. It uses keyword and entity matching
against the known schema. It's fast, deterministic, and free.

Future work could swap this for an LLM-based planner that classifies more
nuanced questions; the synthesizer takes a plan, so the swap is easy.

The planner:
  1. Detects which Company is being asked about (defaults to Apple Inc.
     since that's our only ingested filer for now).
  2. Detects category keywords (supply_chain, cybersecurity, etc.) and
     emits Pattern A.
  3. Detects geography keywords and emits Pattern B.
  4. Detects competitor / "vs" keywords and emits Pattern C/D.
  5. Always emits a vector-search step using the question text — vector
     search is cheap insurance against the planner missing the right pattern.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)


# ---- Plan dataclasses ------------------------------------------------------


class PatternId(StrEnum):
    """Mirrors the methods on GraphRetriever."""

    RISKS_FOR_COMPANY = "risks_for_company"
    RISKS_IN_GEOGRAPHY = "risks_in_geography"
    COMPETITORS_OF = "competitors_of"
    SHARED_RISKS = "shared_risks"


@dataclass
class GraphStep:
    """One graph-retrieval pattern call to make as part of the plan."""

    pattern: PatternId
    params: dict[str, object] = field(default_factory=dict)


@dataclass
class RetrievalPlan:
    """The compiled retrieval recipe for a user question.

    Even when graph_steps is empty, vector_search remains True — semantic
    retrieval is the safety net for questions whose intent we couldn't pin
    down structurally.
    """

    question: str
    graph_steps: list[GraphStep] = field(default_factory=list)
    run_vector: bool = True
    vector_top_k: int = 10
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        steps = ", ".join(f"{s.pattern.value}({s.params})" for s in self.graph_steps) or "—"
        return f"Plan(graph=[{steps}], vector=top_{self.vector_top_k}, notes={self.notes})"


# ---- Knowledge of our schema (kept narrow on purpose) ----------------------

# Risk category keywords. Maps a phrase the user might say to the canonical
# `category` value we stored on RiskFactor nodes. Order matters for the
# longest-match heuristic.
_CATEGORY_KEYWORDS: dict[str, str] = {
    "supply chain": "supply_chain",
    "cybersecurity": "cybersecurity",
    "data privacy": "regulatory",
    "regulatory": "regulatory",
    "financial": "financial",
    "operational": "operational",
    "competitive": "competitive",
    "geopolitical": "geopolitical",
    "environmental": "environmental",
    "legal": "legal",
    "reputational": "reputational",
    "technology": "technology",
    "macroeconomic": "macroeconomic",
}

# Geographies present in our ingested data. Mapping is identity for canonical
# names; the planner just checks substring against this set.
_GEOGRAPHY_KEYWORDS: list[str] = [
    "China", "India", "Japan", "South Korea", "Korea", "Taiwan", "Vietnam",
    "United States", "U.S.", "USA", "Europe", "European Union", "EU",
    "International",
]

# Competitor-question signals.
_COMPETITOR_PATTERNS = re.compile(
    r"\b(competitor|competing|rival|versus|vs\.?|share with|both)\b",
    re.IGNORECASE,
)

# Default reporting company. With multi-doc corpus this becomes detected
# from the question instead of hardcoded.
_DEFAULT_COMPANY = "Apple Inc."


# ---- Planner ---------------------------------------------------------------


class QueryPlanner:
    """Deterministic, rule-based intent classifier for our domain."""

    def __init__(
        self,
        default_company: str = _DEFAULT_COMPANY,
        vector_top_k: int = 10,
    ) -> None:
        self._default_company = default_company
        self._vector_top_k = vector_top_k

    def plan(self, question: str) -> RetrievalPlan:
        plan = RetrievalPlan(question=question, vector_top_k=self._vector_top_k)
        q_lower = question.lower()

        # 1. Identify the focal company (today: always Apple).
        company = self._detect_company(question) or self._default_company
        plan.notes.append(f"focal_company={company}")

        # 2. Risk category? -> Pattern A, parameterized by category.
        category = self._detect_category(q_lower)
        if category:
            plan.notes.append(f"category={category}")
            plan.graph_steps.append(GraphStep(
                pattern=PatternId.RISKS_FOR_COMPANY,
                params={"company": company, "category": category},
            ))

        # 3. Geography? -> Pattern B, parameterized by location.
        geography = self._detect_geography(question)
        if geography:
            plan.notes.append(f"geography={geography}")
            plan.graph_steps.append(GraphStep(
                pattern=PatternId.RISKS_IN_GEOGRAPHY,
                params={"geography": geography},
            ))

        # 4. Competitor language? -> Pattern C (and D once corpus has >1 doc).
        if _COMPETITOR_PATTERNS.search(question):
            plan.notes.append("competitor_intent")
            plan.graph_steps.append(GraphStep(
                pattern=PatternId.COMPETITORS_OF,
                params={"company": company},
            ))

        # 5. If we found nothing structural, fall back to a generic
        #    risks-for-company sweep — the question is probably a broad
        #    'what risks does Apple report?' question.
        if not plan.graph_steps:
            plan.notes.append("fallback_broad_company_sweep")
            plan.graph_steps.append(GraphStep(
                pattern=PatternId.RISKS_FOR_COMPANY,
                params={"company": company, "category": None},
            ))

        logger.info("Plan: %s", plan)
        return plan

    # ---- internal helpers ----

    def _detect_category(self, q_lower: str) -> str | None:
        # Longest-match-wins so 'supply chain' beats a hypothetical 'chain'
        # keyword.
        for phrase in sorted(_CATEGORY_KEYWORDS, key=len, reverse=True):
            if phrase in q_lower:
                return _CATEGORY_KEYWORDS[phrase]
        return None

    def _detect_geography(self, question: str) -> str | None:
        for geo in _GEOGRAPHY_KEYWORDS:
            if geo.lower() in question.lower():
                # Normalize to our canonical names where we know them.
                if geo in {"U.S.", "USA"}:
                    return "United States"
                if geo == "EU":
                    return "European Union"
                if geo == "Korea":
                    return "South Korea"
                return geo
        return None

    def _detect_company(self, question: str) -> str | None:
        # In the multi-doc future this would scan against a known-companies
        # registry. For now: whichever company is named, or None.
        for known in ("Apple Inc.", "Apple", "Microsoft", "Alphabet",
                      "Meta", "Amazon", "Google"):
            if known.lower() in question.lower():
                # Map informal names to canonical filer names if needed.
                mapping = {
                    "apple": "Apple Inc.",
                    "microsoft": "Microsoft Corporation",
                    "alphabet": "Alphabet Inc.",
                    "google": "Alphabet Inc.",
                    "meta": "Meta Platforms, Inc.",
                    "amazon": "Amazon.com, Inc.",
                }
                return mapping.get(known.lower(), known)
        return None
