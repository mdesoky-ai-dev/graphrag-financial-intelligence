"""
Ground-truth evaluation dataset.

Ten questions covering the retrieval patterns our system handles, with
reference answers and expected source chunks. This dataset is the input
to the Ragas evaluation runner.

Distribution:
    Easy   (4): single-pattern questions, clear answers expected
    Medium (4): geography + category combinations
    Hard   (2): narrow specific topics + "sparse signal" cases

Reference answers are drawn from Apple's 10-K FY25 Risk Factors section.
They don't have to be word-perfect — Ragas uses semantic similarity, not
string match — but they should accurately reflect the source material.

Expected chunks are the chunk_ids that a well-functioning retriever should
surface for each question. They're used by Ragas to compute context_precision.
We choose 2-4 chunks per question; not all chunks need to be retrieved, but
the expected ones should appear in the top-K returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TestQuestion:
    """One question in the evaluation set."""

    qid: str
    difficulty: str            # "easy" | "medium" | "hard"
    question: str
    ground_truth: str          # Reference answer
    expected_chunks: list[str] = field(default_factory=list)
    notes: str = ""            # Free-text reminder of what we're testing


# ----------------------------------------------------------------------------
# The 10 questions
# ----------------------------------------------------------------------------


TEST_QUESTIONS: list[TestQuestion] = [
    # ---- EASY: single-pattern, category filter ----
    TestQuestion(
        qid="q01_supply_chain",
        difficulty="easy",
        question="What does Apple say about supply chain risks?",
        ground_truth=(
            "Apple's supply chain risks center on its concentration of manufacturing "
            "in Asia, particularly China, India, Japan, South Korea, Taiwan, and "
            "Vietnam, and its reliance on single or limited sources for many critical "
            "components. The company faces risks of supply shortages, commodity price "
            "fluctuations, and disruptions from natural disasters, trade restrictions, "
            "and geopolitical events. Changes or additions to the supply chain require "
            "considerable time and resources and introduce additional regulatory and "
            "operational risks."
        ),
        expected_chunks=[
            "apple-10k-fy25_chunk_0006",  # Asian manufacturing concentration
            "apple-10k-fy25_chunk_0026",  # Single-source / supply shortages
            "apple-10k-fy25_chunk_0014",  # Single/limited sources for critical components
        ],
        notes="Tests Pattern A (risks_for_company) with category=supply_chain.",
    ),
    TestQuestion(
        qid="q02_competitive",
        difficulty="easy",
        question="What competitive risks does Apple face?",
        ground_truth=(
            "Apple operates in highly competitive global markets characterized by "
            "aggressive price competition, rapid technological change, and frequent "
            "product introductions. Competitors imitate Apple's product features and "
            "intellectual property, and several have resources to provide products at "
            "little or no profit. Apple has minority market share in global smartphone, "
            "personal computer, tablet, and wearables markets, which can affect "
            "third-party developer support."
        ),
        expected_chunks=[],  # Will be auto-filled at runtime if empty
        notes="Tests Pattern A with category=competitive.",
    ),
    TestQuestion(
        qid="q03_cybersecurity",
        difficulty="easy",
        question="What cybersecurity risks does Apple report?",
        ground_truth=(
            "Apple faces cybersecurity risks including ransomware and malicious attacks "
            "on its global supplier network that could disrupt business operations. The "
            "company's products and services rely on complex software and infrastructure "
            "that could be targeted by attackers, and errors or vulnerabilities can be "
            "exploited to compromise the safety and security of user devices."
        ),
        expected_chunks=[],
        notes="Tests Pattern A with category=cybersecurity.",
    ),
    TestQuestion(
        qid="q04_financial",
        difficulty="easy",
        question="What are Apple's main financial risks?",
        ground_truth=(
            "Apple's financial risks include credit and collectibility risk on trade "
            "receivables, the failure of derivative counterparties and financial "
            "institutions, reduced liquidity and limitations on its ability to issue "
            "new debt, and declines in the fair value of financial instruments. Adverse "
            "macroeconomic conditions can amplify each of these."
        ),
        expected_chunks=[],
        notes="Tests Pattern A with category=financial.",
    ),

    # ---- MEDIUM: geography + multi-pattern ----
    TestQuestion(
        qid="q05_china",
        difficulty="medium",
        question="What risks does Apple face related to China?",
        ground_truth=(
            "Apple's risks related to China include heavy concentration of manufacturing "
            "and assembly in China mainland (alongside India, Japan, South Korea, Taiwan, "
            "and Vietnam), exposure to U.S. tariffs on Chinese imports announced in 2025, "
            "and broader geopolitical and trade-dispute risks. Trade restrictions and "
            "retaliatory measures could materially affect costs, component availability, "
            "and supply chain operations."
        ),
        expected_chunks=[],
        notes="Tests Pattern B (geography filter on China) plus Pattern A overlap.",
    ),
    TestQuestion(
        qid="q06_asian_manufacturing",
        difficulty="medium",
        question="How does Apple's reliance on Asian manufacturing affect its business?",
        ground_truth=(
            "A significant majority of Apple's manufacturing is performed by outsourcing "
            "partners in China mainland, India, Japan, South Korea, Taiwan, and Vietnam, "
            "with final assembly of substantially all hardware concentrated in Asia. This "
            "concentration creates vulnerability to supply disruption, geopolitical "
            "tensions, trade restrictions, and natural disasters. Apple cannot easily "
            "substitute alternatives and has reduced direct control over production and "
            "distribution due to outsourcing."
        ),
        expected_chunks=[],
        notes="Tests vector retrieval primarily — phrasing doesn't match a category neatly.",
    ),
    TestQuestion(
        qid="q07_macroeconomic",
        difficulty="medium",
        question="What macroeconomic risks does Apple identify?",
        ground_truth=(
            "Apple identifies macroeconomic risks including slow growth or recession, "
            "high unemployment, inflation, tighter credit, higher interest rates, and "
            "currency fluctuations. These conditions can adversely affect consumer "
            "confidence and spending, demand for Apple's products and services, and the "
            "financial stability of suppliers, contract manufacturers, distributors, and "
            "channel partners."
        ),
        expected_chunks=[],
        notes="Tests Pattern A with category=macroeconomic.",
    ),
    TestQuestion(
        qid="q08_regulatory",
        difficulty="medium",
        question="What regulatory risks does Apple face?",
        ground_truth=(
            "Apple faces regulatory risks including non-compliance with privacy and "
            "data protection laws, government investigations, and restrictions on "
            "international trade such as tariffs and export controls. Regulatory "
            "requirements can force Apple to withdraw or modify products in certain "
            "countries, share innovations with competitors, or pay significant penalties."
        ),
        expected_chunks=[],
        notes="Tests Pattern A with category=regulatory.",
    ),

    # ---- HARD: narrow + sparse-signal tests ----
    TestQuestion(
        qid="q09_third_party_ip",
        difficulty="hard",
        question="What does Apple say about its dependence on third-party intellectual property?",
        ground_truth=(
            "Apple's products and services include technology or intellectual property "
            "that must be licensed from third parties. The company is not always able to "
            "obtain necessary licenses on commercially reasonable terms, which could "
            "require Apple to modify products, preclude it from selling certain products, "
            "or expose it to significant licensing costs. Risk is increased by the use of "
            "machine learning and artificial intelligence."
        ),
        expected_chunks=[],
        notes="Tests narrow specific topic — should rely heavily on vector retrieval.",
    ),
    TestQuestion(
        qid="q10_executives",
        difficulty="hard",
        question="Does Apple report any executive-related risks?",
        ground_truth=(
            "Apple's success depends in part on the talents and continued service of "
            "key personnel including its Chief Executive Officer, executive team, and "
            "highly skilled employees. Loss of key personnel or inability to recruit and "
            "retain talented employees, particularly in Silicon Valley where most key "
            "personnel are located, could materially affect the company. The 10-K does "
            "not name individual executives within the Risk Factors section beyond "
            "referring to the CEO."
        ),
        expected_chunks=[],
        notes="Tests sparse signal — only 1 Executive node in the graph. Faithfulness should remain high.",
    ),
]


def get_questions(difficulty: str | None = None) -> list[TestQuestion]:
    """Return all questions, or filter by difficulty."""
    if difficulty is None:
        return TEST_QUESTIONS
    return [q for q in TEST_QUESTIONS if q.difficulty == difficulty]
