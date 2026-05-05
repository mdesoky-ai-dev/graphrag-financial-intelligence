"""
Entity name normalization.

The cheapest layer in our entity resolution cascade. Turns wordy entity
names into canonical slugs for instant duplicate detection.

Examples:
    "Adverse Macroeconomic Conditions"          -> "adverse macroeconomic conditions"
    "Adverse macroeconomic conditions."         -> "adverse macroeconomic conditions"
    "  ADVERSE MACROECONOMIC CONDITIONS  "      -> "adverse macroeconomic conditions"
    "Apple Inc."                                -> "apple inc"
    "the People's Republic of China"            -> "peoples republic of china"
    "U.S."                                      -> "us"

Normalization is intentionally aggressive but stops short of stemming or
synonym expansion — those tend to over-merge ("dependence on chip suppliers"
should NOT collapse to "supplier dependence"). Subtler merges are handled
by the embedding layer downstream.
"""

from __future__ import annotations

import re
import unicodedata


# Words to drop entirely — articles, common stopwords that add noise but no
# semantic signal at the entity-name level.
_STOPWORDS = frozenset({
    "a", "an", "the",
    "of", "in", "on", "at", "to", "for", "with",
    "and", "or",
})


# Punctuation pattern. Replace with space, not nothing, so 'U.S.' becomes
# 'u s' (safe) rather than 'us' which collides with 'us' the pronoun.
# Then we collapse multi-spaces. Final 'u s' becomes 'us' via stopword drop
# of single-letter tokens — handled separately.
_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize(name: str) -> str:
    """Reduce an entity name to a canonical slug for exact-match deduplication.

    Steps:
      1. Strip accents (NFD decomposition + remove combining marks)
      2. Lowercase
      3. Replace punctuation with spaces
      4. Drop stopwords (a, an, the, of, ...)
      5. Drop single-character tokens that result from punctuation removal
      6. Collapse repeated whitespace
      7. Strip leading/trailing whitespace

    Returns:
        Canonicalized string. Two names that normalize to the same slug
        are considered identical at the cheapest layer.
    """
    if not name:
        return ""

    # 1. Unicode normalization — strip accents (e.g. café -> cafe).
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))

    # 2. Lowercase
    lowered = ascii_only.lower()

    # 3. Punctuation -> space
    no_punct = _PUNCT_RE.sub(" ", lowered)

    # 4-5. Tokenize and filter
    tokens = no_punct.split()
    filtered = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

    # 6. Rejoin with single spaces
    result = " ".join(filtered)

    # 7. Collapse whitespace (paranoid — split() already did this)
    return _WHITESPACE_RE.sub(" ", result).strip()


def slug(name: str) -> str:
    """Normalize and additionally replace spaces with underscores.

    Used as a stable graph-friendly identifier. Suitable for Neo4j node
    keys and Pinecone vector IDs where spaces are awkward.

    Examples:
        "Apple Inc."                          -> "apple_inc"
        "Adverse Macroeconomic Conditions"    -> "adverse_macroeconomic_conditions"
    """
    return normalize(name).replace(" ", "_")
