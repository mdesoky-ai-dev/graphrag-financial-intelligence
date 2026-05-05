"""
LLM-based entity and relationship extractor.

Sends a text chunk to Claude on Bedrock and asks it to return structured
JSON matching ExtractionResponse. Validates with Pydantic, retries on
failure, and returns a typed object.

Usage:
    from ingestion.extractors.llm_extractor import LLMExtractor

    extractor = LLMExtractor()
    response = extractor.extract(chunk_text="Apple Inc. reported...")
    for entity in response.entities:
        print(entity.type, entity.name)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.config import Config
from pydantic import ValidationError

from app.config import settings
from ingestion.extractors.schemas import ExtractionResponse

logger = logging.getLogger(__name__)


@dataclass
class DocContext:
    """Document-level context passed into every extraction call.

    Threaded through the prompt so 'the Company' resolves to the actual
    filing entity. Avoids the v1 problem where 'The Company' was extracted
    as an entity name across docs.
    """

    company_name: str       # e.g. "Apple Inc."
    ticker: str             # e.g. "AAPL"
    fiscal_year: str        # e.g. "FY25"


# ----------------------------------------------------------------------------
# The system prompt
# ----------------------------------------------------------------------------
# Pinned in code (not config) because prompt changes are code changes — they
# should be reviewed in PRs and tracked in git history. Bump PROMPT_VERSION
# whenever the prompt changes meaningfully so we can correlate eval runs.

PROMPT_VERSION = "v2"

SYSTEM_PROMPT = """You are an expert financial analyst extracting structured entities and relationships from SEC 10-K filings.

# Your task

Read the text chunk provided by the user. Extract every entity that fits one of the seven types below, and every relationship between them that the text actually states or strongly implies. Return ONLY a single valid JSON object — no preamble, no commentary, no markdown fences.

# Document context

The user message includes a `<doc_context>` block describing the 10-K being processed (the filing company and fiscal year). When the chunk uses generic phrases like "the Company", "we", "our", "the Registrant", or "the Issuer", resolve them to the filing company named in the doc_context. NEVER emit "The Company", "the Issuer", "we", or any other placeholder as an entity name — always use the real company name from doc_context.

# Entity types

- **Company**: a public company. Use the official name as it appears in the filing (e.g. "Apple Inc.", not "Apple"). Include `ticker` in properties when known. The reporting company itself counts and should usually be present.
- **Executive**: a named officer. The text in a 10-K rarely names individual executives in Risk Factors — only extract if explicitly named. Required property: `role` (e.g. "CEO", "CFO", "Director").
- **RiskFactor**: a discrete risk, threat, or vulnerability the company reports. Be specific — "supply chain disruption in Asia" not "operational risk". Optional property: `category` (one of: supply_chain, cybersecurity, regulatory, financial, operational, competitive, geopolitical, environmental, legal, reputational, technology, macroeconomic, general).
- **Subsidiary**: a named subsidiary or business unit owned by a company.
- **Competitor**: a company named as a competitor. The same name may appear as Company in other chunks — that is fine.
- **FinancialMetric**: a numeric financial fact. Optional properties: `value`, `period`.
- **Geography**: a country, region, or jurisdiction relevant to a company's operations or risks. Use canonical English names ("China" not "the PRC"). Optional property: `kind` ("country", "region", "jurisdiction").

# Relationship types

- HAS_EXECUTIVE      Company → Executive
- REPORTS_RISK       Company → RiskFactor
- OWNS               Company → Subsidiary
- COMPETES_WITH      Company → Competitor (or another Company)
- REPORTS_METRIC     Company → FinancialMetric
- OPERATES_IN        Company → Geography
- RISK_IN            RiskFactor → Geography

# Critical rules

1. **Entity name normalization.** NEVER use generic phrases like "The Company", "the Issuer", "we", or "our" as an entity name. ALWAYS resolve to the filing company named in doc_context.

2. **Source and target of every relationship must match an entity in the same response.** If you mention an entity in a relationship, it must also appear in the `entities` list.

3. **Geography rule — no inference from contrast.** Only emit a Geography entity AND a relationship (OPERATES_IN, RISK_IN) when the text *explicitly states* operations or risks are LOCATED in that geography. Phrases like "outside the U.S.", "non-U.S. operations", "international sales" describe a CONTRAST and do NOT establish the U.S. as the location. In such cases:
   - Emit Geography "International" with kind="region" if the contrast is to "international" / "outside U.S." / "global", or
   - Omit the geography entirely if no specific location is named.
   - Do NOT emit OPERATES_IN/RISK_IN to "United States" merely because the text says "outside the U.S."

4. **Do not hallucinate.** If the chunk does not name a specific risk, do not invent one. Empty lists are correct when the chunk has no extractable content (e.g., section preambles).

5. **Risks must be discrete, specific, and re-encounterable.** Good: "Dependence on single-source component suppliers in Asia". Bad: "Various business risks".

6. **Be deduplicative within the chunk.** If the same risk is discussed in two paragraphs of the same chunk, emit it once.

# Output format

Return EXACTLY this JSON shape, with no surrounding text:

{
  "entities": [
    {"type": "Company", "name": "Apple Inc.", "properties": {"ticker": "AAPL"}},
    {"type": "RiskFactor", "name": "Single-source component supplier dependency", "properties": {"category": "supply_chain"}},
    {"type": "Geography", "name": "China", "properties": {"kind": "country"}}
  ],
  "relationships": [
    {"source": "Apple Inc.", "predicate": "REPORTS_RISK", "target": "Single-source component supplier dependency"},
    {"source": "Apple Inc.", "predicate": "OPERATES_IN", "target": "China"},
    {"source": "Single-source component supplier dependency", "predicate": "RISK_IN", "target": "China"}
  ]
}
"""


# A bedrock-friendly user prompt template
USER_PROMPT_TEMPLATE = """Extract entities and relationships from this 10-K chunk. Return only the JSON object.

<doc_context>
filing_company: {company_name}
ticker: {ticker}
fiscal_year: {fiscal_year}
</doc_context>

<chunk>
{chunk_text}
</chunk>"""


# ----------------------------------------------------------------------------
# Helper: strip markdown fences if Claude includes them despite instructions
# ----------------------------------------------------------------------------

_MARKDOWN_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _extract_json(raw: str) -> str:
    """Pull the JSON object out of a model response, stripping any wrapping."""
    text = _MARKDOWN_FENCE.sub("", raw).strip()
    # If there's leading prose before the JSON, find the first { and last }.
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1:
        return text
    return text[first : last + 1]


# ----------------------------------------------------------------------------
# Extractor
# ----------------------------------------------------------------------------


class LLMExtractor:
    """Bedrock Claude wrapper for structured entity extraction.

    Single instance owns the boto3 client and reuses it across calls.
    Construction is cheap; reuse the same instance across all chunks.
    """

    def __init__(
        self,
        model_id: str | None = None,
        region: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        max_retries: int = 4,
    ) -> None:
        self._model_id = model_id or settings.bedrock_llm_model_id
        self._region = region or settings.aws_region
        self._max_tokens = max_tokens
        self._temperature = temperature

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
        logger.info(
            "LLMExtractor ready: model=%s, region=%s, prompt_version=%s",
            self._model_id, self._region, PROMPT_VERSION,
        )

    def _invoke_claude(self, chunk_text: str, doc_context: DocContext) -> str:
        """Single Bedrock call. Returns the raw text response from Claude."""
        user_prompt = USER_PROMPT_TEMPLATE.format(
            chunk_text=chunk_text,
            company_name=doc_context.company_name,
            ticker=doc_context.ticker,
            fiscal_year=doc_context.fiscal_year,
        )
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
        })
        response = self._client.invoke_model(
            modelId=self._model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read())
        # Claude on Bedrock returns content as a list of blocks
        text_blocks = [b["text"] for b in payload["content"] if b["type"] == "text"]
        return "\n".join(text_blocks)

    def extract(
        self,
        chunk_text: str,
        doc_context: DocContext,
        max_validation_retries: int = 1,
    ) -> ExtractionResponse:
        """Extract entities and relationships from a chunk.

        On JSON or schema validation failure, retry once. If that also fails,
        return an empty response and log — we never write malformed data.
        """
        last_raw: str = ""
        for attempt in range(max_validation_retries + 1):
            try:
                last_raw = self._invoke_claude(chunk_text, doc_context)
                json_str = _extract_json(last_raw)
                data: dict[str, Any] = json.loads(json_str)
                response = ExtractionResponse.model_validate(data)
                # Sanity check: every relationship references an entity we extracted
                names = {e.name for e in response.entities}
                bad_rels = [
                    r for r in response.relationships
                    if r.source not in names or r.target not in names
                ]
                if bad_rels:
                    logger.warning(
                        "Dropping %d relationships referencing missing entities: %s",
                        len(bad_rels),
                        [(r.source, r.predicate.value, r.target) for r in bad_rels[:3]],
                    )
                    response.relationships = [
                        r for r in response.relationships
                        if r.source in names and r.target in names
                    ]
                return response
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(
                    "Extraction validation failed (attempt %d): %s. Raw response head: %r",
                    attempt + 1, type(e).__name__, last_raw[:200],
                )
                if attempt >= max_validation_retries:
                    logger.error("Giving up; returning empty extraction.")
                    return ExtractionResponse()
        return ExtractionResponse()  # unreachable but keeps mypy happy
