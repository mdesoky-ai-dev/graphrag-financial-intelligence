"""
Ragas evaluation runner.

Bridges our Synthesizer to the Ragas framework so we can score the system
on faithfulness, answer relevancy, and context precision against the
ground-truth dataset.

Pipeline per question:
  1. Run Synthesizer.answer(question) -> get answer + retrieved chunks
  2. Build a Ragas EvaluationDataset row: (question, answer, contexts, ground_truth)
  3. After all rows are built, evaluate with Bedrock LLMs as the judge

Why Bedrock for the judge: keeps everything inside one cloud, avoids needing
a separate OpenAI key. Ragas accepts any LangChain-compatible LLM via its
LangchainLLMWrapper, and langchain-aws ships ChatBedrock + BedrockEmbeddings.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset
from botocore.config import Config as BotoConfig
from langchain_aws import BedrockEmbeddings, ChatBedrock
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    faithfulness,
)
from ragas.run_config import RunConfig

from app.config import settings
from evaluation.test_questions import TestQuestion
from retrieval.synthesizer import Synthesizer

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Per-question eval result, with all signals retained for the report."""

    qid: str
    difficulty: str
    question: str
    ground_truth: str
    answer: str
    retrieved_chunk_ids: list[str]
    elapsed_seconds: float
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalSummary:
    """Run-level summary: aggregate scores plus the per-question results."""

    n_questions: int
    avg_scores: dict[str, float]
    avg_elapsed: float
    total_elapsed: float
    per_question: list[EvalResult] = field(default_factory=list)


def _build_judge_llm() -> LangchainLLMWrapper:
    """Bedrock-Claude wrapped as a Ragas-compatible judge."""
    # Boto3 retry policy: Ragas fires many parallel calls and Bedrock
    # on-demand throughput is limited. Adaptive mode uses long exponential
    # backoffs so transient ThrottlingExceptions resolve themselves.
    boto_config = BotoConfig(
        region_name=settings.aws_region,
        retries={"max_attempts": 10, "mode": "adaptive"},
    )
    chat = ChatBedrock(
        model_id=settings.bedrock_llm_model_id,
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id.get_secret_value(),
        aws_secret_access_key=settings.aws_secret_access_key.get_secret_value(),
        # max_tokens=4096: faithfulness can produce 10-20 verdict objects per
        # answer (one per atomic claim, each with statement + reason + verdict).
        # 1024 tokens is enough for short test answers but truncates real ones,
        # causing LLMDidNotFinishException. 4096 has comfortable headroom.
        model_kwargs={"temperature": 0.0, "max_tokens": 8192},
        config=boto_config,
    )
    return LangchainLLMWrapper(chat)


def _build_judge_embeddings() -> LangchainEmbeddingsWrapper:
    """Bedrock-Titan embeddings wrapped for Ragas."""
    embeds = BedrockEmbeddings(
        model_id=settings.bedrock_embedding_model_id,
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id.get_secret_value(),
        aws_secret_access_key=settings.aws_secret_access_key.get_secret_value(),
    )
    return LangchainEmbeddingsWrapper(embeds)


def run_eval(
    questions: list[TestQuestion],
    synthesizer: Synthesizer | None = None,
) -> EvalSummary:
    """Run the full evaluation: retrieve answers, then score with Ragas."""
    synthesizer = synthesizer or Synthesizer()
    overall_t0 = time.time()

    # ---- Step 1: produce answers + contexts for every question ----
    rows: dict[str, list[Any]] = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    per_question: list[EvalResult] = []

    for q in questions:
        logger.info("[%s] %s", q.qid, q.question)
        result = synthesizer.answer(q.question)

        # Build the contexts list — one entry per retrieved chunk's text.
        # Ragas expects strings; we use the chunk excerpts the synthesizer fed
        # to Claude.
        contexts: list[str] = []
        # Re-fetch the chunks to get their text (synthesizer doesn't surface it)
        if result.cited_chunk_ids:
            chunk_texts = synthesizer._vector.fetch(result.cited_chunk_ids)
            contexts = [
                chunk_texts[cid].text_excerpt
                for cid in result.cited_chunk_ids
                if cid in chunk_texts
            ]

        rows["question"].append(q.question)
        rows["answer"].append(result.text)
        rows["contexts"].append(contexts)
        rows["ground_truth"].append(q.ground_truth)

        per_question.append(EvalResult(
            qid=q.qid,
            difficulty=q.difficulty,
            question=q.question,
            ground_truth=q.ground_truth,
            answer=result.text,
            retrieved_chunk_ids=list(result.cited_chunk_ids),
            elapsed_seconds=result.elapsed_seconds,
        ))

    # ---- Step 2: build Hugging Face Dataset and run Ragas ----
    logger.info("Built %d rows for Ragas evaluation. Running judge LLM ...",
                len(rows["question"]))
    dataset = Dataset.from_dict(rows)
    judge_llm = _build_judge_llm()
    judge_embeds = _build_judge_embeddings()

    metrics = [faithfulness, answer_relevancy, context_precision]

    # Cap concurrency hard. Ragas defaults to 16 parallel workers, which blows
    # past Bedrock on-demand quota and triggers ThrottlingException retries.
    # 4 workers + long timeouts is the safest combination on Bedrock.
    run_config = RunConfig(
        max_workers=4,
        max_retries=10,
        max_wait=60,        # seconds between retries
        timeout=180,        # per-call timeout
    )

    scores = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeds,
        run_config=run_config,
        raise_exceptions=True,   # surface errors instead of silently returning 0.0
    )

    # ---- Step 3: attach per-question scores to results ----
    # Ragas returns per-row scores in the order rows were submitted.
    score_df = scores.to_pandas()
    metric_columns = [m.name for m in metrics]
    for i, row in enumerate(per_question):
        for col in metric_columns:
            if col in score_df.columns:
                val = score_df.iloc[i][col]
                # NaN → 0.0 for safety; happens when judge can't decide
                row.scores[col] = float(val) if val == val else 0.0

    avg_scores = {
        col: float(score_df[col].mean()) if col in score_df.columns else 0.0
        for col in metric_columns
    }

    total = time.time() - overall_t0
    summary = EvalSummary(
        n_questions=len(questions),
        avg_scores=avg_scores,
        avg_elapsed=sum(r.elapsed_seconds for r in per_question) / max(1, len(per_question)),
        total_elapsed=total,
        per_question=per_question,
    )
    logger.info("Eval complete in %.1fs (avg per question: %.1fs)",
                total, summary.avg_elapsed)
    logger.info("Average scores: %s", avg_scores)
    return summary
