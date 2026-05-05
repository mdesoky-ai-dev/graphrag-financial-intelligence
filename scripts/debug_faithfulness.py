"""
Diagnose why Ragas faithfulness returns NaN.

Runs faithfulness on a single hand-crafted row with verbose logging and
exception capture so we can see what's actually failing inside Ragas.

Usage:
    python -m scripts.debug_faithfulness

Cost: ~$0.05 per run, ~30 seconds.
"""

from __future__ import annotations

import logging
import sys
import traceback

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness

from app.config import settings
from evaluation.ragas_runner import _build_judge_embeddings, _build_judge_llm

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("debug_faithfulness")


# A small, hand-crafted row that we know is well-formed and grounded.
# If faithfulness can't score this, the bug is in the wiring, not our data.
TEST_ROW = {
    "question": ["What does Apple say about supply chain risks?"],
    "answer": [
        "Apple's supply chain is concentrated in Asia, particularly China, "
        "India, Japan, South Korea, Taiwan, and Vietnam. Apple relies on "
        "single or limited sources for many critical components."
    ],
    "contexts": [[
        # Context 1
        "A significant majority of the Company's manufacturing is performed "
        "in whole or in part by outsourcing partners located primarily in "
        "China mainland, India, Japan, South Korea, Taiwan and Vietnam.",
        # Context 2
        "Because the Company currently obtains certain components from single "
        "or limited sources, the Company is subject to significant supply and "
        "pricing risks.",
    ]],
    "ground_truth": [
        "Apple's supply chain risks include concentration of manufacturing in "
        "Asia and reliance on single or limited sources for critical components."
    ],
}


def main() -> int:
    log.info("Building judge LLM and embeddings ...")
    judge_llm = _build_judge_llm()
    judge_embeds = _build_judge_embeddings()
    log.info("Using LLM model_id=%s, region=%s",
             settings.bedrock_llm_model_id, settings.aws_region)

    dataset = Dataset.from_dict(TEST_ROW)
    log.info("Dataset built: %d row(s)", len(dataset))

    log.info("Running Ragas faithfulness on 1 row ...")
    try:
        scores = evaluate(
            dataset=dataset,
            metrics=[faithfulness],
            llm=judge_llm,
            embeddings=judge_embeds,
            raise_exceptions=True,   # ← surface Ragas's underlying error
        )
        df = scores.to_pandas()
        log.info("Result columns: %s", list(df.columns))
        log.info("Faithfulness: %s", df["faithfulness"].iloc[0])
    except Exception as e:
        log.error("Ragas raised: %s", type(e).__name__)
        log.error("Message: %s", str(e))
        log.error("Traceback:\n%s", traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
