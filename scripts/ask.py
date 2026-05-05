"""
Ask a question against the ingested knowledge graph + vector index.

Usage:
    python -m scripts.ask "What does Apple say about supply chain risks?"
    python -m scripts.ask "What risks does Apple face in China?"
    python -m scripts.ask "What are Apple's main competitive risks?"

What it does:
  1. Builds a retrieval plan from the question
  2. Runs the planned graph patterns
  3. Runs vector search
  4. Fuses results with reciprocal rank fusion
  5. Fetches the top chunks from Pinecone
  6. Asks Claude to synthesize a grounded, cited answer
  7. Prints the answer plus diagnostics
"""

from __future__ import annotations

import logging
import sys

from app.config import settings
from retrieval.synthesizer import Synthesizer

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("ask")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print('Usage: python -m scripts.ask "<your question>"')
        return 2
    question = " ".join(argv[1:]).strip()
    log.info("Q: %s", question)

    synthesizer = Synthesizer()
    try:
        answer = synthesizer.answer(question)
    finally:
        synthesizer.close()

    log.info("==== Plan ====")
    log.info("%s", answer.plan)
    log.info("==== Retrieval ====")
    log.info("Graph hits: %d", len(answer.graph_hits))
    log.info("Vector hits: %d", len(answer.vector_hits))
    log.info("Fused chunks (top %d): %d",
             synthesizer._top_n_chunks, len(answer.fused_chunks))
    log.info("Chunks fed to synthesis: %d", len(answer.cited_chunk_ids))
    log.info("Elapsed: %.1fs", answer.elapsed_seconds)

    log.info("==== Answer ====\n%s\n", answer.text)

    log.info("==== Top fused chunks (by RRF) ====")
    for h in answer.fused_chunks[:10]:
        log.info("  %s  rrf=%.4f  sources=%s  graph_rank=%s  vector_rank=%s",
                 h.chunk_id, h.rrf_score, h.sources, h.graph_rank, h.vector_rank)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
