"""
Run the full Ragas evaluation and write a report.

Usage:
    python -m scripts.run_eval

What it does:
  1. Loads the 10 ground-truth questions
  2. Runs each through the Synthesizer (graph + vector + Claude)
  3. Scores answers with Ragas (faithfulness, answer relevancy, context precision)
  4. Writes:
       - data/eval/eval_<timestamp>.json   (full per-question results)
       - data/eval/eval_<timestamp>.md     (human-readable report)

Cost: roughly $1.50-$2.50 for 10 questions
  ($0.02 per answer × 10 + ~$0.10-$0.20 per Ragas judge round × 10 questions × 3 metrics)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from app.config import settings
from evaluation.ragas_runner import EvalSummary, run_eval
from evaluation.test_questions import get_questions
from retrieval.synthesizer import Synthesizer

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("run_eval")


def write_json_report(summary: EvalSummary, path: Path) -> None:
    payload = {
        "n_questions": summary.n_questions,
        "avg_scores": summary.avg_scores,
        "avg_elapsed_seconds": summary.avg_elapsed,
        "total_elapsed_seconds": summary.total_elapsed,
        "per_question": [
            {
                "qid": r.qid,
                "difficulty": r.difficulty,
                "question": r.question,
                "ground_truth": r.ground_truth,
                "answer": r.answer,
                "retrieved_chunk_ids": r.retrieved_chunk_ids,
                "elapsed_seconds": r.elapsed_seconds,
                "scores": r.scores,
            }
            for r in summary.per_question
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Wrote JSON report: %s (%d KB)", path, path.stat().st_size // 1024)


def write_markdown_report(summary: EvalSummary, path: Path) -> None:
    lines: list[str] = []
    lines.append(f"# RAG Evaluation Report\n")
    lines.append(f"- Questions: **{summary.n_questions}**")
    lines.append(f"- Total elapsed: **{summary.total_elapsed:.1f}s**")
    lines.append(f"- Avg per-question elapsed: **{summary.avg_elapsed:.1f}s**")
    lines.append("")
    lines.append("## Average scores\n")
    lines.append("| Metric | Score |")
    lines.append("| --- | --- |")
    for metric, score in summary.avg_scores.items():
        lines.append(f"| {metric} | **{score:.3f}** |")
    lines.append("")

    lines.append("## Per-question scores\n")
    lines.append("| qid | difficulty | faithfulness | answer_relevancy | context_precision | elapsed (s) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in summary.per_question:
        f = r.scores.get("faithfulness", 0.0)
        a = r.scores.get("answer_relevancy", 0.0)
        c = r.scores.get("context_precision", 0.0)
        lines.append(f"| {r.qid} | {r.difficulty} | {f:.3f} | {a:.3f} | {c:.3f} | {r.elapsed_seconds:.1f} |")
    lines.append("")

    lines.append("## Per-question detail\n")
    for r in summary.per_question:
        lines.append(f"### {r.qid} ({r.difficulty})")
        lines.append(f"**Question:** {r.question}\n")
        lines.append(f"**Ground truth:** {r.ground_truth}\n")
        lines.append(f"**Answer:**\n\n{r.answer}\n")
        lines.append(f"**Retrieved chunks:** {', '.join(r.retrieved_chunk_ids[:8])}"
                     + (" …" if len(r.retrieved_chunk_ids) > 8 else ""))
        lines.append(f"**Scores:** faithfulness={r.scores.get('faithfulness', 0):.3f}, "
                     f"answer_relevancy={r.scores.get('answer_relevancy', 0):.3f}, "
                     f"context_precision={r.scores.get('context_precision', 0):.3f}")
        lines.append("\n---\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote Markdown report: %s (%d KB)", path, path.stat().st_size // 1024)


def main() -> int:
    questions = get_questions()
    log.info("Loaded %d evaluation questions", len(questions))

    synthesizer = Synthesizer()
    try:
        summary = run_eval(questions, synthesizer=synthesizer)
    finally:
        synthesizer.close()

    out_dir = Path("data/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"eval_{ts}.json"
    md_path = out_dir / f"eval_{ts}.md"

    write_json_report(summary, json_path)
    write_markdown_report(summary, md_path)

    log.info("==== FINAL SCORES ====")
    for metric, score in summary.avg_scores.items():
        log.info("  %-22s %.3f", metric, score)

    return 0


if __name__ == "__main__":
    sys.exit(main())
