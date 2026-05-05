# evaluation/

Ragas evaluation harness. The directory that makes this project credible.

- `ground_truth.yaml` — hand-curated Q/A pairs covering the three query types
- `ragas_runner.py` — runs the suite, produces faithfulness / answer-relevancy / context-precision scores
- `regression.py` — diffs scores across commits so regressions are visible

Run:
    python -m evaluation.ragas_runner --output reports/eval-$(date +%Y-%m-%d).json
