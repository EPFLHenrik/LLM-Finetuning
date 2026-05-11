"""Math task evaluation pipeline.

Reuses the shared `evaluate/` package for all answer logic:
  - boxed extraction + math equivalence  → evaluate.extract_answer
  - per-benchmark grader                  → evaluate.benchmarks
  - pass@k                                 → evaluate.pass_at_k
  - end-to-end scorer                      → evaluate.score

Adds on top:
  - math-specific dataset loaders         → math_task.dataset_loader
  - vLLM generation orchestration         → math_task.generation
  - diagnostic metrics (box compliance,
    truncation, avg tokens)               → math_task.evaluation
  - CLI                                    → math_task.run_eval
"""