"""Metric computation.

Four numbers per run:
  - pass@1               : average per-attempt accuracy (mean over n=8 samples per problem)
  - pass@8               : unbiased pass@k estimator from Chen et al. 2021
  - box_compliance_rate  : fraction of completions that emit any \\boxed{...}
  - truncation_rate      : fraction of completions that hit the 4096 ceiling
                           (= finish_reason == 'length') — KEY diagnostic for
                           whether "always thinking mode" is viable for math.

Plus accuracy_given_boxed (pass@1 conditional on box compliance) to separate
"doesn't know the answer" from "doesn't follow the format".
"""
from typing import List, Dict
import numpy as np

from .answer_compare import extract_boxed, answers_match


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from Chen et al. 2021. n = total samples drawn,
    c = correct samples, k = budget. P(at least one of k correct)."""
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def compute_metrics(results: List[Dict]) -> Dict:
    per_problem = []
    for r in results:
        n = len(r["completions"])
        n_correct = n_boxed = n_truncated = 0
        for c in r["completions"]:
            extracted = extract_boxed(c["text"])
            if extracted is not None:
                n_boxed += 1
                if answers_match(extracted, r["gold"]):
                    n_correct += 1
            if c.get("finish_reason") == "length":
                n_truncated += 1
        per_problem.append(dict(n=n, c=n_correct, b=n_boxed, t=n_truncated))

    n_problems = len(per_problem)
    n_total = sum(p["n"] for p in per_problem)
    n_boxed_total = sum(p["b"] for p in per_problem)
    n_correct_total = sum(p["c"] for p in per_problem)

    return {
        "n_problems": n_problems,
        "n_completions_per_problem": n_total // n_problems if n_problems else 0,
        "pass@1": float(np.mean([p["c"] / p["n"] for p in per_problem])),
        "pass@8": float(np.mean([
            pass_at_k(p["n"], p["c"], min(8, p["n"])) for p in per_problem
        ])),
        "box_compliance_rate": n_boxed_total / n_total if n_total else 0.0,
        "truncation_rate": sum(p["t"] for p in per_problem) / n_total if n_total else 0.0,
        "accuracy_given_boxed": (
            n_correct_total / n_boxed_total if n_boxed_total else 0.0
        ),
    }