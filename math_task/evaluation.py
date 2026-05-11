"""Diagnostic metrics on top of pass@k.

`evaluate/` already computes the leaderboard metrics (pass@1, pass@8). These
extras help explain WHY the model is scoring what it scores:

  - box_compliance_rate     : did the model emit \\boxed{...} at all?
                              (uses the same extractor the CI grader uses, so
                               this is exactly "the official grader will get
                               something out of this generation")
  - truncation_rate         : did the model run out of context before answering?
                              (vLLM finish_reason == "length")
  - avg_tokens_per_completion : how long is the model thinking? Useful for
                                deciding if "always thinking mode" is viable
                                under the 4096-token CI ceiling.
"""
from typing import List, Dict

from evaluate.extract_answer import extract_boxed_answer


def compute_extra_metrics(generation_results: List[Dict]) -> Dict[str, float]:
    """Compute diagnostic metrics from raw generation output.

    `generation_results` is the output of `math_task.generation.run_generation`
    — each item has `completions: [{text, finish_reason, n_tokens}, ...]`.
    """
    n_total = 0
    n_boxed = 0
    n_truncated = 0
    token_counts: List[int] = []

    for r in generation_results:
        for c in r["completions"]:
            n_total += 1
            if extract_boxed_answer(c["text"], strip_double_curly_brace=True) is not None:
                n_boxed += 1
            if c.get("finish_reason") == "length":
                n_truncated += 1
            token_counts.append(int(c.get("n_tokens", 0)))

    if n_total == 0:
        return {
            "n_completions_total": 0,
            "box_compliance_rate": 0.0,
            "truncation_rate": 0.0,
            "avg_tokens_per_completion": 0.0,
        }

    return {
        "n_completions_total": n_total,
        "box_compliance_rate": n_boxed / n_total,
        "truncation_rate": n_truncated / n_total,
        "avg_tokens_per_completion": sum(token_counts) / n_total,
    }