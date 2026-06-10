"""Diagnostic metrics that complement pass@k.

`scoring.py` computes the headline metrics (pass@1, pass@8). The metrics here
help explain why a model scores what it scores:

  - box_compliance_rate     : did the model emit \\boxed{...} at all?
                              (uses the same extractor as scoring, so this is
                               exactly "the grader will get something out of
                               this generation")
  - truncation_rate         : did the model run out of context before answering?
                              (vLLM finish_reason == "length")
  - avg_tokens_per_completion : how long is the model thinking? Useful for
                                deciding whether always-on thinking mode fits
                                under the 4096-token budget.
"""
from typing import List, Dict

from .answer_extraction import extract_boxed_answer


def compute_extra_metrics(generation_results: List[Dict]) -> Dict[str, float]:
    """Compute diagnostic metrics from raw generation output.

    `generation_results` is the output of `code.math_model.evaluation.generation.run_generation`
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