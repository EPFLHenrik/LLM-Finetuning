"""Answer extraction and comparison for math benchmarks."""

from __future__ import annotations

from .answer_extraction import extract_boxed_answer, is_equiv, normalize_final_answer


VALID_METHODS = ("boxed", "exact")


def extract_benchmark_answer(text: str, method: str, reference: str) -> str | None:
    """Extract a final answer according to the benchmark's evaluator."""
    if method == "exact":
        return text.strip()
    # Default ("boxed"): take the last \boxed{...}, falling back to raw text.
    extracted = extract_boxed_answer(text, strip_double_curly_brace=True)
    return extracted if extracted is not None else text.strip()


def is_correct_benchmark_answer(
    extracted: str | None,
    reference: str,
    method: str,
) -> bool:
    """Compare an extracted answer to the benchmark reference."""
    if extracted is None:
        return False
    return is_equiv(
        normalize_final_answer(extracted),
        normalize_final_answer(reference),
    )
