r"""Extract \boxed{...} answers and compare to gold.

Conservative implementation: light LaTeX normalization + numeric fallback.
Will under-credit some equivalent answers (e.g. `\sin(\pi/6)` vs `1/2`).
For a more robust comparator later, swap in `math_verify` (HF's open-source
math grader): https://github.com/huggingface/Math-Verify
"""
import re
from typing import Optional


def extract_boxed(text: str) -> Optional[str]:
    r"""Return contents of the LAST \boxed{...} in text, with proper brace
    matching for nested expressions like \boxed{\frac{1}{2}}.
    Returns None if no boxed answer is present or braces are unbalanced.
    """
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start : i - 1].strip()


def normalize(s: Optional[str]) -> str:
    """Strip LaTeX wrappers and whitespace that don't change the value."""
    if s is None:
        return ""
    s = s.strip()
    for tok in ["$", "\\!", "\\,", "\\;", "\\left", "\\right"]:
        s = s.replace(tok, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = s.replace(" ", "").rstrip(".")
    return s.lower()


def answers_match(predicted: Optional[str], gold: str) -> bool:
    """True if predicted == gold after light normalization, with numeric and
    simple-fraction fallbacks."""
    if predicted is None:
        return False
    p, g = normalize(predicted), normalize(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    # Numeric fallback (handles "0.5" vs "0.50")
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        pass
    # Simple-fraction fallback ("1/2" vs "0.5")
    if re.fullmatch(r"-?\d+/\d+", p):
        try:
            num, den = p.split("/")
            return abs(float(num) / float(den) - float(g)) < 1e-6
        except (ValueError, ZeroDivisionError):
            pass
    return False