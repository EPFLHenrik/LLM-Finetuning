"""Configuration for math task evaluation.

Two evaluation budgets are supported:
  - budget="ci"    -> max_model_len = max_new_tokens = 4096
  - budget="final" -> max_model_len = max_new_tokens = 16384
"""
from dataclasses import dataclass, field
from pathlib import Path


# Token budgets for the two evaluation modes.
BUDGET_TOKENS = {
    "ci": 4096,
    "final": 16384,
}


@dataclass
class EvalConfig:
    # --- model ---
    model_id: str = "Qwen/Qwen3-1.7B"
    dtype: str = "float16"

    # --- inference ---
    # Setting `budget` auto-fills max_model_len + max_new_tokens via __post_init__.
    # Pass the lengths explicitly to override (rare).
    budget: str = "final"
    max_model_len: int = 0       # 0 -> filled from budget
    max_new_tokens: int = 0      # 0 -> filled from budget; vLLM caps at max_model_len - prompt_len
    n_samples: int = 8

    # --- sampling — diverse for pass@8, not greedy ---
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 20
    seed: int = 42

    # passed as enable_thinking to the chat template
    thinking_enabled: bool = True

    # Package root (code/math_model), used to resolve results/ and RUNS.md.
    package_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )

    def __post_init__(self) -> None:
        if self.budget not in BUDGET_TOKENS:
            raise ValueError(
                f"budget must be one of {list(BUDGET_TOKENS)}; got {self.budget!r}"
            )
        budget_tokens = BUDGET_TOKENS[self.budget]
        if self.max_model_len == 0:
            self.max_model_len = budget_tokens
        if self.max_new_tokens == 0:
            self.max_new_tokens = budget_tokens
