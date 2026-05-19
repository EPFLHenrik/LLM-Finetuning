"""Configuration for math task evaluation.

Two evaluation budgets are supported:
  - budget="ci"    -> max_model_len = max_new_tokens = 4096   (matches nightly CI)
  - budget="final" -> max_model_len = max_new_tokens = 16384  (matches final grading)

The CI nightly pipeline runs at 4k for throughput. The actual graded run uses 16k.
For the math individual model with thinking ON, the 16k budget is what we
optimize for; the 4k number is a sanity check that the model still loads.

Other defaults mirror the CI evaluation contract:
  - n = 8 completions per problem
  - apply_chat_template(messages, add_generation_prompt=True), nothing else

The `prompt_variant` and `thinking_enabled` fields are ablation knobs used
locally to probe how prompt wording and Qwen3 thinking mode affect pass@k.
CI does NOT pass any of this through — for the final pushed checkpoint,
encode the chosen behavior in chat_template.jinja and/or the SFT data.
"""
from dataclasses import dataclass, field
from pathlib import Path


# Token budgets for the two evaluation modes.
BUDGET_TOKENS = {
    "ci": 4096,       # nightly CI eval (fast iteration signal)
    "final": 16384,   # final grading (what your grade is based on)
}


# Prefix prepended to the user message for each ablation cell.
# "none" stays identical to the CI contract. The other two are ONLY for local
# probing, since CI never adds anything to the user prompt.
PROMPT_VARIANTS = {
    "none": "",
    "boxed": (
        "Answer with the final result in \\boxed{} format.\n\n"
    ),
    "classify": (
        "First classify this problem as one of: math reasoning, safety, "
        "multilinguality, general knowledge. Then solve it and put the "
        "final answer in \\boxed{} format.\n\n"
    ),
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

    # --- ablation knobs (local only — see module docstring) ---
    prompt_variant: str = "none"      # key into PROMPT_VARIANTS
    thinking_enabled: bool = True     # passed as enable_thinking to chat_template

    # --- paths (resolved relative to repo root) ---
    repo_root: Path = field(
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

    @property
    def validation_path(self) -> Path:
        return self.repo_root / "validation_samples" / "math.jsonl"

    @property
    def results_dir(self) -> Path:
        return self.repo_root / "math_task" / "results"