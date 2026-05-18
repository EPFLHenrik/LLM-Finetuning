"""Configuration for math task evaluation.

Defaults mirror the CI evaluation contract from README.md:
  - max_model_len = 4096
  - n = 8 completions per problem
  - apply_chat_template(messages, add_generation_prompt=True), nothing else
so local numbers are directly comparable to the leaderboard.

The `prompt_variant` and `thinking_enabled` fields are ablation knobs used
locally to probe how prompt wording and Qwen3 thinking mode affect pass@k.
CI does NOT pass any of this through — for the final pushed checkpoint,
encode the chosen behavior in chat_template.jinja and/or the SFT data.
"""
from dataclasses import dataclass, field
from pathlib import Path


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

    # --- inference (must match CI exactly) ---
    max_model_len: int = 4096
    max_new_tokens: int = 4096   # vLLM caps at max_model_len - prompt_len
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

    @property
    def validation_path(self) -> Path:
        return self.repo_root / "validation_samples" / "math.jsonl"

    @property
    def results_dir(self) -> Path:
        return self.repo_root / "math_task" / "results"