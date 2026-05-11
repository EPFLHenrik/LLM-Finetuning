"""Configuration for math task evaluation.

Defaults mirror the CI evaluation contract from README.md:
  - max_model_len = 4096
  - n = 8 completions per problem
  - apply_chat_template(messages, add_generation_prompt=True), nothing else
so local numbers are directly comparable to the leaderboard.
"""
from dataclasses import dataclass, field
from pathlib import Path


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