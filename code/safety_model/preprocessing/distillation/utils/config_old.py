from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from constants.system_prompt import (
    TEACHER_MODEL_PROMPT_EXPLICIT,
    TEACHER_MODEL_PROMPT_HIDDEN,
)

ROOT = Path(__file__).resolve().parents[2]
INPUT_DATASET_PATH = (
    ROOT / "data" / "processed" / "sft" / "wildguardmix" / "binary_choice"
)
TEACHER_MODEL = "Qwen/Qwen3-32B-AWQ"


@dataclass(frozen=True)
class RuntimeConfig:
    input_path: Path
    output_path: Path
    fallback_output_path: Path
    error_log_path: Path
    split_name: str
    teacher_model: str
    system_prompt: Callable[[str], str]
    system_prompt_fallback_func: Callable[[str, str], str]
    max_new_tokens: int
    max_distinct_tries: int
    max_total_generations: int
    batch_size: int
    temperature: float
    top_p: float
    seed: int
    overwrite: bool


def build_runtime_config(
    *,
    input_path: Path,
    output_root: Path,
    split: str | None,
    max_new_tokens: int,
    seed: int,
    overwrite: bool,
    teacher_model: str = TEACHER_MODEL,
    system_prompt: Callable[[str], str] = TEACHER_MODEL_PROMPT_HIDDEN,
    system_prompt_fallback_func: Callable[[str, str], str] = TEACHER_MODEL_PROMPT_EXPLICIT,
) -> RuntimeConfig:
    output_path = output_root / split if split else output_root
    fallback_output_path = output_path.parent / f"{output_path.name}_fallback"
    split_name = split if split else "all"
    error_log_path = output_path / f"error_{split_name}.log"

    input_path = input_path / split if split else input_path

    return RuntimeConfig(
        input_path=input_path,
        output_path=output_path,
        fallback_output_path=fallback_output_path,
        error_log_path=error_log_path,
        split_name=split_name,
        teacher_model=teacher_model,
        system_prompt=system_prompt,
        system_prompt_fallback_func=system_prompt_fallback_func,
        max_new_tokens=max_new_tokens,
        max_distinct_tries=5,
        max_total_generations=20,
        batch_size=500,
        temperature=0.7,
        top_p=0.8,
        seed=seed,
        overwrite=overwrite,
    )