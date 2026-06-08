#!/usr/bin/env python3
"""Train a reasoning model with GRPO.

This script is intentionally small and repo-oriented:
- it reads a JSONL/JSON dataset with prompt/answer rows,
- normalizes each row into a conversational `prompt` plus an `answer`,
- feeds the dataset into TRL's `GRPOTrainer`,
- uses a boxed-answer reward based on exact match, and
- saves a merged checkpoint plus tokenizer artifacts at the end.

The trainer works best when the dataset rows are raw prompts, not already
pre-templated chat strings. If a row has `question`/`choices`, the script
builds a multiple-choice prompt; if it already has `prompt`, that text is used.
The final `prompt` field stored in the dataset is a list of chat `messages`, so
TRL applies the model's chat template internally.
"""

from __future__ import annotations

import argparse
import functools
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback

# Prevent PEFT from importing bitsandbytes when not using quantization.
# Define this locally so we can call it before importing `peft`.
def _disable_peft_bnb_dispatch() -> None:
    try:
        import peft.tuners.lora.model as lora_model
        import peft.import_utils as peft_import_utils

        lora_model.is_bnb_available = lambda: False
        lora_model.is_bnb_4bit_available = lambda: False
        peft_import_utils.is_bnb_available = lambda: False
        peft_import_utils.is_bnb_4bit_available = lambda: False
    except Exception:
        pass

# We'll import PEFT symbols after disabling bitsandbytes dispatch to avoid
# unintended bitsandbytes imports when not using quantization.
from trl import GRPOConfig, GRPOTrainer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate.extract_answer import extract_boxed_answer_with_fallbacks

DEFAULT_SYSTEM_MESSAGE = (
    "You are a careful reasoning assistant.\n\n"
    "Reason through the problem when needed.\n\n"
    "Your final answer must appear exactly once inside \\boxed{...}.\n\n"
    "Rules:\n"
    "- For multiple-choice questions, output only the correct capital letter inside \\boxed{...}.\n"
    "- For math questions, output only the final answer inside \\boxed{...}.\n"
    "- Do not include explanations or extra text inside the box.\n"
    "- Never output multiple boxed answers.\n"
    "- Never leave the final answer unboxed.\n\n"
    "Examples:\n"
    "\\boxed{B}\n"
    "\\boxed{42}"
)

DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

ACC_WEIGHT = 0.50
FORMAT_WEIGHT = 0.40
# Repetition penalty weight (multiplicative). 0.0 = no penalty, 1.0 = full erase
REPETITION_WEIGHT = 0.20

# Module-level mutable so reward functions can read CLI-overridden values.
_acc_weight: float = ACC_WEIGHT
_format_weight: float = FORMAT_WEIGHT
_repetition_weight: float = REPETITION_WEIGHT


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise SystemExit(f"{path}:{lineno}: each row must be a JSON object")
                rows.append(row)
        return rows

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list) and all(isinstance(row, dict) for row in payload):
            return list(payload)
        raise SystemExit(f"{path}: JSON files must contain a list of objects")

    raise SystemExit(f"Unsupported dataset format: {path.suffix}. Use .jsonl or .json")


def _build_messages(prompt_text: str, system_message: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt_text},
    ]


def _normalize_row(row: dict[str, Any], system_message: str) -> dict[str, Any]:
    prompt_text = str(row.get("prompt", "")).strip()
    if not prompt_text:
        raise SystemExit(f"Missing 'prompt' in row: {row}")
    answer = str(row.get("answer", "")).strip()
    if not answer:
        raise SystemExit(f"Missing 'answer' in row: {row}")

    normalized = dict(row)
    normalized["answer"] = answer
    normalized["prompt"] = _build_messages(prompt_text, system_message)
    return normalized


def _extract_answer_column(kwargs: dict[str, Any]) -> list[str] | None:
    for key in ("answer", "answers", "label", "labels", "reference", "references", "target", "targets"):
        value = kwargs.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            return [str(item).strip() for item in value]
        return [str(value).strip()]
    return None


def _normalize_completion(completion: Any) -> str:
    if not isinstance(completion, str):
        try:
            return " ".join(map(str, completion))
        except Exception:
            return str(completion)
    return completion


def _repetition_score(completion: str, n: int = 3) -> float:
    """Return a repetition score in [0,1]: 1 - (unique_ngrams / total_ngrams)."""
    text = completion.strip()
    if not text:
        return 0.0
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (unique / total)))


def reward_format(prompts: list[Any], completions: list[str], **kwargs: Any) -> list[float]:
    """Binary reward for presence of a \\boxed{} answer."""
    rewards = []
    for completion in completions:
        text = _normalize_completion(completion)
        extracted, _ = extract_boxed_answer_with_fallbacks(text)
        rewards.append(_format_weight * (1.0 if extracted is not None else 0.0))
    return rewards


def reward_accuracy(prompts: list[Any], completions: list[str], **kwargs: Any) -> list[float]:
    """Binary reward for exact-match between boxed answer and reference."""
    answers = _extract_answer_column(kwargs)
    if answers is None:
        raise SystemExit("reward_accuracy did not receive an answer column from the dataset")
    if len(answers) != len(completions):
        raise SystemExit(f"Reward input mismatch: {len(answers)} answers for {len(completions)} completions")
    rewards = []
    for completion, answer in zip(completions, answers, strict=True):
        text = _normalize_completion(completion)
        reference = answer.strip()
        extracted, _ = extract_boxed_answer_with_fallbacks(text)
        if extracted is not None and reference:
            r = 1.0 if extracted.strip().upper() == reference.strip().upper() else 0.0
        else:
            r = 0.0
        rewards.append(_acc_weight * r)
    return rewards


def reward_repetition(prompts: list[Any], completions: list[str], **kwargs: Any) -> list[float]:
    """Additive penalty for 3-gram repetition (returns negative values)."""
    rewards = []
    for completion in completions:
        text = _normalize_completion(completion)
        rewards.append(-_repetition_weight * _repetition_score(text))
    return rewards


def _make_accumulating_reward_func(func: Any, acc: list[float]) -> Any:
    """Wrap a reward function to accumulate its values for logging at each logging step."""

    @functools.wraps(func)
    def wrapper(prompts: list[Any], completions: list[str], **kwargs: Any) -> list[float]:
        rewards = func(prompts, completions, **kwargs)
        acc.extend(rewards)
        return rewards

    return wrapper


class _WandbRewardCallback(TrainerCallback):
    """Logs mean per-reward-function values to wandb at each logging step."""

    def __init__(self, accumulators: dict[str, list[float]]) -> None:
        self._acc = accumulators

    def on_log(self, args: Any, state: Any, control: Any, logs: dict | None = None, **kwargs: Any) -> None:
        try:
            import wandb

            if wandb.run is None:
                return
            metrics = {
                f"rewards/{name}": sum(values) / len(values)
                for name, values in self._acc.items()
                if values
            }
            for values in self._acc.values():
                values.clear()
            if metrics:
                wandb.log(metrics, step=state.global_step, commit=False)
        except Exception:
            pass


def _save_merged_model(trainer: GRPOTrainer, tokenizer: AutoTokenizer, output_dir: Path) -> None:
    from peft import PeftModel
    output_dir.mkdir(parents=True, exist_ok=True)
    model = trainer.model
    save_dtype = (
        torch.bfloat16 if trainer.args.bf16 else torch.float16 if trainer.args.fp16 else None
    )
    if isinstance(model, PeftModel) or hasattr(model, "merge_and_unload"):
        merged_model = model.merge_and_unload()
        if save_dtype is not None:
            merged_model = merged_model.to(save_dtype)
        merged_model.save_pretrained(output_dir, safe_serialization=True)
        if getattr(merged_model, "generation_config", None) is not None:
            merged_model.generation_config.save_pretrained(output_dir)
    else:
        if save_dtype is not None:
            trainer.model.to(save_dtype)
        trainer.save_model(output_dir)

    tokenizer.save_pretrained(output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a model with GRPO on prompt/answer data.")
    parser.add_argument("--model-folder", required=True, type=Path, help="Base model folder to train from")
    parser.add_argument("--dataset-file", required=True, type=Path, help="Input JSONL/JSON file with prompt/answer rows")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where to write the final merged checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=80)
    parser.add_argument("--save-strategy", choices=("epoch", "steps", "no"), default="epoch")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default="cs552-mystery-machine")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-key", type=str, default=None)
    parser.add_argument("--skip-wandb", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", action="store_false", dest="use_lora")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str, default=",".join(DEFAULT_LORA_TARGET_MODULES))
    parser.add_argument("--adapter-folder", type=Path, default=None, help="Existing LoRA adapter checkpoint to continue training from (--model-folder must be the matching base model)")
    parser.add_argument("--dry-run", action="store_true", help="Print a sample of the normalized dataset and exit")
    parser.add_argument("--beta", type=float, default=0.04, help="KL penalty coefficient (0 = no penalty, higher = stay closer to reference)")
    parser.add_argument("--acc-weight", type=float, default=ACC_WEIGHT, help="Reward weight for accuracy")
    parser.add_argument("--format-weight", type=float, default=FORMAT_WEIGHT, help="Reward weight for \\boxed{} format presence")
    parser.add_argument("--repetition-weight", type=float, default=REPETITION_WEIGHT, help="Penalty weight for 3-gram repetition")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not args.model_folder.exists():
        raise SystemExit(f"Model folder not found: {args.model_folder}")
    if args.adapter_folder and not args.adapter_folder.exists():
        raise SystemExit(f"Adapter folder not found: {args.adapter_folder}")
    if not args.dataset_file.exists():
        raise SystemExit(f"Dataset file not found: {args.dataset_file}")

    if args.output_dir.exists():
        if not args.overwrite_output:
            raise SystemExit(f"Output dir already exists: {args.output_dir} (pass --overwrite-output)")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("This script expects a CUDA-capable GPU for GRPO training.")

    global _acc_weight, _format_weight, _repetition_weight
    _acc_weight = args.acc_weight
    _format_weight = args.format_weight
    _repetition_weight = args.repetition_weight

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    rows = _read_json_rows(args.dataset_file)
    normalized_rows = [_normalize_row(row, DEFAULT_SYSTEM_MESSAGE) for row in rows]
    if not normalized_rows:
        raise SystemExit("No usable rows found after normalization")

    if args.dry_run:
        print(json.dumps(normalized_rows[0], indent=2, ensure_ascii=False))
        print(f"[dry-run] normalized {len(normalized_rows)} rows")
        return 0

    tokenizer_source = args.model_folder
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit(f"Tokenizer does not define a chat template: {args.model_folder}")

    # Prevent PEFT from attempting to dispatch bitsandbytes when we're not
    # quantizing; mirrors behavior in lora_post_train.py
    _disable_peft_bnb_dispatch()

    train_dataset = Dataset.from_list(normalized_rows)
    # Shuffle the training dataset to ensure random ordering each run (seeded)
    train_dataset = train_dataset.shuffle(seed=args.seed)
    if args.max_train_samples > 0:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))

    grpo_config = GRPOConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
        beta=args.beta,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        report_to=[] if args.skip_wandb else ["wandb"],
        run_name=args.output_dir.name,
        num_generations=args.num_generations,
        generation_batch_size=args.per_device_train_batch_size * args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        use_vllm=True,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    # Disable PEFT bitsandbytes dispatch before importing PEFT symbols.
    _disable_peft_bnb_dispatch()
    from peft import LoraConfig, PeftConfig, PeftModel, TaskType, set_peft_model_state_dict, load_peft_weights

    lora_config = None
    if args.use_lora or args.adapter_folder:
        if args.adapter_folder:
            # Read config from the checkpoint so r/alpha/target_modules match exactly.
            peft_cfg = PeftConfig.from_pretrained(str(args.adapter_folder))
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=peft_cfg.r,
                lora_alpha=peft_cfg.lora_alpha,
                lora_dropout=peft_cfg.lora_dropout,
                target_modules=list(peft_cfg.target_modules),
            )
        else:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=[module.strip() for module in args.lora_target_modules.split(",") if module.strip()],
            )

    if not args.skip_wandb:
        import wandb

        try:
            wandb.login(key=args.wandb_key)
        except Exception:
            pass
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name="GRPO test run")

    # Always pass the base model as a string path so vLLM can initialize properly.
    # If resuming from an adapter, we inject the weights after the trainer is built.
    model_arg = str(args.model_folder)

    _base_reward_funcs = [reward_format, reward_accuracy, reward_repetition]
    accumulators: dict[str, list[float]] = {f.__name__: [] for f in _base_reward_funcs}
    reward_funcs = [_make_accumulating_reward_func(f, accumulators[f.__name__]) for f in _base_reward_funcs]
    trainer = GRPOTrainer(
        model=model_arg,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[_WandbRewardCallback(accumulators)],
    )

    if args.adapter_folder:
        # Load the SFT adapter weights into the trainer's LoRA model.
        # vLLM already has the base model; TRL will sync the LoRA weights to it before generation.
        adapter_weights = load_peft_weights(str(args.adapter_folder))
        set_peft_model_state_dict(trainer.model, adapter_weights)

    trainer.train()
    _save_merged_model(trainer, tokenizer, args.output_dir)

    if not args.skip_wandb:
        import wandb

        wandb.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
