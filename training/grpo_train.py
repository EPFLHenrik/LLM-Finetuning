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
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

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
        print("[info] Disabled PEFT bitsandbytes dispatch.")
    except Exception as exc:
        print(f"[warn] Could not disable PEFT bitsandbytes dispatch: {exc}")

# We'll import PEFT symbols after disabling bitsandbytes dispatch to avoid
# unintended bitsandbytes imports when not using quantization.
from trl import GRPOConfig, GRPOTrainer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate.benchmarks import extract_benchmark_answer, is_correct_benchmark_answer
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

QUESTION_KEYS = ("question", "prompt", "input", "text")
ANSWER_KEYS = ("answer", "reference", "label", "target", "answerKey", "gold_answer")

DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Reward weighting and length normalization
# Format (boxed answer) is most important because without correct format
# the answer cannot be evaluated. We weight format highest and reduce
# accuracy slightly while keeping a small length bonus.
ACC_WEIGHT = 0.25
FORMAT_WEIGHT = 0.70
LENGTH_WEIGHT = 0.05
# Repetition penalty weight (multiplicative). 0.0 = no penalty, 1.0 = full erase
REPETITION_WEIGHT = 0.20


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


def _get_first(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _index_to_label(index: int) -> str:
    # Supports the 2-20 option range in the project, but also handles larger sets.
    label = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def _normalize_choices(choices: Any) -> list[str]:
    if isinstance(choices, list):
        return [str(choice).strip() for choice in choices if str(choice).strip()]
    if isinstance(choices, dict):
        labels = choices.get("label")
        texts = choices.get("text")
        if isinstance(labels, list) and isinstance(texts, list) and len(labels) == len(texts):
            return [f"{str(label).strip()}) {str(text).strip()}" for label, text in zip(labels, texts)]
    return []


def _build_prompt_text(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")).strip() == "user":
                content = str(message.get("content", "")).strip()
                if content:
                    return content

    prompt = _get_first(row, QUESTION_KEYS)
    if prompt is None:
        raise SystemExit(f"Missing prompt/question in row: {row}")

    choices = _normalize_choices(row.get("choices"))
    if not choices:
        return prompt

    lines = [prompt, ""]
    for i, choice in enumerate(choices):
        if ") " in choice and choice[0].isalpha():
            # Preserve pre-labeled choice text if it already looks like "A) ...".
            lines.append(choice)
        else:
            lines.append(f"{_index_to_label(i)}) {choice}")
    return "\n".join(lines)


def _normalize_answer(answer: Any, choices: Any) -> str:
    text = str(answer).strip() if answer is not None else ""
    if not text:
        raise SystemExit("Missing answer in row")

    if len(text) == 1 and text.isalpha():
        return text.upper()

    normalized_choices = _normalize_choices(choices)
    for index, choice_text in enumerate(normalized_choices):
        if text == choice_text:
            return _index_to_label(index)
        if text == choice_text.split(") ", 1)[-1].strip():
            return _index_to_label(index)

    return text


def _build_messages(prompt_text: str, system_message: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt_text},
    ]


def _normalize_row(row: dict[str, Any], system_message: str) -> dict[str, Any]:
    prompt_text = _build_prompt_text(row)
    answer_value = _get_first(row, ANSWER_KEYS)
    if answer_value is None:
        raise SystemExit(f"Missing answer-like field in row: {row}")

    normalized = dict(row)
    normalized["prompt_text"] = prompt_text
    normalized["answer"] = _normalize_answer(answer_value, row.get("choices"))
    if isinstance(row.get("messages"), list):
        normalized["prompt"] = row["messages"]
    else:
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


def _reward_from_completion(completion: str, answer: str) -> float:
    # Keep existing single-purpose reward for compatibility: use boxed extraction
    # Normalize completion to text in case TRL passes token lists or other types.
    if not isinstance(completion, str):
        try:
            completion_text = " ".join(map(str, completion))
        except Exception:
            completion_text = str(completion)
    else:
        completion_text = completion

    reference = answer.strip()
    if not reference:
        return 0.0

    extracted, _ = extract_boxed_answer_with_fallbacks(completion_text)
    if extracted is None:
        return 0.0
    return 1.0 if extracted.upper() == reference.upper() else 0.0


def grpo_reward(prompts: list[Any], completions: list[str], **kwargs: Any) -> list[float | None]:
    answers = _extract_answer_column(kwargs)
    if answers is None:
        raise SystemExit("GRPO reward function did not receive an answer column from the dataset")

    if len(answers) != len(completions):
        # TRL repeats the prompt/answer columns to match completions; if that ever changes,
        # fail loudly rather than silently misaligning labels.
        raise SystemExit(f"Reward input mismatch: {len(answers)} answers for {len(completions)} completions")

    def _length_reward(completion: str) -> float:
        if not isinstance(completion, str):
            try:
                completion_text = " ".join(map(str, completion))
            except Exception:
                completion_text = str(completion)
        else:
            completion_text = completion

        l = len(completion_text.strip())
        if l <= 0:
            return 0.0
        # Reward shorter completions without an arbitrary max; use a
        # diminishing return so very short outputs get a larger bonus.
        return 1.0 / (1.0 + float(l))

    def _repetition_score(completion: str, n: int = 3) -> float:
        """Return a repetition score in [0,1] based on n-gram uniqueness.

        Higher means more repetition. We compute 1 - (unique_ngrams / total_ngrams).
        """
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
        score = 1.0 - (unique / total)
        return max(0.0, min(1.0, score))

    rewards: list[float] = []
    for completion, answer in zip(completions, answers, strict=True):
        reference = answer.strip()

        # Normalize completion to text for extraction and length checks
        if not isinstance(completion, str):
            try:
                completion_text = " ".join(map(str, completion))
            except Exception:
                completion_text = str(completion)
        else:
            completion_text = completion

        # Format reward: is there a boxed answer at all?
        extracted, _ = extract_boxed_answer_with_fallbacks(completion_text)
        format_r = 1.0 if extracted is not None else 0.0

        # Accuracy reward: 1 if boxed extraction equals reference, else 0
        accuracy_r = 0.0
        if extracted is not None and reference:
            accuracy_r = 1.0 if extracted.strip().upper() == reference.strip().upper() else 0.0

        # Length reward: shorter is better (normalized), small weight
        length_r = _length_reward(completion_text)

        base = ACC_WEIGHT * accuracy_r + FORMAT_WEIGHT * format_r + LENGTH_WEIGHT * length_r

        # Apply multiplicative repetition penalty so high repetition reduces reward
        rep_score = _repetition_score(completion_text)
        final = base * (1.0 - REPETITION_WEIGHT * rep_score)
        rewards.append(final)
    return rewards


def _save_merged_model(trainer: GRPOTrainer, tokenizer: AutoTokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = trainer.model
    if isinstance(model, PeftModel) or hasattr(model, "merge_and_unload"):
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_dir, safe_serialization=True)
        if getattr(merged_model, "generation_config", None) is not None:
            merged_model.generation_config.save_pretrained(output_dir)
    else:
        trainer.save_model(output_dir)

    tokenizer.save_pretrained(output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a model with GRPO on prompt/answer data.")
    parser.add_argument("--model-folder", required=True, type=Path, help="Base model folder to train from")
    parser.add_argument("--dataset-file", required=True, type=Path, help="Input JSONL/JSON file with prompt/answer rows")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where to write the final merged checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", choices=("epoch", "steps", "no"), default="epoch")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=1000)
    parser.add_argument("--wandb-project", type=str, default="cs552-mystery-machine")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-key", type=str, default=None)
    parser.add_argument("--skip-wandb", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", action="store_false", dest="use_lora")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str, default=",".join(DEFAULT_LORA_TARGET_MODULES))
    parser.add_argument("--peft-adapter", type=Path, default=None, help="Path to a PEFT adapter checkpoint to load on top of a base model")
    parser.add_argument("--base-model", type=Path, default=None, help="Path to the base (unmerged) model when loading a PEFT adapter")
    parser.add_argument("--dry-run", action="store_true", help="Print a sample of the normalized dataset and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not args.model_folder.exists():
        raise SystemExit(f"Model folder not found: {args.model_folder}")
    if not args.dataset_file.exists():
        raise SystemExit(f"Dataset file not found: {args.dataset_file}")

    if args.output_dir.exists():
        if not args.overwrite_output:
            raise SystemExit(f"Output dir already exists: {args.output_dir} (pass --overwrite-output)")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("This script expects a CUDA-capable GPU for GRPO training.")

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

    # If a PEFT adapter will be loaded on top of a base model, prefer the
    # tokenizer from the base model; otherwise use the provided model folder.
    tokenizer_source = args.base_model if getattr(args, "peft_adapter", None) else args.model_folder
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit(f"Tokenizer does not define a chat template: {args.model_folder}")

    # Prevent PEFT from attempting to dispatch bitsandbytes when we're not
    # quantizing; mirrors behavior in lora_post_train.py
    try:
        _disable_peft_bnb_dispatch()
    except Exception:
        pass

    train_dataset = Dataset.from_list(normalized_rows)
    # Shuffle the training dataset to ensure random ordering each run (seeded)
    train_dataset = train_dataset.shuffle(seed=args.seed)
    if args.max_train_samples > 0:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))

    grpo_config = GRPOConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
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
        # Ensure generation_batch_size is divisible by num_generations.
        # Use num_generations so each generation can be produced in one batch.
        generation_batch_size=args.num_generations,
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
    )
    # Disable PEFT bitsandbytes dispatch before importing PEFT symbols.
    _disable_peft_bnb_dispatch()
    from peft import LoraConfig, PeftModel, TaskType

    lora_config = None
    if args.use_lora:
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
        except Exception as exc:
            print(f"[warn] wandb.login() failed: {exc}")
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=args.output_dir.name)

    # If a PEFT adapter path is provided, load the base model object and
    # attach the adapter so we pass a fully prepared model object to the trainer.
    model_arg = str(args.model_folder)
    if getattr(args, "peft_adapter", None):
        if not args.base_model:
            raise SystemExit("When using --peft-adapter you must pass --base-model pointing to the base model folder")
        base_model = AutoModelForCausalLM.from_pretrained(str(args.base_model), trust_remote_code=True)
        model_arg = PeftModel.from_pretrained(base_model, str(args.peft_adapter))

    trainer = GRPOTrainer(
        model=model_arg,
        reward_funcs=grpo_reward,
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    _save_merged_model(trainer, tokenizer, args.output_dir)

    if not args.skip_wandb:
        import wandb

        wandb.finish()

    print(f"Saved merged GRPO model to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
