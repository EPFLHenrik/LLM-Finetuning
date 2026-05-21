#!/usr/bin/env python3
"""LoRA post-training for chat-style Qwen checkpoints.

This script:
1. loads a base model folder and a JSONL dataset file;
2. fine-tunes the model with LoRA on chat-formatted prompt/answer pairs;
3. evaluates generated answers with the project answer extractor; and
4. saves a merged checkpoint plus the tokenizer artifacts to the output dir.

The evaluator uses `extract_boxed_answer_with_fallbacks` so failed extraction is
penalized in stages, with `no_answer_found` treated as the worst outcome.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate.extract_answer import extract_boxed_answer_with_fallbacks, is_equiv


PROMPT_KEYS = ("prompt", "question", "input", "text")
ANSWER_KEYS = ("answer", "reference", "label", "target")

DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

WANDB_PROJECT = "cs552-mystery-machine"

PENALTY_BY_STRATEGY = {
    "boxed_exact_capital": 0.0,
    "boxed_contains_capital": 0.5,
    "last_letter_capital": 1.0,
    "tail10_capital": 1.5,
    "no_answer_found": 3.0,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise SystemExit(f"{path}:{lineno}: each row must be a JSON object")
            rows.append(item)
    if not rows:
        raise SystemExit(f"{path}: file is empty")
    return rows


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".json"}:
        rows = _read_jsonl(path) if path.suffix.lower() == ".jsonl" else json.loads(path.read_text(encoding="utf-8"))
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
        raise SystemExit(f"{path}: JSON files must contain a list of objects")
    raise SystemExit(f"Unsupported dataset format: {path.suffix}. Use .jsonl or .json")


def _get_first(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            value = str(row[key]).strip()
            if value:
                return value
    raise SystemExit(f"Missing one of {keys} in row: {row}")


def _is_choice_answer(answer: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]", str(answer).strip().upper()))


def _split_rows(rows: list[dict[str, Any]], validation_split: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 <= validation_split < 1.0:
        raise SystemExit("--validation-split must be in [0, 1)")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    if validation_split == 0.0 or len(shuffled) < 2:
        return shuffled, []
    val_count = max(1, int(round(len(shuffled) * validation_split)))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def _disable_peft_bnb_dispatch() -> None:
    """Prevent PEFT from importing bitsandbytes when not quantizing."""
    try:
        import peft.tuners.lora.model as lora_model

        lora_model.is_bnb_available = lambda: False
        lora_model.is_bnb_4bit_available = lambda: False
        print("[info] Disabled PEFT bitsandbytes dispatch.")
    except Exception as exc:
        print(f"[warn] Could not disable PEFT bitsandbytes dispatch: {exc}")


def _format_example(tokenizer, row: dict[str, Any]) -> str:
    prompt = _get_first(row, PROMPT_KEYS)
    answer = _get_first(row, ANSWER_KEYS)
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer.strip()},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )


def _evaluation_penalty(strategy: str) -> float:
    return PENALTY_BY_STRATEGY.get(strategy, PENALTY_BY_STRATEGY["no_answer_found"])


def _extract_prediction_text(tokenizer, model, prompt: str, *, max_new_tokens: int, temperature: float, top_p: float) -> str:
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            **gen_kwargs,
        )
    prompt_length = inputs["input_ids"].shape[-1]
    completion = generated[0][prompt_length:]
    return tokenizer.decode(completion, skip_special_tokens=False)


def _score_prediction(prediction: str, reference: str) -> tuple[bool, float, str, str | None]:
    extracted, strategy = extract_boxed_answer_with_fallbacks(prediction)
    reference = reference.strip()

    if _is_choice_answer(reference):
        correct = extracted is not None and extracted.upper() == reference.upper()
        penalty = 0.0 if correct else _evaluation_penalty(strategy)
        return correct, penalty, strategy, extracted

    boxed_match = re.search(r"\\boxed\{(.*)\}", prediction)
    if boxed_match:
        candidate = boxed_match.group(1).strip()
        correct = is_equiv(candidate, reference)
    else:
        candidate = extracted
        correct = candidate is not None and is_equiv(candidate, reference)
    penalty = 0.0 if correct else _evaluation_penalty(strategy)
    return correct, penalty, strategy, extracted


def _evaluate_model(
    tokenizer,
    model,
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    if not rows:
        return {
            "n_examples": 0,
            "accuracy": 0.0,
            "mean_penalty": 0.0,
            "strategy_counts": {},
            "details": [],
        }

    model.eval()
    correct_count = 0
    penalty_sum = 0.0
    strategy_counts: Counter[str] = Counter()
    details: list[dict[str, Any]] = []

    for row in rows:
        prompt = _get_first(row, PROMPT_KEYS)
        reference = _get_first(row, ANSWER_KEYS)
        prediction = _extract_prediction_text(
            tokenizer,
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        correct, penalty, strategy, extracted = _score_prediction(prediction, reference)
        correct_count += int(correct)
        penalty_sum += penalty
        strategy_counts[strategy] += 1
        details.append(
            {
                "prompt": prompt,
                "reference": reference,
                "prediction": prediction,
                "extracted": extracted,
                "strategy": strategy,
                "correct": correct,
                "penalty": penalty,
            }
        )

    n_examples = len(rows)
    return {
        "n_examples": n_examples,
        "accuracy": correct_count / n_examples,
        "mean_penalty": penalty_sum / n_examples,
        "strategy_counts": dict(strategy_counts),
        "details": details,
    }


def _save_tokenizer_artifacts(tokenizer, output_dir: Path) -> None:
    tokenizer.save_pretrained(output_dir)

    tokenizer_config_path = output_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        data = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        data["is_local"] = False
        data.pop("chat_template", None)
        tokenizer_config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if tokenizer.chat_template:
        (output_dir / "chat_template.jinja").write_text(tokenizer.chat_template, encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA post-training for JSONL prompt/answer datasets.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional W&B run name")
    parser.add_argument("--model-folder", required=True, type=Path, help="Base model folder to train from")
    parser.add_argument("--dataset-file", required=True, type=Path, help="JSONL/JSON file with prompt/answer rows")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where to write the merged checkpoint")
    parser.add_argument("--validation-file", type=Path, default=None, help="Optional validation JSONL/JSON file")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Holdout fraction when no validation file is provided")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature for evaluation")
    parser.add_argument("--top-p", type=float, default=0.95, help="Generation top-p for evaluation")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", choices=("epoch", "steps", "no"), default="epoch")
    parser.add_argument("--eval-steps", type=int, default=0, help="Only used when save-strategy is steps")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default=WANDB_PROJECT)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-key", type=str, default=None, help="Optional W&B API key (sets WANDB_API_KEY and logs in)")
    parser.add_argument("--skip-wandb", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--merge-and-save", action="store_true", default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.model_folder.exists():
        raise SystemExit(f"Model folder not found: {args.model_folder}")
    if not args.dataset_file.exists():
        raise SystemExit(f"Dataset file not found: {args.dataset_file}")
    if args.validation_file is not None and not args.validation_file.exists():
        raise SystemExit(f"Validation file not found: {args.validation_file}")

    if args.output_dir.exists():
        if not args.overwrite_output:
            raise SystemExit(f"Output dir already exists: {args.output_dir} (pass --overwrite-output)")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("This script expects a CUDA-capable GPU for LoRA post-training.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_folder, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    run_name = args.run_name or args.output_dir.name
    if not args.skip_wandb:
        # If the user provided an API key on the CLI, set it and call wandb.login().
        # This mirrors `wandb login` behaviour in non-interactive scripts.
        if args.wandb_key:
            os.environ["WANDB_API_KEY"] = args.wandb_key
            try:
                wandb.login(key=args.wandb_key)
            except Exception as exc:
                print(f"[warn] wandb.login() failed: {exc}")
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_folder,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(DEFAULT_LORA_TARGET_MODULES),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_rows = _load_rows(args.dataset_file)
    if args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]

    if args.validation_file is not None:
        eval_rows = _load_rows(args.validation_file)
    else:
        train_rows, eval_rows = _split_rows(train_rows, args.validation_split, args.seed)

    if args.max_eval_samples > 0:
        eval_rows = eval_rows[: args.max_eval_samples]

    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None

    sft_kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir / "lora_adapter"),
        "num_train_epochs": args.num_train_epochs,
        "max_steps": -1,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "bf16": torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": True,
        "optim": "adamw_torch",
        "logging_steps": args.logging_steps,
        "report_to": "wandb" if not args.skip_wandb else "none",
        "run_name": run_name,
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
        "do_train": True,
        "do_eval": eval_dataset is not None,
        "eval_on_start": eval_dataset is not None,
        "seed": args.seed,
        "data_seed": args.seed,
        "max_length": args.max_seq_length,
        "packing": False,
        "assistant_only_loss": True,
    }
    if eval_dataset is not None:
        sft_kwargs.update(
            {
                "eval_strategy": "steps",
                "eval_steps": args.eval_steps,
                "save_strategy": "steps",
                "save_steps": args.save_steps,
                "save_total_limit": 3,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        )
    else:
        sft_kwargs.update(
            {
                "eval_strategy": "no",
                "save_strategy": "no",
                "load_best_model_at_end": False,
            }
        )

    sft_config = SFTConfig(**sft_kwargs)

    _disable_peft_bnb_dispatch()

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=lambda example: _format_example(tokenizer, example),
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()

    eval_report: dict[str, Any] = {"n_examples": 0, "accuracy": 0.0, "mean_penalty": 0.0, "strategy_counts": {}, "details": []}
    if eval_rows:
        eval_report = _evaluate_model(
            tokenizer,
            trainer.model,
            eval_rows,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    if args.merge_and_save:
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(args.output_dir, safe_serialization=True)
        _save_tokenizer_artifacts(tokenizer, args.output_dir)

    report = {
        "dataset_file": str(args.dataset_file),
        "validation_file": str(args.validation_file) if args.validation_file is not None else None,
        "n_train_examples": len(train_rows),
        "n_eval_examples": len(eval_rows),
        "evaluation": eval_report,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": list(DEFAULT_LORA_TARGET_MODULES),
        },
    }
    report_path = args.output_dir / "post_training_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not args.skip_wandb:
        wandb.log(
            {
                "evaluation/accuracy": eval_report["accuracy"],
                "evaluation/mean_penalty": eval_report["mean_penalty"],
                "evaluation/n_examples": eval_report["n_examples"],
            }
        )
        wandb.finish()

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote merged checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()