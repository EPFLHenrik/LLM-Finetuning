#!/usr/bin/env python3
"""Train a model with DPO on (prompt, chosen, rejected) pairs.

Expected input format (JSONL, one row per line):
  {
    "prompt":   "<question text>",
    "chosen":   "<correct assistant completion, may include <think>...</think>>",
    "rejected": "<wrong assistant completion>"
  }

Use scripts/build_dpo_dataset.py to generate this file from student run files
and the teacher distillation corpus.
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def _build_prompt_messages(prompt_text: str, system_message: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt_text},
    ]


def _normalize_row(row: dict[str, Any], system_message: str) -> dict[str, Any]:
    prompt_text = str(row.get("prompt", "")).strip()
    chosen = str(row.get("chosen", "")).strip()
    rejected = str(row.get("rejected", "")).strip()
    if not prompt_text:
        raise SystemExit(f"Row missing 'prompt': {row}")
    if not chosen:
        raise SystemExit(f"Row missing 'chosen': {row}")
    if not rejected:
        raise SystemExit(f"Row missing 'rejected': {row}")
    return {
        "prompt": _build_prompt_messages(prompt_text, system_message),
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


def _save_merged_model(trainer: DPOTrainer, tokenizer: AutoTokenizer, output_dir: Path) -> None:
    _disable_peft_bnb_dispatch()
    from peft import PeftModel
    output_dir.mkdir(parents=True, exist_ok=True)
    model = trainer.model
    save_dtype = (
        torch.bfloat16 if trainer.args.bf16 else torch.float16 if trainer.args.fp16 else None
    )
    if isinstance(model, PeftModel) or hasattr(model, "merge_and_unload"):
        merged = model.merge_and_unload()
        if save_dtype is not None:
            merged = merged.to(save_dtype)
        merged.save_pretrained(str(output_dir), safe_serialization=True)
        if getattr(merged, "generation_config", None) is not None:
            merged.generation_config.save_pretrained(str(output_dir))
    else:
        if save_dtype is not None:
            trainer.model.to(save_dtype)
        trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DPO fine-tuning on (prompt, chosen, rejected) pairs.")
    parser.add_argument("--model-folder", required=True, type=Path)
    parser.add_argument("--dataset-file", required=True, type=Path,
                        help="JSONL with prompt/chosen/rejected rows (from build_dpo_dataset.py)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096,
                        help="Max total sequence length (prompt + completion)")
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-strategy", choices=("epoch", "steps", "no"), default="epoch")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty coefficient (default: 0.1)")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", action="store_false", dest="use_lora")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str,
                        default=",".join(DEFAULT_LORA_TARGET_MODULES))
    parser.add_argument("--wandb-project", type=str, default="cs552-mystery-machine")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-key", type=str, default=None)
    parser.add_argument("--skip-wandb", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
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
        raise SystemExit("This script expects a CUDA-capable GPU.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    rows = _read_jsonl(args.dataset_file)
    normalized = [_normalize_row(r, DEFAULT_SYSTEM_MESSAGE) for r in rows]
    if not normalized:
        raise SystemExit("No usable rows found after normalization")

    dataset = Dataset.from_list(normalized).shuffle(seed=args.seed)
    if args.max_train_samples > 0:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))
    print(f"Training on {len(dataset)} DPO pairs")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_folder), use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit(f"Tokenizer has no chat template: {args.model_folder}")

    if not args.skip_wandb:
        import wandb
        try:
            wandb.login(key=args.wandb_key)
        except Exception:
            pass
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=args.output_dir.name,
        )

    dpo_config = DPOConfig(
        output_dir=str(args.output_dir),
        seed=args.seed,
        beta=args.beta,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        report_to=[] if args.skip_wandb else ["wandb"],
        run_name=args.output_dir.name,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    _disable_peft_bnb_dispatch()
    lora_config = None
    if args.use_lora:
        from peft import LoraConfig, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
        )

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_folder),
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    _save_merged_model(trainer, tokenizer, args.output_dir)

    if not args.skip_wandb:
        import wandb
        wandb.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
