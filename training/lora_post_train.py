#!/usr/bin/env python3
"""LoRA post-training for chat-style Qwen checkpoints.

This script expects the dataset to already be converted to conversational
chat-style JSONL where each row contains a `messages` array of the form
[{'role':'system','content':...},{'role':'user','content':...},{'role':'assistant','content':'\\boxed{...}'}].
Use `scripts/preprocess_mcq_chat.py` to convert flat MCQ JSONL into that format
before running this trainer.

The script:
1. loads a base model folder and a preprocessed JSONL dataset file;
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
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Evaluation utilities moved to scripts/evaluate_training.py

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



def _disable_peft_bnb_dispatch() -> None:
    """Prevent PEFT from importing bitsandbytes when not quantizing."""
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



def _ensure_training_chat_template(tokenizer) -> None:
    """Ensure the tokenizer's chat template contains a training generation marker.

    TRL requires chat templates to include prefix-preservation or a
    `{% generation %}` marker. If the loaded tokenizer template is missing
    the marker we append a generation placeholder so the template becomes
    training-compatible.
    """
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\\n' }}"
        "{% if message['role'] == 'assistant' %}"
        "{% generation %}{{ message['content'] }}{% endgeneration %}"
        "{% else %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{{ '<|im_end|>\\n' }}"
        "{% endfor %}"
    )


def _save_tokenizer_artifacts(tokenizer, output_dir: Path, source_dir: Path) -> None:
    tokenizer.save_pretrained(output_dir)

    tokenizer_config_path = output_dir / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        data = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        data["is_local"] = False
        data.pop("chat_template", None)
        tokenizer_config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    source_chat_template = source_dir / "chat_template.jinja"
    if not source_chat_template.exists():
        raise SystemExit(f"Source chat template not found: {source_chat_template}")
    shutil.copyfile(source_chat_template, output_dir / "chat_template.jinja")


def _merge_saved_adapter_checkpoints(base_model_folder: Path, adapter_root: Path, merged_root: Path, tokenizer, dtype: torch.dtype) -> list[Path]:
    checkpoint_dirs = sorted(
        path for path in adapter_root.iterdir()
        if path.is_dir() and (path / "adapter_config.json").exists()
    )
    if not checkpoint_dirs:
        print(f"[info] No adapter checkpoints found under {adapter_root}; skipping merged checkpoint export.")
        return []

    merged_root.mkdir(parents=True, exist_ok=True)
    merged_dirs: list[Path] = []
    for checkpoint_dir in checkpoint_dirs:
        merged_dir = merged_root / checkpoint_dir.name
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        print(f"[info] Merging {checkpoint_dir.name} -> {merged_dir}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_folder,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        peft_model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
        merged_model.config.save_pretrained(merged_dir)
        if getattr(merged_model, "generation_config", None) is not None:
            merged_model.generation_config.save_pretrained(merged_dir)
            _save_tokenizer_artifacts(tokenizer, merged_dir, base_model_folder)
        merged_dirs.append(merged_dir)

    return merged_dirs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA post-training for JSONL prompt/answer datasets.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional W&B run name")
    parser.add_argument("--model-folder", required=True, type=Path, help="Base model folder to train from")
    parser.add_argument("--dataset-file", required=True, type=Path, help="JSONL/JSON file with prompt/answer rows")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where to write the merged checkpoint")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", choices=("epoch", "steps", "no"), default="epoch")
    parser.add_argument("--save-steps", type=int, default=500, help="Checkpoint save interval when save-strategy is steps")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-train-samples", type=int, default=0)
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

    if args.output_dir.exists():
        if not args.overwrite_output:
            raise SystemExit(f"Output dir already exists: {args.output_dir} (pass --overwrite-output)")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("This script expects a CUDA-capable GPU for LoRA post-training.")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_folder, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if not tokenizer.chat_template:
        raise SystemExit(f"Tokenizer does not define a chat template: {args.model_folder}")

    _ensure_training_chat_template(tokenizer)        

    _disable_peft_bnb_dispatch()

    run_name = args.run_name or args.output_dir.name
    if not args.skip_wandb:
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

    # Expect the dataset to already be preprocessed into conversational JSONL
    # (use `scripts/preprocess_mcq_chat.py` to convert flat MCQ JSONL into the
    # required `messages` shape). Do not normalize in-trainer.
    train_rows = _load_rows(args.dataset_file)
    random.shuffle(train_rows)
    if args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]

    # For training, prefer purely conversational examples (no top-level 'prompt' key)
    train_rows_for_trainer: list[dict[str, Any]] = []
    for r in train_rows:
        copied = dict(r)
        copied.pop("prompt", None)
        train_rows_for_trainer.append(copied)

    train_dataset = Dataset.from_list(train_rows_for_trainer)
    eval_dataset = None

    if train_rows_for_trainer:
        preview_text = tokenizer.apply_chat_template(train_rows_for_trainer[0]["messages"], tokenize=False, add_generation_prompt=False)
        print("\n--- Sample formatted prompt ---")
        print(preview_text)
        print("------------------------------\n")

    sft_kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir / "lora_adapter"),
        "num_train_epochs": args.num_train_epochs,
        "max_steps": -1,
        "per_device_train_batch_size": args.per_device_train_batch_size,
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
        "do_eval": False,
        "eval_on_start": False,
        "max_length": args.max_seq_length,
        "packing": False,
        "assistant_only_loss": True,
    }
    # Configure saving strategy independently of evaluation
    if args.save_strategy == "steps":
        save_steps = args.save_steps if args.save_steps > 0 else args.logging_steps
        if save_steps <= 0:
            raise SystemExit("--save-steps must be positive when --save-strategy is steps")
        sft_kwargs.update({"save_strategy": "steps", "save_steps": save_steps, "save_total_limit": 3})
    else:
        sft_kwargs.update({"save_strategy": args.save_strategy, "save_total_limit": 3})

    sft_config = SFTConfig(**sft_kwargs)

    _disable_peft_bnb_dispatch()

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    if args.merge_and_save:
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(args.output_dir, safe_serialization=True)
        _save_tokenizer_artifacts(tokenizer, args.output_dir, args.model_folder)
        merged_checkpoints_dir = args.output_dir / "merged_checkpoints"
        merged_checkpoint_dirs = _merge_saved_adapter_checkpoints(
            args.model_folder,
            args.output_dir / "lora_adapter",
            merged_checkpoints_dir,
            tokenizer,
            dtype,
        )
        if merged_checkpoint_dirs:
            print(f"[info] Wrote {len(merged_checkpoint_dirs)} merged checkpoints to {merged_checkpoints_dir}")

    report = {
        "dataset_file": str(args.dataset_file),
        "n_train_examples": len(train_rows),
        "n_eval_examples": 0,
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
        wandb.finish()

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote merged checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()