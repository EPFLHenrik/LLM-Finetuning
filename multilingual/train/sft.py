"""multilingual/train/sft.py

LoRA SFT training using TRL SFTTrainer + SFTConfig (TRL 1.3.0).

Usage:
    # Debug run
    python -m multilingual.train.sft \
        --run_name debug_run \
        --max_steps 20 --eval_steps 10 --save_steps 10 --log_steps 1 --skip_wandb

    # Full run
    python -m multilingual.train.sft --run_name my_run
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from multilingual.config import REPO as MLREPO

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

BASE_MODEL    = "Qwen/Qwen3-1.7B"
WANDB_PROJECT = "MNLP-Multilingual"

SFT_DATA_DIR       = MLREPO / "multilingual" / "data" / "sft"
DEFAULT_TRAIN_FILE = SFT_DATA_DIR / "mixed" / "train_mixed.jsonl"
DEFAULT_VAL_FILE   = SFT_DATA_DIR / "mixed" / "val_mixed.jsonl"

LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05 #
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

DEFAULT_EPOCHS       = 3
DEFAULT_BATCH_SIZE   = 1
DEFAULT_GRAD_ACCUM   = 16
DEFAULT_LR           = 2e-4
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_MAX_SEQ_LEN  = 4096
DEFAULT_SAVE_STEPS   = 100
DEFAULT_EVAL_STEPS   = 100
DEFAULT_LOG_STEPS    = 10


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> Dataset:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name",   type=str,  required=True)
    parser.add_argument("--train_file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--val_file",   type=Path, default=DEFAULT_VAL_FILE)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--epochs",       type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size",   type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad_accum",   type=int,   default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lr",           type=float, default=DEFAULT_LR)
    parser.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)
    parser.add_argument("--lora_r",       type=int,   default=LORA_R)
    parser.add_argument("--lora_alpha",   type=int,   default=LORA_ALPHA)
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--max_seq_len",  type=int,   default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--save_steps",   type=int,   default=DEFAULT_SAVE_STEPS)
    parser.add_argument("--eval_steps",   type=int,   default=DEFAULT_EVAL_STEPS)
    parser.add_argument("--log_steps",    type=int,   default=DEFAULT_LOG_STEPS)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--max_steps",    type=int,   default=-1,
                        help="Override epochs. Use with small values for debug runs.")
    parser.add_argument("--wandb_project", type=str, default=WANDB_PROJECT)
    parser.add_argument("--wandb_entity",  type=str, default=None)
    parser.add_argument("--skip_wandb",    action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _disable_peft_bnb_dispatch() -> None:
    """Prevent PEFT from importing bitsandbytes when not quantizing."""
    try:
        import peft.tuners.lora.model as lora_model
        lora_model.is_bnb_available = lambda: False
        lora_model.is_bnb_4bit_available = lambda: False
        print("[info] Disabled PEFT bitsandbytes dispatch.")
    except Exception as e:
        print(f"[warn] Could not disable PEFT bitsandbytes dispatch: {e}")


def main():
    args = parse_args()

    # save_steps must equal eval_steps for load_best_model_at_end to work
    if args.save_steps != args.eval_steps:
        print(f"[warn] save_steps ({args.save_steps}) != eval_steps ({args.eval_steps}). "
              f"Setting save_steps = eval_steps = {args.eval_steps}.")
        args.save_steps = args.eval_steps

    output_dir = args.output_dir or (
        MLREPO / "multilingual" / "results" / "checkpoints" / args.run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    hf_token = os.environ.get("HF_TOKEN", "")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # formatting_func defined here to capture tokenizer without a global
    def formatting_func(example: dict) -> str:
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

    # Debug: print one formatted example to verify template output
    train_dataset = _load_jsonl(args.train_file)
    val_dataset   = _load_jsonl(args.val_file)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print("\n--- Sample formatted example ---")
    print(formatting_func(train_dataset[0]))
    print("--------------------------------\n")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        logging_steps=args.log_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if not args.skip_wandb else "none",
        run_name=args.run_name,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        do_train=True,
        do_eval=True,
        eval_on_start=True,
        seed=42,
        data_seed=42,
        # SFTConfig-specific
        max_length=args.max_seq_len,
        packing=False,
        assistant_only_loss=True,
    )

    _disable_peft_bnb_dispatch()

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_func,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    trainer.train()

    best_adapter_dir = output_dir / "best_adapter"
    trainer.model.save_pretrained(best_adapter_dir)
    tokenizer.save_pretrained(best_adapter_dir)
    print(f"Best adapter saved: {best_adapter_dir}")

    if not args.skip_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()