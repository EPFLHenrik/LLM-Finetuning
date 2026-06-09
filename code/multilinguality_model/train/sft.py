"""code/multilinguality_model/train/sft.py

LoRA SFT training using TRL SFTTrainer + SFTConfig.
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

from multilinguality_model.config import (
    REPO as MLREPO,
    SYSTEM_PROMPT_THINKING,
    SYSTEM_PROMPT_NON_THINKING,
    SYSTEM_PROMPT_NON_THINKING_AUGMENTED,
)

BASE_MODEL = "Qwen/Qwen3-1.7B"
WANDB_PROJECT = "MNLP-Multilingual"

SFT_DATA_DIR = MLREPO / "data" / "multilinguality" / "sft"
DEFAULT_TRAIN_FILE = SFT_DATA_DIR / "mixed" / "train_mixed.jsonl"
DEFAULT_VAL_FILE = SFT_DATA_DIR / "mixed" / "val_mixed.jsonl"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 8
DEFAULT_LR = 1e-4
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_SAVE_STEPS = 100
DEFAULT_EVAL_STEPS = 100
DEFAULT_LOG_STEPS = 10


def _load_jsonl(path: Path) -> Dataset:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def _select_system_prompt(name: str) -> str:
    if name == "thinking":
        return SYSTEM_PROMPT_THINKING
    if name == "non_thinking":
        return SYSTEM_PROMPT_NON_THINKING
    if name == "non_thinking_augmented":
        return SYSTEM_PROMPT_NON_THINKING_AUGMENTED
    raise ValueError(f"Unknown system_prompt: {name}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--train_file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--val_file", type=Path, default=DEFAULT_VAL_FILE)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad_accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)
    parser.add_argument("--lora_r", type=int, default=LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--warmup_ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--save_steps", type=int, default=DEFAULT_SAVE_STEPS)
    parser.add_argument("--eval_steps", type=int, default=DEFAULT_EVAL_STEPS)
    parser.add_argument("--log_steps", type=int, default=DEFAULT_LOG_STEPS)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Override epochs. Use with small values for debug runs.",
    )
    parser.add_argument("--wandb_project", type=str, default=WANDB_PROJECT)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--skip_wandb", action="store_true")
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="non_thinking",
        choices=["thinking", "non_thinking", "non_thinking_augmented"],
        help=(
            "System prompt variant. Ignored when --preserve_system_prompt is set."
        ),
    )
    parser.add_argument(
        "--preserve_system_prompt",
        action="store_true",
        help=(
            "Keep the system prompt already stored in each training example. "
            "Use this for reproducing historical SFT data such as v1."
        ),
    )
    parser.add_argument(
        "--use_qwen_default_chat_template",
        action="store_true",
        help=(
            "Explicitly use the default chat template from Qwen/Qwen3-1.7B. "
            "This is already the training default; the flag is kept for reproducibility clarity."
        ),
    )
    return parser.parse_args()


def _disable_peft_bnb_dispatch() -> None:
    try:
        import peft.tuners.lora.model as lora_model
        lora_model.is_bnb_available = lambda: False
        lora_model.is_bnb_4bit_available = lambda: False
        print("[info] Disabled PEFT bitsandbytes dispatch.")
    except Exception as e:
        print(f"[warn] Could not disable PEFT bitsandbytes dispatch: {e}")


def main():
    args = parse_args()

    if args.save_steps != args.eval_steps:
        print(
            f"[warn] save_steps ({args.save_steps}) != eval_steps ({args.eval_steps}). "
            f"Setting save_steps = eval_steps = {args.eval_steps}."
        )
        args.save_steps = args.eval_steps

    output_dir = args.output_dir or (
        MLREPO / "checkpoints" / "multilinguality_model" / args.run_name
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

    if args.use_qwen_default_chat_template:
        print("[info] Using Qwen/Qwen3-1.7B default chat template.")
    else:
        print("[info] Using Qwen/Qwen3-1.7B default chat template.")

    train_dataset = _load_jsonl(args.train_file)
    val_dataset = _load_jsonl(args.val_file)

    if args.preserve_system_prompt:
        print("[info] Preserving system prompts from data.")
    else:
        selected_system_prompt = _select_system_prompt(args.system_prompt)

        def _override_system(ex):
            msgs = list(ex.get("messages", []))
            if msgs and msgs[0].get("role") == "system":
                msgs[0] = {"role": "system", "content": selected_system_prompt}
            else:
                msgs = [{"role": "system", "content": selected_system_prompt}] + msgs
            ex["messages"] = msgs
            return ex

        train_dataset = train_dataset.map(_override_system)
        val_dataset = val_dataset.map(_override_system)
        print(f"[info] System prompt overridden with: {args.system_prompt}")

    print("[info] enable_thinking=False")

    def formatting_func(example: dict) -> str:
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

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
        eos_token="<|im_end|>",
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