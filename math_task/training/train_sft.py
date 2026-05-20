"""LoRA SFT for Qwen3-1.7B on math reasoning data.

Run from the pod with the JSONL produced by `math_task.data.prepare_sft`:

    python -m math_task.training.train_sft \\
        --data /scratch/<user>/sft_data/openr1_math_5k.jsonl \\
        --output /scratch/<user>/ckpt/math_sft_v1 \\
        --epochs 2 \\
        --lr 2e-4

Saves a LoRA adapter every `--save-steps` (default 100), keeping ALL
checkpoints so we can pick the best by held-out eval. Adapter only -- the
base model isn't copied, so each checkpoint is ~50-100 MB.

After training, merge + push the chosen checkpoint with:
    python -m math_task.training.push_checkpoint <ckpt-dir>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


BASE_MODEL = "Qwen/Qwen3-1.7B"
CHAT_TEMPLATE_PATH = Path(__file__).parent / "chat_template.jinja"
DEFAULT_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "gate_proj", "down_proj",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True,
                    help="Path to the SFT JSONL produced by prepare_sft.py.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Directory for LoRA checkpoints. Will be created.")

    # Training hyperparameters
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="LoRA LR. 2e-4 is standard for LoRA on small models.")
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--per-device-eval-batch", type=int, default=1,
                    help="Trainer's default is 8, which OOMs at high "
                         "max_length × vocab on a 40GB A100. Keep at 1.")
    ap.add_argument("--grad-accum", type=int, default=16,
                    help="Effective batch = per-device-batch * grad-accum.")
    ap.add_argument("--max-length", type=int, default=8192,
                    help="Sequence length cap. 8192 fits comfortably under "
                         "the 16k inference budget.")
    ap.add_argument("--warmup-ratio", type=float, default=0.03)

    # LoRA config
    ap.add_argument("--lora-r", type=int, default=32,
                    help="LoRA rank. 32 is generous but still cheap at 1.7B.")
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    # Checkpointing
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--eval-frac", type=float, default=0.02,
                    help="Fraction of data held out for eval loss.")

    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb-project", default="cs552-math-sft",
                    help="W&B project name. Set WANDB_API_KEY to enable.")
    args = ap.parse_args()

    # Lazy imports so --help works without the heavy stack.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    if os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        report_to = "wandb"
    else:
        report_to = "none"
        print("[train] WANDB_API_KEY not set; metrics will only print to stdout.")

    args.output.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    print(f"[data] loading {args.data}")
    ds_full = load_dataset("json", data_files=str(args.data), split="train")
    print(f"[data] {len(ds_full)} examples")
    split = ds_full.train_test_split(test_size=args.eval_frac, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"[data] train={len(train_ds)}  eval={len(eval_ds)}")

    # --------------------------------------------------------------- tokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    # Use the team's thinking-ON chat template so training EXACTLY matches
    # what gets pushed to HF and served by vLLM at eval time.
    tok.chat_template = CHAT_TEMPLATE_PATH.read_text()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Sanity-render to confirm template + tokenizer agree.
    sample_msgs = [{"role": "user", "content": "What is 2+2?"}]
    rendered = tok.apply_chat_template(sample_msgs, tokenize=False, add_generation_prompt=True)
    print(f"[tokenizer] template renders to (last 80 chars): ...{rendered[-80:]!r}")
    assert rendered.endswith("<|im_start|>assistant\n<think>\n"), (
        "Chat template doesn't end with thinking-ON generation prompt."
    )

    # ------------------------------------------------------------------ model
    print(f"[model] loading {BASE_MODEL} (bf16, flash-attn-2 if available)")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False  # incompatible with grad checkpointing

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=DEFAULT_LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ----------------------------------------------------------------- config
    sft_args: Dict[str, Any] = dict(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,

        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_eval_batch,
        gradient_accumulation_steps=args.grad_accum,
        eval_accumulation_steps=1,    # offload eval predictions to CPU one batch at a time

        max_length=args.max_length,
        packing=False,                  # incompatible with completion-only loss
        completion_only_loss=True,      # mask user prompt from loss

        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=None,          # KEEP ALL checkpoints
        save_only_model=True,           # adapter only; drops optimizer state

        eval_strategy="steps",
        eval_steps=args.eval_steps,

        seed=args.seed,
        report_to=report_to,
        run_name=args.output.name,
    )

    cfg = SFTConfig(**sft_args)

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        peft_config=peft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
    )

    print(f"[train] starting; checkpoints will land in {args.output}")
    trainer.train()
    trainer.save_model()

    # Dump the final training config for reproducibility.
    with open(args.output / "training_args.json", "w") as f:
        json.dump(
            {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            f, indent=2,
        )
    print(f"[done] training complete; final adapter in {args.output}")
    print(f"[done] all checkpoint subdirs (for evaluation):")
    for sub in sorted(args.output.glob("checkpoint-*")):
        print(f"        {sub}")


if __name__ == "__main__":
    main()
