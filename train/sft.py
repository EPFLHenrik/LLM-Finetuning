"""Supervised Fine-Tuning (SFT) with LoRA for Qwen3-1.7B.

Usage:
    uv run train/sft.py \\
        --datasets safetybench wildguardmix \\
        [--run-id my_run] \\
        [--resume checkpoints/20260506-143022/step-200]

Dependencies (add to pyproject.toml if missing):
    peft>=0.14

Checkpoint layout:
    checkpoints/
      <run-id>/              # e.g. 20260506-143022  or  --run-id value
        step-<N>/
          adapter/           # LoRA adapter weights (adapter_model.safetensors + config)
          optimizer.pt       # AdamW state
          scheduler.pt       # LR-scheduler state
          state.json         # {global_step, epoch, best_eval_loss, args}
"""

import argparse
import json
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from constants.global_variables import DatasetFieldNames

# ── Constants ──────────────────────────────────────────────────────────────────

BASE_MODEL   = "Qwen/Qwen3-1.7B"
DATA_DIR     = Path("data/processed/sft")
CHECKPOINT_DIR = Path("checkpoints")

# LoRA
LORA_R              = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training defaults
DEFAULT_EPOCHS      = 3
DEFAULT_BATCH_SIZE  = 4
DEFAULT_GRAD_ACCUM  = 8
DEFAULT_LR          = 2e-4
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_SAVE_STEPS  = 100


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT with LoRA on Qwen3-1.7B")
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        help=(
            "Dataset names to train on (e.g. safetybench wildguardmix). "
            "Each must exist under data/processed/sft/<name>/train and .../test."
        ),
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Run identifier used as the checkpoint folder name (default: timestamp).",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a specific step checkpoint to resume from (e.g. checkpoints/run/step-200). "
             "If omitted, the latest checkpoint in the run folder is used automatically.",
    )
    parser.add_argument("--epochs",        type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size",    type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum",    type=int,   default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lr",            type=float, default=DEFAULT_LR)
    parser.add_argument("--warmup-ratio",  type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--max-seq-len",   type=int,   default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument(
        "--save-steps", type=int, default=DEFAULT_SAVE_STEPS,
        help="Save a checkpoint every N optimizer steps.",
    )
    return parser.parse_args()


# ── Data ───────────────────────────────────────────────────────────────────────

def load_split(dataset_names: list[str], split: str):
    """Load and concatenate one split across all requested datasets."""
    parts = []
    for name in dataset_names:
        path = DATA_DIR / name / split
        if not path.exists():
            raise FileNotFoundError(f"Dataset split not found: {path}")
        parts.append(load_from_disk(str(path)))
    return concatenate_datasets(parts) if len(parts) > 1 else parts[0]


def encode_example(tokenizer, max_seq_len: int, prompt: str, target: str) -> dict:
    """
    Build input_ids and labels for one (prompt, target) pair.
    The prompt portion is masked with -100 so the loss is computed on the
    assistant response only.
    """
    messages_full   = [{"role": "user", "content": prompt}, {"role": "assistant", "content": target}]
    messages_prompt = [{"role": "user", "content": prompt}]

    full_text   = tokenizer.apply_chat_template(messages_full,   tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)

    full_ids   = tokenizer(full_text,   add_special_tokens=False)["input_ids"]
    prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

    input_ids = full_ids[:max_seq_len]
    labels    = ([-100] * prompt_len + full_ids[prompt_len:])[:max_seq_len]

    return {"input_ids": input_ids, "labels": labels}


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, max_seq_len: int):
        self.examples = [
            encode_example(
                tokenizer, max_seq_len,
                row[DatasetFieldNames.SFT_INPUT],
                row[DatasetFieldNames.SFT_TARGET],
            )
            for row in hf_dataset
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    """Right-align sequences (left-pad) so attention is on real tokens."""
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attention_mask = [], [], []
    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids.append(     [pad_token_id] * pad_len + x["input_ids"])
        labels.append(        [-100]         * pad_len + x["labels"])
        attention_mask.append([0]            * pad_len + [1] * len(x["input_ids"]))
    return {
        "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
        "labels":         torch.tensor(labels,         dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


# ── Checkpointing ──────────────────────────────────────────────────────────────

def save_checkpoint(run_dir: Path, step: int, model, optimizer, scheduler, state: dict):
    ckpt_dir = run_dir / f"step-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(ckpt_dir / "adapter")                          # LoRA adapter
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
    with open(ckpt_dir / "state.json", "w") as f:
        json.dump(state, f, indent=2)

    print(f"  [checkpoint] saved → {ckpt_dir}")


def load_checkpoint(ckpt_path: Path, model, optimizer, scheduler) -> dict:
    """Restore adapter weights and training state from a checkpoint folder."""
    print(f"Resuming from: {ckpt_path}")
    model.load_adapter(str(ckpt_path / "adapter"), adapter_name="default")
    optimizer.load_state_dict(torch.load(ckpt_path / "optimizer.pt", weights_only=True))
    scheduler.load_state_dict(torch.load(ckpt_path / "scheduler.pt", weights_only=True))
    with open(ckpt_path / "state.json") as f:
        return json.load(f)


def latest_checkpoint(run_dir: Path) -> Path | None:
    """Return the step folder with the highest step number inside run_dir, or None."""
    if not run_dir.exists():
        return None
    candidates = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("step-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    return candidates[-1] if candidates else None


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model, loader: DataLoader, device) -> float:
    model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=      batch["input_ids"].to(device),
                attention_mask= batch["attention_mask"].to(device),
                labels=         batch["labels"].to(device),
            )
            total_loss += outputs.loss.item()
            n_batches  += 1
    model.train()
    return total_loss / n_batches if n_batches else float("inf")


# ── Training loop ──────────────────────────────────────────────────────────────

def train(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id  = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = CHECKPOINT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("SFT TRAINING")
    print(f"  Datasets   : {args.datasets}")
    print(f"  Base model : {BASE_MODEL}")
    print(f"  Run dir    : {run_dir}")
    print(f"  Device     : {device}")
    print("=" * 56)

    # ── Tokenizer & model ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # ── Datasets & loaders ─────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_ds = SFTDataset(load_split(args.datasets, "train"), tokenizer, args.max_seq_len)
    test_ds  = SFTDataset(load_split(args.datasets, "test"),  tokenizer, args.max_seq_len)
    print(f"  Train examples : {len(train_ds)}")
    print(f"  Test  examples : {len(test_ds)}\n")

    _collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=_collate)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=_collate)

    # ── Optimizer & scheduler ──────────────────────────────────────────────────
    optimizer    = AdamW(model.parameters(), lr=args.lr)
    total_steps  = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # ── Resume from checkpoint if available ────────────────────────────────────
    global_step    = 0
    start_epoch    = 0
    best_eval_loss = float("inf")

    resume_path = Path(args.resume) if args.resume else latest_checkpoint(run_dir)
    if resume_path and resume_path.exists():
        saved          = load_checkpoint(resume_path, model, optimizer, scheduler)
        global_step    = saved["global_step"]
        start_epoch    = saved["epoch"]
        best_eval_loss = saved.get("best_eval_loss", float("inf"))
        print(f"Resumed at global_step={global_step}, epoch={start_epoch}\n")

    # ── Main loop ──────────────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()

    for epoch in range(start_epoch, args.epochs):
        print(f"── Epoch {epoch + 1}/{args.epochs} ──────────────────────────────")
        accum_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            outputs = model(
                input_ids=      batch["input_ids"].to(device),
                attention_mask= batch["attention_mask"].to(device),
                labels=         batch["labels"].to(device),
            )
            loss = outputs.loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0:
                    print(
                        f"  step {global_step:>6} | "
                        f"train_loss {accum_loss:.4f} | "
                        f"lr {scheduler.get_last_lr()[0]:.2e}"
                    )
                    accum_loss = 0.0

                if global_step % args.save_steps == 0:
                    eval_loss  = evaluate(model, test_loader, device)
                    is_best    = eval_loss < best_eval_loss
                    best_eval_loss = min(best_eval_loss, eval_loss)
                    print(f"  [eval]  loss {eval_loss:.4f}{' ← best' if is_best else ''}")
                    save_checkpoint(
                        run_dir, global_step, model, optimizer, scheduler,
                        state={
                            "global_step":    global_step,
                            "epoch":          epoch,
                            "best_eval_loss": best_eval_loss,
                            "args":           vars(args),
                        },
                    )

    # ── Final checkpoint ───────────────────────────────────────────────────────
    eval_loss = evaluate(model, test_loader, device)
    print(f"\nFinal eval loss: {eval_loss:.4f}")
    save_checkpoint(
        run_dir, global_step, model, optimizer, scheduler,
        state={
            "global_step":    global_step,
            "epoch":          args.epochs,
            "best_eval_loss": min(best_eval_loss, eval_loss),
            "args":           vars(args),
        },
    )
    print("Training complete.")


if __name__ == "__main__":
    train(parse_args())
