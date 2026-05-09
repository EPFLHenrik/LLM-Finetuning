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
import importlib
import json
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import torch
from datasets import concatenate_datasets, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from constants.global_variables import DatasetFieldNames

try:
    _vllm_mod = importlib.import_module("vllm")
    _vllm_lora_mod = importlib.import_module("vllm.lora.request")
    LLM = _vllm_mod.LLM
    SamplingParams = _vllm_mod.SamplingParams
    LoRARequest = _vllm_lora_mod.LoRARequest
    HAS_VLLM = True
except Exception:
    LLM = None
    SamplingParams = None
    LoRARequest = None
    HAS_VLLM = False


def disable_peft_bitsandbytes_dispatch() -> None:
    """Prevent PEFT LoRA from importing bitsandbytes when not using quantization.

    In some CUDA/container setups, a broken bitsandbytes install is importable
    but fails at runtime. PEFT checks these helpers before importing its bnb
    dispatch path, so forcing them to False keeps standard LoRA on torch layers.
    """
    try:
        import peft.tuners.lora.model as lora_model

        lora_model.is_bnb_available = lambda: False
        lora_model.is_bnb_4bit_available = lambda: False
        print("[info] Disabled PEFT bitsandbytes dispatch for this run.")
    except Exception as exc:
        print(f"[warn] Failed to disable PEFT bitsandbytes dispatch: {exc}")

# ── Constants ──────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen3-1.7B"
DATA_DIR = Path("data/processed/sft")
CHECKPOINT_DIR = Path("checkpoints")

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Training defaults
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 32
DEFAULT_LR = 2e-4
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_MAX_SEQ_LEN = 2048
DEFAULT_SAVE_STEPS = 100
DEFAULT_MAX_EVAL_SAMPLES = 256


# ── Argument parsing ───────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT with LoRA on Qwen3-1.7B")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help=(
            "Dataset names to train on (e.g. safetybench wildguardmix). "
            "Each must exist under data/processed/sft/<name>/train and .../test."
        ),
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier used as the checkpoint folder name (default: timestamp).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a specific step checkpoint to resume from (e.g. checkpoints/run/step-200). "
        "If omitted, the latest checkpoint in the run folder is used automatically.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument(
        "--save-steps",
        type=int,
        default=DEFAULT_SAVE_STEPS,
        help="Save a checkpoint every N optimizer steps.",
    )
    parser.add_argument(
        "--eval-backend",
        choices=["torch", "vllm"],
        default="vllm",
        help=(
            "Evaluation backend. 'vllm' is generally faster for inference-style evaluation, "
            "but SFT training still uses Transformers/PEFT. Falls back to torch on errors."
        ),
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=DEFAULT_MAX_EVAL_SAMPLES,
        help="Maximum number of test examples to use for periodic/final evaluation.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Fraction of GPU memory to reserve for vLLM when --eval-backend=vllm.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable gradient checkpointing to reduce activation memory during training "
            "(default: enabled)."
        ),
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


def maybe_limit_dataset(hf_dataset, max_examples: int):
    if max_examples <= 0 or len(hf_dataset) <= max_examples:
        return hf_dataset
    return hf_dataset.select(range(max_examples))


def encode_example(tokenizer, max_seq_len: int, prompt: str, target: str) -> dict:
    """
    Build input_ids and labels for one (prompt, target) pair.
    The prompt portion is masked with -100 so the loss is computed on the
    assistant response only.
    """
    messages_full = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target},
    ]
    messages_prompt = [{"role": "user", "content": prompt}]

    full_text = tokenizer.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False
    )
    prompt_text = tokenizer.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True
    )

    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

    input_ids = full_ids[:max_seq_len]
    labels = ([-100] * prompt_len + full_ids[prompt_len:])[:max_seq_len]

    return {"input_ids": input_ids, "labels": labels}


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, max_seq_len: int):
        self.examples = [
            encode_example(
                tokenizer,
                max_seq_len,
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
        input_ids.append([pad_token_id] * pad_len + x["input_ids"])
        labels.append([-100] * pad_len + x["labels"])
        attention_mask.append([0] * pad_len + [1] * len(x["input_ids"]))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


# ── Checkpointing ──────────────────────────────────────────────────────────────


def save_checkpoint(run_dir: Path, step: int, model, optimizer, scheduler, state: dict):
    ckpt_dir = run_dir / f"step-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(ckpt_dir / "adapter")  # LoRA adapter
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
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            total_loss += outputs.loss.item()
            n_batches += 1
    model.train()
    return total_loss / n_batches if n_batches else float("inf")


def _extract_token_logprob(
    logprob_dict: dict[int, Any] | None, token_id: int
) -> float | None:
    if not logprob_dict or token_id not in logprob_dict:
        return None
    entry = logprob_dict[token_id]
    if hasattr(entry, "logprob"):
        return float(entry.logprob)
    try:
        return float(entry)
    except (TypeError, ValueError):
        return None


def evaluate_with_vllm(
    tokenizer,
    hf_dataset,
    max_seq_len: int,
    base_model: str,
    adapter_path: Path,
    gpu_memory_utilization: float,
) -> float:
    if not HAS_VLLM:
        raise RuntimeError("vLLM is not available in this environment.")

    eval_records = []
    for row in hf_dataset:
        prompt = row[DatasetFieldNames.SFT_INPUT]
        target = row[DatasetFieldNames.SFT_TARGET]
        messages_full = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ]
        messages_prompt = [{"role": "user", "content": prompt}]

        full_text = tokenizer.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )
        prompt_text = tokenizer.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )

        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

        full_ids = full_ids[:max_seq_len]
        if len(full_ids) <= 1:
            continue

        prompt_len = min(prompt_len, len(full_ids) - 1)
        truncated_text = tokenizer.decode(full_ids, skip_special_tokens=False)
        eval_records.append((truncated_text, prompt_len))

    if not eval_records:
        return float("inf")

    llm = LLM(
        model=base_model,
        enable_lora=True,
        dtype="auto",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_seq_len,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=1,
    )
    lora_request = LoRARequest("sft_adapter", 1, str(adapter_path))

    prompts = [record[0] for record in eval_records]
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_request,
        use_tqdm=False,
    )

    total_nll = 0.0
    total_tokens = 0
    for output, (_, prompt_len) in zip(outputs, eval_records):
        token_ids = output.prompt_token_ids
        token_logprobs = output.prompt_logprobs or []

        usable = min(len(token_ids), len(token_logprobs))
        for pos in range(prompt_len, usable):
            logprob = _extract_token_logprob(token_logprobs[pos], token_ids[pos])
            if logprob is None:
                continue
            total_nll -= logprob
            total_tokens += 1

    if total_tokens == 0:
        return float("inf")
    return total_nll / total_tokens


# ── Training loop ──────────────────────────────────────────────────────────────


def train(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = CHECKPOINT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print("SFT TRAINING")
    print(f"  Datasets   : {args.datasets}")
    print(f"  Base model : {BASE_MODEL}")
    print(f"  Run dir    : {run_dir}")
    print(f"  Device     : {device}")
    print(f"  Eval mode  : {args.eval_backend}")
    print("=" * 56)

    if args.eval_backend == "vllm" and not HAS_VLLM:
        print("[warn] vLLM not installed/available; falling back to torch evaluation.")
        args.eval_backend = "torch"
    if args.eval_backend == "vllm" and device.type != "cuda":
        print("[warn] vLLM evaluation requires CUDA; falling back to torch evaluation.")
        args.eval_backend = "torch"

    # ── Tokenizer & model ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        device_map={"": device},
    )

    disable_peft_bitsandbytes_dispatch()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Required for checkpointing with frozen embeddings in many causal LMs.
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    # ── Datasets & loaders ─────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_hf = load_split(args.datasets, "train")
    test_hf = load_split(args.datasets, "test")
    test_eval_hf = maybe_limit_dataset(test_hf, args.max_eval_samples)

    train_ds = SFTDataset(train_hf, tokenizer, args.max_seq_len)
    test_ds = SFTDataset(test_eval_hf, tokenizer, args.max_seq_len)
    print(f"  Train examples : {len(train_ds)}")
    print(f"  Test  examples : {len(test_hf)}")
    if len(test_eval_hf) != len(test_hf):
        print(f"  Eval subset   : {len(test_eval_hf)}")
    print()

    _collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate
    )

    # ── Optimizer & scheduler ──────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Resume from checkpoint if available ────────────────────────────────────
    global_step = 0
    start_epoch = 0
    best_eval_loss = float("inf")

    resume_path = Path(args.resume) if args.resume else latest_checkpoint(run_dir)
    if resume_path and resume_path.exists():
        saved = load_checkpoint(resume_path, model, optimizer, scheduler)
        global_step = saved["global_step"]
        start_epoch = saved["epoch"]
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
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
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
                    eval_loss = evaluate(model, test_loader, device)
                    is_best = eval_loss < best_eval_loss
                    best_eval_loss = min(best_eval_loss, eval_loss)
                    print(
                        f"  [eval]  loss {eval_loss:.4f}{' ← best' if is_best else ''}"
                    )
                    save_checkpoint(
                        run_dir,
                        global_step,
                        model,
                        optimizer,
                        scheduler,
                        state={
                            "global_step": global_step,
                            "epoch": epoch,
                            "best_eval_loss": best_eval_loss,
                            "args": vars(args),
                        },
                    )

    # ── Final checkpoint ───────────────────────────────────────────────────────
    final_eval_loss = evaluate(model, test_loader, device)
    eval_metric_name = "torch_nll"

    if args.eval_backend == "vllm":
        final_eval_adapter_dir = run_dir / ".final_eval_adapter"
        try:
            if final_eval_adapter_dir.exists():
                shutil.rmtree(final_eval_adapter_dir)
            model.save_pretrained(final_eval_adapter_dir)

            vllm_eval = evaluate_with_vllm(
                tokenizer=tokenizer,
                hf_dataset=test_eval_hf,
                max_seq_len=args.max_seq_len,
                base_model=BASE_MODEL,
                adapter_path=final_eval_adapter_dir,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            )
            final_eval_loss = vllm_eval
            eval_metric_name = "vllm_nll"
        except Exception as exc:
            print(f"[warn] vLLM evaluation failed ({exc}); keeping torch final eval.")
        finally:
            if final_eval_adapter_dir.exists():
                shutil.rmtree(final_eval_adapter_dir, ignore_errors=True)

    print(f"\nFinal eval ({eval_metric_name}): {final_eval_loss:.4f}")
    save_checkpoint(
        run_dir,
        global_step,
        model,
        optimizer,
        scheduler,
        state={
            "global_step": global_step,
            "epoch": args.epochs,
            "best_eval_loss": min(best_eval_loss, final_eval_loss),
            "args": vars(args),
        },
    )
    print("Training complete.")


if __name__ == "__main__":
    train(parse_args())
