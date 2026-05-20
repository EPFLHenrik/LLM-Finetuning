"""Simple SFT entry point with per-dataset metrics and W&B logging.

This script keeps training orchestration readable while delegating details to:
- train/sft_data.py
- train/sft_export.py

Usage example:
  uv run train/sft.py \
    --datasets safetybench \
    --run_name safetybench_cot \
    --epochs 1
"""

from __future__ import annotations

import argparse
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from constants.global_variables import Random
from prompts.system_prompt import SYSTEM_PROMPT_MCQ
from train.sft_data import DatasetBundle, load_dataset_bundles
from train.sft_export import export_merged_checkpoint

wandb: Any
try:
    import wandb

    HAS_WANDB = True
except Exception:
    wandb = cast(Any, None)
    HAS_WANDB = False


BASE_MODEL = "Qwen/Qwen3-1.7B"

# Simple, conservative defaults.
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 16
DEFAULT_LR = 2e-4
DEFAULT_WARMUP_RATIO = 0.05

# LoRA defaults (gold-standard starter values for 1-2B models).
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


class NamedTrainDataset(Dataset):
    def __init__(self, dataset: Dataset, dataset_name: str):
        self.dataset = dataset
        self.dataset_name = dataset_name

    def __len__(self) -> int:
        return len(cast(Any, self.dataset))

    def __getitem__(self, idx: int) -> dict:
        example = dict(self.dataset[idx])
        example["dataset_name"] = self.dataset_name
        return example


def disable_peft_bitsandbytes_dispatch() -> None:
    """Avoid accidental PEFT bitsandbytes path when not quantizing."""
    try:
        import peft.tuners.lora.model as lora_model

        setattr(lora_model, "is_bnb_available", lambda: False)
        setattr(lora_model, "is_bnb_4bit_available", lambda: False)
        print("[info] Disabled PEFT bitsandbytes dispatch for this run.")
    except Exception as exc:
        print(f"[warn] Failed to disable PEFT bitsandbytes dispatch: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple SFT with LoRA on Qwen3-1.7B")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names under data/processed/sft/<name> with train and test splits.",
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional per-dataset train sample caps aligned with --datasets order. "
            "Use -1 for full dataset. Values above dataset size are clamped to full dataset."
        ),
    )
    parser.add_argument(
        "--test-sizes",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional per-dataset test sample caps aligned with --datasets order. "
            "Use -1 for full dataset. Values above dataset size are clamped to full dataset."
        ),
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help=(
            "Human-readable run name. Also used as default checkpoint folder name "
            "under checkpoints/<run_name>."
        ),
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Number of training samples between evaluations.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--best-checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Export path for best checkpoint. Defaults to checkpoints/<run_name> "
            "(or checkpoints/<run-id> when --run_name is not set)."
        ),
    )
    parser.add_argument(
        "--export-temperature",
        type=float,
        default=0.7,
        help="temperature to store in exported generation_config.json",
    )
    parser.add_argument(
        "--export-top-p",
        type=float,
        default=0.95,
        help="top_p to store in exported generation_config.json",
    )
    parser.add_argument(
        "--export-top-k",
        type=int,
        default=50,
        help="top_k to store in exported generation_config.json",
    )
    parser.add_argument(
        "--skip_wandb",
        action="store_true",
        help="Disable W&B logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="MNLP-Safety",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
    )
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_loss(
    model,
    loader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            batch_loss = float(outputs.loss.item())
            total_loss += batch_loss
            n_batches += 1
    model.train()
    return total_loss / n_batches if n_batches else float("inf")


def weighted_average(loss_by_name: dict[str, float], weight_by_name: dict[str, int]) -> float:
    num = 0.0
    den = 0
    for name, loss in loss_by_name.items():
        w = weight_by_name[name]
        num += loss * w
        den += w
    return num / den if den else float("inf")


def resolve_run_configuration(args: argparse.Namespace) -> tuple[str, str, Path]:
    run_id = args.run_id or args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or args.wandb_run_name or f"sft-{run_id}"

    if args.best_checkpoint_dir is not None:
        best_checkpoint_dir = args.best_checkpoint_dir
    else:
        default_dir_name = args.run_name or run_id
        best_checkpoint_dir = Path("checkpoints") / default_dir_name

    return run_id, run_name, best_checkpoint_dir


def resolve_checkpoint_subdirs(checkpoint_root: Path) -> tuple[Path, Path]:
    hf_model_dir = checkpoint_root / "hf_model"
    lora_dir = checkpoint_root / "LoRA"
    return hf_model_dir, lora_dir


def resolve_effective_train_size(requested_size: int, available_size: int) -> int:
    if requested_size < -1:
        raise ValueError("Each --train-sizes value must be -1 or a non-negative integer.")
    if requested_size == -1:
        return available_size
    return min(requested_size, available_size)


def apply_train_size_caps(
    bundles: list[DatasetBundle],
    train_sizes: list[int] | None,
    seed: int,
) -> list[DatasetBundle]:
    if train_sizes is None:
        return bundles

    if len(train_sizes) != len(bundles):
        raise ValueError(
            "--train-sizes must provide exactly one integer per dataset in --datasets. "
            f"Received {len(train_sizes)} sizes for {len(bundles)} datasets."
        )

    resized_bundles: list[DatasetBundle] = []
    for idx, (bundle, requested_size) in enumerate(zip(bundles, train_sizes)):
        target_train_size = resolve_effective_train_size(requested_size, bundle.train_size)
        train_dataset = bundle.train_loader.dataset

        if target_train_size < bundle.train_size:
            indices = list(range(bundle.train_size))
            rng = random.Random(seed + idx)
            rng.shuffle(indices)
            train_dataset = Subset(train_dataset, indices[:target_train_size])

        resized_bundles.append(
            DatasetBundle(
                name=bundle.name,
                train_loader=DataLoader(
                    train_dataset,
                    batch_size=bundle.train_loader.batch_size,
                    shuffle=True,
                    collate_fn=bundle.train_loader.collate_fn,
                ),
                test_loader=bundle.test_loader,
                train_size=target_train_size,
                test_size=bundle.test_size,
            )
        )

    return resized_bundles


def apply_test_size_caps(
    bundles: list[DatasetBundle],
    test_sizes: list[int] | None,
    seed: int,
) -> list[DatasetBundle]:
    if test_sizes is None:
        return bundles

    if len(test_sizes) != len(bundles):
        raise ValueError(
            "--test-sizes must provide exactly one integer per dataset in --datasets. "
            f"Received {len(test_sizes)} sizes for {len(bundles)} datasets."
        )

    resized_bundles: list[DatasetBundle] = []
    for idx, (bundle, requested_size) in enumerate(zip(bundles, test_sizes)):
        target_test_size = resolve_effective_train_size(requested_size, bundle.test_size)
        test_dataset = bundle.test_loader.dataset

        if target_test_size < bundle.test_size:
            indices = list(range(bundle.test_size))
            rng = random.Random(seed + idx)
            rng.shuffle(indices)
            test_dataset = Subset(test_dataset, indices[:target_test_size])

        resized_bundles.append(
            DatasetBundle(
                name=bundle.name,
                train_loader=bundle.train_loader,
                test_loader=DataLoader(
                    test_dataset,
                    batch_size=bundle.test_loader.batch_size,
                    shuffle=False,
                    collate_fn=bundle.test_loader.collate_fn,
                ),
                train_size=bundle.train_size,
                test_size=target_test_size,
            )
        )

    return resized_bundles


def build_mixed_train_loader(bundles: list[DatasetBundle], seed: int) -> DataLoader:
    if not bundles:
        raise ValueError("At least one dataset bundle is required for training.")

    train_datasets = [
        NamedTrainDataset(bundle.train_loader.dataset, bundle.name) for bundle in bundles
    ]
    mixed_dataset = ConcatDataset(train_datasets)

    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed)

    def mixed_collate_fn(batch: list[dict]) -> dict:
        dataset_names = [example["dataset_name"] for example in batch]
        model_batch = cast(
            dict[str, Any],
            bundles[0].train_loader.collate_fn(
            [
                {
                    "input_ids": example["input_ids"],
                    "labels": example["labels"],
                }
                for example in batch
            ]
            ),
        )
        model_batch["dataset_name"] = dataset_names
        return model_batch

    return DataLoader(
        mixed_dataset,
        batch_size=bundles[0].train_loader.batch_size,
        shuffle=True,
        generator=shuffle_generator,
        collate_fn=mixed_collate_fn,
    )


def maybe_init_wandb(args: argparse.Namespace, run_id: str, run_name: str):
    if args.skip_wandb:
        return None
    if not HAS_WANDB:
        raise RuntimeError("wandb is not installed but --skip_wandb was not provided.")

    wb_run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=run_name,
        group="sft",
        job_type="training",
        config=vars(args),
    )

    wandb.define_metric("train/sample")
    wandb.define_metric("train/batch_loss", step_metric="train/sample")
    wandb.define_metric("train/running_loss", step_metric="train/sample")

    wandb.define_metric("test/sample")
    wandb.define_metric("test/*", step_metric="test/sample")

    wandb.define_metric("other/global_step")
    wandb.define_metric("other/*", step_metric="other/global_step")
    return wb_run


def build_model_and_tokenizer(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any]:
    tokenizer: Any = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model: Any = AutoModelForCausalLM.from_pretrained(
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
    model: Any = get_peft_model(base_model, lora_cfg)
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model.print_trainable_parameters()
    return model, tokenizer


def train(args: argparse.Namespace) -> None:
    if args.grad_accum <= 0:
        raise ValueError("--grad-accum must be > 0")
    if args.train_sizes is not None and len(args.train_sizes) != len(args.datasets):
        raise ValueError(
            "--train-sizes must have the same number of values as --datasets."
        )
    if args.test_sizes is not None and len(args.test_sizes) != len(args.datasets):
        raise ValueError(
            "--test-sizes must have the same number of values as --datasets."
        )
    if args.eval_interval <= 0:
        raise ValueError("--eval-interval must be > 0")

    set_seeds(Random.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id, run_name, best_checkpoint_dir = resolve_run_configuration(args)
    best_hf_model_dir, best_lora_dir = resolve_checkpoint_subdirs(best_checkpoint_dir)
    wb_run = maybe_init_wandb(args, run_id, run_name)

    print("SFT TRAINING")
    print(f"  Base model           : {BASE_MODEL}")
    print(f"  Datasets             : {args.datasets}")
    print(f"  Train sizes          : {args.train_sizes}")
    print(f"  Eval interval        : every {args.eval_interval} samples")
    print(f"  Device               : {device}")
    print(f"  Run name             : {run_name}")
    print(f"  Best checkpoint path : {best_checkpoint_dir}")
    print(f"  - HF merged model    : {best_hf_model_dir}")
    print(f"  - LoRA adapter       : {best_lora_dir}")
    print("=" * 60)

    model, tokenizer = build_model_and_tokenizer(args, device)

    bundles: list[DatasetBundle] = load_dataset_bundles(
        dataset_names=args.datasets,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        system_prompt=SYSTEM_PROMPT_MCQ,
    )
    bundles = apply_train_size_caps(
        bundles=bundles,
        train_sizes=args.train_sizes,
        seed=Random.SEED,
    )
    bundles = apply_test_size_caps(
        bundles=bundles,
        test_sizes=args.test_sizes,
        seed=Random.SEED,
    )

    print("\nLoaded datasets:")
    for bundle in bundles:
        print(
            f"  - {bundle.name}: train={bundle.train_size} test={bundle.test_size}"
        )

    if wb_run is not None:
        wb_run.config.update(
            {"effective_train_sizes": {b.name: b.train_size for b in bundles}},
            allow_val_change=True,
        )

    mixed_train_loader = build_mixed_train_loader(bundles, seed=Random.SEED)

    total_micro_batches = len(mixed_train_loader) * args.epochs
    total_optimizer_steps = max(1, total_micro_batches // args.grad_accum)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_optimizer_steps * args.warmup_ratio),
        num_training_steps=total_optimizer_steps,
    )

    best_weighted_test = float("inf")
    best_epoch = -1
    best_step = -1
    global_step = 0
    grad_counter = 0
    samples_seen = 0
    next_eval_at_samples = args.eval_interval
    eval_pending = False

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        model.train()

        test_loss_by_dataset: dict[str, float] = {}
        test_weight_by_dataset: dict[str, int] = {}
        running_loss = 0.0
        n_batches = 0

        for batch in mixed_train_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            raw_loss = float(outputs.loss.item())
            loss = outputs.loss / args.grad_accum
            loss.backward()

            batch_samples = int(batch["input_ids"].shape[0])
            samples_seen += batch_samples
            while samples_seen >= next_eval_at_samples:
                eval_pending = True
                next_eval_at_samples += args.eval_interval

            running_loss += raw_loss
            n_batches += 1
            grad_counter += 1

            if wb_run is not None:
                log_payload = {
                    "train/sample": samples_seen,
                    "train/batch_loss": raw_loss,
                    "train/running_loss": running_loss / n_batches,
                }
                wb_run.log(log_payload, commit=True)

            if grad_counter % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if wb_run is not None:
                    wb_run.log(
                        {
                            "other/global_step": global_step,
                            "other/epoch": epoch,
                            "other/samples_seen": samples_seen,
                            "other/learning_rate": scheduler.get_last_lr()[0],
                        },
                        commit=True,
                    )

                # Evaluation is scheduled by sample count and executed after a model update.
                if eval_pending:
                    eval_pending = False
                    print(
                        f"[eval] Step {global_step} (samples={samples_seen}): "
                        "Running evaluation..."
                    )
                    for bundle in bundles:
                        test_avg_loss = evaluate_loss(
                            model,
                            bundle.test_loader,
                            device,
                        )
                        test_loss_by_dataset[bundle.name] = test_avg_loss
                        test_weight_by_dataset[bundle.name] = bundle.test_size

                    test_weighted = weighted_average(test_loss_by_dataset, test_weight_by_dataset)
                    print(f"[eval] Step {global_step}: test_weighted_loss={test_weighted:.4f}")

                    # Log eval to wandb
                    if wb_run is not None:
                        eval_log = {
                            "test/sample": samples_seen,
                            "test/average_loss": test_weighted,
                        }
                        for name in args.datasets:
                            eval_log[f"test/{name}_loss"] = test_loss_by_dataset[name]
                        wb_run.log(eval_log, commit=True)

                    # Save checkpoint if best
                    if test_weighted < best_weighted_test:
                        best_weighted_test = test_weighted
                        best_epoch = epoch
                        best_step = global_step
                        print(f"[info] New best checkpoint at step {global_step}. Exporting merged model + LoRA adapter...")
                        export_merged_checkpoint(
                            model=model,
                            tokenizer=tokenizer,
                            output_dir=best_hf_model_dir,
                            base_model=BASE_MODEL,
                            system_prompt=SYSTEM_PROMPT_MCQ,
                            temperature=args.export_temperature,
                            top_p=args.export_top_p,
                            top_k=args.export_top_k,
                        )

                        if best_lora_dir.exists():
                            shutil.rmtree(best_lora_dir)
                        model.save_pretrained(str(best_lora_dir))

        if grad_counter % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if wb_run is not None:
                wb_run.log(
                    {
                        "other/global_step": global_step,
                        "other/epoch": epoch,
                        "other/samples_seen": samples_seen,
                        "other/learning_rate": scheduler.get_last_lr()[0],
                    },
                    commit=True,
                )

            if eval_pending:
                eval_pending = False
                print(
                    f"[eval] Step {global_step} (samples={samples_seen}): "
                    "Running evaluation..."
                )
                for bundle in bundles:
                    test_avg_loss = evaluate_loss(
                        model,
                        bundle.test_loader,
                        device,
                    )
                    test_loss_by_dataset[bundle.name] = test_avg_loss
                    test_weight_by_dataset[bundle.name] = bundle.test_size

                test_weighted = weighted_average(test_loss_by_dataset, test_weight_by_dataset)
                print(f"[eval] Step {global_step}: test_weighted_loss={test_weighted:.4f}")

                if wb_run is not None:
                    eval_log = {
                        "test/sample": samples_seen,
                        "test/average_loss": test_weighted,
                    }
                    for name in args.datasets:
                        eval_log[f"test/{name}_loss"] = test_loss_by_dataset[name]
                    wb_run.log(eval_log, commit=True)

                if test_weighted < best_weighted_test:
                    best_weighted_test = test_weighted
                    best_epoch = epoch
                    best_step = global_step
                    print(f"[info] New best checkpoint at step {global_step}. Exporting merged model + LoRA adapter...")
                    export_merged_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        output_dir=best_hf_model_dir,
                        base_model=BASE_MODEL,
                        system_prompt=SYSTEM_PROMPT_MCQ,
                        temperature=args.export_temperature,
                        top_p=args.export_top_p,
                        top_k=args.export_top_k,
                    )

                    if best_lora_dir.exists():
                        shutil.rmtree(best_lora_dir)
                    model.save_pretrained(str(best_lora_dir))

        train_running = running_loss / n_batches if n_batches else float("inf")
        print(f"train_running_loss={train_running:.4f}")

        if wb_run is not None:
            wb_run.summary["best_test_average_loss"] = best_weighted_test
            wb_run.summary["best_epoch"] = best_epoch
            wb_run.summary["best_step"] = best_step
            wb_run.summary["best_checkpoint_dir"] = str(best_checkpoint_dir)
            wb_run.summary["best_hf_model_dir"] = str(best_hf_model_dir)
            wb_run.summary["best_lora_dir"] = str(best_lora_dir)

    print("\nTraining complete.")

    print("[eval] End of training: Running final evaluation...")
    final_test_loss_by_dataset: dict[str, float] = {}
    final_test_weight_by_dataset: dict[str, int] = {}
    for bundle in bundles:
        test_avg_loss = evaluate_loss(
            model,
            bundle.test_loader,
            device,
        )
        final_test_loss_by_dataset[bundle.name] = test_avg_loss
        final_test_weight_by_dataset[bundle.name] = bundle.test_size

    final_test_weighted = weighted_average(final_test_loss_by_dataset, final_test_weight_by_dataset)
    print(f"[eval] Final test_weighted_loss={final_test_weighted:.4f}")

    if wb_run is not None:
        final_eval_log = {
            "test/sample": samples_seen,
            "test/average_loss": final_test_weighted,
        }
        for name in args.datasets:
            final_eval_log[f"test/{name}_loss"] = final_test_loss_by_dataset[name]
        wb_run.log(final_eval_log, commit=True)

    if final_test_weighted < best_weighted_test:
        best_weighted_test = final_test_weighted
        best_epoch = args.epochs
        best_step = global_step
        print("[info] Final evaluation is the new best checkpoint. Exporting merged model + LoRA adapter...")
        export_merged_checkpoint(
            model=model,
            tokenizer=tokenizer,
            output_dir=best_hf_model_dir,
            base_model=BASE_MODEL,
            system_prompt=SYSTEM_PROMPT_MCQ,
            temperature=args.export_temperature,
            top_p=args.export_top_p,
            top_k=args.export_top_k,
        )

        if best_lora_dir.exists():
            shutil.rmtree(best_lora_dir)
        model.save_pretrained(str(best_lora_dir))

    if wb_run is not None:
        wb_run.summary["best_test_average_loss"] = best_weighted_test
        wb_run.summary["best_epoch"] = best_epoch
        wb_run.summary["best_step"] = best_step

    print(f"Best checkpoint root: {best_checkpoint_dir} (epoch {best_epoch}, step {best_step})")
    print(f"  HF merged model: {best_hf_model_dir}")
    print(f"  LoRA adapter   : {best_lora_dir}")

    if wb_run is not None:
        wb_run.summary["final_epoch"] = args.epochs
        wb_run.finish()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()