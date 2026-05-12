"""Simple SFT entry point with per-dataset metrics, W&B logging, and benchmark hooks.

This script keeps training orchestration readable while delegating details to:
- train/sft_data.py
- train/sft_benchmark.py
- train/sft_export.py

Usage example:
  uv run train/sft.py \
    --datasets safetybench \
    --run_name safetybench_cot \
    --benchmarks safety \
    --epochs 1
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constants.global_variables import Random
from prompts.system_prompt import SYSTEM_PROMPT_MCQ
from train.sft_benchmark import run_benchmark_eval
from train.sft_data import DatasetBundle, load_dataset_bundles
from train.sft_export import export_merged_checkpoint

try:
    import wandb

    HAS_WANDB = True
except Exception:
    wandb = None
    HAS_WANDB = False


BASE_MODEL = "Qwen/Qwen3-1.7B"

# Simple, conservative defaults.
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 16
DEFAULT_LR = 2e-4
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_MAX_SEQ_LEN = 16384  # max token defined in the report.
DEFAULT_BENCHMARK_INTERVAL = 1
DEFAULT_BENCHMARK_LIMIT = None

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


def disable_peft_bitsandbytes_dispatch() -> None:
    """Avoid accidental PEFT bitsandbytes path when not quantizing."""
    try:
        import peft.tuners.lora.model as lora_model

        lora_model.is_bnb_available = lambda: False
        lora_model.is_bnb_4bit_available = lambda: False
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
        "--benchmarks",
        nargs="*",
        default=[],
        help="Benchmark names under data/processed/benchmark/<name>.",
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
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument(
        "--benchmark-interval",
        type=int,
        default=DEFAULT_BENCHMARK_INTERVAL,
        help="Run benchmark evaluation every N epochs.",
    )
    parser.add_argument(
        "--benchmark-limit",
        type=int,
        default=DEFAULT_BENCHMARK_LIMIT,
        help="Optional benchmark subset size for faster eval.",
    )
    parser.add_argument(
        "--benchmark-use-vllm",
        action="store_true",
        help="Use vLLM inside benchmark evaluation.",
    )
    parser.add_argument(
        "--benchmark-think",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable thinking mode during benchmark evaluation.",
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
    wb_run=None,
    epoch: int | None = None,
    dataset_name: str | None = None,
    eval_step: int = 0,
) -> tuple[float, int]:
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

            if wb_run is not None and epoch is not None and dataset_name is not None:
                eval_step += 1
                wb_run.log(
                    {
                        "epoch": epoch,
                        "eval_step": eval_step,
                        "batch/test_loss": batch_loss,
                        "batch/test_running_loss": total_loss / n_batches,
                        f"batch/test_loss/{dataset_name}": batch_loss,
                        f"batch/test_running_loss/{dataset_name}": total_loss / n_batches,
                    },
                    commit=True,
                )
    model.train()
    return (total_loss / n_batches if n_batches else float("inf"), eval_step)


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

    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("test/*", step_metric="epoch")
    wandb.define_metric("benchmark/*", step_metric="epoch")
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("batch/train*", step_metric="train_step")
    wandb.define_metric("batch/test*", step_metric="eval_step")
    return wb_run


def build_model_and_tokenizer(args: argparse.Namespace, device: torch.device):
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
        model.enable_input_require_grads()

    model.print_trainable_parameters()
    return model, tokenizer


def run_benchmarks_for_epoch(
    args: argparse.Namespace,
    run_id: str,
    epoch: int,
    model,
    tokenizer,
) -> dict[str, dict]:
    if not args.benchmarks:
        return {}

    temp_eval_ckpt = Path("checkpoints/.sft-current-benchmark-eval")
    export_merged_checkpoint(
        model=model,
        tokenizer=tokenizer,
        output_dir=temp_eval_ckpt,
        base_model=BASE_MODEL,
        system_prompt=SYSTEM_PROMPT_MCQ,
        temperature=args.export_temperature,
        top_p=args.export_top_p,
        top_k=args.export_top_k,
    )

    benchmark_payload: dict[str, dict] = {}
    for benchmark in args.benchmarks:
        payload = run_benchmark_eval(
            checkpoint_path=temp_eval_ckpt,
            benchmark=benchmark,
            think_mode=args.benchmark_think,
            use_vllm=args.benchmark_use_vllm,
            limit=args.benchmark_limit,
            run_name=run_id,
            epoch=epoch,
        )
        benchmark_payload[benchmark] = payload

    return benchmark_payload


def train(args: argparse.Namespace) -> None:
    if args.grad_accum <= 0:
        raise ValueError("--grad-accum must be > 0")
    if args.benchmark_interval <= 0:
        raise ValueError("--benchmark-interval must be > 0")

    set_seeds(Random.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id, run_name, best_checkpoint_dir = resolve_run_configuration(args)
    wb_run = maybe_init_wandb(args, run_id, run_name)

    print("SFT TRAINING")
    print(f"  Base model           : {BASE_MODEL}")
    print(f"  Datasets             : {args.datasets}")
    print(f"  Benchmarks           : {args.benchmarks}")
    print(f"  Device               : {device}")
    print(f"  Run name             : {run_name}")
    print(f"  Best checkpoint path : {best_checkpoint_dir}")
    print("=" * 60)

    model, tokenizer = build_model_and_tokenizer(args, device)

    bundles: list[DatasetBundle] = load_dataset_bundles(
        dataset_names=args.datasets,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        system_prompt=SYSTEM_PROMPT_MCQ,
    )

    print("\nLoaded datasets:")
    for bundle in bundles:
        print(
            f"  - {bundle.name}: train={bundle.train_size} test={bundle.test_size}"
        )

    total_micro_batches = sum(len(bundle.train_loader) for bundle in bundles) * args.epochs
    total_optimizer_steps = max(1, total_micro_batches // args.grad_accum)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_optimizer_steps * args.warmup_ratio),
        num_training_steps=total_optimizer_steps,
    )

    best_weighted_test = float("inf")
    best_epoch = -1
    global_step = 0
    grad_counter = 0
    train_step = 0
    eval_step = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        model.train()

        train_loss_by_dataset: dict[str, float] = {}
        test_loss_by_dataset: dict[str, float] = {}
        train_weight_by_dataset: dict[str, int] = {}
        test_weight_by_dataset: dict[str, int] = {}

        for bundle in bundles:
            running_loss = 0.0
            n_batches = 0

            for batch in bundle.train_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                raw_loss = float(outputs.loss.item())
                loss = outputs.loss / args.grad_accum
                loss.backward()

                running_loss += raw_loss
                n_batches += 1
                grad_counter += 1
                train_step += 1

                if wb_run is not None:
                    wb_run.log(
                        {
                            "epoch": epoch,
                            "train_step": train_step,
                            "batch/train_loss": raw_loss,
                            "batch/train_running_loss": running_loss / n_batches,
                            f"batch/train_loss/{bundle.name}": raw_loss,
                            f"batch/train_running_loss/{bundle.name}": running_loss / n_batches,
                            "train/global_step": global_step,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                        },
                        commit=True,
                    )

                if grad_counter % args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            train_loss_by_dataset[bundle.name] = (
                running_loss / n_batches if n_batches else float("inf")
            )
            train_weight_by_dataset[bundle.name] = bundle.train_size

        if grad_counter % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        for bundle in bundles:
            test_avg_loss, eval_step = evaluate_loss(
                model,
                bundle.test_loader,
                device,
                wb_run=wb_run,
                epoch=epoch,
                dataset_name=bundle.name,
                eval_step=eval_step,
            )
            test_loss_by_dataset[bundle.name] = test_avg_loss
            test_weight_by_dataset[bundle.name] = bundle.test_size

        train_weighted = weighted_average(train_loss_by_dataset, train_weight_by_dataset)
        test_weighted = weighted_average(test_loss_by_dataset, test_weight_by_dataset)

        print(f"train_weighted_loss={train_weighted:.4f} test_weighted_loss={test_weighted:.4f}")
        for name in args.datasets:
            print(
                f"  {name}: train={train_loss_by_dataset[name]:.4f} "
                f"test={test_loss_by_dataset[name]:.4f}"
            )

        if test_weighted < best_weighted_test:
            best_weighted_test = test_weighted
            best_epoch = epoch
            print("[info] New best checkpoint. Exporting merged baseline-format model...")
            export_merged_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=best_checkpoint_dir,
                base_model=BASE_MODEL,
                system_prompt=SYSTEM_PROMPT_MCQ,
                temperature=args.export_temperature,
                top_p=args.export_top_p,
                top_k=args.export_top_k,
            )

        benchmark_payload: dict[str, dict] = {}
        if args.benchmarks and (epoch % args.benchmark_interval == 0):
            print("[info] Running benchmark evaluation hook...")
            benchmark_payload = run_benchmarks_for_epoch(
                args=args,
                run_id=run_id,
                epoch=epoch,
                model=model,
                tokenizer=tokenizer,
            )

        if wb_run is not None:
            log_payload = {
                "epoch": epoch,
                "train/weighted_loss": train_weighted,
                "test/weighted_loss": test_weighted,
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train/global_step": global_step,
            }

            for name in args.datasets:
                log_payload[f"train/{name}_loss"] = train_loss_by_dataset[name]
                log_payload[f"test/{name}_loss"] = test_loss_by_dataset[name]

            for benchmark, payload in benchmark_payload.items():
                metrics = payload["metrics"]
                prefix = f"benchmark/{benchmark}"
                log_payload[f"{prefix}/pass@1"] = metrics.get("pass@1")
                log_payload[f"{prefix}/box_compliance_rate"] = metrics.get("box_compliance_rate")
                log_payload[f"{prefix}/pass@1_among_extracted"] = metrics.get("pass@1_among_extracted")

            wb_run.log(log_payload, commit=True)
            wb_run.summary["best_test_weighted_loss"] = best_weighted_test
            wb_run.summary["best_epoch"] = best_epoch
            wb_run.summary["best_checkpoint_dir"] = str(best_checkpoint_dir)

    print("\nTraining complete.")
    print(f"Best checkpoint: {best_checkpoint_dir} (epoch {best_epoch})")

    if wb_run is not None:
        wb_run.summary["final_epoch"] = args.epochs
        wb_run.finish()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()