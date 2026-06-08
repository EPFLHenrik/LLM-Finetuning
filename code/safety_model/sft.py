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
import gc
import json
import random
import shutil
from pathlib import Path
from typing import Any, cast

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from code.safety_model.constants.global_variables import DatasetFieldNames, Random
from code.safety_model.constants.system_prompt import SYSTEM_PROMPT_MCQ
from code.safety_model.utils.train.training_common import (
    HomogeneousBatchSampler,
    build_tokenizer_only,
    resolve_checkpoint_subdirs,
    resolve_run_configuration,
    set_seeds,
)
from code.safety_model.utils.train.sft_data import DatasetBundle, load_dataset_bundles
from code.safety_model.utils.train.sft_export import export_merged_checkpoint

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
DEFAULT_LR = 2e-4
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 16
DEFAULT_MAX_SEQ_LEN = 15000 # Safer cap than 16384 to avoid OOMs from outliers in the training data.

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
        help="Dataset names under data/safety/processed/sft/<name> with train and test splits.",
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
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help="Maximum token length per encoded training example after chat templating.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        help=(
            "Number of micro-batches to accumulate before an optimizer step. "
            "Updates are dataset-homogeneous: all accumulated samples come from one dataset."
        ),
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help=(
            "Deprecated in this version. Training no longer runs generation eval mid-epoch; "
            "checkpoints are saved per epoch and evaluated after training to avoid VRAM conflicts."
        ),
    )
    parser.add_argument(
        "--skip_training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Skip model training and assume per-epoch checkpoints already exist under "
            "--epoch-checkpoints-dir (or checkpoints/<run_name>/epochs). Useful as a "
            "fallback to re-run post-training validation/selection only."
        ),
    )
    parser.add_argument(
        "--save-epoch-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export a merged HF checkpoint and LoRA adapter at the end of each epoch.",
    )
    parser.add_argument(
        "--epoch-checkpoints-dir",
        type=Path,
        default=None,
        help="Directory for per-epoch checkpoints. Defaults to checkpoints/<run_name>/epochs.",
    )
    parser.add_argument(
        "--select-best-after-training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After training, free training VRAM, evaluate each epoch checkpoint on the SFT "
            "validation/test split with vLLM, then copy the best epoch to the final checkpoint path."
        ),
    )
    parser.add_argument(
        "--validation-eval-limit",
        type=int,
        default=None,
        help="Optional total number of validation examples for post-training checkpoint selection.",
    )
    parser.add_argument(
        "--validation-output-dir",
        type=Path,
        default=Path("code/safety_model/runs/sft_validation"),
        help="Where validation generations and metric JSON files are written.",
    )
    parser.add_argument("--validation-think", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-temperature", type=float, default=0.6)
    parser.add_argument("--validation-top-p", type=float, default=0.95)
    parser.add_argument("--validation-top-k", type=int, default=20)
    parser.add_argument("--validation-max-tokens", type=int, default=16384)
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
                train_max_tokens=bundle.train_max_tokens,
                test_max_tokens=bundle.test_max_tokens,
                train_dropped_count=bundle.train_dropped_count,
                test_dropped_count=bundle.test_dropped_count,
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
                train_max_tokens=bundle.train_max_tokens,
                test_max_tokens=bundle.test_max_tokens,
                train_dropped_count=bundle.train_dropped_count,
                test_dropped_count=bundle.test_dropped_count,
            )
        )

    return resized_bundles


def build_mixed_train_loader(
    bundles: list[DatasetBundle],
    seed: int,
    gradient_accumulation_steps: int,
) -> DataLoader:
    if not bundles:
        raise ValueError("At least one dataset bundle is required for training.")

    train_datasets = [bundle.train_loader.dataset for bundle in bundles]
    mixed_dataset = ConcatDataset(train_datasets)

    batch_size = bundles[0].train_loader.batch_size
    if batch_size is None:
        raise ValueError("Expected train DataLoader to define batch_size.")

    batch_sampler = HomogeneousBatchSampler(
        dataset_sizes=[bundle.train_size for bundle in bundles],
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
    )

    return DataLoader(
        mixed_dataset,
        batch_sampler=batch_sampler,
        collate_fn=bundles[0].train_loader.collate_fn,
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
    wandb.define_metric("validation/epoch")
    wandb.define_metric("validation/*", step_metric="validation/epoch")

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


def epoch_checkpoint_dirs(epoch_root: Path, epoch: int) -> tuple[Path, Path, Path]:
    epoch_dir = epoch_root / f"epoch_{epoch:02d}"
    return epoch_dir, epoch_dir / "hf_model", epoch_dir / "LoRA"


def first_supervised_token_index(labels: Any) -> int:
    if isinstance(labels, torch.Tensor):
        label_list = labels.detach().cpu().tolist()
    else:
        label_list = list(labels)

    for idx, value in enumerate(label_list):
        if int(value) != -100:
            return idx
    return len(label_list)


def materialize_validation_examples(
    bundles: list[DatasetBundle],
    tokenizer: Any,
    limit: int | None,
) -> list[dict[str, str]]:
    """Build raw, already chat-templated prompts from SFT validation examples.

    This avoids needing to know the original raw question field. It assumes the usual
    SFT convention: labels are -100 on the prompt tokens and supervised on the
    assistant answer tokens. The generated prompt can be passed directly to vLLM.
    """
    examples: list[dict[str, str]] = []

    for bundle in bundles:
        dataset = bundle.test_loader.dataset
        for idx in range(len(cast(Any, dataset))):
            row = dict(dataset[idx])
            if "answer" not in row:
                raise KeyError(
                    f"Validation example from {bundle.name} has no 'answer' field. "
                    "The post-training MCQ evaluator needs gold answers."
                )
            if "input_ids" not in row or "labels" not in row:
                raise KeyError(
                    f"Validation example from {bundle.name} must contain input_ids and labels."
                )

            input_ids = row["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                input_id_list = input_ids.detach().cpu().tolist()
            else:
                input_id_list = list(input_ids)

            prompt_end = first_supervised_token_index(row["labels"])
            prompt_ids = input_id_list[:prompt_end]
            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)

            examples.append(
                {
                    "dataset": bundle.name,
                    "prompt": prompt_text,
                    "answer": str(row["answer"]).strip(),
                }
            )

            if limit is not None and len(examples) >= limit:
                return examples

    return examples


def evaluate_checkpoint_on_validation_with_vllm(
    checkpoint_path: Path,
    validation_examples: list[dict[str, str]],
    output_dir: Path,
    run_name: str,
    epoch: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    """Generate validation completions with vLLM and score them with evaluate.score.

    This intentionally runs only after the training model has been deleted and CUDA
    cache has been cleared, so vLLM never competes with the training graph/model.
    """
    from code.safety_model.utils.evaluate.score import main as evaluate_score
    from vllm import LLM, SamplingParams

    outputs_dir = output_dir / "outputs"
    results_dir = output_dir / "results"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # run_name may contain path separators (e.g. "safety_model/sft").
    # Keep filenames flat under outputs/results to avoid missing parent dirs.
    safe_run_name = run_name.replace("/", "_").replace("\\", "_")
    jsonl_path = outputs_dir / f"{safe_run_name}_epoch_{epoch:02d}_validation.jsonl"
    results_path = results_dir / f"{safe_run_name}_epoch_{epoch:02d}_validation_results.json"

    print(f"[validation] Loading checkpoint with vLLM: {checkpoint_path}")
    llm = LLM(model=str(checkpoint_path), dtype="auto", seed=seed)
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        seed=seed,
    )

    prompts = [ex["prompt"] for ex in validation_examples]
    completions: list[str] = []
    generation_batch_size = 500

    for start in range(0, len(prompts), generation_batch_size):
        end = min(start + generation_batch_size, len(prompts))
        print(
            f"[validation] Generating batch {start}-{end - 1} "
            f"of {len(prompts)} for epoch {epoch:02d}"
        )
        outputs = llm.generate(prompts[start:end], sampling_params, use_tqdm=True)
        completions.extend(output.outputs[0].text.strip() for output in outputs)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ex, completion in zip(validation_examples, completions):
            output_row = {
                DatasetFieldNames.BENCHMARK_INPUT: ex["prompt"],
                DatasetFieldNames.BENCHMARK_COMPLETIONS: [completion],
                DatasetFieldNames.BENCHMARK_ANSWER: ex["answer"],
                "dataset": ex["dataset"],
            }
            f.write(json.dumps(output_row, ensure_ascii=False) + "\n")

    # Your run.py uses evaluate.score with benchmark="safety" for these metrics.
    evaluate_score(
        [
            "--generations", str(jsonl_path),
            "--benchmark", "safety",
            "--output", str(results_path),
        ]
    )

    with open(results_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    metrics = payload.get("metrics", {})
    print(
        f"[validation] epoch={epoch:02d} "
        f"pass@1={metrics.get('pass@1')} "
        f"box={metrics.get('box_compliance_rate')} "
        f"pass@1_extracted={metrics.get('pass@1_among_extracted')}"
    )

    # Drop vLLM objects before the next checkpoint.
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "epoch": epoch,
        "checkpoint_path": str(checkpoint_path),
        "generations_path": str(jsonl_path),
        "results_path": str(results_path),
        "metrics": metrics,
    }


def copy_best_checkpoint(
    best_record: dict[str, Any],
    epoch_root: Path,
    best_hf_model_dir: Path,
    best_lora_dir: Path,
) -> None:
    epoch = int(best_record["epoch"])
    _, epoch_hf_dir, epoch_lora_dir = epoch_checkpoint_dirs(epoch_root, epoch)

    if best_hf_model_dir.exists():
        shutil.rmtree(best_hf_model_dir)
    if best_lora_dir.exists():
        shutil.rmtree(best_lora_dir)

    shutil.copytree(epoch_hf_dir, best_hf_model_dir)
    if epoch_lora_dir.exists():
        shutil.copytree(epoch_lora_dir, best_lora_dir)


def train(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.max_seq_len <= 0:
        raise ValueError("--max-seq-len must be > 0")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be > 0")
    if args.train_sizes is not None and len(args.train_sizes) != len(args.datasets):
        raise ValueError("--train-sizes must have the same number of values as --datasets.")
    if args.test_sizes is not None and len(args.test_sizes) != len(args.datasets):
        raise ValueError("--test-sizes must have the same number of values as --datasets.")

    set_seeds(Random.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id, run_name, best_checkpoint_dir = resolve_run_configuration(args)
    best_hf_model_dir, best_lora_dir = resolve_checkpoint_subdirs(best_checkpoint_dir)
    epoch_root = args.epoch_checkpoints_dir or (best_checkpoint_dir / "epochs")
    wb_run = maybe_init_wandb(args, run_id, run_name)

    print("SFT TRAINING")
    print(f"  Base model           : {BASE_MODEL}")
    print(f"  Datasets             : {args.datasets}")
    print(f"  Train sizes          : {args.train_sizes}")
    print(f"  Max seq len          : {args.max_seq_len}")
    print(f"  Batch size           : {args.batch_size}")
    print(f"  Grad accumulation    : {args.gradient_accumulation_steps}")
    print(f"  Effective update size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Device               : {device}")
    print(f"  Run name             : {run_name}")
    print(f"  Epoch checkpoints    : {epoch_root}")
    print(f"  Final/best checkpoint: {best_checkpoint_dir}")
    print("=" * 60)

    model: Any | None = None
    optimizer: Any | None = None
    scheduler: Any | None = None
    mixed_train_loader: Any | None = None

    if args.skip_training:
        print("[info] --skip_training enabled; skipping model training and using existing epoch checkpoints.")
        tokenizer = build_tokenizer_only(BASE_MODEL)
    else:
        model, tokenizer = build_model_and_tokenizer(args, device)

    bundles: list[DatasetBundle] = load_dataset_bundles(
        dataset_names=args.datasets,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        system_prompt=SYSTEM_PROMPT_MCQ,
        max_seq_len=args.max_seq_len,
    )
    bundles = apply_train_size_caps(bundles, args.train_sizes, seed=Random.SEED)
    bundles = apply_test_size_caps(bundles, args.test_sizes, seed=Random.SEED)

    print("\nLoaded datasets:")
    for bundle in bundles:
        print(
            "  - "
            f"{bundle.name}: "
            f"train={bundle.train_size} "
            f"test={bundle.test_size} "
            f"train_max_tokens={bundle.train_max_tokens} "
            f"test_max_tokens={bundle.test_max_tokens} "
            f"train_dropped={bundle.train_dropped_count} "
            f"test_dropped={bundle.test_dropped_count}"
        )

    if wb_run is not None:
        wb_run.config.update(
            {
                "effective_train_sizes": {b.name: b.train_size for b in bundles},
                "effective_test_sizes": {b.name: b.test_size for b in bundles},
                "epoch_checkpoints_dir": str(epoch_root),
            },
            allow_val_change=True,
        )

    global_step = 0
    samples_seen = 0
    saved_epochs: list[int] = []

    if args.skip_training:
        saved_epochs = list(range(1, args.epochs + 1))
    else:
        mixed_train_loader = build_mixed_train_loader(
            bundles,
            seed=Random.SEED,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        batch_sampler = cast(HomogeneousBatchSampler, mixed_train_loader.batch_sampler)
        optimizer_steps_per_epoch = max(1, batch_sampler.optimizer_steps_per_epoch())
        total_optimizer_steps = max(1, optimizer_steps_per_epoch * args.epochs)

        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_optimizer_steps * args.warmup_ratio),
            num_training_steps=total_optimizer_steps,
        )

        for epoch in range(1, args.epochs + 1):
            print(f"\n=== Epoch {epoch}/{args.epochs} ===")
            model.train()
            running_loss = 0.0
            n_batches = 0
            optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(mixed_train_loader):
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                raw_loss = float(outputs.loss.item())
                (outputs.loss / args.gradient_accumulation_steps).backward()

                batch_samples = int(batch["input_ids"].shape[0])
                samples_seen += batch_samples
                running_loss += raw_loss
                n_batches += 1

                if wb_run is not None:
                    wb_run.log(
                        {
                            "train/sample": samples_seen,
                            "train/batch_loss": raw_loss,
                            "train/running_loss": running_loss / n_batches,
                        },
                        commit=True,
                    )

                if batch_sampler.update_ends_at(batch_idx):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
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

            train_running = running_loss / n_batches if n_batches else float("inf")
            print(f"train_running_loss={train_running:.4f}")

            if args.save_epoch_checkpoints:
                epoch_dir, epoch_hf_dir, epoch_lora_dir = epoch_checkpoint_dirs(epoch_root, epoch)
                print(f"[checkpoint] Exporting epoch {epoch} to {epoch_dir}")
                export_merged_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    output_dir=epoch_hf_dir,
                    base_model=BASE_MODEL,
                    system_prompt=SYSTEM_PROMPT_MCQ,
                    temperature=args.export_temperature,
                    top_p=args.export_top_p,
                    top_k=args.export_top_k,
                )
                if epoch_lora_dir.exists():
                    shutil.rmtree(epoch_lora_dir)
                model.save_pretrained(str(epoch_lora_dir))
                saved_epochs.append(epoch)

            if wb_run is not None:
                wb_run.summary["last_epoch_train_running_loss"] = train_running
                wb_run.summary["last_epoch"] = epoch
                wb_run.summary["last_global_step"] = global_step

    print("\nTraining complete.")

    if not saved_epochs:
        if args.skip_training:
            raise RuntimeError(
                "--skip_training was set but no epochs were selected for evaluation. "
                "Provide --epochs > 0."
            )
        print("[checkpoint] No per-epoch checkpoints were saved; exporting final model as best checkpoint.")
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
        best_epoch = args.epochs
        validation_records: list[dict[str, Any]] = []
    else:
        validation_records = []
        best_epoch = saved_epochs[-1]

        if args.select_best_after_training:
            print("[validation] Materializing SFT validation prompts before freeing tokenizer.")
            validation_examples = materialize_validation_examples(
                bundles=bundles,
                tokenizer=tokenizer,
                limit=args.validation_eval_limit,
            )
            print(f"[validation] Prepared {len(validation_examples)} validation examples.")

            # Free the training model/optimizer before starting vLLM. This is the key OOM-avoidance step.
            del model, optimizer, scheduler, mixed_train_loader, bundles, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            best_record: dict[str, Any] | None = None
            for epoch in saved_epochs:
                _, epoch_hf_dir, _ = epoch_checkpoint_dirs(epoch_root, epoch)
                record = evaluate_checkpoint_on_validation_with_vllm(
                    checkpoint_path=epoch_hf_dir,
                    validation_examples=validation_examples,
                    output_dir=args.validation_output_dir / run_name,
                    run_name=run_name,
                    epoch=epoch,
                    temperature=args.validation_temperature,
                    top_p=args.validation_top_p,
                    top_k=args.validation_top_k,
                    max_tokens=args.validation_max_tokens,
                    seed=Random.SEED,
                )
                validation_records.append(record)

                metrics = record["metrics"]
                score = float(metrics.get("pass@1") or 0.0)
                box = float(metrics.get("box_compliance_rate") or 0.0)
                extracted = float(metrics.get("pass@1_among_extracted") or 0.0)

                if best_record is None:
                    best_record = record
                else:
                    best_metrics = best_record["metrics"]
                    best_key = (
                        float(best_metrics.get("pass@1") or 0.0),
                        float(best_metrics.get("box_compliance_rate") or 0.0),
                        float(best_metrics.get("pass@1_among_extracted") or 0.0),
                    )
                    if (score, box, extracted) > best_key:
                        best_record = record

                if wb_run is not None:
                    wb_run.log(
                        {
                            "validation/epoch": epoch,
                            "validation/pass@1": score,
                            "validation/box_compliance_rate": box,
                            "validation/pass@1_among_extracted": extracted,
                        },
                        commit=True,
                    )

            assert best_record is not None
            best_epoch = int(best_record["epoch"])
            print(f"[validation] Best epoch by pass@1: {best_epoch}")
            copy_best_checkpoint(best_record, epoch_root, best_hf_model_dir, best_lora_dir)

            if wb_run is not None:
                best_metrics = best_record["metrics"]
                wb_run.summary["best_epoch"] = best_epoch
                wb_run.summary["best_validation_pass@1"] = best_metrics.get("pass@1")
                wb_run.summary["best_validation_box_compliance_rate"] = best_metrics.get("box_compliance_rate")
                wb_run.summary["best_validation_pass@1_among_extracted"] = best_metrics.get("pass@1_among_extracted")
                wb_run.summary["best_validation_results_path"] = best_record.get("results_path")
        else:
            print("[checkpoint] Post-training validation disabled; using final epoch as best checkpoint.")
            final_record = {"epoch": saved_epochs[-1]}
            copy_best_checkpoint(final_record, epoch_root, best_hf_model_dir, best_lora_dir)

    summary_path = best_checkpoint_dir / "validation_selection_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "best_epoch": best_epoch,
                "best_checkpoint_dir": str(best_checkpoint_dir),
                "epoch_checkpoints_dir": str(epoch_root),
                "records": validation_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Best checkpoint root: {best_checkpoint_dir} (epoch {best_epoch})")
    print(f"  HF merged model: {best_hf_model_dir}")
    print(f"  LoRA adapter   : {best_lora_dir}")
    print(f"  Selection summary: {summary_path}")

    if wb_run is not None:
        wb_run.summary["final_epoch"] = args.epochs
        wb_run.summary["best_checkpoint_dir"] = str(best_checkpoint_dir)
        wb_run.summary["best_hf_model_dir"] = str(best_hf_model_dir)
        wb_run.summary["best_lora_dir"] = str(best_lora_dir)
        wb_run.finish()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
