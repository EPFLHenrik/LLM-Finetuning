from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from peft import PeftModel
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from constants.global_variables import DatasetFieldNames, Random
from evaluate.score import main as evaluate_score
from prompts.system_prompt import SYSTEM_PROMPT_MCQ
from train.sft_data import DatasetBundle, collate_fn, encode_example
from train.sft_export import export_merged_checkpoint
from train.training_common import (
    HomogeneousBatchSampler,
    build_tokenizer_only,
    resolve_checkpoint_subdirs,
    resolve_run_configuration,
    set_seeds,
)

wandb: Any
try:
    import wandb  # type: ignore[import-not-found]

    HAS_WANDB = True
except Exception:
    wandb = cast(Any, None)
    HAS_WANDB = False


BASE_MODEL = "checkpoints/Qwen3-1.7B-baseline"

DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 1
DEFAULT_LR = 1e-5
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 16
DEFAULT_MAX_SEQ_LEN = 4096
DEFAULT_BETA = 0.03


def disable_peft_bitsandbytes_dispatch() -> None:
    """Avoid accidental PEFT bitsandbytes path when not quantizing."""
    try:
        import peft.tuners.lora.model as lora_model

        setattr(lora_model, "is_bnb_available", lambda: False)
        setattr(lora_model, "is_bnb_4bit_available", lambda: False)
        print("[info] Disabled PEFT bitsandbytes dispatch for this run.")
    except Exception as exc:
        print(f"[warn] Failed to disable PEFT bitsandbytes dispatch: {exc}")


@dataclass
class PreferenceDatasetBundle:
    name: str
    train_loader: DataLoader
    train_size: int
    train_max_tokens: int = 0
    train_dropped_count: int = 0
    train_fallback_chosen_count: int = 0


class DPOPreferenceDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        tokenizer,
        system_prompt: str,
        max_seq_len: int | None = None,
        drop_missing_chosen: bool = False,
    ):
        self.examples: list[dict[str, Any]] = []
        self.max_tokens = 0
        self.dropped_count = 0
        self.fallback_chosen_count = 0

        for row in hf_dataset:
            prompt = str(row[DatasetFieldNames.DS_INPUT])
            rejected = str(row[DatasetFieldNames.DPO_REJECTED]).strip()
            chosen = str(row.get(DatasetFieldNames.DPO_CHOSEN, "")).strip()
            fallback_target = str(row.get(DatasetFieldNames.SFT_TARGET, "")).strip()
            if not chosen:
                if drop_missing_chosen:
                    self.dropped_count += 1
                    continue
                chosen = fallback_target
                if chosen:
                    self.fallback_chosen_count += 1

            if not rejected:
                raise ValueError(
                    "Each DPO row must contain a non-empty rejected response."
                )
            if not chosen:
                raise ValueError(
                    "Each DPO row must contain either a chosen response or an SFT target fallback."
                )

            ex = encode_preference_example(
                tokenizer=tokenizer,
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
                system_prompt=system_prompt,
                max_seq_len=max_seq_len,
            )
            if ex is None:
                self.dropped_count += 1
                continue

            if "answer" in row:
                ex["answer"] = row["answer"]

            self.max_tokens = max(
                self.max_tokens,
                len(ex["chosen_input_ids"]),
                len(ex["rejected_input_ids"]),
            )
            self.examples.append(ex)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DPO on an existing SFT LoRA adapter for Qwen3-1.7B"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names under --train-data-root/<name> with train and test splits.",
    )
    parser.add_argument(
        "--train-data-root",
        type=Path,
        default=Path("data/processed/dpo"),
        help="Root directory containing processed DPO/SFT datasets for training and validation.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        required=True,
        help="Path to the existing LoRA adapter to continue training. No merged model is expected.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=BASE_MODEL,
        help="Base model checkpoint used to load the adapter; defaults to the local Qwen3-1.7B baseline.",
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
        "--max-seq-len",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help="Maximum token length per encoded preference example after chat templating.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        help="Number of micro-batches to accumulate before an optimizer step.",
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument(
        "--skip_training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip DPO training and assume per-epoch checkpoints already exist.",
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
        help="Evaluate every epoch checkpoint after training and keep the best one.",
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
        default=Path("runs/dpo_validation"),
        help="Where validation generations and metric JSON files are written.",
    )
    parser.add_argument(
        "--validation-think", action=argparse.BooleanOptionalAction, default=True
    )
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
        "--dpo-beta",
        type=float,
        default=DEFAULT_BETA,
        help="Beta coefficient for the DPO objective.",
    )
    parser.add_argument(
        "--drop-missing-chosen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Drop preference rows that have no explicit 'chosen' response instead of "
            "falling back to the SFT target."
        ),
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


def resolve_effective_size(requested_size: int, available_size: int) -> int:
    if requested_size < -1:
        raise ValueError("Each size value must be -1 or a non-negative integer.")
    if requested_size == -1:
        return available_size
    return min(requested_size, available_size)


def encode_preference_example(
    tokenizer,
    prompt: str,
    chosen: str,
    rejected: str,
    system_prompt: str,
    max_seq_len: int | None = None,
) -> dict | None:
    messages_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    messages_chosen = messages_prompt + [{"role": "assistant", "content": chosen}]
    messages_rejected = messages_prompt + [{"role": "assistant", "content": rejected}]

    prompt_text = tokenizer.apply_chat_template(
        messages_prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

    chosen_text = tokenizer.apply_chat_template(
        messages_chosen,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    rejected_text = tokenizer.apply_chat_template(
        messages_rejected,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )

    chosen_ids = tokenizer(chosen_text, add_special_tokens=False)["input_ids"]
    rejected_ids = tokenizer(rejected_text, add_special_tokens=False)["input_ids"]

    if max_seq_len is not None and max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive when provided.")

    if max_seq_len is not None and (
        len(chosen_ids) > max_seq_len or len(rejected_ids) > max_seq_len
    ):
        return None

    chosen_labels = [-100] * prompt_len + chosen_ids[prompt_len:]
    rejected_labels = [-100] * prompt_len + rejected_ids[prompt_len:]

    return {
        "chosen_input_ids": chosen_ids,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_ids,
        "rejected_labels": rejected_labels,
        "prompt": prompt,
        "chosen_text": chosen,
        "rejected_text": rejected,
    }


def collate_preference_fn(batch: list[dict], pad_token_id: int) -> dict:
    def _pad_left(seqs: list[list[int]], pad_value: int) -> list[list[int]]:
        max_len = max(len(seq) for seq in seqs)
        return [[pad_value] * (max_len - len(seq)) + seq for seq in seqs]

    def _attention_mask(seqs: list[list[int]]) -> list[list[int]]:
        max_len = max(len(seq) for seq in seqs)
        return [[0] * (max_len - len(seq)) + [1] * len(seq) for seq in seqs]

    chosen_input_ids_raw = [x["chosen_input_ids"] for x in batch]
    chosen_labels_raw = [x["chosen_labels"] for x in batch]
    rejected_input_ids_raw = [x["rejected_input_ids"] for x in batch]
    rejected_labels_raw = [x["rejected_labels"] for x in batch]

    chosen_input_ids = _pad_left(chosen_input_ids_raw, pad_token_id)
    chosen_labels = _pad_left(chosen_labels_raw, -100)
    rejected_input_ids = _pad_left(rejected_input_ids_raw, pad_token_id)
    rejected_labels = _pad_left(rejected_labels_raw, -100)

    batch_out = {
        "chosen_input_ids": torch.tensor(chosen_input_ids, dtype=torch.long),
        "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
        "chosen_attention_mask": torch.tensor(
            _attention_mask(chosen_input_ids_raw), dtype=torch.long
        ),
        "rejected_input_ids": torch.tensor(rejected_input_ids, dtype=torch.long),
        "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
        "rejected_attention_mask": torch.tensor(
            _attention_mask(rejected_input_ids_raw), dtype=torch.long
        ),
    }

    if "reference_chosen_logps" in batch[0] and "reference_rejected_logps" in batch[0]:
        batch_out["reference_chosen_logps"] = torch.tensor(
            [float(x["reference_chosen_logps"]) for x in batch], dtype=torch.float32
        )
        batch_out["reference_rejected_logps"] = torch.tensor(
            [float(x["reference_rejected_logps"]) for x in batch], dtype=torch.float32
        )

    if "answer" in batch[0]:
        batch_out["answer"] = [x["answer"] for x in batch]

    return batch_out


def _load_preference_bundle(
    data_root: Path,
    name: str,
    tokenizer: Any,
    batch_size: int,
    system_prompt: str,
    max_seq_len: int | None,
    drop_missing_chosen: bool,
) -> PreferenceDatasetBundle:
    train_path = data_root / name / "train"
    if not train_path.exists():
        raise FileNotFoundError(f"Expected train split at {train_path}")

    train_hf = load_from_disk(str(train_path))
    train_ds = DPOPreferenceDataset(
        train_hf,
        tokenizer,
        system_prompt,
        max_seq_len=max_seq_len,
        drop_missing_chosen=drop_missing_chosen,
    )

    _collate = partial(collate_preference_fn, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
    )

    return PreferenceDatasetBundle(
        name=name,
        train_loader=train_loader,
        train_size=len(train_ds),
        train_max_tokens=train_ds.max_tokens,
        train_dropped_count=train_ds.dropped_count,
        train_fallback_chosen_count=train_ds.fallback_chosen_count,
    )


def load_preference_dataset_bundles(
    data_root: Path,
    dataset_names: list[str],
    tokenizer: Any,
    batch_size: int,
    system_prompt: str,
    max_seq_len: int | None = None,
    drop_missing_chosen: bool = False,
) -> list[PreferenceDatasetBundle]:
    return [
        _load_preference_bundle(
            data_root=data_root,
            name=name,
            tokenizer=tokenizer,
            batch_size=batch_size,
            system_prompt=system_prompt,
            max_seq_len=max_seq_len,
            drop_missing_chosen=drop_missing_chosen,
        )
        for name in dataset_names
    ]


class _SFTDataset(Dataset):
    def __init__(
        self, hf_dataset, tokenizer, system_prompt: str, max_seq_len: int | None = None
    ):
        self.examples = []
        self.max_tokens = 0
        self.dropped_count = 0
        for row in hf_dataset:
            ex = encode_example(
                tokenizer,
                row[DatasetFieldNames.SFT_INPUT],
                row[DatasetFieldNames.SFT_TARGET],
                system_prompt,
                max_seq_len=max_seq_len,
            )
            if ex is None:
                self.dropped_count += 1
                continue
            if "answer" in row:
                ex["answer"] = row["answer"]
            self.max_tokens = max(self.max_tokens, len(ex["input_ids"]))
            self.examples.append(ex)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_sft_dataset_bundles(
    data_root: Path,
    dataset_names: list[str],
    tokenizer: Any,
    batch_size: int,
    system_prompt: str,
    max_seq_len: int | None = None,
) -> list[DatasetBundle]:
    bundles: list[DatasetBundle] = []
    _collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)

    for name in dataset_names:
        train_path = data_root / name / "train"
        test_path = data_root / name / "test"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"Expected train/test splits at {train_path} and {test_path}"
            )

        train_hf = load_from_disk(str(train_path))
        test_hf = load_from_disk(str(test_path))

        train_ds = _SFTDataset(
            train_hf, tokenizer, system_prompt, max_seq_len=max_seq_len
        )
        test_ds = _SFTDataset(
            test_hf, tokenizer, system_prompt, max_seq_len=max_seq_len
        )

        bundles.append(
            DatasetBundle(
                name=name,
                train_loader=DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=_collate,
                ),
                test_loader=DataLoader(
                    test_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=_collate,
                ),
                train_size=len(train_ds),
                test_size=len(test_ds),
                train_max_tokens=train_ds.max_tokens,
                test_max_tokens=test_ds.max_tokens,
                train_dropped_count=train_ds.dropped_count,
                test_dropped_count=test_ds.dropped_count,
            )
        )

    return bundles


def build_model_and_tokenizer(
    args: argparse.Namespace, device: torch.device
) -> tuple[Any, Any]:
    disable_peft_bitsandbytes_dispatch()

    tokenizer: Any = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model: Any = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )

    model: Any = PeftModel.from_pretrained(
        base_model, str(args.adapter_path), is_trainable=True
    )
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model.print_trainable_parameters()
    return model, tokenizer


def maybe_init_wandb(args: argparse.Namespace, run_id: str, run_name: str):
    if args.skip_wandb:
        return None
    if not HAS_WANDB:
        raise RuntimeError("wandb is not installed but --skip_wandb was not provided.")

    wb_run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=run_name,
        group="dpo",
        job_type="training",
        config=vars(args),
    )

    wandb.define_metric("train/sample")
    wandb.define_metric("train/batch_loss", step_metric="train/sample")
    wandb.define_metric("train/dpo_loss", step_metric="train/sample")
    wandb.define_metric("train/preference_accuracy", step_metric="train/sample")
    wandb.define_metric("train/reward_margin", step_metric="train/sample")
    wandb.define_metric("train/policy_margin", step_metric="train/sample")
    wandb.define_metric("train/reference_margin", step_metric="train/sample")

    wandb.define_metric("validation/epoch")
    wandb.define_metric("validation/*", step_metric="validation/epoch")

    wandb.define_metric("other/global_step")
    wandb.define_metric("other/*", step_metric="other/global_step")
    return wb_run


def _sequence_logps_from_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~mask, 0)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * mask).sum(dim=-1)


def _compute_logps_for_batch(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        labels=labels.to(device),
    )
    return _sequence_logps_from_logits(outputs.logits, labels.to(device))


def _attach_reference_logps(
    bundle: PreferenceDatasetBundle,
    model: Any,
    device: torch.device,
    dataset_name: str,
) -> None:
    dataset = cast(DPOPreferenceDataset, bundle.train_loader.dataset)
    loader = DataLoader(
        dataset,
        batch_size=bundle.train_loader.batch_size,
        shuffle=False,
        collate_fn=bundle.train_loader.collate_fn,
    )

    total_examples = len(dataset)
    if total_examples == 0:
        print(f"[precompute] {dataset_name}: no examples to precompute.")
        return

    print(f"[precompute] {dataset_name}: 0% ({0}/{total_examples})")
    next_report_pct = 5

    cursor = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            chosen_logps = _compute_logps_for_batch(
                model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
                device,
            )
            rejected_logps = _compute_logps_for_batch(
                model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
                device,
            )

            batch_size = int(chosen_logps.shape[0])
            for i in range(batch_size):
                dataset.examples[cursor + i]["reference_chosen_logps"] = float(
                    chosen_logps[i].item()
                )
                dataset.examples[cursor + i]["reference_rejected_logps"] = float(
                    rejected_logps[i].item()
                )
            cursor += batch_size

            progress_pct = int((cursor * 100) / total_examples)
            while progress_pct >= next_report_pct and next_report_pct <= 100:
                print(
                    f"[precompute] {dataset_name}: "
                    f"{next_report_pct}% ({min(cursor, total_examples)}/{total_examples})"
                )
                next_report_pct += 5

    model.train()
    print(f"[precompute] {dataset_name}: 100% ({total_examples}/{total_examples})")


def build_mixed_train_loader(
    bundles: list[PreferenceDatasetBundle],
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


def apply_train_size_caps(
    bundles: list[PreferenceDatasetBundle],
    train_sizes: list[int] | None,
    seed: int,
) -> list[PreferenceDatasetBundle]:
    if train_sizes is None:
        return bundles

    if len(train_sizes) != len(bundles):
        raise ValueError(
            "--train-sizes must provide exactly one integer per dataset in --datasets. "
            f"Received {len(train_sizes)} sizes for {len(bundles)} datasets."
        )

    resized_bundles: list[PreferenceDatasetBundle] = []
    for idx, (bundle, requested_size) in enumerate(zip(bundles, train_sizes)):
        target_train_size = resolve_effective_size(requested_size, bundle.train_size)
        train_dataset = bundle.train_loader.dataset

        if target_train_size < bundle.train_size:
            indices = list(range(bundle.train_size))
            rng = random.Random(seed + idx)
            rng.shuffle(indices)
            train_dataset = Subset(train_dataset, indices[:target_train_size])

        resized_bundles.append(
            PreferenceDatasetBundle(
                name=bundle.name,
                train_loader=DataLoader(
                    train_dataset,
                    batch_size=bundle.train_loader.batch_size,
                    shuffle=True,
                    collate_fn=bundle.train_loader.collate_fn,
                ),
                train_size=target_train_size,
                train_max_tokens=bundle.train_max_tokens,
                train_dropped_count=bundle.train_dropped_count,
                train_fallback_chosen_count=bundle.train_fallback_chosen_count,
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
        target_test_size = resolve_effective_size(requested_size, bundle.test_size)
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
    examples: list[dict[str, str]] = []

    for bundle in bundles:
        dataset = bundle.test_loader.dataset
        for idx in range(len(cast(Any, dataset))):
            row = dict(dataset[idx])
            if "answer" not in row:
                raise KeyError(
                    f"Validation example from {bundle.name} has no 'answer' field."
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
    from vllm import LLM, SamplingParams

    outputs_dir = output_dir / "outputs"
    results_dir = output_dir / "results"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = outputs_dir / f"{run_name}_epoch_{epoch:02d}_validation.jsonl"
    results_path = results_dir / f"{run_name}_epoch_{epoch:02d}_validation_results.json"

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

    evaluate_score(
        [
            "--generations",
            str(jsonl_path),
            "--benchmark",
            "safety",
            "--output",
            str(results_path),
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


def epoch_checkpoint_dirs(epoch_root: Path, epoch: int) -> tuple[Path, Path, Path]:
    epoch_dir = epoch_root / f"epoch_{epoch:02d}"
    return epoch_dir, epoch_dir / "hf_model", epoch_dir / "LoRA"


def _dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    policy_margin = policy_chosen_logps - policy_rejected_logps
    reference_margin = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_margin - reference_margin)
    losses = -F.logsigmoid(logits)
    preference_accuracy = (logits > 0).float()
    return losses, policy_margin, reference_margin, preference_accuracy


def train(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.max_seq_len <= 0:
        raise ValueError("--max-seq-len must be > 0")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be > 0")
    if args.train_sizes is not None and len(args.train_sizes) != len(args.datasets):
        raise ValueError(
            "--train-sizes must have the same number of values as --datasets."
        )
    if args.test_sizes is not None and len(args.test_sizes) != len(args.datasets):
        raise ValueError(
            "--test-sizes must have the same number of values as --datasets."
        )

    set_seeds(Random.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id, run_name, best_checkpoint_dir = resolve_run_configuration(args)
    best_hf_model_dir, best_lora_dir = resolve_checkpoint_subdirs(best_checkpoint_dir)
    epoch_root = args.epoch_checkpoints_dir or (best_checkpoint_dir / "epochs")
    wb_run = maybe_init_wandb(args, run_id, run_name)

    print("DPO TRAINING")
    print(f"  Base model           : {args.base_model}")
    print(f"  Adapter path         : {args.adapter_path}")
    print(f"  Train datasets       : {args.datasets}")
    print(f"  Eval datasets        : {args.datasets}")
    print(f"  Train sizes          : {args.train_sizes}")
    print(f"  Max seq len          : {args.max_seq_len}")
    print(f"  Batch size           : {args.batch_size}")
    print(f"  Grad accumulation    : {args.gradient_accumulation_steps}")
    print(
        f"  Effective update size: {args.batch_size * args.gradient_accumulation_steps}"
    )
    print(f"  Device               : {device}")
    print(f"  Run name             : {run_name}")
    print(f"  Epoch checkpoints    : {epoch_root}")
    print(f"  Final/best checkpoint: {best_checkpoint_dir}")
    print(f"  DPO beta             : {args.dpo_beta}")
    print(f"  Drop missing chosen  : {args.drop_missing_chosen}")
    print("=" * 60)

    model: Any | None = None
    optimizer: Any | None = None
    scheduler: Any | None = None
    mixed_train_loader: Any | None = None

    if args.skip_training:
        print(
            "[info] --skip_training enabled; skipping model training and using existing epoch checkpoints."
        )
        tokenizer = build_tokenizer_only(args.base_model)
    else:
        model, tokenizer = build_model_and_tokenizer(args, device)

    preference_bundles = load_preference_dataset_bundles(
        data_root=args.train_data_root,
        dataset_names=args.datasets,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        system_prompt=SYSTEM_PROMPT_MCQ,
        max_seq_len=args.max_seq_len,
        drop_missing_chosen=args.drop_missing_chosen,
    )

    eval_bundles = load_sft_dataset_bundles(
        data_root=args.train_data_root,
        dataset_names=args.datasets,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        system_prompt=SYSTEM_PROMPT_MCQ,
        max_seq_len=args.max_seq_len,
    )

    if not args.skip_training:
        assert model is not None
        for bundle in preference_bundles:
            _attach_reference_logps(bundle, model, device, dataset_name=bundle.name)

    preference_bundles = apply_train_size_caps(
        preference_bundles, args.train_sizes, seed=Random.SEED
    )
    eval_bundles = apply_test_size_caps(eval_bundles, args.test_sizes, seed=Random.SEED)

    print("\nLoaded preference datasets:")
    for bundle in preference_bundles:
        fallback_ratio = (
            bundle.train_fallback_chosen_count / bundle.train_size
            if bundle.train_size > 0
            else 0.0
        )
        print(
            "  - "
            f"{bundle.name}: "
            f"train={bundle.train_size} "
            f"train_max_tokens={bundle.train_max_tokens} "
            f"train_dropped={bundle.train_dropped_count} "
            f"fallback_chosen={bundle.train_fallback_chosen_count} "
            f"fallback_ratio={fallback_ratio:.2%}"
        )
        if fallback_ratio > 0.2:
            print(
                "    [warn] High fallback chosen ratio can add noisy preferences. "
                "Consider --drop-missing-chosen."
            )

    print("\nLoaded evaluation datasets:")
    for bundle in eval_bundles:
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
                "effective_train_sizes": {
                    b.name: b.train_size for b in preference_bundles
                },
                "effective_eval_test_sizes": {
                    b.name: b.test_size for b in eval_bundles
                },
                "epoch_checkpoints_dir": str(epoch_root),
                "adapter_path": str(args.adapter_path),
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
            preference_bundles,
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
            running_preference_accuracy = 0.0
            running_policy_margin = 0.0
            running_reference_margin = 0.0
            n_batches = 0
            optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(mixed_train_loader):
                policy_chosen_logps = _compute_logps_for_batch(
                    model,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["chosen_labels"],
                    device,
                )
                policy_rejected_logps = _compute_logps_for_batch(
                    model,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch["rejected_labels"],
                    device,
                )
                reference_chosen_logps = batch["reference_chosen_logps"].to(device)
                reference_rejected_logps = batch["reference_rejected_logps"].to(device)

                dpo_losses, policy_margin, reference_margin, preference_accuracy = (
                    _dpo_loss(
                        policy_chosen_logps=policy_chosen_logps,
                        policy_rejected_logps=policy_rejected_logps,
                        reference_chosen_logps=reference_chosen_logps,
                        reference_rejected_logps=reference_rejected_logps,
                        beta=args.dpo_beta,
                    )
                )
                loss = dpo_losses.mean()
                (loss / args.gradient_accumulation_steps).backward()

                raw_loss = float(loss.item())
                batch_samples = int(batch["chosen_input_ids"].shape[0])
                samples_seen += batch_samples
                running_loss += raw_loss
                running_preference_accuracy += float(preference_accuracy.mean().item())
                running_policy_margin += float(policy_margin.mean().item())
                running_reference_margin += float(reference_margin.mean().item())
                n_batches += 1

                if wb_run is not None:
                    wb_run.log(
                        {
                            "train/sample": samples_seen,
                            "train/batch_loss": raw_loss,
                            "train/dpo_loss": running_loss / n_batches,
                            "train/preference_accuracy": running_preference_accuracy
                            / n_batches,
                            "train/reward_margin": (
                                running_policy_margin - running_reference_margin
                            )
                            / n_batches,
                            "train/policy_margin": running_policy_margin / n_batches,
                            "train/reference_margin": running_reference_margin
                            / n_batches,
                            "train/policy_chosen_logp": float(
                                policy_chosen_logps.mean().item()
                            ),
                            "train/policy_rejected_logp": float(
                                policy_rejected_logps.mean().item()
                            ),
                            "train/reference_chosen_logp": float(
                                reference_chosen_logps.mean().item()
                            ),
                            "train/reference_rejected_logp": float(
                                reference_rejected_logps.mean().item()
                            ),
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
                        wb_run.log({"other/global_step": global_step}, commit=True)

            train_running = running_loss / n_batches if n_batches else float("inf")
            print(f"train_running_dpo_loss={train_running:.4f}")

            if args.save_epoch_checkpoints:
                epoch_dir, epoch_hf_dir, epoch_lora_dir = epoch_checkpoint_dirs(
                    epoch_root, epoch
                )
                print(f"[checkpoint] Exporting epoch {epoch} to {epoch_dir}")
                export_merged_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    output_dir=epoch_hf_dir,
                    base_model=args.base_model,
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
                wb_run.summary["last_epoch_train_running_dpo_loss"] = train_running
                wb_run.summary["last_epoch"] = epoch
                wb_run.summary["last_global_step"] = global_step

    print("\nTraining complete.")

    if not saved_epochs:
        if args.skip_training:
            raise RuntimeError(
                "--skip_training was set but no epochs were selected for evaluation. "
                "Provide --epochs > 0."
            )
        print(
            "[checkpoint] No per-epoch checkpoints were saved; exporting final model as best checkpoint."
        )
        export_merged_checkpoint(
            model=model,
            tokenizer=tokenizer,
            output_dir=best_hf_model_dir,
            base_model=args.base_model,
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
            print(
                "[validation] Materializing SFT validation prompts before freeing tokenizer."
            )
            validation_examples = materialize_validation_examples(
                bundles=eval_bundles,
                tokenizer=tokenizer,
                limit=args.validation_eval_limit,
            )
            print(
                f"[validation] Prepared {len(validation_examples)} validation examples."
            )

            del (
                model,
                optimizer,
                scheduler,
                mixed_train_loader,
                preference_bundles,
                eval_bundles,
                tokenizer,
            )
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
                current_tuple = (score, box, extracted)

                if best_record is None:
                    best_record = record
                else:
                    best_metrics = best_record["metrics"]
                    best_tuple = (
                        float(best_metrics.get("pass@1") or 0.0),
                        float(best_metrics.get("box_compliance_rate") or 0.0),
                        float(best_metrics.get("pass@1_among_extracted") or 0.0),
                    )
                    if current_tuple > best_tuple:
                        best_record = record

                if wb_run is not None:
                    wb_run.log(
                        {
                            "validation/epoch": epoch,
                            "validation/pass@1": metrics.get("pass@1"),
                            "validation/box_compliance_rate": metrics.get(
                                "box_compliance_rate"
                            ),
                            "validation/pass@1_among_extracted": metrics.get(
                                "pass@1_among_extracted"
                            ),
                        },
                        commit=True,
                    )

            assert best_record is not None
            best_epoch = int(best_record["epoch"])
            print(f"[validation] Best epoch by pass@1: {best_epoch}")
            copy_best_checkpoint(
                best_record, epoch_root, best_hf_model_dir, best_lora_dir
            )

            if wb_run is not None:
                best_metrics = best_record["metrics"]
                wb_run.summary["best_epoch"] = best_epoch
                wb_run.summary["best_validation_pass@1"] = best_metrics.get("pass@1")
                wb_run.summary["best_validation_box_compliance_rate"] = (
                    best_metrics.get("box_compliance_rate")
                )
                wb_run.summary["best_validation_pass@1_among_extracted"] = (
                    best_metrics.get("pass@1_among_extracted")
                )
                wb_run.summary["best_validation_results_path"] = best_record.get(
                    "results_path"
                )
        else:
            print(
                "[checkpoint] Post-training validation disabled; using final epoch as best checkpoint."
            )
            final_record = {"epoch": saved_epochs[-1]}
            copy_best_checkpoint(
                final_record, epoch_root, best_hf_model_dir, best_lora_dir
            )

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
