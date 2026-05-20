"""Simple RLVR entry point with per-dataset metrics and W&B logging.

This script mirrors train/sft.py structure and dependencies while optimizing
sequence-level rewards (REINFORCE) for MCQ tasks with extractable outputs.

Usage example:
  uv run train/rlvr.py \
    --datasets safetybench \
    --adapter-path checkpoints/sft_run/LoRA \
    --run_name safetybench_rlvr \
    --epochs 1
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
from datasets import load_from_disk
from peft import PeftModel
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    get_cosine_schedule_with_warmup,
)

from constants.global_variables import DatasetFieldNames, Random
from evaluate.extract_answer import extract_boxed_answer
from prompts.system_prompt import SYSTEM_PROMPT_MCQ
from train.sft_export import export_merged_checkpoint

wandb: Any
try:
    import wandb

    HAS_WANDB = True
except Exception:
    wandb = cast(Any, None)
    HAS_WANDB = False


DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_DATA_DIR = Path("data/processed/sft")

DEFAULT_EPOCHS = 2
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 16
DEFAULT_LR = 5e-5
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_MAX_NEW_TOKENS = 16384


class BoxedAnswerStoppingCriteria(StoppingCriteria):
    """Stop generation once all batch items contain a boxed answer."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: Any) -> bool:
        for row in input_ids:
            decoded = self.tokenizer.decode(row, skip_special_tokens=True)
            if extract_boxed_answer(decoded) is None:
                return False
        return True


@dataclass
class RLVRDatasetBundle:
    name: str
    train_loader: DataLoader
    test_loader: DataLoader
    train_size: int
    test_size: int


class RLVRPromptDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_seq_len: int, system_prompt: str):
        self.examples: list[dict[str, Any]] = []

        for row in hf_dataset:
            prompt = row[DatasetFieldNames.RLVR_INPUT]
            answer = row[DatasetFieldNames.RLVR_ANSWER]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

            self.examples.append(
                {
                    "prompt_ids": prompt_ids,
                    "gold_answer": str(answer),
                    "prompt_text": prompt,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


class NamedTrainDataset(Dataset):
    def __init__(self, dataset: Dataset, dataset_name: str):
        self.dataset = dataset
        self.dataset_name = dataset_name

    def __len__(self) -> int:
        return len(cast(Any, self.dataset))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = dict(self.dataset[idx])
        example["dataset_name"] = self.dataset_name
        return example


def collate_prompt_batch(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    max_len = max(len(x["prompt_ids"]) for x in batch)
    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    gold_answers: list[str] = []
    dataset_names: list[str] = []

    for x in batch:
        pad_len = max_len - len(x["prompt_ids"])
        input_ids.append([pad_token_id] * pad_len + x["prompt_ids"])
        attention_mask.append([0] * pad_len + [1] * len(x["prompt_ids"]))
        gold_answers.append(x["gold_answer"])
        if "dataset_name" in x:
            dataset_names.append(x["dataset_name"])

    result: dict[str, Any] = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "gold_answers": gold_answers,
    }
    if dataset_names:
        result["dataset_name"] = dataset_names
    return result


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
    parser = argparse.ArgumentParser(description="Simple RLVR with LoRA adapter on MCQ datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names under data/processed/sft/<name> with train and test splits.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        required=True,
        help="Path to a pre-trained LoRA adapter directory used as RLVR initialization.",
    )
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
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
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Number of training samples between evaluations.",
    )
    parser.add_argument(
        "--reward-correct",
        type=float,
        default=1.0,
        help="Reward assigned when extracted answer matches the gold MCQ answer.",
    )
    parser.add_argument(
        "--reward-incorrect",
        type=float,
        default=0.0,
        help="Reward assigned when extracted answer does not match.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature used during RL rollouts.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Sampling top-p used during RL rollouts.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Sampling top-k used during RL rollouts.",
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
        "--export-merged",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also export merged HF checkpoint for the best model (default: false).",
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


def resolve_run_configuration(args: argparse.Namespace) -> tuple[str, str, Path]:
    run_id = args.run_id or args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or args.wandb_run_name or f"rlvr-{run_id}"

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


def resolve_effective_size(requested_size: int, available_size: int, arg_name: str) -> int:
    if requested_size < -1:
        raise ValueError(f"Each {arg_name} value must be -1 or a non-negative integer.")
    if requested_size == -1:
        return available_size
    return min(requested_size, available_size)


def apply_train_size_caps(
    bundles: list[RLVRDatasetBundle],
    train_sizes: list[int] | None,
    seed: int,
) -> list[RLVRDatasetBundle]:
    if train_sizes is None:
        return bundles

    if len(train_sizes) != len(bundles):
        raise ValueError(
            "--train-sizes must provide exactly one integer per dataset in --datasets. "
            f"Received {len(train_sizes)} sizes for {len(bundles)} datasets."
        )

    resized_bundles: list[RLVRDatasetBundle] = []
    for idx, (bundle, requested_size) in enumerate(zip(bundles, train_sizes)):
        target_train_size = resolve_effective_size(requested_size, bundle.train_size, "--train-sizes")
        train_dataset = bundle.train_loader.dataset

        if target_train_size < bundle.train_size:
            indices = list(range(bundle.train_size))
            rng = random.Random(seed + idx)
            rng.shuffle(indices)
            train_dataset = Subset(train_dataset, indices[:target_train_size])

        resized_bundles.append(
            RLVRDatasetBundle(
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
    bundles: list[RLVRDatasetBundle],
    test_sizes: list[int] | None,
    seed: int,
) -> list[RLVRDatasetBundle]:
    if test_sizes is None:
        return bundles

    if len(test_sizes) != len(bundles):
        raise ValueError(
            "--test-sizes must provide exactly one integer per dataset in --datasets. "
            f"Received {len(test_sizes)} sizes for {len(bundles)} datasets."
        )

    resized_bundles: list[RLVRDatasetBundle] = []
    for idx, (bundle, requested_size) in enumerate(zip(bundles, test_sizes)):
        target_test_size = resolve_effective_size(requested_size, bundle.test_size, "--test-sizes")
        test_dataset = bundle.test_loader.dataset

        if target_test_size < bundle.test_size:
            indices = list(range(bundle.test_size))
            rng = random.Random(seed + idx)
            rng.shuffle(indices)
            test_dataset = Subset(test_dataset, indices[:target_test_size])

        resized_bundles.append(
            RLVRDatasetBundle(
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


def load_dataset_bundles(
    dataset_names: list[str],
    data_dir: Path,
    tokenizer,
    max_seq_len: int,
    batch_size: int,
    system_prompt: str,
) -> list[RLVRDatasetBundle]:
    bundles: list[RLVRDatasetBundle] = []
    _collate = partial(collate_prompt_batch, pad_token_id=tokenizer.pad_token_id)

    for name in dataset_names:
        train_path = data_dir / name / "train"
        test_path = data_dir / name / "test"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"Expected train/test splits at {train_path} and {test_path}"
            )

        train_hf = load_from_disk(str(train_path))
        test_hf = load_from_disk(str(test_path))

        train_ds = RLVRPromptDataset(train_hf, tokenizer, max_seq_len, system_prompt)
        test_ds = RLVRPromptDataset(test_hf, tokenizer, max_seq_len, system_prompt)

        bundles.append(
            RLVRDatasetBundle(
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
            )
        )

    return bundles


def build_mixed_train_loader(bundles: list[RLVRDatasetBundle], seed: int) -> DataLoader:
    if not bundles:
        raise ValueError("At least one dataset bundle is required for training.")

    train_datasets = [
        NamedTrainDataset(bundle.train_loader.dataset, bundle.name) for bundle in bundles
    ]
    mixed_dataset = ConcatDataset(train_datasets)

    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed)

    def mixed_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        dataset_names = [example["dataset_name"] for example in batch]
        model_batch = cast(dict[str, Any], bundles[0].train_loader.collate_fn(batch))
        model_batch["dataset_name"] = dataset_names
        return model_batch

    return DataLoader(
        mixed_dataset,
        batch_size=bundles[0].train_loader.batch_size,
        shuffle=True,
        generator=shuffle_generator,
        collate_fn=mixed_collate_fn,
    )


def maybe_init_wandb(args: argparse.Namespace, run_name: str):
    if args.skip_wandb:
        return None
    if not HAS_WANDB:
        raise RuntimeError("wandb is not installed but --skip_wandb was not provided.")

    wb_run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=run_name,
        group="rlvr",
        job_type="training",
        config=vars(args),
    )

    wandb.define_metric("train/sample")
    wandb.define_metric("train/*", step_metric="train/sample")

    wandb.define_metric("test/sample")
    wandb.define_metric("test/*", step_metric="test/sample")

    wandb.define_metric("other/global_step")
    wandb.define_metric("other/*", step_metric="other/global_step")
    return wb_run


def build_model_and_tokenizer(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any]:
    tokenizer: Any = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model: Any = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map={"": device},
    )

    disable_peft_bitsandbytes_dispatch()

    model: Any = PeftModel.from_pretrained(
        base_model,
        str(args.adapter_path),
        is_trainable=True,
    )
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model.print_trainable_parameters()
    return model, tokenizer


def canonical_letter(value: str | None) -> str | None:
    if value is None:
        return None
    candidate = str(value).strip().upper()
    boxed = extract_boxed_answer(candidate)
    if boxed is not None:
        candidate = boxed.strip().upper()

    single = re.findall(r"[A-Z]", candidate)
    if not single:
        return None
    return single[-1]


def compute_rewards(
    generations: list[str],
    gold_answers: list[str],
    reward_correct: float,
    reward_incorrect: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    reward_values: list[float] = []
    correct_values: list[float] = []

    for pred_text, gold_text in zip(generations, gold_answers):
        pred_letter = canonical_letter(extract_boxed_answer(pred_text) or pred_text)
        gold_letter = canonical_letter(gold_text)
        is_correct = pred_letter is not None and gold_letter is not None and pred_letter == gold_letter
        reward_values.append(reward_correct if is_correct else reward_incorrect)
        correct_values.append(1.0 if is_correct else 0.0)

    rewards = torch.tensor(reward_values, dtype=torch.float32)
    correct = torch.tensor(correct_values, dtype=torch.float32)
    return rewards, correct


def sample_rollouts(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
    stopping_criteria = StoppingCriteriaList([BoxedAnswerStoppingCriteria(tokenizer)])
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            use_cache=True,
        )

    prompt_lens = attention_mask.sum(dim=1)
    full_attention = (generated != tokenizer.pad_token_id).long()

    generated_texts: list[str] = []
    generated_token_counts: list[int] = []
    for i in range(generated.shape[0]):
        pl = int(prompt_lens[i].item())
        total = int(full_attention[i].sum().item())
        start = max(0, pl)
        end = max(start, total)
        gen_ids = generated[i, start:end]
        generated_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
        generated_token_counts.append(max(0, end - start))

    gen_lengths = torch.tensor(generated_token_counts, dtype=torch.long, device=generated.device)
    return generated, full_attention, generated_texts, prompt_lens, gen_lengths


def compute_reinforce_loss(
    model,
    generated_ids: torch.Tensor,
    full_attention: torch.Tensor,
    prompt_lens: torch.Tensor,
    rewards: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    outputs = model(input_ids=generated_ids, attention_mask=full_attention)
    logits = outputs.logits

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_tokens = generated_ids[:, 1:]
    token_log_probs = log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)

    bsz, seq_minus_one = token_log_probs.shape
    token_mask = torch.zeros((bsz, seq_minus_one), device=token_log_probs.device, dtype=token_log_probs.dtype)

    for i in range(bsz):
        prompt_len = int(prompt_lens[i].item())
        total_len = int(full_attention[i].sum().item())
        # token_log_probs[t] corresponds to token at position t+1.
        start = max(0, prompt_len - 1)
        end = max(start, total_len - 1)
        if end > start:
            token_mask[i, start:end] = 1.0

    log_prob_sum = (token_log_probs * token_mask).sum(dim=1)
    token_count = token_mask.sum(dim=1).clamp_min(1.0)
    seq_log_prob = log_prob_sum / token_count

    rewards = rewards.to(seq_log_prob.device)
    advantages = rewards - rewards.mean()

    pg_loss = -(advantages * seq_log_prob).mean()

    metrics = {
        "mean_reward": float(rewards.mean().item()),
        "reward_std": float(rewards.std(unbiased=False).item()),
        "mean_advantage": float(advantages.mean().item()),
        "mean_seq_logprob": float(seq_log_prob.mean().item()),
        "mean_gen_tokens": float(token_count.mean().item()),
    }
    return pg_loss, metrics


def evaluate_accuracy(
    model,
    tokenizer,
    loader,
    device: torch.device,
    max_new_tokens: int,
) -> tuple[float, float]:
    stopping_criteria = StoppingCriteriaList([BoxedAnswerStoppingCriteria(tokenizer)])
    model.eval()
    total_reward = 0.0
    total_correct = 0.0
    total_count = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gold_answers = batch["gold_answers"]

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
                use_cache=True,
            )

        prompt_lens = attention_mask.sum(dim=1)
        generated_texts: list[str] = []
        for i in range(generated.shape[0]):
            pl = int(prompt_lens[i].item())
            gen_ids = generated[i, pl:]
            generated_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True))

        rewards, correct = compute_rewards(
            generated_texts,
            gold_answers,
            reward_correct=1.0,
            reward_incorrect=0.0,
        )

        batch_size = len(gold_answers)
        total_reward += float(rewards.sum().item())
        total_correct += float(correct.sum().item())
        total_count += batch_size

    model.train()
    if total_count == 0:
        return 0.0, 0.0
    return total_reward / total_count, total_correct / total_count


def weighted_average(metric_by_name: dict[str, float], weight_by_name: dict[str, int]) -> float:
    num = 0.0
    den = 0
    for name, metric in metric_by_name.items():
        w = weight_by_name[name]
        num += metric * w
        den += w
    return num / den if den else 0.0


def save_best_checkpoint(
    args: argparse.Namespace,
    model,
    tokenizer,
    best_hf_model_dir: Path,
    best_lora_dir: Path,
) -> None:
    if best_lora_dir.exists():
        shutil.rmtree(best_lora_dir)
    model.save_pretrained(str(best_lora_dir))

    if args.export_merged:
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


def train(args: argparse.Namespace) -> None:
    if args.grad_accum <= 0:
        raise ValueError("--grad-accum must be > 0")
    if args.eval_interval <= 0:
        raise ValueError("--eval-interval must be > 0")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")
    if not args.adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {args.adapter_path}")
    if args.train_sizes is not None and len(args.train_sizes) != len(args.datasets):
        raise ValueError("--train-sizes must have the same number of values as --datasets.")
    if args.test_sizes is not None and len(args.test_sizes) != len(args.datasets):
        raise ValueError("--test-sizes must have the same number of values as --datasets.")

    set_seeds(Random.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id, run_name, best_checkpoint_dir = resolve_run_configuration(args)
    best_hf_model_dir, best_lora_dir = resolve_checkpoint_subdirs(best_checkpoint_dir)
    wb_run = maybe_init_wandb(args, run_name)

    print("RLVR TRAINING")
    print(f"  Base model           : {args.base_model}")
    print(f"  Adapter source       : {args.adapter_path}")
    print(f"  Datasets             : {args.datasets}")
    print(f"  Train sizes          : {args.train_sizes}")
    print(f"  Eval interval        : every {args.eval_interval} samples")
    print(f"  Device               : {device}")
    print(f"  Run name             : {run_name}")
    print(f"  Best checkpoint path : {best_checkpoint_dir}")
    print(f"  - LoRA adapter       : {best_lora_dir}")
    print(f"  - HF merged model    : {best_hf_model_dir} (enabled={args.export_merged})")
    print("=" * 60)

    model, tokenizer = build_model_and_tokenizer(args, device)

    bundles = load_dataset_bundles(
        dataset_names=args.datasets,
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
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
        print(f"  - {bundle.name}: train={bundle.train_size} test={bundle.test_size}")

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

    best_weighted_acc = -1.0
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

        train_running_loss = 0.0
        train_running_reward = 0.0
        train_running_accuracy = 0.0
        n_batches = 0

        for batch in mixed_train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gold_answers = batch["gold_answers"]

            generated_ids, full_attention, generated_texts, prompt_lens, _ = sample_rollouts(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )

            rewards_cpu, correct_cpu = compute_rewards(
                generations=generated_texts,
                gold_answers=gold_answers,
                reward_correct=args.reward_correct,
                reward_incorrect=args.reward_incorrect,
            )

            loss, loss_metrics = compute_reinforce_loss(
                model=model,
                generated_ids=generated_ids,
                full_attention=full_attention,
                prompt_lens=prompt_lens,
                rewards=rewards_cpu,
            )

            scaled_loss = loss / args.grad_accum
            scaled_loss.backward()

            batch_samples = int(input_ids.shape[0])
            samples_seen += batch_samples
            while samples_seen >= next_eval_at_samples:
                eval_pending = True
                next_eval_at_samples += args.eval_interval

            batch_reward = float(rewards_cpu.mean().item())
            batch_acc = float(correct_cpu.mean().item())
            train_running_loss += float(loss.item())
            train_running_reward += batch_reward
            train_running_accuracy += batch_acc
            n_batches += 1
            grad_counter += 1

            if wb_run is not None:
                wb_run.log(
                    {
                        "train/sample": samples_seen,
                        "train/loss": float(loss.item()),
                        "train/reward": batch_reward,
                        "train/accuracy": batch_acc,
                        "train/running_loss": train_running_loss / n_batches,
                        "train/running_reward": train_running_reward / n_batches,
                        "train/running_accuracy": train_running_accuracy / n_batches,
                        "train/mean_gen_tokens": loss_metrics["mean_gen_tokens"],
                        "train/mean_seq_logprob": loss_metrics["mean_seq_logprob"],
                    },
                    commit=True,
                )

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

                if eval_pending:
                    eval_pending = False
                    print(
                        f"[eval] Step {global_step} (samples={samples_seen}): "
                        "Running evaluation..."
                    )

                    eval_acc_by_dataset: dict[str, float] = {}
                    eval_reward_by_dataset: dict[str, float] = {}
                    eval_weight_by_dataset: dict[str, int] = {}

                    for bundle in bundles:
                        avg_reward, avg_acc = evaluate_accuracy(
                            model=model,
                            tokenizer=tokenizer,
                            loader=bundle.test_loader,
                            device=device,
                            max_new_tokens=args.max_new_tokens,
                        )
                        eval_reward_by_dataset[bundle.name] = avg_reward
                        eval_acc_by_dataset[bundle.name] = avg_acc
                        eval_weight_by_dataset[bundle.name] = bundle.test_size

                    weighted_acc = weighted_average(eval_acc_by_dataset, eval_weight_by_dataset)
                    weighted_reward = weighted_average(eval_reward_by_dataset, eval_weight_by_dataset)
                    print(
                        f"[eval] Step {global_step}: "
                        f"test_weighted_accuracy={weighted_acc:.4f} "
                        f"test_weighted_reward={weighted_reward:.4f}"
                    )

                    if wb_run is not None:
                        eval_log = {
                            "test/sample": samples_seen,
                            "test/average_accuracy": weighted_acc,
                            "test/average_reward": weighted_reward,
                        }
                        for name in args.datasets:
                            eval_log[f"test/{name}_accuracy"] = eval_acc_by_dataset[name]
                            eval_log[f"test/{name}_reward"] = eval_reward_by_dataset[name]
                        wb_run.log(eval_log, commit=True)

                    if weighted_acc > best_weighted_acc:
                        best_weighted_acc = weighted_acc
                        best_epoch = epoch
                        best_step = global_step
                        print(
                            f"[info] New best checkpoint at step {global_step}. "
                            "Exporting LoRA adapter..."
                        )
                        save_best_checkpoint(
                            args=args,
                            model=model,
                            tokenizer=tokenizer,
                            best_hf_model_dir=best_hf_model_dir,
                            best_lora_dir=best_lora_dir,
                        )

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

                eval_acc_by_dataset = {}
                eval_reward_by_dataset = {}
                eval_weight_by_dataset = {}

                for bundle in bundles:
                    avg_reward, avg_acc = evaluate_accuracy(
                        model=model,
                        tokenizer=tokenizer,
                        loader=bundle.test_loader,
                        device=device,
                        max_new_tokens=args.max_new_tokens,
                    )
                    eval_reward_by_dataset[bundle.name] = avg_reward
                    eval_acc_by_dataset[bundle.name] = avg_acc
                    eval_weight_by_dataset[bundle.name] = bundle.test_size

                weighted_acc = weighted_average(eval_acc_by_dataset, eval_weight_by_dataset)
                weighted_reward = weighted_average(eval_reward_by_dataset, eval_weight_by_dataset)
                print(
                    f"[eval] Step {global_step}: "
                    f"test_weighted_accuracy={weighted_acc:.4f} "
                    f"test_weighted_reward={weighted_reward:.4f}"
                )

                if wb_run is not None:
                    eval_log = {
                        "test/sample": samples_seen,
                        "test/average_accuracy": weighted_acc,
                        "test/average_reward": weighted_reward,
                    }
                    for name in args.datasets:
                        eval_log[f"test/{name}_accuracy"] = eval_acc_by_dataset[name]
                        eval_log[f"test/{name}_reward"] = eval_reward_by_dataset[name]
                    wb_run.log(eval_log, commit=True)

                if weighted_acc > best_weighted_acc:
                    best_weighted_acc = weighted_acc
                    best_epoch = epoch
                    best_step = global_step
                    print(
                        f"[info] New best checkpoint at step {global_step}. "
                        "Exporting LoRA adapter..."
                    )
                    save_best_checkpoint(
                        args=args,
                        model=model,
                        tokenizer=tokenizer,
                        best_hf_model_dir=best_hf_model_dir,
                        best_lora_dir=best_lora_dir,
                    )

        epoch_loss = train_running_loss / n_batches if n_batches else 0.0
        epoch_reward = train_running_reward / n_batches if n_batches else 0.0
        epoch_acc = train_running_accuracy / n_batches if n_batches else 0.0
        print(
            f"train_running_loss={epoch_loss:.4f} "
            f"train_running_reward={epoch_reward:.4f} "
            f"train_running_accuracy={epoch_acc:.4f}"
        )

        if wb_run is not None:
            wb_run.summary["best_test_average_accuracy"] = best_weighted_acc
            wb_run.summary["best_epoch"] = best_epoch
            wb_run.summary["best_step"] = best_step
            wb_run.summary["best_checkpoint_dir"] = str(best_checkpoint_dir)
            wb_run.summary["best_lora_dir"] = str(best_lora_dir)
            if args.export_merged:
                wb_run.summary["best_hf_model_dir"] = str(best_hf_model_dir)

    print("\nTraining complete.")

    print("[eval] End of training: Running final evaluation...")
    final_acc_by_dataset: dict[str, float] = {}
    final_reward_by_dataset: dict[str, float] = {}
    final_weight_by_dataset: dict[str, int] = {}

    for bundle in bundles:
        avg_reward, avg_acc = evaluate_accuracy(
            model=model,
            tokenizer=tokenizer,
            loader=bundle.test_loader,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        final_reward_by_dataset[bundle.name] = avg_reward
        final_acc_by_dataset[bundle.name] = avg_acc
        final_weight_by_dataset[bundle.name] = bundle.test_size

    final_weighted_acc = weighted_average(final_acc_by_dataset, final_weight_by_dataset)
    final_weighted_reward = weighted_average(final_reward_by_dataset, final_weight_by_dataset)
    print(
        f"[eval] Final test_weighted_accuracy={final_weighted_acc:.4f} "
        f"test_weighted_reward={final_weighted_reward:.4f}"
    )

    if wb_run is not None:
        final_eval_log = {
            "test/sample": samples_seen,
            "test/average_accuracy": final_weighted_acc,
            "test/average_reward": final_weighted_reward,
        }
        for name in args.datasets:
            final_eval_log[f"test/{name}_accuracy"] = final_acc_by_dataset[name]
            final_eval_log[f"test/{name}_reward"] = final_reward_by_dataset[name]
        wb_run.log(final_eval_log, commit=True)

    if final_weighted_acc > best_weighted_acc:
        best_weighted_acc = final_weighted_acc
        best_epoch = args.epochs
        best_step = global_step
        print("[info] Final evaluation is the new best checkpoint. Exporting LoRA adapter...")
        save_best_checkpoint(
            args=args,
            model=model,
            tokenizer=tokenizer,
            best_hf_model_dir=best_hf_model_dir,
            best_lora_dir=best_lora_dir,
        )

    if wb_run is not None:
        wb_run.summary["best_test_average_accuracy"] = best_weighted_acc
        wb_run.summary["best_epoch"] = best_epoch
        wb_run.summary["best_step"] = best_step
        wb_run.summary["final_epoch"] = args.epochs
        wb_run.finish()

    print(f"Best checkpoint root: {best_checkpoint_dir} (epoch {best_epoch}, step {best_step})")
    print(f"  LoRA adapter   : {best_lora_dir}")
    if args.export_merged:
        print(f"  HF merged model: {best_hf_model_dir}")


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
