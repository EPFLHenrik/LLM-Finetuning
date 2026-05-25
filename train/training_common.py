from __future__ import annotations

import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Sampler
from transformers import AutoTokenizer


BASE_MODEL_FALLBACK = "checkpoints/Qwen3-1.7B-baseline"


class HomogeneousBatchSampler(Sampler[list[int]]):
	"""Keep each optimizer update within one dataset partition."""

	def __init__(
		self,
		dataset_sizes: list[int],
		batch_size: int,
		gradient_accumulation_steps: int,
		seed: int,
	):
		self.dataset_sizes = dataset_sizes
		self.batch_size = batch_size
		self.gradient_accumulation_steps = gradient_accumulation_steps
		self.seed = seed
		self._iter_count = 0
		self._update_end_flags: list[bool] = []

	def __len__(self) -> int:
		return sum(math.ceil(size / self.batch_size) for size in self.dataset_sizes)

	def optimizer_steps_per_epoch(self) -> int:
		micro_batches_per_update = self.batch_size * self.gradient_accumulation_steps
		return sum(math.ceil(size / micro_batches_per_update) for size in self.dataset_sizes)

	def update_ends_at(self, batch_idx: int) -> bool:
		if batch_idx < 0 or batch_idx >= len(self._update_end_flags):
			return False
		return self._update_end_flags[batch_idx]

	def __iter__(self):
		rng = random.Random(self.seed + self._iter_count)
		self._iter_count += 1

		update_blocks: list[list[list[int]]] = []
		offset = 0
		for size in self.dataset_sizes:
			dataset_indices = list(range(offset, offset + size))
			rng.shuffle(dataset_indices)

			micro_batches: list[list[int]] = []
			for start in range(0, size, self.batch_size):
				micro_batch = dataset_indices[start : start + self.batch_size]
				if micro_batch:
					micro_batches.append(micro_batch)

			for start in range(0, len(micro_batches), self.gradient_accumulation_steps):
				update_block = micro_batches[start : start + self.gradient_accumulation_steps]
				if update_block:
					update_blocks.append(update_block)

			offset += size

		rng.shuffle(update_blocks)

		flattened_batches: list[list[int]] = []
		update_end_flags: list[bool] = []
		for block in update_blocks:
			for idx, micro_batch in enumerate(block):
				flattened_batches.append(micro_batch)
				update_end_flags.append(idx == len(block) - 1)

		self._update_end_flags = update_end_flags
		yield from flattened_batches


def set_seeds(seed: int) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def resolve_run_configuration(args: Any) -> tuple[str, str, Path]:
	run_id = args.run_id or args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
	run_name = args.run_name or args.wandb_run_name or f"run-{run_id}"

	if args.best_checkpoint_dir is not None:
		best_checkpoint_dir = args.best_checkpoint_dir
	else:
		default_dir_name = args.run_name or run_id
		best_checkpoint_dir = Path("checkpoints") / default_dir_name

	return run_id, run_name, best_checkpoint_dir


def resolve_checkpoint_subdirs(checkpoint_root: Path) -> tuple[Path, Path]:
	return checkpoint_root / "hf_model", checkpoint_root / "LoRA"


def build_tokenizer_only(base_model: str = BASE_MODEL_FALLBACK):
	tokenizer: Any = AutoTokenizer.from_pretrained(base_model)
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token_id = tokenizer.eos_token_id
	return tokenizer