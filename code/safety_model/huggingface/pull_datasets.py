"""Download the safety dataset repository from Hugging Face.

This script materializes every Hub config locally with ``load_dataset(... )`` and
``save_to_disk()`` so the resulting folders are Arrow-backed and can be consumed
directly by the existing loaders:
- `train/sft_data.py` expects `data/safety/processed/sft/<name>/train` and `.../test`
- `benchmark/run.py` scans for `.arrow` files under `data/safety/processed/benchmark`

By default the repo configs are written under `data/safety/processed/` using the config
name as the local folder name.
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk
from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "cs-552-2026-mystery-machine/safety"
DEFAULT_OUTPUT_ROOT = Path("data/safety/processed")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Download and mirror the safety dataset repository from Hugging Face."
	)
	parser.add_argument(
		"--repo-id",
		default=DEFAULT_REPO_ID,
		help="Hugging Face dataset repository to mirror.",
	)
	parser.add_argument(
		"--revision",
		default="main",
		help="Hub revision, branch, tag, or commit hash to download.",
	)
	parser.add_argument(
		"--output-root",
		type=Path,
		default=DEFAULT_OUTPUT_ROOT,
		help=(
			"Local directory to mirror the repository into. "
			"The remote folder structure is preserved under this root."
		),
	)
	parser.add_argument(
		"--token",
		default=None,
		help="Explicit Hugging Face token. Defaults to HF_TOKEN or HUGGINGFACE_HUB_TOKEN.",
	)
	parser.add_argument(
		"--force-download",
		action="store_true",
		help="Re-download files even if a cached local mirror exists.",
	)
	parser.add_argument(
		"--skip-validation",
		action="store_true",
		help="Skip local load_from_disk validation after download.",
	)
	return parser.parse_args()


def resolve_token(args: argparse.Namespace) -> str | None:
	if args.token:
		return args.token
	return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def group_parquet_files(folder: Path) -> dict[str, list[Path]]:
	groups: dict[str, list[Path]] = {}
	for parquet_path in sorted(folder.glob("*.parquet")):
		stem = parquet_path.name.split("-")[0]
		groups.setdefault(stem, []).append(parquet_path)
	return groups


def load_parquet_group(parquet_files: list[Path]) -> Dataset:
	dataset_dict = load_dataset(
		"parquet",
		data_files={"train": [str(path) for path in parquet_files]},
	)
	train_split = dataset_dict["train"]
	if not isinstance(train_split, Dataset):
		raise TypeError(f"Expected Dataset from parquet group, got {type(train_split)}")
	return train_split


def materialize_benchmark_dataset(folder: Path, parquet_files: list[Path], force_download: bool) -> None:
	temp_dir = folder.with_name(f".{folder.name}.materialized")
	if temp_dir.exists():
		shutil.rmtree(temp_dir)
	if force_download and folder.exists():
		for child in folder.iterdir():
			if child.is_dir():
				shutil.rmtree(child)
			else:
				child.unlink()

	dataset = load_parquet_group(parquet_files)
	dataset.save_to_disk(str(temp_dir))

	for parquet_file in parquet_files:
		parquet_file.unlink()
	if folder.exists():
		shutil.rmtree(folder)
	temp_dir.rename(folder)


def materialize_split_dataset(split_dir: Path, parquet_files: list[Path]) -> None:
	split_dir.parent.mkdir(parents=True, exist_ok=True)
	if split_dir.exists():
		shutil.rmtree(split_dir)

	dataset = load_parquet_group(parquet_files)
	dataset.save_to_disk(str(split_dir))

	for parquet_file in parquet_files:
		parquet_file.unlink()


def materialize_snapshot(output_root: Path, force_download: bool) -> None:
	leaf_dirs = sorted({parquet_path.parent for parquet_path in output_root.rglob("*.parquet")}, key=lambda p: len(p.parts), reverse=True)
	for leaf_dir in leaf_dirs:
		parquet_groups = group_parquet_files(leaf_dir)
		if not parquet_groups:
			continue

		relative_parts = leaf_dir.relative_to(output_root).parts
		top_level = relative_parts[0] if relative_parts else leaf_dir.name

		if top_level == "benchmark":
			if len(parquet_groups) != 1:
				raise RuntimeError(f"Expected a single parquet group for benchmark dataset {leaf_dir}, found {sorted(parquet_groups)}")
			print(f"[materialize] Benchmark dataset: {leaf_dir.relative_to(output_root)}")
			materialize_benchmark_dataset(leaf_dir, next(iter(parquet_groups.values())), force_download)
			continue

		print(f"[materialize] Split dataset container: {leaf_dir.relative_to(output_root)}")
		for split_name, parquet_files in parquet_groups.items():
			split_dir = leaf_dir / split_name
			materialize_split_dataset(split_dir, parquet_files)


def download_repository(
	repo_id: str,
	revision: str,
	output_root: Path,
	token: str | None,
	force_download: bool,
) -> Path:
	output_root.mkdir(parents=True, exist_ok=True)
	print(f"[download] Mirroring {repo_id}@{revision} -> {output_root}")
	local_dir = snapshot_download(
		repo_id=repo_id,
		revision=revision,
		repo_type="dataset",
		local_dir=str(output_root),
		local_dir_use_symlinks=False,
		token=token,
		force_download=force_download,
	)
	materialize_snapshot(Path(local_dir), force_download=force_download)
	return Path(local_dir)


def discover_dataset_roots(root: Path) -> list[Path]:
	candidates: set[Path] = set()
	if root.is_dir() and ((root / "state.json").exists() or (root / "dataset_info.json").exists()):
		candidates.add(root)

	for arrow_path in root.rglob("*.arrow"):
		candidates.add(arrow_path.parent)

	valid_roots = []
	for candidate in sorted(candidates):
		state_path = candidate / "state.json"
		dataset_info_path = candidate / "dataset_info.json"
		if state_path.exists() or dataset_info_path.exists():
			valid_roots.append(candidate)

	return valid_roots


def validate_dataset_roots(dataset_roots: list[Path]) -> None:
	if not dataset_roots:
		print("[validate] No dataset directories with .arrow files were found.")
		return

	print(f"[validate] Found {len(dataset_roots)} dataset directories with .arrow files:")
	for dataset_root in dataset_roots:
		print(f"  - {dataset_root}")

	for dataset_root in dataset_roots:
		try:
			loaded = load_from_disk(str(dataset_root))
			length = len(loaded) if hasattr(loaded, "__len__") else "unknown"
			print(f"[validate] OK {dataset_root} (rows={length})")
		except Exception as exc:
			print(f"[validate] FAIL {dataset_root}: {exc}")
			raise


def main() -> None:
	args = parse_args()
	token = resolve_token(args)

	try:
		local_dir = download_repository(
			repo_id=args.repo_id,
			revision=args.revision,
			output_root=args.output_root,
			token=token,
			force_download=args.force_download,
		)
	except Exception as exc:
		print(f"[error] Failed to download {args.repo_id}: {exc}")
		sys.exit(1)

	print(f"[download] Local mirror available at: {local_dir}")

	if args.skip_validation:
		return

	try:
		dataset_roots = discover_dataset_roots(local_dir)
		validate_dataset_roots(dataset_roots)
	except Exception as exc:
		print(f"[error] Validation failed: {exc}")
		sys.exit(1)


if __name__ == "__main__":
	main()
