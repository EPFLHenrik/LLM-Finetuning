"""Validate and push a dataset folder to Hugging Face.

Default target repo: <org>/safety (dataset repo type)

Usage:
  uv run huggingface/push_dataset.py --dataset_dir <local-dataset-folder>
  uv run huggingface/push_dataset.py --dataset_dir <local-dataset-folder> --check_only
  uv run huggingface/push_dataset.py --dataset_dir <local-dataset-folder> --dataset_name safety
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, load_from_disk
from huggingface_hub import create_repo


DEFAULT_ORG = "cs-552-2026-mystery-machine"
DEFAULT_DATASET_NAME = "safety"


@dataclass
class CheckResult:
	name: str
	ok: bool
	details: str


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Validate local dataset folder and push it to Hugging Face Hub."
	)
	parser.add_argument(
		"--org",
		default=DEFAULT_ORG,
		help="Hugging Face organization/user namespace.",
	)
	parser.add_argument(
		"--dataset_name",
		default=DEFAULT_DATASET_NAME,
		help="Dataset repository name inside org (default: safety). Can be set to safety/<custom-name>, e.g. safety/benchmark.",
	)
	parser.add_argument(
		"--dataset_dir",
		type=Path,
		required=True,
		help="Local dataset directory to validate/upload.",
	)
	parser.add_argument(
		"--commit_message",
		help="Commit message for the upload commit.",
	)
	parser.add_argument(
		"--check_only",
		action="store_true",
		help="Run checks only, do not upload.",
	)
	parser.add_argument(
		"--private",
		action="store_true",
		help="Create dataset repo as private.",
	)
	return parser.parse_args()


def check_dataset_dir_exists(dataset_dir: Path) -> CheckResult:
	return CheckResult(
		name="dataset_dir_exists",
		ok=dataset_dir.exists() and dataset_dir.is_dir(),
		details=str(dataset_dir),
	)


def check_dataset_dir_non_empty(dataset_dir: Path) -> CheckResult:
	has_files = any(dataset_dir.iterdir()) if dataset_dir.exists() else False
	return CheckResult(
		name="dataset_dir_non_empty",
		ok=has_files,
		details="contains files/directories" if has_files else "empty or missing",
	)


def check_loadability(dataset_dir: Path) -> CheckResult:
	# Support either a single dataset folder or a folder containing split subfolders.
	try:
		load_from_disk(str(dataset_dir))
		return CheckResult(
			name="dataset_loadability",
			ok=True,
			details="load_from_disk succeeded on dataset root",
		)
	except Exception as root_exc:
		split_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir()) if dataset_dir.exists() else []
		if not split_dirs:
			return CheckResult(
				name="dataset_loadability",
				ok=False,
				details=f"root load failed and no split dirs found: {root_exc}",
			)

		loaded = []
		failed = []
		for split_dir in split_dirs:
			try:
				load_from_disk(str(split_dir))
				loaded.append(split_dir.name)
			except Exception as split_exc:
				failed.append(f"{split_dir.name}: {split_exc}")

		if failed:
			return CheckResult(
				name="dataset_loadability",
				ok=False,
				details="split load failures -> " + "; ".join(failed),
			)

		return CheckResult(
			name="dataset_loadability",
			ok=True,
			details=f"root load failed, but split loads succeeded: {', '.join(loaded)}",
		)


def run_checks(dataset_dir: Path) -> list[CheckResult]:
	return [
		check_dataset_dir_exists(dataset_dir),
		check_dataset_dir_non_empty(dataset_dir),
		check_loadability(dataset_dir),
	]


def print_checks(results: list[CheckResult]) -> None:
	print("\n[Checks]\n")
	for item in results:
		status = "PASS" if item.ok else "FAIL"
		print(f"[{status}] {item.name}")
		print(f"       {item.details}")
	passed = sum(1 for r in results if r.ok)
	print(f"\n[Summary] {passed}/{len(results)} checks passed")


def resolve_dataset_target(org: str, dataset_name: str) -> tuple[str, str | None]:
	parts = [p for p in dataset_name.strip("/").split("/") if p]
	if not parts:
		raise ValueError("dataset_name must not be empty")

	repo_name = parts[0]
	config_name = "/".join(parts[1:]) if len(parts) > 1 else None
	repo_id = f"{org}/{repo_name}"
	return repo_id, config_name


def load_dataset_for_hub(dataset_dir: Path) -> Dataset | DatasetDict:
	"""Load a dataset saved with datasets.save_to_disk for Hub-native publishing."""
	try:
		loaded = load_from_disk(str(dataset_dir))
		if isinstance(loaded, (Dataset, DatasetDict)):
			return loaded
		raise TypeError(f"Unsupported loaded object type: {type(loaded)}")
	except Exception:
		split_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
		if not split_dirs:
			raise

		splits: dict[str, Dataset] = {}
		for split_dir in split_dirs:
			obj = load_from_disk(str(split_dir))
			if isinstance(obj, DatasetDict):
				if len(obj) != 1:
					raise ValueError(
						f"Split folder {split_dir} loaded as DatasetDict with "
						f"{len(obj)} splits; expected 1"
					)
				split_name, split_ds = next(iter(obj.items()))
				splits[split_name] = split_ds
			elif isinstance(obj, Dataset):
				splits[split_dir.name] = obj
			else:
				raise TypeError(f"Unsupported split object type: {type(obj)}")

		return DatasetDict(splits)


def harmonize_datasetdict_features(dataset_dict: DatasetDict) -> DatasetDict:
	"""Make all splits share identical columns/features for push_to_hub."""
	if not dataset_dict:
		return dataset_dict

	ordered_columns: list[str] = []
	column_features: dict[str, object] = {}

	for split_name in dataset_dict:
		ds = dataset_dict[split_name]
		for col in ds.column_names:
			if col not in column_features:
				ordered_columns.append(col)
				column_features[col] = ds.features[col]

	unified_features = Features({col: column_features[col] for col in ordered_columns})
	harmonized: dict[str, Dataset] = {}

	for split_name in dataset_dict:
		ds = dataset_dict[split_name]
		for col in ordered_columns:
			if col not in ds.column_names:
				ds = ds.add_column(col, [None] * len(ds), feature=column_features[col])

		ds = ds.select_columns(ordered_columns)
		ds = ds.cast(unified_features)
		harmonized[split_name] = ds

	return DatasetDict(harmonized)


def upload_dataset_folder(
	repo_id: str,
	config_name: str | None,
	dataset_dir: Path,
	private: bool,
	commit_message: str,
) -> None:
	token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
	if not token:
		raise RuntimeError(
			"Missing auth token. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN in your shell."
		)

	print(f"\n[Upload] Ensuring dataset repo exists: {repo_id}")
	create_repo(
		repo_id=repo_id,
		token=token,
		repo_type="dataset",
		exist_ok=True,
		private=private,
	)

	print(f"[Upload] Preparing Hub-native dataset payload from: {dataset_dir}")
	dataset_obj = load_dataset_for_hub(dataset_dir)
	if isinstance(dataset_obj, DatasetDict):
		dataset_obj = harmonize_datasetdict_features(dataset_obj)

	push_kwargs: dict[str, object] = {
		"repo_id": repo_id,
		"token": token,
		"private": private,
	}
	if config_name:
		push_kwargs["config_name"] = config_name

	print(
		"[Upload] Pushing with datasets.push_to_hub "
		"(Hub-native layout for Data Viewer compatibility)"
	)
	try:
		dataset_obj.push_to_hub(commit_message=commit_message, **push_kwargs)
	except TypeError:
		# Backward-compat for datasets versions lacking commit_message support.
		dataset_obj.push_to_hub(**push_kwargs)

	canonical_url = f"https://huggingface.co/datasets/{repo_id}"
	org_datasets_url = f"https://huggingface.co/{repo_id.split('/')[0]}/datasets"
	print(f"[Upload] Done: {canonical_url}")
	if config_name:
		print(f"[Upload] Config: {config_name}")
		print(f"[Upload] Data Viewer URL: {canonical_url}?config={config_name}")
	print(f"[Upload] Org datasets page: {org_datasets_url}")


def main() -> None:
	args = parse_args()
	dataset_dir = args.dataset_dir.resolve()

	try:
		repo_id, config_name = resolve_dataset_target(args.org, args.dataset_name)
	except ValueError as exc:
		print(f"[Error] Invalid --dataset_name: {exc}")
		sys.exit(1)

	if not args.commit_message:
		date_format = "%Y-%m-%d %H:%M:%S"
		stat = dataset_dir.stat() if dataset_dir.exists() else None
		if stat is None:
			created_at = datetime.datetime.now()
		else:
			creation_time = getattr(stat, "st_birthtime", None)
			if creation_time is None:
				creation_time = getattr(stat, "st_ctime", stat.st_mtime)
			created_at = datetime.datetime.fromtimestamp(creation_time)
		args.commit_message = (
			f"Upload dataset from {args.dataset_dir}. "
			f"Date of creation: {created_at.strftime(date_format)}"
		)

	print(f"[Info] Dataset dir: {dataset_dir}")
	print(f"[Info] Target dataset repo: {repo_id}")
	print(f"[Info] Dataset config: {config_name or '<default>'}")
	print(f"[Info] Canonical URL: https://huggingface.co/datasets/{repo_id}")

	if not dataset_dir.exists():
		print(f"[Error] Dataset dir does not exist: {dataset_dir}")
		sys.exit(1)

	results = run_checks(dataset_dir)
	print_checks(results)

	all_ok = all(r.ok for r in results)
	if not all_ok:
		print("\n[Stop] Validation failed. Not uploading.")
		sys.exit(1)

	if args.check_only:
		print("\n[Done] Validation passed. check-only mode, no upload performed.")
		return

	upload_dataset_folder(
		repo_id=repo_id,
		config_name=config_name,
		dataset_dir=dataset_dir,
		private=args.private,
		commit_message=args.commit_message,
	)


if __name__ == "__main__":
	main()
