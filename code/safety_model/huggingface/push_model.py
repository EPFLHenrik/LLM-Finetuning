"""Validate and push a model checkpoint to Hugging Face.

Default target repo: <org>/safety_model

This script performs README-aligned checks before upload:
1) Required root files exist (config, generation config, safetensors)
2) Tokenizer has a chat_template
3) Base model identity appears to be Qwen/Qwen3-1.7B (best-effort)
4) Optional strict vLLM loadability check

Usage:
  uv run huggingface/push_model.py --org <your-org>
  uv run huggingface/push_model.py --org <your-org> --check-only
  uv run huggingface/push_model.py --org <your-org> --checkpoint-dir checkpoints/Qwen3-1.7B-base
"""

from __future__ import annotations


import argparse
import json
import os
import sys
import datetime
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from transformers import AutoConfig, AutoTokenizer


EXPECTED_BASE_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_CHECKPOINT_DIR = Path("checkpoints") / "Qwen3-1.7B-base"
DEFAULT_MODEL_NAME = "safety_model"


@dataclass
class CheckResult:
	name: str
	ok: bool
	details: str


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Validate local checkpoint and push it to Hugging Face Hub."
	)
	parser.add_argument(
		"--org",
		default="cs-552-2026-mystery-machine",
		help="Hugging Face organization/user namespace.",
	)
	parser.add_argument(
		"--model_name",
		default=DEFAULT_MODEL_NAME,
		help="Repository name inside org (default: safety_model).",
	)
	parser.add_argument(
		"--commit_message",
		help="Commit message for the upload commit.",
	)
	parser.add_argument(
		"--checkpoint_dir",
		type=Path,
		required=True,
		help="Local checkpoint directory to validate/upload.",
	)
	parser.add_argument(
		"--expected_base_model",
		default=EXPECTED_BASE_MODEL,
		help="Expected upstream base model identity.",
	)
	parser.add_argument(
		"--strict_vllm",
		action=argparse.BooleanOptionalAction,
		default=True,
		help=(
			"Require successful vLLM import + model load validation. "
			"Disable with --no-strict-vllm if your local env cannot run vLLM."
		),
	)
	parser.add_argument(
		"--check_only",
		action="store_true",
		help="Run checks only, do not upload.",
	)
	parser.add_argument(
		"--private",
		action="store_true",
		help="Create repo as private (README suggests public repos for this course).",
	)
	return parser.parse_args()


def _read_json(path: Path) -> dict:
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def check_required_files(checkpoint_dir: Path) -> list[CheckResult]:
	required = ["config.json", "generation_config.json", "tokenizer_config.json"]
	results = []

	for filename in required:
		file_path = checkpoint_dir / filename
		results.append(
			CheckResult(
				name=f"required_file:{filename}",
				ok=file_path.exists(),
				details=str(file_path),
			)
		)

	has_safetensors = any(checkpoint_dir.glob("*.safetensors"))
	safetensor_names = [p.name for p in checkpoint_dir.glob("*.safetensors")]
	results.append(
		CheckResult(
			name="required_file:model_safetensors",
			ok=has_safetensors,
			details=", ".join(safetensor_names) if safetensor_names else "none",
		)
	)
	return results


def check_chat_template(checkpoint_dir: Path) -> CheckResult:
	try:
		tok = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
		chat_template = getattr(tok, "chat_template", None)
		ok = bool(chat_template and str(chat_template).strip())
		details = "chat_template present" if ok else "chat_template missing/empty"
		return CheckResult(name="tokenizer_chat_template", ok=ok, details=details)
	except Exception as exc:
		return CheckResult(
			name="tokenizer_chat_template",
			ok=False,
			details=f"tokenizer load failed: {exc}",
		)


def check_generation_config(checkpoint_dir: Path) -> CheckResult:
	path = checkpoint_dir / "generation_config.json"
	if not path.exists():
		return CheckResult(
			name="generation_config_semantics",
			ok=False,
			details="generation_config.json missing",
		)

	cfg = _read_json(path)
	keys = ["temperature", "top_p", "top_k"]
	missing = [k for k in keys if k not in cfg]
	if missing:
		return CheckResult(
			name="generation_config_semantics",
			ok=False,
			details=f"missing keys: {missing}",
		)

	return CheckResult(
		name="generation_config_semantics",
		ok=True,
		details=(
			f"temperature={cfg.get('temperature')} top_p={cfg.get('top_p')} "
			f"top_k={cfg.get('top_k')}"
		),
	)


def check_base_model_identity(checkpoint_dir: Path, expected_base_model: str) -> CheckResult:
	try:
		cfg = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
		raw = _read_json(checkpoint_dir / "config.json")
		raw_name = str(raw.get("_name_or_path", ""))
		model_type = str(getattr(cfg, "model_type", "unknown"))

		qwen3_ok = model_type.lower() in {"qwen3", "qwen2"}
		base_hint_ok = expected_base_model in raw_name or "Qwen3-1.7B" in raw_name

		# `_name_or_path` is a hint and may not always survive checkpoint export,
		# so model_type is the stronger local check.
		ok = qwen3_ok
		details = (
			f"model_type={model_type}, _name_or_path={raw_name or '<missing>'}, "
			f"expected={expected_base_model}, base_hint_match={base_hint_ok}"
		)
		return CheckResult(name="base_model_identity", ok=ok, details=details)
	except Exception as exc:
		return CheckResult(
			name="base_model_identity",
			ok=False,
			details=f"config load failed: {exc}",
		)


def check_vllm_loadability(checkpoint_dir: Path, strict_vllm: bool) -> CheckResult:
	try:
		from vllm import LLM

		llm = LLM(
			model=str(checkpoint_dir),
			tokenizer=str(checkpoint_dir),
			trust_remote_code=True,
			tensor_parallel_size=1,
			max_model_len=512,
			gpu_memory_utilization=0.80,
		)
		del llm
		return CheckResult(
			name="vllm_loadability",
			ok=True,
			details="vLLM loaded the checkpoint successfully",
		)
	except Exception as exc:
		return CheckResult(
			name="vllm_loadability",
			ok=not strict_vllm,
			details=(
				f"vLLM check failed: {exc}. "
				f"strict_vllm={strict_vllm}"
			),
		)


def run_checks(checkpoint_dir: Path, expected_base_model: str, strict_vllm: bool) -> list[CheckResult]:
	results: list[CheckResult] = []
	results.extend(check_required_files(checkpoint_dir))
	results.append(check_chat_template(checkpoint_dir))
	results.append(check_generation_config(checkpoint_dir))
	results.append(check_base_model_identity(checkpoint_dir, expected_base_model))
	results.append(check_vllm_loadability(checkpoint_dir, strict_vllm))
	return results


def print_checks(results: list[CheckResult]) -> None:
	print("\n[Checks]\n")
	for item in results:
		status = "PASS" if item.ok else "FAIL"
		print(f"[{status}] {item.name}")
		print(f"       {item.details}")
	passed = sum(1 for r in results if r.ok)
	print(f"\n[Summary] {passed}/{len(results)} checks passed")


def upload_folder(repo_id: str, checkpoint_dir: Path, private: bool, commit_message: str) -> None:
	token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
	if not token:
		raise RuntimeError(
			"Missing auth token. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN in your shell."
		)

	print(f"\n[Upload] Ensuring repo exists: {repo_id}")
	create_repo(repo_id=repo_id, token=token, exist_ok=True, private=private)

	print(f"[Upload] Pushing folder: {checkpoint_dir}")
	api = HfApi(token=token)
	api.upload_folder(
		repo_id=repo_id,
		folder_path=str(checkpoint_dir),
		path_in_repo=".",
		commit_message=commit_message,
	)
	print(f"[Upload] Done: https://huggingface.co/{repo_id}")


def main() -> None:
	args = parse_args()
	checkpoint_dir = args.checkpoint_dir.resolve()
	repo_id = f"{args.org}/{args.model_name}"

	if not args.commit_message:
		# Example Commit: "Upload checkpoint from Qwen3-1.7B-base. Date of creation: 2024-06-01 12:34:56"

		   date_format = "%Y-%m-%d %H:%M:%S"
		   # Get the creation time of the checkpoint directory (best-effort, fallback to mtime if ctime is unavailable)
		   stat = checkpoint_dir.stat()
		   # On Unix, st_ctime is the inode change time, but often used as creation time if available
		   creation_time = getattr(stat, 'st_birthtime', None)
		   if creation_time is None:
			   # Fallback: use st_ctime (inode change time) or st_mtime (modification time)
			   creation_time = getattr(stat, 'st_ctime', stat.st_mtime)
		   date = datetime.datetime.fromtimestamp(creation_time).strftime(date_format)
		   args.commit_message = f"Upload checkpoint from {args.checkpoint_dir}. Date of creation: {date}"

	print(f"[Info] Checkpoint dir: {checkpoint_dir}")
	print(f"[Info] Target repo: {repo_id}")
	print(f"[Info] Strict vLLM check: {args.strict_vllm}")

	if not checkpoint_dir.exists():
		print(f"[Error] Checkpoint dir does not exist: {checkpoint_dir}")
		sys.exit(1)

	results = run_checks(
		checkpoint_dir=checkpoint_dir,
		expected_base_model=args.expected_base_model,
		strict_vllm=args.strict_vllm,
	)
	print_checks(results)

	all_ok = all(r.ok for r in results)
	if not all_ok:
		print("\n[Stop] Validation failed. Not uploading.")
		sys.exit(1)

	if args.check_only:
		print("\n[Done] Validation passed. check-only mode, no upload performed.")
		return

	upload_folder(repo_id=repo_id, checkpoint_dir=checkpoint_dir, private=args.private, commit_message=args.commit_message)


if __name__ == "__main__":
	main()
