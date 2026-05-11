"""Create a local base checkpoint for Qwen3-1.7B with CI-required files.

This script creates:
  checkpoints/Qwen3-1.7B-base/
	- config.json
	- generation_config.json
	- model*.safetensors (sharded if needed)
	- tokenizer files
	- chat_template.jinja

Usage:
  uv run train/base_model_checkpoint_creation.py

Optional:
  uv run train/base_model_checkpoint_creation.py --force
  uv run train/base_model_checkpoint_creation.py --temperature 0.7 --top-p 0.95 --top-k 50
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_OUTPUT_DIR = Path("checkpoints") / "Qwen3-1.7B-base"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Create a local base checkpoint folder compliant with project requirements."
	)
	parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument(
		"--dtype",
		choices=["auto", "float16", "bfloat16", "float32"],
		default="auto",
		help="Torch dtype used while loading model before saving safetensors.",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=0.7,
		help="Saved into generation_config.json.",
	)
	parser.add_argument(
		"--top-p",
		type=float,
		default=0.95,
		help="Saved into generation_config.json.",
	)
	parser.add_argument(
		"--top-k",
		type=int,
		default=50,
		help="Saved into generation_config.json.",
	)
	parser.add_argument(
		"--force",
		action="store_true",
		help="Overwrite output directory if it already exists.",
	)
	return parser.parse_args()


def resolve_dtype(dtype: str):
	if dtype == "float16":
		return torch.float16
	if dtype == "bfloat16":
		return torch.bfloat16
	if dtype == "float32":
		return torch.float32
	return "auto"


def ensure_clean_output_dir(path: Path, force: bool) -> None:
	if path.exists() and any(path.iterdir()):
		if not force:
			raise FileExistsError(
				f"Output directory already exists and is not empty: {path}. "
				"Use --force to overwrite."
			)
		shutil.rmtree(path)
	path.mkdir(parents=True, exist_ok=True)


def force_thinking_on(chat_template: str) -> str:
	"""Ensure the template defaults to Qwen thinking mode ON in CI.

	The CI does not pass enable_thinking; setting this variable at template
	level guarantees deterministic behavior.
	"""
	forced_line = "{%- set enable_thinking = true %}"
	pattern = r"\{%-\s*set\s+enable_thinking\s*=\s*(true|false)\s*%\}"

	if re.search(pattern, chat_template):
		return re.sub(pattern, forced_line, chat_template, count=1)

	return forced_line + "\n" + chat_template


def write_chat_template_file(output_dir: Path, chat_template: str) -> None:
	template_path = output_dir / "chat_template.jinja"
	template_path.write_text(chat_template, encoding="utf-8")


def save_generation_config(model, output_dir: Path, temperature: float, top_p: float, top_k: int) -> None:
	gen_cfg = model.generation_config
	gen_cfg.temperature = temperature
	gen_cfg.top_p = top_p
	gen_cfg.top_k = top_k
	gen_cfg.do_sample = True
	gen_cfg.save_pretrained(output_dir)


def validate_checkpoint(output_dir: Path) -> None:
	required_files = [
		"config.json",
		"generation_config.json",
		"tokenizer_config.json",
		"chat_template.jinja",
	]

	missing = [name for name in required_files if not (output_dir / name).exists()]
	if missing:
		raise RuntimeError(f"Missing required files: {missing}")

	has_safetensors = any(output_dir.glob("*.safetensors"))
	if not has_safetensors:
		raise RuntimeError("No .safetensors model weights were found in output directory.")

	tok = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
	if not getattr(tok, "chat_template", None):
		raise RuntimeError("Tokenizer has no chat_template after save.")

	template_text = (output_dir / "chat_template.jinja").read_text(encoding="utf-8")
	if "set enable_thinking = true" not in template_text:
		raise RuntimeError(
			"chat_template.jinja does not explicitly force enable_thinking=true."
		)


def write_manifest(output_dir: Path, model_id: str) -> None:
	manifest = {
		"base_model": model_id,
		"format": "safetensors",
		"notes": (
			"Generated locally from the official base model. "
			"Includes generation_config.json and tokenizer chat template."
		),
	}
	(output_dir / "checkpoint_manifest.json").write_text(
		json.dumps(manifest, indent=2), encoding="utf-8"
	)


def main() -> None:
	args = parse_args()
	output_dir: Path = args.output_dir

	ensure_clean_output_dir(output_dir, force=args.force)

	print(f"[info] Loading tokenizer: {args.model_id}")
	tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

	if not getattr(tokenizer, "chat_template", None):
		raise RuntimeError(
			"Base tokenizer has no chat_template. Cannot satisfy CI requirement."
		)

	tokenizer.chat_template = force_thinking_on(tokenizer.chat_template)

	print(f"[info] Loading model: {args.model_id}")
	model = AutoModelForCausalLM.from_pretrained(
		args.model_id,
		trust_remote_code=True,
		torch_dtype=resolve_dtype(args.dtype),
		low_cpu_mem_usage=True,
	)

	print(f"[info] Saving model + tokenizer to: {output_dir}")
	model.save_pretrained(output_dir, safe_serialization=True)
	tokenizer.save_pretrained(output_dir)
	write_chat_template_file(output_dir, tokenizer.chat_template)
	save_generation_config(model, output_dir, args.temperature, args.top_p, args.top_k)
	write_manifest(output_dir, args.model_id)

	print("[info] Validating generated checkpoint...")
	validate_checkpoint(output_dir)

	print("[ok] Base checkpoint created successfully.")
	print(f"[ok] Path: {output_dir}")


if __name__ == "__main__":
	main()
