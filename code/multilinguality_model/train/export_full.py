"""code/multilinguality_model/train/export_full.py
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM


def _copytree_overwrite(src: Path, dst: Path, overwrite: bool) -> None:
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"model_path does not exist or is not a directory: {src}")

    if dst.exists():
        if not overwrite:
            raise FileExistsError(
                f"output_dir already exists: {dst}\n"
                f"Pass --overwrite to replace it."
            )
        shutil.rmtree(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    print(f"[info] Copied full model checkpoint:")
    print(f"       from: {src}")
    print(f"       to:   {dst}")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _apply_chat_template(output_dir: Path, chat_template_path: Path) -> None:
    if not chat_template_path.exists():
        raise FileNotFoundError(f"chat_template file not found: {chat_template_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(output_dir), trust_remote_code=True)
    tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
    tokenizer.save_pretrained(str(output_dir))

    print(f"[info] Applied chat template: {chat_template_path}")


def _apply_generation_config(output_dir: Path, generation_config_path: Path) -> None:
    if not generation_config_path.exists():
        raise FileNotFoundError(f"generation_config file not found: {generation_config_path}")

    cfg = _load_json(generation_config_path)
    _write_json(output_dir / "generation_config.json", cfg)

    print(f"[info] Applied generation config: {generation_config_path}")


def _verify_full_model(output_dir: Path, skip_model_load: bool) -> None:
    print("\n[info] Verifying exported full model...")

    required = ["config.json"]
    for name in required:
        path = output_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing required file after export: {path}")
        print(f"  {name}: OK")

    weight_files = list(output_dir.glob("*.safetensors")) + list(output_dir.glob("*.bin"))
    index_files = list(output_dir.glob("*.safetensors.index.json")) + list(output_dir.glob("*.bin.index.json"))

    if not weight_files and not index_files:
        raise FileNotFoundError(
            f"No model weight files found in {output_dir}. "
            f"Expected *.safetensors, *.bin, or index json."
        )
    print("  model weights: OK")

    tok = AutoTokenizer.from_pretrained(str(output_dir), trust_remote_code=True)
    print("  AutoTokenizer.from_pretrained: OK")
    print(f"  chat_template present: {tok.chat_template is not None}")

    if not skip_model_load:
        AutoModelForCausalLM.from_pretrained(
            str(output_dir),
            device_map="cpu",
            dtype="auto",
            trust_remote_code=True,
        )
        print("  AutoModelForCausalLM.from_pretrained: OK")

    print("[info] Verification passed.")


def _update_manifest(
    output_dir: Path,
    *,
    model_path: Path,
    run_name: str | None,
    chat_template: Path | None,
    generation_config: Path | None,
) -> None:
    manifest_path = output_dir / "checkpoint_manifest.json"

    if manifest_path.exists():
        try:
            manifest = _load_json(manifest_path)
        except Exception:
            manifest = {}
    else:
        manifest = {}

    manifest.update(
        {
            "run_name": run_name or output_dir.name,
            "source_model_path": str(model_path),
            "source_model_name": model_path.name,
            "output_dir": str(output_dir),
            "format": "full_hf_model",
            "merged_lora": False,
            "already_full_model": True,
            "chat_template_override": str(chat_template) if chat_template else None,
            "generation_config_override": str(generation_config) if generation_config else None,
        }
    )

    _write_json(manifest_path, manifest)
    print(f"[info] Updated manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an already-merged/full Hugging Face model checkpoint."
    )
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument(
        "--chat_template",
        type=Path,
        default=None,
        help="Optional .jinja chat template to save into tokenizer. If omitted, tokenizer files are left unchanged.",
    )
    parser.add_argument(
        "--generation_config",
        type=Path,
        default=None,
        help="Optional generation_config.json to copy into output_dir. If omitted, generation_config is left unchanged.",
    )

    parser.add_argument("--hf_repo", type=str, default=None)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_verify", action="store_true")
    parser.add_argument(
        "--skip_model_load",
        action="store_true",
        help="Skip CPU AutoModelForCausalLM load verification. Tokenizer/config checks still run.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "")

    model_path = args.model_path.resolve()
    output_dir = args.output_dir.resolve()

    print("=" * 80)
    print("FULL MODEL EXPORT")
    print(f"Source model:       {model_path}")
    print(f"Output dir:         {output_dir}")
    print(f"Run name:           {args.run_name or output_dir.name}")
    print(f"Chat template:      {args.chat_template}")
    print(f"Generation config:  {args.generation_config}")
    print(f"Push:               {args.push}")
    print("=" * 80)

    _copytree_overwrite(model_path, output_dir, overwrite=args.overwrite)

    if args.chat_template is not None:
        _apply_chat_template(output_dir, args.chat_template)

    if args.generation_config is not None:
        _apply_generation_config(output_dir, args.generation_config)

    _update_manifest(
        output_dir,
        model_path=model_path,
        run_name=args.run_name,
        chat_template=args.chat_template,
        generation_config=args.generation_config,
    )

    if not args.skip_verify:
        _verify_full_model(output_dir, skip_model_load=args.skip_model_load)

    if args.push:
        if not args.hf_repo:
            raise ValueError("--hf_repo is required when --push is set.")

        print(f"\n[info] Pushing to HF Hub: {args.hf_repo}")
        api = HfApi()
        api.create_repo(
            repo_id=args.hf_repo,
            repo_type="model",
            token=hf_token,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=args.hf_repo,
            repo_type="model",
            token=hf_token,
        )
        print(f"[info] Pushed: https://huggingface.co/{args.hf_repo}")
    else:
        print("\n[info] Local export complete. No HF push requested.")


if __name__ == "__main__":
    main()
