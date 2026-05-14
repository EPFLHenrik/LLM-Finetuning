"""multilingual/train/export.py

Merges a LoRA adapter into Qwen/Qwen3-1.7B and saves a vLLM-loadable
checkpoint, then optionally pushes to HF Hub.

Usage:
    python -m multilingual.train.export \
        --adapter_path multilingual/results/checkpoints/debug_sft/best_adapter \
        --output_dir multilingual/results/exports/debug_export

    # With HF push
    python -m multilingual.train.export \
        --adapter_path multilingual/results/checkpoints/sft_v1/best_adapter \
        --output_dir multilingual/results/exports/sft_v1 \
        --hf_repo cs-552-2026-mystery-machine/multilingual_model \
        --push
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

BASE_MODEL = "Qwen/Qwen3-1.7B"

DEFAULT_GENERATION_CONFIG = {
    "bos_token_id": 151643,
    "do_sample": False,          # greedy for MCQ/pass@1
    "eos_token_id": [151645, 151643],
    "pad_token_id": 151643,
    "temperature": 1.0,
    "top_k": 1,
    "top_p": 1.0,
    "transformers_version": "4.51.0",
}


def _patch_chat_template_no_think(tokenizer) -> None:
    """Prepend `{%- set enable_thinking = false %}` to force thinking off."""
    template = tokenizer.chat_template
    if template is None:
        print("[warn] No chat template found; skipping patch.")
        return

    prefix = "{%- set enable_thinking = false %}\n"
    if prefix.strip() in template:
        print("[info] Chat template already has enable_thinking=false; skipping.")
        return

    tokenizer.chat_template = prefix + template
    print("[info] Chat template patched: enable_thinking forced to false.")


def disable_peft_bitsandbytes_dispatch() -> None:
    """Avoid PEFT importing bitsandbytes dispatch when not quantizing."""
    try:
        import peft.tuners.lora.model as lora_model

        lora_model.is_bnb_available = lambda: False
        lora_model.is_bnb_4bit_available = lambda: False
        print("[info] Disabled PEFT bitsandbytes dispatch.")
    except Exception as exc:
        print(f"[warn] Failed to disable PEFT bitsandbytes dispatch: {exc}")

def _verify_export(output_dir: Path, hf_token: str) -> None:
    """Smoke-check the exported directory before pushing."""
    print("\nVerifying export...")

    # Check generation_config.json exists
    cfg_path = output_dir / "generation_config.json"
    assert cfg_path.exists(), "generation_config.json missing!"
    print(f"  generation_config.json: OK")

    # Check tokenizer loads
    tok = AutoTokenizer.from_pretrained(str(output_dir), token=hf_token)
    print(f"  AutoTokenizer.from_pretrained: OK")

    # Check chat template has no-think prefix
    template = tok.chat_template or ""
    assert "enable_thinking = false" in template, \
        "Chat template does not contain enable_thinking=false!"
    print(f"  Chat template no-think patch: OK")

    # Check apply_chat_template works and has assistant turn
    test_messages = [
        {"role": "system",  "content": "You are a test assistant."},
        {"role": "user",    "content": "What is 2+2?"},
    ]
    rendered = tok.apply_chat_template(
        test_messages, tokenize=False, add_generation_prompt=True
    )
    assert "<|im_start|>assistant" in rendered, \
        "apply_chat_template output missing assistant turn!"
    print(f"  apply_chat_template: OK")
    print(f"  Sample output (last 200 chars): ...{rendered[-200:]}")

    # Check model loads (CPU, no weights loaded — just config)
    AutoModelForCausalLM.from_pretrained(
        str(output_dir), token=hf_token, device_map="cpu",
        dtype=torch.bfloat16,
    )
    print(f"  AutoModelForCausalLM.from_pretrained: OK")
    print("Verification passed.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=Path, required=True)
    parser.add_argument("--output_dir",   type=Path, required=True)
    parser.add_argument("--base_model",   type=str,  default=BASE_MODEL)
    parser.add_argument("--do_sample",    action="store_true",
                        help="Enable sampling in generation_config (default: greedy).")
    parser.add_argument("--temperature",  type=float, default=0.6)
    parser.add_argument("--top_p",        type=float, default=0.95)
    parser.add_argument("--top_k",        type=int,   default=20)
    parser.add_argument("--hf_repo",      type=str,   default=None)
    parser.add_argument("--push",         action="store_true")
    parser.add_argument("--skip_verify",  action="store_true",
                        help="Skip post-export verification (faster, not recommended).")
    return parser.parse_args()


def main():
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base model:  {args.base_model}")
    print(f"Adapter:     {args.adapter_path}")
    print(f"Output:      {args.output_dir}")
    print(f"Sampling:    do_sample={args.do_sample}")

    # Load and merge
    print("\nLoading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )
    disable_peft_bitsandbytes_dispatch()
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, str(args.adapter_path))
    print("Merging...")
    model = model.merge_and_unload()
    model.eval()

    # Save merged model
    print("Saving merged model...")
    model.save_pretrained(args.output_dir, safe_serialization=True)

    # Save tokenizer with no-think patch
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
    _patch_chat_template_no_think(tokenizer)
    tokenizer.save_pretrained(args.output_dir)

    # Save generation_config.json
    gen_config = dict(DEFAULT_GENERATION_CONFIG)
    if args.do_sample:
        gen_config["do_sample"]   = True
        gen_config["temperature"] = args.temperature
        gen_config["top_p"]       = args.top_p
        gen_config["top_k"]       = args.top_k

    with open(args.output_dir / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(gen_config, f, indent=2)
    print("Saved generation_config.json")

    manifest = {
        "base_model": args.base_model,
        "adapter_path": str(args.adapter_path),
        "format": "safetensors",
        "merged_lora": True,
        "thinking_mode": "off",
        "generation_config": "generation_config.json",
        "notes": (
            "Generated from LoRA SFT by merging adapter into Qwen/Qwen3-1.7B. "
            "Includes generation_config.json and tokenizer chat template."
        ),
    }

    with open(args.output_dir / "checkpoint_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Saved checkpoint_manifest.json")

    # Verify
    if not args.skip_verify:
        _verify_export(args.output_dir, hf_token)

    # Push via upload_folder so generation_config.json is included
    if args.push:
        if not args.hf_repo:
            raise ValueError("--hf_repo is required when --push is set.")
        print(f"\nPushing to HF Hub: {args.hf_repo}")
        api = HfApi()
        api.create_repo(
            repo_id=args.hf_repo, repo_type="model",
            token=hf_token, exist_ok=True,
        )
        api.upload_folder(
            folder_path=str(args.output_dir),
            repo_id=args.hf_repo,
            repo_type="model",
            token=hf_token,
        )
        print(f"Pushed: https://huggingface.co/{args.hf_repo}")
    else:
        print(f"\nTo push later:")
        print(f"  python -m multilingual.train.export "
              f"--adapter_path {args.adapter_path} "
              f"--output_dir {args.output_dir} "
              f"--hf_repo <repo> --push")


if __name__ == "__main__":
    main()