"""multilingual/train/push_base_tokenizer.py

Push the base Qwen3-1.7B tokenizer with the correct system prompt
patched into the chat template, without any LoRA merging.

Usage:
    python -m multilingual.train.push_base_tokenizer \
        --hf_repo cs-552-2026-mystery-machine/multilingual_model
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from multilingual.config import SYSTEM_PROMPT

BASE_MODEL = "Qwen/Qwen3-1.7B"


def _patch_system_prompt(tokenizer, system_prompt: str) -> None:
    """
    Patch the Qwen3 default chat template to inject a default system prompt
    when no system message is passed (e.g. during CI evaluation).
    """
    template = tokenizer.chat_template
    if template is None:
        raise ValueError("Tokenizer has no chat template.")

    # Qwen3 default template checks messages[0].role == 'system'
    # We add an else branch that injects our system prompt
    old_block = (
        "{%- if messages[0].role == 'system' %}\n"
        "        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n"
        "    {%- endif %}"
    )
    escaped   = system_prompt.replace("\\", "\\\\").replace("'", "\\'")
    new_block = (
        "{%- if messages[0].role == 'system' %}\n"
        "        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n"
        "    {%- else %}\n"
        f"        {{{{- '<|im_start|>system\\n{escaped}<|im_end|>\\n' }}}}\n"
        "    {%- endif %}"
    )

    if old_block in template:
        tokenizer.chat_template = template.replace(old_block, new_block)
        print("[info] System prompt patched into chat template.")
    else:
        print("[warn] Could not find system block — template may have changed.")
        print("[warn] Saving tokenizer without system prompt patch.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_repo",     type=str, required=True)
    parser.add_argument("--base_model",  type=str, default=BASE_MODEL)
    parser.add_argument("--output_dir",  type=Path, default=None,
                        help="Local dir to save to before pushing. "
                             "Default: /tmp/base_tokenizer_push")
    parser.add_argument("--push_model",  action="store_true",
                        help="Also push the full base model weights (slow).")
    return parser.parse_args()


def main():
    args     = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "")
    out_dir  = args.output_dir or Path("/tmp/base_tokenizer_push")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base model: {args.base_model}")
    print(f"HF repo:    {args.hf_repo}")
    print(f"Output dir: {out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
    _patch_system_prompt(tokenizer, SYSTEM_PROMPT)
    tokenizer.save_pretrained(out_dir)
    print("Tokenizer saved.")

    if args.push_model:
        print("Loading model weights (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, token=hf_token
        )
        model.save_pretrained(out_dir, safe_serialization=True)
        print("Model saved.")

    print(f"\nPushing to HF Hub: {args.hf_repo}")
    api = HfApi()
    api.create_repo(repo_id=args.hf_repo, repo_type="model",
                    token=hf_token, exist_ok=True)
    api.upload_folder(folder_path=str(out_dir), repo_id=args.hf_repo,
                      repo_type="model", token=hf_token)
    print(f"Done: https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()