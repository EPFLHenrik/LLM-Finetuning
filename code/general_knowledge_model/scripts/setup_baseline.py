#!/usr/bin/env python3
"""Prepare a baseline checkpoint directory for the project.

This script loads a model and tokenizer from the Hugging Face Hub using
`from_pretrained(...)`, saves them into the output directory, and ensures the
required root files are present: `config.json`, `generation_config.json`,
`tokenizer.json`, `tokenizer_config.json`, and `chat_template.jinja`.
It does not attempt to convert weights (large and architecture-specific); it
copies whatever the source model provides. If you need conversion, run a custom
conversion step separately.

Usage:
  python scripts/setup_baseline.py --dest models/qwen3_baseline

Optional args:
  --repo REPO_ID   Hugging Face repo id to copy from (default: Qwen/Qwen3-1.7B)
  --dest DIR       Destination dir (default: models/qwen3_baseline)
  --yes            Overwrite without prompt
"""
import argparse
import os
import shutil
import json
from datetime import datetime
import torch
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import os as _os
import re


DEFAULT_REPO = "Qwen/Qwen3-1.7B"
REFERENCE_CONFIG_REPO = "cs-552-2026-mystery-machine/multilingual_model"
REFERENCE_TOKENIZER_CONFIG_REPO = "cs-552-2026-mystery-machine/safety_model"

SYSTEM_MESSAGE = (
    "You are a helpful assistant solving MCQ problems. At the end of your reasoning, you must include your answer in \\boxed{LETTER}. "
    "For example, if the answer is A, you must end your response with \\boxed{A}."
)


def _enforce_enable_thinking(chat_template: str) -> str:
    forced_line = "{%- set enable_thinking = false %}"
    pattern = r"\{%-\s*set\s+enable_thinking\s*=\s*(true|false)\s*%\}" 
    if re.search(pattern, chat_template):
        return re.sub(pattern, forced_line, chat_template, count=1)

    return forced_line + "\n" + chat_template


def _enforce_system_message(chat_template: str) -> str:
    # Insert system_message as a Jinja block to preserve literal newlines
    forced_block = "{%- set system_message %}\n" + SYSTEM_MESSAGE + "\n{%- endset %}"
    # Remove any existing system_message assignment, whether single-line or block.
    chat_template = re.sub(
        r"(?ms)^\{%-\s*set\s+system_message\s*=\s*.*?%}\s*\n?",
        "",
        chat_template,
        count=1,
    )
    chat_template = re.sub(
        r"(?ms)^\{%-\s*set\s+system_message\s*%}\n.*?^\{%-\s*endset\s*%}\s*\n?",
        "",
        chat_template,
        count=1,
    )
    return forced_block + "\n" + chat_template


GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
    "max_new_tokens": 256
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def copy_reference_file(repo_id, filename, dest_path):
    print(f"Copying {filename} from {repo_id}...")
    src_path = hf_hub_download(repo_id=repo_id, filename=filename)
    ensure_dir(os.path.dirname(dest_path) or ".")
    shutil.copy2(src_path, dest_path)
    print(f"Wrote {dest_path}")


def write_config_json(dest):
    path = os.path.join(dest, "config.json")
    copy_reference_file(REFERENCE_CONFIG_REPO, "config.json", path)


def write_generation_config(dest):
    path = os.path.join(dest, "generation_config.json")
    if os.path.exists(path):
        print("generation_config.json already exists, skipping write")
        return
    with open(path, "w") as f:
        json.dump(GENERATION_CONFIG, f, indent=2)
    print("Wrote generation_config.json")


def write_chat_template(dest):
    print("Wrote chat_template.jinja")


def write_tokenizer_files(dest, source_dir):
    """Ensure tokenizer.json and tokenizer_config.json exist in the checkpoint.

    tokenizer.json is copied from the source snapshot when available. The
    tokenizer_config.json file is copied verbatim from the safety model repo so
    the checkpoint matches the expected tokenizer metadata.
    """
    source_tokenizer_json = os.path.join(source_dir, "tokenizer.json")
    dest_tokenizer_json = os.path.join(dest, "tokenizer.json")
    dest_tokenizer_config = os.path.join(dest, "tokenizer_config.json")

    if os.path.exists(source_tokenizer_json) and not os.path.exists(dest_tokenizer_json):
        shutil.copy2(source_tokenizer_json, dest_tokenizer_json)
        print("Copied tokenizer.json")

    copy_reference_file(REFERENCE_TOKENIZER_CONFIG_REPO, "tokenizer_config.json", dest_tokenizer_config)


def materialize_checkpoint(base_model, output_dir, hf_token=None):
    ensure_dir(output_dir)
    print(f"Loading model from {base_model} via from_pretrained(...)")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )
    model.save_pretrained(output_dir, safe_serialization=True)
    print(f"Saved model weights into {output_dir}")

    print(f"Loading tokenizer from {base_model} via from_pretrained(...)")
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token, trust_remote_code=True, use_fast=True)
    # Enforce enable_thinking and prepend the required system message, then
    # assign into the tokenizer object so it is saved with the tokenizer files.
    existing_chat = tokenizer.chat_template
    enforced = _enforce_system_message(existing_chat)
    final_chat = _enforce_enable_thinking(enforced)
    tokenizer.chat_template = final_chat

    tokenizer.save_pretrained(output_dir)
    print(f"Saved tokenizer files into {output_dir}")


def write_root_layout(dest, source_dir):
    tokenizer_dir = os.path.join(dest, "tokenizer")
    if os.path.isdir(tokenizer_dir):
        shutil.rmtree(tokenizer_dir)
        print("Removed tokenizer/ directory")

    write_config_json(dest)
    write_generation_config(dest)
    write_chat_template(dest)
    write_tokenizer_files(dest, source_dir)


def write_checkpoint_manifest(dest, base_model, adapter_path=None, best_eval_loss=None):
    run_key = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    manifest = {
        "run_name": run_key,
        "base_model": base_model,
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "best_eval_loss": best_eval_loss,
        "format": "safetensors",
        "merged_lora": False,
        "thinking_mode": "off",
        "generation_config": "generation_config.json",
    }
    with open(os.path.join(dest, "checkpoint_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("Saved checkpoint_manifest.json")


def push_to_hf(repo_id, source_dir, commit_message="Add model artifacts"):
    """Upload files from `source_dir` directly to the HF repo using the Hub API.

    Uses the `HF_TOKEN` environment variable if present, or relies on
    `huggingface-cli login` session.
    """
    api = HfApi()
    token = _os.environ.get("HF_TOKEN")
    print(f"Uploading folder {source_dir} to {repo_id}...")
    api.upload_folder(folder_path=source_dir,
                      repo_id=repo_id,
                      repo_type="model",
                      commit_message=commit_message,
                      token=token)
    print("Upload complete.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", "--repo", dest="base_model", default=DEFAULT_REPO)
    p.add_argument("--output-dir", "--dest", dest="output_dir", default="models/qwen3_baseline2")
    p.add_argument("--push-repo", default=None,
                   help="If set, push the resulting files to this HF repo id (e.g. user/model)")
    p.add_argument("--commit-message", default="Add model artifacts from setup_baseline",
                   help="Commit message used when uploading files to the HF repo")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.output_dir) and not args.yes:
        ans = input(f"Destination '{args.output_dir}' exists. Overwrite? [y/N] ")
        if ans.lower() not in ("y", "yes"):
            print("Aborted")
            return
        shutil.rmtree(args.output_dir)

    hf_token = _os.environ.get("HF_TOKEN")
    materialize_checkpoint(args.base_model, args.output_dir, hf_token=hf_token)
    write_root_layout(args.output_dir, args.output_dir)
    write_checkpoint_manifest(args.output_dir, base_model=args.base_model)

    if args.push_repo:
        try:
            print("Preparing to push files to HF repo:", args.push_repo)
            push_to_hf(args.push_repo, args.output_dir, commit_message=args.commit_message)
        except Exception as e:
            print("Failed to push to HF repo:", e)

    print("Done. Inspect the directory:")
    print("  ", args.output_dir)
    print("If weights are not safetensors or vLLM-loadable, you'll need to convert them separately.")


if __name__ == "__main__":
    main()
