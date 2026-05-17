#!/usr/bin/env python3
"""Prepare a baseline checkpoint directory for the project.

This script downloads files from the HF repo `Qwen/Qwen3-1.7B` (or a custom repo),
copies them into `models/qwen3_baseline`, and ensures the minimum required files
are present: `generation_config.json` and a `chat_template.jinja` under the
tokenizer folder. It does not attempt to convert weights (large and architecture-
specific); it copies whatever the source repo provides. If you need conversion,
run a custom conversion step separately.

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
from huggingface_hub import snapshot_download, HfApi
import os as _os


DEFAULT_REPO = "Qwen/Qwen3-1.7B"


CHAT_TEMPLATE = r"""{%- set enable_thinking = true %}
{%- set system_message = "You are a helpful assistant. When you produce the final answer, wrap it exactly in \\boxed{...} with no surrounding text. Use the <think>...</think> block for internal reasoning if enabled." %}

{%- if enable_thinking %}
<think>{{ system_message }}</think>
{%- endif %}

{{ user_message }}
{%- if add_generation_prompt %}
\nA:
{%- endif %}
"""


GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
    "max_new_tokens": 256
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def copy_snapshot(repo_id, dest, allow_patterns=None):
    print(f"Downloading snapshot from {repo_id} (this may take a while)...")
    local = snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns)
    print("Snapshot downloaded to:", local)
    ensure_dir(dest)
    # copy selected files (all) — user can clean up if needed
    for name in os.listdir(local):
        src = os.path.join(local, name)
        dst = os.path.join(dest, name)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    return local


def write_generation_config(dest):
    path = os.path.join(dest, "generation_config.json")
    if os.path.exists(path):
        print("generation_config.json already exists, skipping write")
        return
    with open(path, "w") as f:
        json.dump(GENERATION_CONFIG, f, indent=2)
    print("Wrote generation_config.json")


def write_chat_template(dest):
    tok_dir = os.path.join(dest, "tokenizer")
    ensure_dir(tok_dir)
    path = os.path.join(tok_dir, "chat_template.jinja")
    if os.path.exists(path):
        print("chat_template.jinja already exists, skipping write")
        return
    with open(path, "w") as f:
        f.write(CHAT_TEMPLATE)
    print("Wrote tokenizer/chat_template.jinja")


def write_tokenizer_files(dest, source_dir):
    """Ensure tokenizer.json and tokenizer_config.json exist in the checkpoint.

    If the source snapshot already has these files, copy them. Otherwise, try to
    materialize them from a Transformers tokenizer saved from the source model.
    """
    tok_dir = os.path.join(dest, "tokenizer")
    ensure_dir(tok_dir)

    source_tokenizer_json = os.path.join(source_dir, "tokenizer.json")
    source_tokenizer_config = os.path.join(source_dir, "tokenizer_config.json")
    dest_tokenizer_json = os.path.join(tok_dir, "tokenizer.json")
    dest_tokenizer_config = os.path.join(tok_dir, "tokenizer_config.json")

    if os.path.exists(source_tokenizer_json) and not os.path.exists(dest_tokenizer_json):
        shutil.copy2(source_tokenizer_json, dest_tokenizer_json)
        print("Copied tokenizer/tokenizer.json")

    if os.path.exists(source_tokenizer_config) and not os.path.exists(dest_tokenizer_config):
        shutil.copy2(source_tokenizer_config, dest_tokenizer_config)
        print("Copied tokenizer/tokenizer_config.json")

    if not os.path.exists(dest_tokenizer_config):
        fallback = {
            "chat_template": CHAT_TEMPLATE,
            "model_max_length": 32768,
            "padding_side": "left",
            "truncation_side": "left",
            "use_fast": True
        }
        with open(dest_tokenizer_config, "w") as f:
            json.dump(fallback, f, indent=2)
        print("Wrote fallback tokenizer/tokenizer_config.json")

    if not os.path.exists(dest_tokenizer_json):
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(source_dir, trust_remote_code=True)
            tokenizer.save_pretrained(tok_dir)
            print("Generated tokenizer/tokenizer.json via Transformers")
        except Exception as e:
            print("Could not generate tokenizer/tokenizer.json automatically:", e)


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
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--dest", default="models/qwen3_baseline")
    p.add_argument("--push-repo", default=None,
                   help="If set, push the resulting files to this HF repo id (e.g. user/model)")
    p.add_argument("--commit-message", default="Add model artifacts from setup_baseline",
                   help="Commit message used when uploading files to the HF repo")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()

    if os.path.exists(args.dest) and not args.yes:
        ans = input(f"Destination '{args.dest}' exists. Overwrite? [y/N] ")
        if ans.lower() not in ("y", "yes"):
            print("Aborted")
            return
        shutil.rmtree(args.dest)

    source_dir = copy_snapshot(args.repo, args.dest)
    write_generation_config(args.dest)
    write_chat_template(args.dest)
    write_tokenizer_files(args.dest, source_dir)

    if args.push_repo:
        try:
            print("Preparing to push files to HF repo:", args.push_repo)
            push_to_hf(args.push_repo, args.dest, commit_message=args.commit_message)
        except Exception as e:
            print("Failed to push to HF repo:", e)

    print("Done. Inspect the directory:")
    print("  ", args.dest)
    print("If weights are not safetensors or vLLM-loadable, you'll need to convert them separately.")


if __name__ == "__main__":
    main()
