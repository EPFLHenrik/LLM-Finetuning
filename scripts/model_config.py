#!/usr/bin/env python3
"""Load a local checkpoint, apply in-file config edits, and save a new checkpoint.

Edit constants in this file (SYSTEM_MESSAGE, GENERATION_CONFIG, etc.), then run:
1. Load model + tokenizer with from_pretrained(...)
2. Apply selected built-in edits
3. Save model, tokenizer, and configs into a new output directory
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer


CONFIG_FILE = "config.json"
GENERATION_CONFIG_FILE = "generation_config.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
CHAT_TEMPLATE_FILE = "chat_template.jinja"


SYSTEM_MESSAGE = (
    "You are a helpful assistant solving MCQ problems. At the end of your reasoning, "
    "you must include your answer in \\boxed{LETTER}. "
    "For example, if the answer is A, you must end your response with \\boxed{A}."
)

ENABLE_THINKING = True

GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95,
}

# Set these to dict values if you want to force specific config values on output.
# Keep as None to preserve the loaded source checkpoint values.
CONFIG_OVERRIDE: dict[str, Any] | None = None
TOKENIZER_CONFIG_OVERRIDE: dict[str, Any] | None = None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _enforce_enable_thinking(chat_template: str, enable_thinking: bool) -> str:
    forced_line = f"{{%- set enable_thinking = {'true' if enable_thinking else 'false'} %}}"
    pattern = r"\{%-\s*set\s+enable_thinking\s*=\s*(true|false)\s*%\}"
    if re.search(pattern, chat_template):
        return re.sub(pattern, forced_line, chat_template, count=1)
    return forced_line + "\n" + chat_template


def _build_chat_template(tokenizer) -> str:
    existing_chat = tokenizer.chat_template or ""
    enforced = _enforce_enable_thinking(existing_chat, enable_thinking=ENABLE_THINKING)
    system_line = "{%- set system_message = " + json.dumps(SYSTEM_MESSAGE) + " %}"
    return system_line + "\n" + enforced


def _should_apply(cmd: str, target: str) -> bool:
    return cmd in ("all", target)


def _safe_overwrite_output(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def adapt_checkpoint(
    model_path: Path,
    output_dir: Path,
    *,
    cmd: str,
    token: str | None,
    trust_remote_code: bool,
    use_fast: bool,
) -> None:
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=token,
        trust_remote_code=trust_remote_code,
    )

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=token,
        trust_remote_code=trust_remote_code,
        use_fast=use_fast,
    )

    # Apply chat template to tokenizer object before save_pretrained so tokenizer
    # artifacts stay coherent with the new template.
    if _should_apply(cmd, "chat_template"):
        tokenizer.chat_template = _build_chat_template(tokenizer)
        print("Applied chat_template from in-file constants")

    print(f"Saving model weights/config to {output_dir}...")
    model.save_pretrained(output_dir, safe_serialization=True)
    print(f"Saving tokenizer files to {output_dir}...")
    tokenizer.save_pretrained(output_dir)

    # Override selected config files after save_pretrained.
    if _should_apply(cmd, "config") and CONFIG_OVERRIDE is not None:
        _write_json(output_dir / CONFIG_FILE, CONFIG_OVERRIDE)
        print(f"Wrote {output_dir / CONFIG_FILE}")

    if _should_apply(cmd, "generation_config"):
        _write_json(output_dir / GENERATION_CONFIG_FILE, GENERATION_CONFIG)
        print(f"Wrote {output_dir / GENERATION_CONFIG_FILE}")

    if _should_apply(cmd, "tokenizer_config") and TOKENIZER_CONFIG_OVERRIDE is not None:
        _write_json(output_dir / TOKENIZER_CONFIG_FILE, TOKENIZER_CONFIG_OVERRIDE)
        print(f"Wrote {output_dir / TOKENIZER_CONFIG_FILE}")

    # Keep chat_template.jinja and tokenizer_config.chat_template in sync.
    if _should_apply(cmd, "chat_template"):
        chat_template = _build_chat_template(tokenizer)
        chat_template_path = output_dir / CHAT_TEMPLATE_FILE
        chat_template_path.write_text(chat_template, encoding="utf-8")
        print(f"Wrote {chat_template_path}")

        tokenizer_config_path = output_dir / TOKENIZER_CONFIG_FILE
        tokenizer_config = _read_json(tokenizer_config_path) if tokenizer_config_path.exists() else {}
        tokenizer_config["chat_template"] = chat_template
        _write_json(tokenizer_config_path, tokenizer_config)
        print(f"Updated {tokenizer_config_path} with chat_template")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Source model directory")
    parser.add_argument("--output-dir", required=True, help="Target directory for the adapted checkpoint")
    parser.add_argument(
        "--cmd",
        default="all",
        choices=["all", "config", "generation_config", "tokenizer_config", "chat_template"],
        help="Which built-in override to apply. 'all' applies all built-in overrides.",
    )
    parser.add_argument("--token", default=None, help="HF token for private models")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--use-fast", action="store_true", default=True)
    parser.add_argument("--no-fast", dest="use_fast", action="store_false")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    output_dir = Path(args.output_dir)
    _safe_overwrite_output(output_dir)

    adapt_checkpoint(
        model_path=model_path,
        output_dir=output_dir,
        cmd=args.cmd,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
        use_fast=args.use_fast,
    )

    print("Done")
    print(f"Source: {model_path}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()