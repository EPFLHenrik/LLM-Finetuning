from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM


def force_thinking_on(chat_template: str) -> str:
    forced_line = "{%- set enable_thinking = true %}"
    pattern = r"\{%-\s*set\s+enable_thinking\s*=\s*(true|false)\s*%\}"

    if re.search(pattern, chat_template):
        return re.sub(pattern, forced_line, chat_template, count=1)

    return forced_line + "\n" + chat_template


def inject_system_prompt(chat_template: str, system_message: str) -> str:
    escaped = system_message.replace("\\", "\\\\").replace('"', '\\"')
    set_line = f'{{%- set system_message = "{escaped}" %}}'

    system_msg_pattern = r"\{%-\s*set\s+system_message\s*=.*?%\}"
    thinking_pattern = r"(\{%-\s*set\s+enable_thinking\s*=\s*(?:true|false)\s*%\})"

    if re.search(system_msg_pattern, chat_template):
        chat_template = re.sub(system_msg_pattern, set_line, chat_template, count=1)
    elif re.search(thinking_pattern, chat_template):
        chat_template = re.sub(thinking_pattern, r"\1\n" + set_line, chat_template, count=1)
    else:
        chat_template = set_line + "\n" + chat_template

    old_block = (
        "    {%- if messages[0].role == 'system' %}\n"
        "        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n"
        "    {%- endif %}"
    )
    new_block = (
        "    {%- if messages[0].role == 'system' %}\n"
        "        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n"
        "    {%- else %}\n"
        "        {{- '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}\n"
        "    {%- endif %}"
    )
    if old_block in chat_template:
        chat_template = chat_template.replace(old_block, new_block, 1)

    return chat_template


def write_manifest(output_dir: Path, base_model: str) -> None:
    manifest = {
        "base_model": base_model,
        "format": "safetensors",
        "notes": (
            "Generated from SFT LoRA training by merging adapter into base model. "
            "Includes generation_config.json and tokenizer chat template."
        ),
    }
    (output_dir / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def validate_exported_checkpoint(output_dir: Path) -> None:
    required_files = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "checkpoint_manifest.json",
    ]

    missing = [name for name in required_files if not (output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Missing required files in exported checkpoint: {missing}")

    if not any(output_dir.glob("*.safetensors")):
        raise RuntimeError("No .safetensors files found in exported checkpoint.")

    template_text = (output_dir / "chat_template.jinja").read_text(encoding="utf-8")
    if "set enable_thinking = true" not in template_text:
        raise RuntimeError("chat_template.jinja must force enable_thinking=true.")
    if "set system_message" not in template_text:
        raise RuntimeError("chat_template.jinja must contain a default system_message.")


def export_merged_checkpoint(
    model,
    tokenizer,
    output_dir: Path,
    base_model: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    top_k: int,
) -> None:
    output_dir = Path(output_dir)
    temp_adapter_dir = output_dir.parent / ".tmp_sft_adapter_export"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if temp_adapter_dir.exists():
        shutil.rmtree(temp_adapter_dir)
    temp_adapter_dir.mkdir(parents=True, exist_ok=True)

    try:
        model.save_pretrained(temp_adapter_dir)

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="auto",
            device_map="cpu",
            trust_remote_code=True,
        )
        peft_model = PeftModel.from_pretrained(base, str(temp_adapter_dir))
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(output_dir, safe_serialization=True)

        tokenizer.chat_template = force_thinking_on(tokenizer.chat_template)
        tokenizer.chat_template = inject_system_prompt(tokenizer.chat_template, system_prompt)
        tokenizer.save_pretrained(output_dir)
        (output_dir / "chat_template.jinja").write_text(
            tokenizer.chat_template,
            encoding="utf-8",
        )

        gen_cfg = merged.generation_config
        gen_cfg.temperature = temperature
        gen_cfg.top_p = top_p
        gen_cfg.top_k = top_k
        gen_cfg.do_sample = True
        gen_cfg.save_pretrained(output_dir)

        write_manifest(output_dir, base_model)
        validate_exported_checkpoint(output_dir)
    finally:
        if temp_adapter_dir.exists():
            shutil.rmtree(temp_adapter_dir, ignore_errors=True)