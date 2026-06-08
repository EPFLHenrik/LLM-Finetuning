from __future__ import annotations

import json
import shutil
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM


STATIC_CHAT_TEMPLATE_PATH = Path("code/safety_model/constants/chat_template.jinja")


def load_static_chat_template(template_path: Path = STATIC_CHAT_TEMPLATE_PATH) -> str:
    if not template_path.exists():
        raise RuntimeError(f"Static chat template not found: {template_path}")
    return template_path.read_text(encoding="utf-8")


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
    source_template_text = load_static_chat_template()
    if template_text != source_template_text:
        raise RuntimeError(
            "chat_template.jinja in exported checkpoint must exactly match constants/chat_template.jinja."
        )


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
    _ = system_prompt  # Export now always uses constants/chat_template.jinja verbatim.
    output_dir = Path(output_dir)
    temp_adapter_dir = output_dir.parent / ".tmp_sft_adapter_export"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if temp_adapter_dir.exists():
        shutil.rmtree(temp_adapter_dir)
    temp_adapter_dir.mkdir(parents=True, exist_ok=True)

    try:
        static_chat_template = load_static_chat_template()

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

        tokenizer.chat_template = static_chat_template
        tokenizer.save_pretrained(output_dir)
        (output_dir / "chat_template.jinja").write_text(
            static_chat_template,
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