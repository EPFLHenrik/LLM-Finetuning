#!/usr/bin/env python3
"""Merge a LoRA adapter into a base model and save the result.

Applies LoRA weights manually (W_new = W + scaling * B @ A) so that
bitsandbytes/CUDA is not required.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file as safetensors_load
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_adapter_weights(adapter_path: Path) -> dict[str, torch.Tensor]:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        p = adapter_path / name
        if p.exists():
            if name.endswith(".safetensors"):
                return safetensors_load(str(p))
            return torch.load(str(p), map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"No adapter weights found in {adapter_path}")


def merge(adapter_path: Path, base_model_path: Path, output_path: Path) -> None:
    # Load adapter config for scaling factor
    with open(adapter_path / "adapter_config.json") as f:
        cfg = json.load(f)
    lora_alpha: float = cfg["lora_alpha"]
    r: int = cfg["r"]
    scaling = lora_alpha / r
    print(f"LoRA scaling = {lora_alpha}/{r} = {scaling}")

    print(f"Loading base model from {base_model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))

    print(f"Loading adapter weights from {adapter_path} ...")
    adapter_weights = load_adapter_weights(adapter_path)

    # Build a map: module_key -> {lora_A: tensor, lora_B: tensor}
    lora_pairs: dict[str, dict[str, torch.Tensor]] = {}
    # Keys look like: base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight
    for key, tensor in adapter_weights.items():
        m = re.match(r"^base_model\.model\.(.*)\.(lora_[AB])\.weight$", key)
        if not m:
            continue
        module_key, ab = m.group(1), m.group(2)
        lora_pairs.setdefault(module_key, {})[ab] = tensor

    print(f"Found {len(lora_pairs)} LoRA modules to merge ...")
    merged = 0
    model_state = dict(model.named_parameters())
    for module_key, ab_dict in lora_pairs.items():
        if "lora_A" not in ab_dict or "lora_B" not in ab_dict:
            print(f"  WARNING: incomplete pair for {module_key}, skipping")
            continue
        param_key = f"{module_key}.weight"
        if param_key not in model_state:
            print(f"  WARNING: {param_key} not found in base model, skipping")
            continue
        param = model_state[param_key]
        A = ab_dict["lora_A"].to(param.dtype)
        B = ab_dict["lora_B"].to(param.dtype)
        # delta = scaling * B @ A  (B is out_features x r, A is r x in_features)
        delta = scaling * (B @ A)
        param.data += delta
        merged += 1

    print(f"Merged {merged} LoRA modules.")

    print(f"Saving merged model to {output_path} ...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Copy chat template if present
    chat_template = base_model_path / "chat_template.jinja"
    if chat_template.exists():
        shutil.copy(chat_template, output_path / "chat_template.jinja")

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("adapter", type=Path)
    parser.add_argument("base_model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    merge(args.adapter.resolve(), args.base_model.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
