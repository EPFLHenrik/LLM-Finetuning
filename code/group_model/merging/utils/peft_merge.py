from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import yaml


def dtype_from_string(dtype: str):
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(dtype.lower(), torch.bfloat16)


def adapter_name_from_path(path: str, index: int) -> str:
    name = Path(path).name.strip().replace("-", "_").replace(" ", "_")
    return name if name else f"adapter_{index}"


def build_mergekit_config(
    args: Any,
    merge_method: str,
    *,
    model_parameters_builder: Callable[[str, float], Mapping[str, Any]],
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "merge_method": merge_method,
        "base_model": args.base_model,
        "models": [
            {
                "model": model,
                "parameters": dict(model_parameters_builder(model, weight)),
            }
            for model, weight in zip(args.merging_targets, args.weights)
        ],
        "dtype": args.dtype,
    }
    if parameters is not None:
        config["parameters"] = dict(parameters)
    return config


def run_mergekit_yaml(args: Any, config: dict[str, Any]) -> None:
    config_source = yaml.safe_dump(config, sort_keys=False)

    with open(args.config_path, "w", encoding="utf-8") as handle:
        handle.write(config_source)

    from mergekit.architecture import base as mergekit_arch_base
    from mergekit.config import MergeConfiguration
    from mergekit.merge import run_merge
    from mergekit.options import MergeOptions

    types_namespace = {"torch": torch}
    mergekit_arch_base.ConfiguredModuleArchitecture.model_rebuild(
        _types_namespace=types_namespace
    )
    mergekit_arch_base.ConfiguredModelArchitecture.model_rebuild(
        _types_namespace=types_namespace
    )

    merge_config = MergeConfiguration.model_validate(config)
    use_cuda = torch.cuda.is_available()
    merge_options = MergeOptions(
        cuda=use_cuda,
        device=None if use_cuda else "cpu",
    )
    merge_options.apply_global_options()

    run_merge(
        merge_config,
        args.output_path,
        options=merge_options,
        config_source=config_source,
    )


def run_mergekit_merge(
    args: Any,
    merge_method: str,
    *,
    model_parameters_builder: Callable[[str, float], Mapping[str, Any]],
    parameters: Mapping[str, Any] | None = None,
) -> None:
    config = build_mergekit_config(
        args,
        merge_method,
        model_parameters_builder=model_parameters_builder,
        parameters=parameters,
    )
    run_mergekit_yaml(args, config)


def run_peft_lora_merge(
    *,
    base_model_path: str,
    adapter_paths: list[str],
    output_path: str,
    weights: list[float],
    dtype: str,
    combination_type: str,
    density: float | None = None,
    merged_adapter_name: str = "merged_adapter",
) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not adapter_paths:
        raise ValueError("adapter_paths must contain at least one LoRA adapter path")

    base_dtype = dtype_from_string(dtype)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=base_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    adapter_names = [
        adapter_name_from_path(path, i) for i, path in enumerate(adapter_paths)
    ]
    model: Any = PeftModel.from_pretrained(
        base_model,
        adapter_paths[0],
        adapter_name=adapter_names[0],
    )

    for name, path in zip(adapter_names[1:], adapter_paths[1:]):
        model.load_adapter(path, adapter_name=name)

    model.add_weighted_adapter(
        adapters=adapter_names,
        weights=weights,
        adapter_name=merged_adapter_name,
        combination_type=combination_type,
        **({"density": density} if density is not None else {}),
    )
    model.set_adapter(merged_adapter_name)

    merged: Any = model.merge_and_unload()
    merged.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
