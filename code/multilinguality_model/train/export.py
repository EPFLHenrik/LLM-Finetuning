"""code/multilinguality_model/train/export.py

Merge a LoRA adapter into the base model, save a vLLM-loadable checkpoint,
and optionally push it to the Hugging Face Hub.
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
from pathlib import Path
import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from multilinguality_model.config import SYSTEM_PROMPT
from multilinguality_model.paths import get_package_dir, get_repo_root

REPO = get_repo_root()
BASE_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_TEMPLATE = get_package_dir() / "assets" / "chat_template_multilingual-nonthinking.jinja"
_QWEN_DEFAULT_SENTINEL = "__qwen_default__"

DEFAULT_GENERATION_CONFIG = {
    "bos_token_id": 151643,
    "do_sample": False,
    "eos_token_id": [151645, 151643],
    "pad_token_id": 151643,
    "max_new_tokens": 32,
    "transformers_version": "4.51.0",
}

THINKING_GENERATION_CONFIG = {
    "bos_token_id": 151643,
    "do_sample": True,
    "eos_token_id": [151645, 151643],
    "pad_token_id": 151643,
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "transformers_version": "4.51.0",
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _get_best_eval_loss(checkpoint_dir: Path) -> float | None:
    for candidate in [
        checkpoint_dir / "trainer_state.json",
        checkpoint_dir / "best_adapter" / "trainer_state.json",
        checkpoint_dir.parent / "trainer_state.json",
    ]:
        if candidate.exists():
            state = _load_json(candidate)
            val = state.get("best_metric")
            if val is not None:
                return float(val)
    print(f"[warn] trainer_state.json not found near {checkpoint_dir}")
    return None


def _patch_system_prompt(tokenizer, system_prompt: str) -> None:
    """Inject default system prompt into Qwen3 inference template."""
    template = tokenizer.chat_template
    if template is None:
        raise ValueError("Tokenizer has no chat template.")
    old_block = (
        "{%- if messages[0].role == 'system' %}\n"
        "        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n"
        "    {%- endif %}"
    )
    escaped = system_prompt.replace("\\", "\\\\").replace("'", "\\'")
    new_block = (
        "{%- if messages[0].role == 'system' %}\n"
        "        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n"
        "    {%- else %}\n"
        f"        {{{{- '<|im_start|>system\\n{escaped}<|im_end|>\\n' }}}}\n"
        "    {%- endif %}"
    )
    if old_block in template:
        tokenizer.chat_template = template.replace(old_block, new_block)
        print("[info] System prompt patched into Qwen3 default template.")
    else:
        print("[warn] Could not find system block; saving template without patch.")


def _disable_peft_bnb_dispatch() -> None:
    try:
        import peft.tuners.lora.model as lora_model
        lora_model.is_bnb_available = lambda: False  # type: ignore[attr-defined]
        lora_model.is_bnb_4bit_available = lambda: False  # type: ignore[attr-defined]
        print("[info] Disabled PEFT bitsandbytes dispatch.")
    except Exception as exc:
        print(f"[warn] Failed to disable PEFT bitsandbytes dispatch: {exc}")


def _copy_checkpoint_files(adapter_path: Path, output_dir: Path) -> None:
    """Copy all non-weight files from checkpoint to output_dir."""
    exclude_suffixes = {".safetensors", ".bin", ".pt"}
    exclude_names = {"adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"}
    for src in adapter_path.iterdir():
        if src.is_dir():
            continue
        if src.suffix in exclude_suffixes:
            continue
        if src.name in exclude_names:
            continue
        shutil.copy2(src, output_dir / src.name)
        print(f"  Copied {src.name}")


def _apply_chat_template(tokenizer, override: str | None) -> None:
    """
    Apply chat template.

    override is None:
        use code/multilinguality_model/assets/chat_template_multilingual-nonthinking.jinja
    override == _QWEN_DEFAULT_SENTINEL:
        use Qwen3 default template with system prompt patch
    override is a path:
        use that .jinja file
    """
    if override is None:
        if not DEFAULT_TEMPLATE.exists():
            raise FileNotFoundError(
                f"Default chat template not found: {DEFAULT_TEMPLATE}\n"
                f"Pass --override_chat_template to use the Qwen3 default template, "
                f"or pass --override_chat_template PATH."
            )
        tokenizer.chat_template = DEFAULT_TEMPLATE.read_text(encoding="utf-8")
        print(f"[info] Using default non-thinking chat template: {DEFAULT_TEMPLATE}")
        return
    if override == _QWEN_DEFAULT_SENTINEL:
        _patch_system_prompt(tokenizer, SYSTEM_PROMPT)
        return
    tpl_path = Path(override)
    if not tpl_path.exists():
        raise FileNotFoundError(f"Chat template not found: {tpl_path}")
    tokenizer.chat_template = tpl_path.read_text(encoding="utf-8")
    print(f"[info] Using custom chat template from: {tpl_path}")


def _build_generation_config(args: argparse.Namespace) -> dict:
    gen_config = THINKING_GENERATION_CONFIG.copy() if args.thinking else DEFAULT_GENERATION_CONFIG.copy()
    if args.do_sample:
        gen_config["do_sample"] = True
        gen_config["temperature"] = args.temperature
        gen_config["top_p"] = args.top_p
        gen_config["top_k"] = args.top_k
    return gen_config


def _verify_export(output_dir: Path, hf_token: str, skip_model_load: bool) -> None:
    print("\n[info] Verifying export...")
    if not (output_dir / "config.json").exists():
        raise FileNotFoundError(f"Missing config.json in {output_dir}")
    print("  config.json: OK")

    weight_files = list(output_dir.glob("*.safetensors")) + list(output_dir.glob("*.bin"))
    index_files = list(output_dir.glob("*.safetensors.index.json")) + list(output_dir.glob("*.bin.index.json"))
    if not weight_files and not index_files:
        raise FileNotFoundError(
            f"No model weight files found in {output_dir}. "
            f"Expected *.safetensors, *.bin, or an index JSON."
        )
    print("  model weights: OK")

    generation_config_path = output_dir / "generation_config.json"
    if generation_config_path.exists():
        print("  generation_config.json: OK")
    else:
        print("  generation_config.json: absent")

    tok = AutoTokenizer.from_pretrained(str(output_dir), token=hf_token, trust_remote_code=True)
    print("  AutoTokenizer.from_pretrained: OK")
    if tok.chat_template is None:
        raise ValueError("No chat template in saved tokenizer.")
    print("  chat_template: OK")

    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if "<|im_start|>assistant" not in rendered:
        raise ValueError("Assistant turn missing from rendered chat template.")
    print("  apply_chat_template: OK")
    print(f"  Sample rendered suffix: ...{rendered[-200:]}")

    if not skip_model_load:
        AutoModelForCausalLM.from_pretrained(
            str(output_dir),
            token=hf_token,
            device_map="cpu",
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print("  AutoModelForCausalLM.from_pretrained: OK")
    print("[info] Verification passed.")


def _resolve_adapter_path(adapter_path: Path) -> Path:
    if (adapter_path / "adapter_config.json").exists():
        return adapter_path
    candidate = adapter_path / "best_adapter"
    if (candidate / "adapter_config.json").exists():
        print(f"[info] Using best_adapter/: {candidate}")
        return candidate
    raise FileNotFoundError(
        f"adapter_config.json not found at {adapter_path} or {candidate}"
    )


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"output_dir already exists: {output_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _write_manifest(
    output_dir: Path,
    *,
    run_name: str,
    base_model: str,
    adapter_path: Path,
    best_eval_loss: float | None,
    thinking: bool,
    chat_template_override: str | None,
    generation_config_exported: bool,
) -> None:
    manifest = {
        "run_name": run_name,
        "base_model": base_model,
        "adapter_path": str(adapter_path),
        "best_eval_loss": best_eval_loss,
        "format": "full_hf_model",
        "merged_lora": True,
        "thinking_mode": thinking,
        "default_chat_template": str(DEFAULT_TEMPLATE),
        "chat_template_override": chat_template_override,
        "generation_config_exported": generation_config_exported,
    }
    _write_json(output_dir / "checkpoint_manifest.json", manifest)
    print("Saved checkpoint_manifest.json")


def _push_to_hf(output_dir: Path, hf_repo: str, hf_token: str) -> None:
    print(f"\n[info] Pushing to HF Hub: {hf_repo}")
    api = HfApi()
    api.create_repo(
        repo_id=hf_repo,
        repo_type="model",
        token=hf_token,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=hf_repo,
        repo_type="model",
        token=hf_token,
    )
    print(f"[info] Pushed: https://huggingface.co/{hf_repo}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter into the base model and export a full Hugging Face checkpoint."
    )
    parser.add_argument("--adapter_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
    parser.add_argument(
        "--override_chat_template",
        nargs="?",
        const=_QWEN_DEFAULT_SENTINEL,
        default=None,
        type=str,
        metavar="PATH",
        help=(
            "Chat template override.\n"
            "  no flag       -> use code/multilinguality_model/assets/chat_template_multilingual-nonthinking.jinja\n"
            "  no argument   -> use Qwen3 default template + system prompt patch\n"
            "  PATH          -> use the .jinja file at PATH"
        ),
    )
    parser.add_argument(
        "--export_generation_config",
        action="store_true",
        help="Write generation_config.json. By default, no generation config is exported.",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="If exporting generation config, use the thinking-mode defaults.",
    )
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
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
    output_dir = args.output_dir.resolve()
    adapter_path = _resolve_adapter_path(args.adapter_path.resolve())
    run_key = args.run_name or adapter_path.parent.name if adapter_path.name == "best_adapter" else adapter_path.name

    _prepare_output_dir(output_dir, overwrite=args.overwrite)

    print("=" * 80)
    print("LoRA EXPORT")
    print(f"Base model:          {args.base_model}")
    print(f"Adapter:             {adapter_path}")
    print(f"Output dir:          {output_dir}")
    print(f"Run name:            {run_key}")
    print(f"Default template:    {DEFAULT_TEMPLATE}")
    print(f"Template override:   {args.override_chat_template}")
    print(f"Export gen config:   {args.export_generation_config}")
    print(f"Push:                {args.push}")
    print("=" * 80)

    print("\nCopying checkpoint files...")
    _copy_checkpoint_files(adapter_path, output_dir)

    print("\nLoading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )

    _disable_peft_bnb_dispatch()

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, str(adapter_path))

    print("Merging...")
    merged = model.merge_and_unload()
    merged.eval()

    print("Saving merged model...")
    merged.save_pretrained(output_dir, safe_serialization=True)

    print("\nSaving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token, trust_remote_code=True)
    _apply_chat_template(tokenizer, args.override_chat_template)
    tokenizer.save_pretrained(output_dir)

    if args.export_generation_config:
        gen_config = _build_generation_config(args)
        _write_json(output_dir / "generation_config.json", gen_config)
        print(f"Saved generation_config.json ({'thinking' if args.thinking else 'non-thinking'})")
    else:
        print("No generation_config.json exported. Pass --export_generation_config to write one.")

    current_loss = _get_best_eval_loss(adapter_path)
    _write_manifest(
        output_dir,
        run_name=run_key,
        base_model=args.base_model,
        adapter_path=adapter_path,
        best_eval_loss=current_loss,
        thinking=args.thinking,
        chat_template_override=args.override_chat_template,
        generation_config_exported=args.export_generation_config,
    )

    if not args.skip_verify:
        _verify_export(output_dir, hf_token, skip_model_load=args.skip_model_load)

    if args.push:
        if not args.hf_repo:
            raise ValueError("--hf_repo is required when --push is set.")
        _push_to_hf(output_dir, args.hf_repo, hf_token)
    else:
        print("\n[info] Local export complete. No HF push requested.")


if __name__ == "__main__":
    main()