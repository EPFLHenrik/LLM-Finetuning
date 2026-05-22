"""multilingual/train/export.py

Merges a LoRA adapter into Qwen/Qwen3-1.7B, saves a vLLM-loadable
checkpoint, and optionally pushes to HF Hub.

Chat template (default: the one saved in the checkpoint by the SFT script):
  --override_chat_template              use assets/chat_template.jinja
  --override_chat_template <path>       use a custom .jinja file

Push behaviour:
  --push                push only if eval loss beats all previously logged runs
  --push --force_push   push unconditionally
  (no --push)           export locally only; prints the manual push command

Usage:
    python -m multilingual.train.export \
        --adapter_path multilingual/results/checkpoints/sft_v7 \
        --output_dir   multilingual/results/exports/sft_v7 \
        --run_name     sft_v7 \
        --hf_repo      cs-552-2026-mystery-machine/multilingual_model \
        --push
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from multilingual.config import SYSTEM_PROMPT

BASE_MODEL       = "Qwen/Qwen3-1.7B"
BEST_RUNS_LOG    = REPO / "multilingual" / "results" / "best_runs.json"
DEFAULT_TEMPLATE = REPO / "multilingual" / "assets" / "chat_template.jinja"

# Filename used to save/load the chat template inside a checkpoint dir
CHECKPOINT_TEMPLATE_NAME = "chat_template.jinja"

DEFAULT_GENERATION_CONFIG = {
    "bos_token_id": 151643,
    "do_sample": False,
    "eos_token_id": [151645, 151643],
    "pad_token_id": 151643,
    "temperature": 1.0,
    "top_k": 1,
    "top_p": 1.0,
    "transformers_version": "4.51.0",
}

# ---------------------------------------------------------------------------
# Chat template helpers
# ---------------------------------------------------------------------------

def _load_template(path: Path, system_prompt: str | None = None) -> str:
    """Load a .jinja template, optionally replacing its system_prompt block."""
    template = path.read_text(encoding="utf-8")
    if system_prompt is not None:
        pattern = r"(\{%-?\s*set system_prompt\s*%\}).*?(\{%-?\s*endset\s*%\})"
        replacement = rf"\g<1>\n{system_prompt}\n\g<2>"
        template, n = re.subn(pattern, replacement, template, flags=re.DOTALL)
        if n == 0:
            print("[warn] system_prompt block not found in template — ignored.")
        else:
            print("[info] System prompt injected into template.")
    return template


def _resolve_export_template(
    adapter_path: Path,
    override: Path | None,
) -> tuple[Path, str]:
    """
    Resolve which chat template to use for export.

    Priority:
      1. --override_chat_template <path>  explicit override
      2. --override_chat_template (flag)  assets/chat_template.jinja
      3. <adapter_path>/chat_template.jinja  saved by SFT script (default)
      4. assets/chat_template.jinja  fallback if nothing else found

    Returns (source_path, template_text).
    """
    # override supplied (either flag-only → DEFAULT_TEMPLATE, or explicit path)
    if override is not None:
        src = override
        print(f"[info] Using override chat template: {src}")
    else:
        # look for the template saved during training
        checkpoint_template = adapter_path / CHECKPOINT_TEMPLATE_NAME
        if checkpoint_template.exists():
            src = checkpoint_template
            print(f"[info] Using training chat template: {src}")
        else:
            src = DEFAULT_TEMPLATE
            print(f"[warn] No training chat template found; "
                  f"falling back to default: {src}")

    if not src.exists():
        raise FileNotFoundError(f"Chat template not found: {src}")
    return src, _load_template(src)


# ---------------------------------------------------------------------------
# Best-runs tracking
# ---------------------------------------------------------------------------

def _load_best_runs() -> dict:
    if BEST_RUNS_LOG.exists():
        with open(BEST_RUNS_LOG) as f:
            return json.load(f)
    return {}


def _save_best_runs(runs: dict) -> None:
    BEST_RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_RUNS_LOG, "w") as f:
        json.dump(runs, f, indent=2)
    print(f"Updated best_runs.json: {BEST_RUNS_LOG}")


def _get_best_eval_loss(checkpoint_dir: Path) -> float | None:
    for candidate in [
        checkpoint_dir / "trainer_state.json",
        checkpoint_dir / "best_adapter" / "trainer_state.json",
        checkpoint_dir.parent / "trainer_state.json",
    ]:
        if candidate.exists():
            with open(candidate) as f:
                state = json.load(f)
            val = state.get("best_metric")
            if val is not None:
                return float(val)
    print(f"[warn] trainer_state.json not found near {checkpoint_dir}")
    return None


def _should_push(run_name: str, current_loss: float | None, best_runs: dict) -> bool:
    if current_loss is None:
        print("[warn] Could not determine eval loss — pushing anyway.")
        return True
    global_best = min(best_runs.values()) if best_runs else float("inf")
    best_key    = min(best_runs, key=lambda k: best_runs[k]) if best_runs else "none"
    if current_loss < global_best:
        print(f"[info] New best eval loss: {current_loss:.4f} "
              f"(previous best: {global_best:.4f} from {best_key})")
        return True
    print(f"[skip] Eval loss {current_loss:.4f} not better than "
          f"global best {global_best:.4f} ({best_key}). Not pushing.")
    return False


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _disable_peft_bnb_dispatch() -> None:
    try:
        import peft.tuners.lora.model as lora_model
        lora_model.is_bnb_available      = lambda: False  # type: ignore[attr-defined]
        lora_model.is_bnb_4bit_available = lambda: False  # type: ignore[attr-defined]
        print("[info] Disabled PEFT bitsandbytes dispatch.")
    except Exception as exc:
        print(f"[warn] Failed to disable PEFT bitsandbytes dispatch: {exc}")


def _copy_checkpoint_files(adapter_path: Path, output_dir: Path) -> None:
    """
    Copy all non-model files from the adapter checkpoint to the output dir.
    This includes tokenizer.json, tokenizer_config.json, generation_config.json,
    special_tokens_map.json, chat_template.jinja, and any other config files.
    Model weights (*.safetensors, *.bin, adapter_*) are excluded since those
    get written by save_pretrained.
    """
    exclude_suffixes = {".safetensors", ".bin", ".pt"}
    exclude_names    = {"adapter_config.json", "adapter_model.safetensors",
                        "adapter_model.bin"}
    for src in adapter_path.iterdir():
        if src.is_dir():
            continue
        if src.suffix in exclude_suffixes:
            continue
        if src.name in exclude_names:
            continue
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        print(f"  Copied {src.name}")


def _verify_export(output_dir: Path, hf_token: str, system_prompt: str) -> None:
    print("\nVerifying export...")
    assert (output_dir / "generation_config.json").exists(), \
        "generation_config.json missing!"
    print("  generation_config.json: OK")

    tok = AutoTokenizer.from_pretrained(str(output_dir), token=hf_token)
    print("  AutoTokenizer.from_pretrained: OK")

    probe = system_prompt.strip().splitlines()[0][:40]
    assert probe in (tok.chat_template or ""), \
        f"System prompt not found in saved chat template: {probe!r}"
    print("  System prompt in chat template: OK")

    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        tokenize=False, add_generation_prompt=True,
    )
    assert "<|im_start|>system" in rendered, "System block missing!"
    assert probe in rendered, "System prompt missing from rendered output!"
    assert "<|im_start|>assistant" in rendered, "Assistant turn missing!"
    print("  apply_chat_template (no system msg): OK")
    print(f"  Sample (last 300 chars): ...{rendered[-300:]}")

    AutoModelForCausalLM.from_pretrained(
        str(output_dir), token=hf_token,
        device_map="cpu", dtype=torch.bfloat16,
    )
    print("  AutoModelForCausalLM.from_pretrained: OK")
    print("Verification passed.")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=Path, required=True,
                        help="Path to checkpoint dir (best_adapter/ resolved automatically).")
    parser.add_argument("--output_dir",   type=Path, required=True)
    parser.add_argument("--run_name",     type=str,  default=None)
    parser.add_argument("--base_model",   type=str,  default=BASE_MODEL)

    # Chat template override — optional path argument
    parser.add_argument(
        "--override_chat_template",
        nargs="?",               # 0 or 1 args: flag-only → None, with path → Path
        const=DEFAULT_TEMPLATE,  # flag-only: use assets default
        default=None,            # not passed: use training checkpoint template
        type=Path,
        metavar="PATH",
        help="Override the training chat template. "
             "No value → use assets/chat_template.jinja. "
             "With path → use that .jinja file.",
    )

    # Generation config
    parser.add_argument("--do_sample",   action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p",       type=float, default=0.95)
    parser.add_argument("--top_k",       type=int,   default=20)

    # Push options
    parser.add_argument("--hf_repo",     type=str, default=None)
    parser.add_argument("--push",        action="store_true")
    parser.add_argument("--force_push",  action="store_true")
    parser.add_argument("--skip_verify", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args     = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_key  = args.run_name or str(args.adapter_path)

    # Resolve adapter path
    adapter_path = args.adapter_path
    if not (adapter_path / "adapter_config.json").exists():
        candidate = adapter_path / "best_adapter"
        if (candidate / "adapter_config.json").exists():
            print(f"[info] Using best_adapter/: {candidate}")
            adapter_path = candidate
        else:
            raise FileNotFoundError(
                f"adapter_config.json not found at {adapter_path} "
                f"or {adapter_path / 'best_adapter'}"
            )

    # Resolve chat template
    template_src, template = _resolve_export_template(
        adapter_path, args.override_chat_template
    )

    # Extract effective system prompt for verification
    match = re.search(
        r"\{%-?\s*set system_prompt\s*%\}(.*?)\{%-?\s*endset\s*%\}",
        template, flags=re.DOTALL,
    )
    effective_system_prompt = match.group(1).strip() if match else SYSTEM_PROMPT

    print(f"Base model:     {args.base_model}")
    print(f"Adapter:        {adapter_path}")
    print(f"Output:         {args.output_dir}")
    print(f"Run name:       {run_key}")
    print(f"Chat template:  {template_src}")

    # Copy all checkpoint config files first (tokenizer, generation_config, etc.)
    print("\nCopying checkpoint files...")
    _copy_checkpoint_files(adapter_path, args.output_dir)

    # Load base model + merge adapter
    print("\nLoading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16,
        device_map="auto", token=hf_token,
    )
    _disable_peft_bnb_dispatch()
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, str(adapter_path))
    print("Merging...")
    import torch.nn as nn
    model = model.merge_and_unload()
    cast_model: nn.Module = model  # type: ignore[assignment]
    cast_model.eval()

    print("Saving merged model...")
    cast_model.save_pretrained(args.output_dir, safe_serialization=True)

    # Apply chat template to tokenizer (overrides whatever was copied)
    print("Applying chat template to tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(args.output_dir), token=hf_token)
    tokenizer.chat_template = template
    tokenizer.save_pretrained(args.output_dir)

    # Generation config — override only if do_sample explicitly set
    if args.do_sample:
        gen_cfg_path = args.output_dir / "generation_config.json"
        gen_config = DEFAULT_GENERATION_CONFIG.copy()
        gen_config.update({
            "do_sample":   True,
            "temperature": args.temperature,
            "top_p":       args.top_p,
            "top_k":       args.top_k,
        })
        with open(gen_cfg_path, "w") as f:
            json.dump(gen_config, f, indent=2)
        print("Saved updated generation_config.json (do_sample=True)")

    # Manifest
    current_loss = _get_best_eval_loss(adapter_path)
    manifest = {
        "run_name":          run_key,
        "base_model":        args.base_model,
        "adapter_path":      str(adapter_path),
        "best_eval_loss":    current_loss,
        "format":            "safetensors",
        "merged_lora":       True,
        "thinking_mode":     "enabled",
        "chat_template":     str(template_src),
        "system_prompt":     effective_system_prompt,
    }
    with open(args.output_dir / "checkpoint_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Saved checkpoint_manifest.json")

    if not args.skip_verify:
        _verify_export(args.output_dir, hf_token, effective_system_prompt)

    # Log run
    best_runs = _load_best_runs()
    if current_loss is not None:
        best_runs[run_key] = current_loss
        _save_best_runs(best_runs)
        print("\nAll logged runs:")
        for k, v in sorted(best_runs.items(), key=lambda x: x[1]):
            marker = " ← current" if k == run_key else ""
            print(f"  {v:.4f}  {k}{marker}")

    # Push
    if args.push:
        if not args.hf_repo:
            raise ValueError("--hf_repo is required when --push is set.")
        other_runs = {k: v for k, v in best_runs.items() if k != run_key}
        if args.force_push or _should_push(run_key, current_loss, other_runs):
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
        print(f"\nTo push manually: python -m multilingual.train.export "
              f"--adapter_path {args.adapter_path} "
              f"--output_dir {args.output_dir} "
              f"--run_name {run_key} "
              f"--hf_repo <repo> --push")


if __name__ == "__main__":
    main()