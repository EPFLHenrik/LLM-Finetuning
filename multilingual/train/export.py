"""multilingual/train/export.py
Merges a LoRA adapter into Qwen/Qwen3-1.7B, saves a vLLM-loadable
checkpoint, and optionally pushes to HF Hub.

Chat template behaviour:
  (no flag)                      → use assets/chat_template.jinja  [DEFAULT]
  --override_chat_template       → use Qwen3 default + system prompt patch
  --override_chat_template PATH  → use the .jinja file at PATH

Push behaviour:
  --push                push only if eval loss beats all previously logged runs
  --push --force_push   push unconditionally
  (no --push)           export locally only

Usage:
    python -m multilingual.train.export \
        --adapter_path multilingual/results/checkpoints/sft_v7 \
        --output_dir   multilingual/results/exports/sft_v7 \
        --run_name     sft_v7 \
        --hf_repo      cs-552-2026-mystery-machine/multilingual_model \
        --push --force_push
"""

from __future__ import annotations

import argparse
import json
import os
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

# Sentinel value used when --override_chat_template is passed with no argument
# (meaning: fall back to Qwen3 default + system prompt patch)
_QWEN_DEFAULT_SENTINEL = "__qwen_default__"
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
        print("[info] System prompt patched into Qwen3 default template.")
    else:
        print("[warn] Could not find system block — saving template without patch.")


def _disable_peft_bnb_dispatch() -> None:
    try:
        import peft.tuners.lora.model as lora_model
        lora_model.is_bnb_available      = lambda: False  # type: ignore[attr-defined]
        lora_model.is_bnb_4bit_available = lambda: False  # type: ignore[attr-defined]
        print("[info] Disabled PEFT bitsandbytes dispatch.")
    except Exception as exc:
        print(f"[warn] Failed to disable PEFT bitsandbytes dispatch: {exc}")


def _copy_checkpoint_files(adapter_path: Path, output_dir: Path) -> None:
    """Copy all non-weight files from checkpoint to output dir."""
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
        shutil.copy2(src, output_dir / src.name)
        print(f"  Copied {src.name}")


def _apply_chat_template(tokenizer, override: str | None) -> None:
    """
    Apply the correct chat template to the tokenizer according to:

      override is None (no flag)           → assets/chat_template.jinja  [DEFAULT]
      override == _QWEN_DEFAULT_SENTINEL   → Qwen3 default + system prompt patch
      override is a file path string       → load that .jinja file
    """
    if override is None:
        # Default: use our custom assets/chat_template.jinja
        if not DEFAULT_TEMPLATE.exists():
            raise FileNotFoundError(
                f"Default chat template not found: {DEFAULT_TEMPLATE}\n"
                f"Either create it or pass --override_chat_template to use the Qwen3 default."
            )
        tokenizer.chat_template = DEFAULT_TEMPLATE.read_text(encoding="utf-8")
        print(f"[info] Using custom chat template: {DEFAULT_TEMPLATE}")

    elif override == _QWEN_DEFAULT_SENTINEL:
        # --override_chat_template with no argument → Qwen3 default + patch
        _patch_system_prompt(tokenizer, SYSTEM_PROMPT)

    else:
        # --override_chat_template /some/path.jinja
        tpl_path = Path(override)
        if not tpl_path.exists():
            raise FileNotFoundError(f"Chat template not found: {tpl_path}")
        tokenizer.chat_template = tpl_path.read_text(encoding="utf-8")
        print(f"[info] Using custom chat template from: {tpl_path}")


def _verify_export(output_dir: Path, hf_token: str) -> None:
    print("\nVerifying export...")
    assert (output_dir / "generation_config.json").exists(), \
        "generation_config.json missing!"
    print("  generation_config.json: OK")

    tok = AutoTokenizer.from_pretrained(str(output_dir), token=hf_token)
    print("  AutoTokenizer.from_pretrained: OK")
    assert tok.chat_template is not None, "No chat template in saved tokenizer!"
    print("  chat_template: OK")

    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        tokenize=False, add_generation_prompt=True,
    )
    assert "<|im_start|>system" in rendered, "System block missing — default system prompt not injected!"
    assert "<|im_start|>assistant" in rendered, "Assistant turn missing!"
    print("  apply_chat_template (no system msg): OK — default system prompt injected")
    print(f"  Sample (last 200 chars): ...{rendered[-200:]}")

    import torch.nn as nn
    m = AutoModelForCausalLM.from_pretrained(
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
    parser.add_argument("--adapter_path", type=Path, required=True)
    parser.add_argument("--output_dir",   type=Path, required=True)
    parser.add_argument("--run_name",     type=str,  default=None)
    parser.add_argument("--base_model",   type=str,  default=BASE_MODEL)
    parser.add_argument(
        "--override_chat_template",
        nargs="?",
        const=_QWEN_DEFAULT_SENTINEL,   # --override_chat_template (no arg) → Qwen3 default
        default=None,                   # flag absent → use assets/chat_template.jinja
        type=str,
        metavar="PATH",
        help=(
            "Chat template override.\n"
            "  (no flag)      → use multilingual/assets/chat_template.jinja  [DEFAULT]\n"
            "  (no argument)  → use Qwen3 default template + system prompt patch\n"
            "  PATH           → use the .jinja file at PATH"
        ),
    )
    parser.add_argument("--do_sample",   action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p",       type=float, default=0.95)
    parser.add_argument("--top_k",       type=int,   default=20)
    parser.add_argument("--hf_repo",     type=str,   default=None)
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

    print(f"Base model:  {args.base_model}")
    print(f"Adapter:     {adapter_path}")
    print(f"Output:      {args.output_dir}")
    print(f"Run name:    {run_key}")

    # Copy all checkpoint config files (tokenizer, generation_config, etc.)
    print("\nCopying checkpoint files...")
    _copy_checkpoint_files(adapter_path, args.output_dir)

    # Merge adapter into base model
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
    merged: nn.Module = model.merge_and_unload()  # type: ignore[assignment]
    merged.eval()
    print("Saving merged model...")
    merged.save_pretrained(args.output_dir, safe_serialization=True)  # type: ignore[operator]

    # Load fresh inference tokenizer from base model and apply chat template
    print("\nSaving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
    _apply_chat_template(tokenizer, args.override_chat_template)
    tokenizer.save_pretrained(args.output_dir)

    # Generation config
    if args.do_sample:
        gen_config = DEFAULT_GENERATION_CONFIG.copy()
        gen_config.update({
            "do_sample":   True,
            "temperature": args.temperature,
            "top_p":       args.top_p,
            "top_k":       args.top_k,
        })
        with open(args.output_dir / "generation_config.json", "w") as f:
            json.dump(gen_config, f, indent=2)
        print("Saved updated generation_config.json")

    # Manifest
    current_loss = _get_best_eval_loss(adapter_path)
    manifest = {
        "run_name":       run_key,
        "base_model":     args.base_model,
        "adapter_path":   str(adapter_path),
        "best_eval_loss": current_loss,
        "format":         "safetensors",
        "merged_lora":    True,
        "thinking_mode":  True,
    }
    with open(args.output_dir / "checkpoint_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Saved checkpoint_manifest.json")

    if not args.skip_verify:
        _verify_export(args.output_dir, hf_token)

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
            api.create_repo(repo_id=args.hf_repo, repo_type="model",
                            token=hf_token, exist_ok=True)
            api.upload_folder(folder_path=str(args.output_dir),
                              repo_id=args.hf_repo, repo_type="model",
                              token=hf_token)
            print(f"Pushed: https://huggingface.co/{args.hf_repo}")
    else:
        print(f"\nTo push manually: python -m multilingual.train.export "
              f"--adapter_path {args.adapter_path} "
              f"--output_dir {args.output_dir} "
              f"--run_name {run_key} "
              f"--hf_repo <repo> --push")


if __name__ == "__main__":
    main()