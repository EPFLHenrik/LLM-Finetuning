"""multilingual/train/export.py

Merges a LoRA adapter into Qwen/Qwen3-1.7B and saves a vLLM-loadable
checkpoint, then optionally pushes to HF Hub only if this run's best
eval loss beats all previously logged runs.

Usage:
    python -m multilingual.train.export \
        --adapter_path multilingual/results/checkpoints/sft_v1/best_adapter \
        --output_dir multilingual/results/exports/sft_v1 \
        --run_name sft_v1 \
        --hf_repo cs-552-2026-mystery-machine/multilingual_model \
        --push
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from multilingual.config import SYSTEM_PROMPT

BASE_MODEL     = "Qwen/Qwen3-1.7B"
BEST_RUNS_LOG  = REPO / "multilingual" / "results" / "best_runs.json"

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
    """Read best_metric from trainer_state.json in the checkpoint directory."""
    for candidate in [
        checkpoint_dir / "trainer_state.json",
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
    """Return True if current_loss is better than all previously logged runs."""
    if current_loss is None:
        print("[warn] Could not determine eval loss — pushing anyway.")
        return True
    global_best = min(best_runs.values()) if best_runs else float("inf")
    if current_loss < global_best:
        print(f"[info] New best eval loss: {current_loss:.4f} "
              f"(previous best: {global_best:.4f} from "
              f"{min(best_runs, key=best_runs.get) if best_runs else 'none'})")
        return True
    print(f"[skip] Eval loss {current_loss:.4f} is not better than "
          f"global best {global_best:.4f} "
          f"({min(best_runs, key=best_runs.get) if best_runs else 'none'}). "
          f"Not pushing to HF.")
    return False


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _disable_peft_bnb_dispatch() -> None:
    try:
        import peft.tuners.lora.model as lora_model
        lora_model.is_bnb_available = lambda: False
        lora_model.is_bnb_4bit_available = lambda: False
        print("[info] Disabled PEFT bitsandbytes dispatch.")
    except Exception as exc:
        print(f"[warn] Failed to disable PEFT bitsandbytes dispatch: {exc}")


def _patch_chat_template(tokenizer, system_prompt: str) -> None:
    """
    Patch the Qwen3 chat template to inject a default system prompt when
    the CI passes no system message. Thinking mode is kept enabled (default).
    """
    template = tokenizer.chat_template
    if template is None:
        print("[warn] No chat template found; skipping patch.")
        return

    # The block to find in the no-tools branch of the Qwen3 template
    old_block = (
        "{%- if messages[0].role == 'system' %}\n"
        "        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n"
        "    {%- endif %}"
    )

    # Escape backslashes and single quotes for safe Jinja string embedding
    escaped = system_prompt.replace("\\", "\\\\").replace("'", "\\'")
    default_line = f"        {{{{- '<|im_start|>system\\n{escaped}<|im_end|>\\n' }}}}"

    new_block = (
        "{%- if messages[0].role == 'system' %}\n"
        "        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n"
        "    {%- else %}\n"
        f"{default_line}\n"
        "    {%- endif %}"
    )

    if old_block in template:
        tokenizer.chat_template = template.replace(old_block, new_block)
        print("[info] Chat template patched: default system prompt injected.")
        print(f"[info] System prompt: {system_prompt[:80]}...")
    else:
        print("[warn] Could not find expected system block in template.")
        print("[warn] Template may have changed — check manually.")


def _verify_export(output_dir: Path, hf_token: str) -> None:
    print("\nVerifying export...")
    cfg_path = output_dir / "generation_config.json"
    assert cfg_path.exists(), "generation_config.json missing!"
    print("  generation_config.json: OK")

    tok = AutoTokenizer.from_pretrained(str(output_dir), token=hf_token)
    print("  AutoTokenizer.from_pretrained: OK")

    # Verify system prompt is in template
    template = tok.chat_template or ""
    assert "multilingual reasoning assistant" in template, \
        "System prompt not found in chat template!"
    print("  System prompt in chat template: OK")

    # Test with no system message (simulates CI behavior)
    test_messages = [
        {"role": "user", "content": "What is 2+2?"},
    ]
    rendered = tok.apply_chat_template(
        test_messages, tokenize=False, add_generation_prompt=True
    )
    assert "<|im_start|>system" in rendered, \
        "System prompt not rendered when no system message passed!"
    assert "multilingual reasoning assistant" in rendered, \
        "System prompt content missing from rendered output!"
    assert "<|im_start|>assistant" in rendered, \
        "apply_chat_template output missing assistant turn!"
    print("  apply_chat_template (no system msg): OK — default system prompt injected")
    print(f"  Sample output (last 300 chars): ...{rendered[-300:]}")

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
    parser.add_argument("--adapter_path", type=Path, required=True)
    parser.add_argument("--output_dir",   type=Path, required=True)
    parser.add_argument("--run_name",     type=str,  default=None,
                        help="Run name used as key in best_runs.json.")
    parser.add_argument("--base_model",   type=str,  default=BASE_MODEL)
    parser.add_argument("--do_sample",    action="store_true")
    parser.add_argument("--temperature",  type=float, default=0.6)
    parser.add_argument("--top_p",        type=float, default=0.95)
    parser.add_argument("--top_k",        type=int,   default=20)
    parser.add_argument("--hf_repo",      type=str,   default=None)
    parser.add_argument("--push",         action="store_true")
    parser.add_argument("--force_push",   action="store_true",
                        help="Push regardless of eval loss comparison.")
    parser.add_argument("--skip_verify",  action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_key = args.run_name or str(args.adapter_path)

    print(f"Base model:  {args.base_model}")
    print(f"Adapter:     {args.adapter_path}")
    print(f"Output:      {args.output_dir}")
    print(f"Run name:    {run_key}")
    print(f"Sampling:    do_sample={args.do_sample}")

    # Load and merge
    print("\nLoading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )
    _disable_peft_bnb_dispatch()
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, str(args.adapter_path))
    print("Merging...")
    model = model.merge_and_unload()
    model.eval()

    # Save merged model
    print("Saving merged model...")
    model.save_pretrained(args.output_dir, safe_serialization=True)

    # Save tokenizer with system prompt patched into chat template
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
    _patch_chat_template(tokenizer, SYSTEM_PROMPT)
    tokenizer.save_pretrained(args.output_dir)

    # Save generation_config.json
    gen_config = dict(DEFAULT_GENERATION_CONFIG)
    if args.do_sample:
        gen_config["do_sample"]   = True
        gen_config["temperature"] = args.temperature
        gen_config["top_p"]       = args.top_p
        gen_config["top_k"]       = args.top_k

    with open(args.output_dir / "generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)
    print("Saved generation_config.json")

    # Save manifest
    current_loss = _get_best_eval_loss(args.adapter_path)
    manifest = {
        "run_name":          run_key,
        "base_model":        args.base_model,
        "adapter_path":      str(args.adapter_path),
        "best_eval_loss":    current_loss,
        "format":            "safetensors",
        "merged_lora":       True,
        "thinking_mode":     "enabled",
        "generation_config": "generation_config.json",
        "system_prompt":     SYSTEM_PROMPT,
    }
    with open(args.output_dir / "checkpoint_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("Saved checkpoint_manifest.json")

    # Verify
    if not args.skip_verify:
        _verify_export(args.output_dir, hf_token)

    # Log this run regardless of whether we push
    best_runs = _load_best_runs()
    if current_loss is not None:
        best_runs[run_key] = current_loss
        _save_best_runs(best_runs)
        print(f"\nAll logged runs:")
        for k, v in sorted(best_runs.items(), key=lambda x: x[1]):
            marker = " ← current" if k == run_key else ""
            print(f"  {v:.4f}  {k}{marker}")

    # Push decision
    if args.push:
        if not args.hf_repo:
            raise ValueError("--hf_repo is required when --push is set.")
        if args.force_push or _should_push(run_key, current_loss,
                                            {k: v for k, v in best_runs.items()
                                             if k != run_key}):
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
        print(f"\nTo push manually:")
        print(f"  python -m multilingual.train.export "
              f"--adapter_path {args.adapter_path} "
              f"--output_dir {args.output_dir} "
              f"--run_name {run_key} "
              f"--hf_repo <repo> --push")


if __name__ == "__main__":
    main()