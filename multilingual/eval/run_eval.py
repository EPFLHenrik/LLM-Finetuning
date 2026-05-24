"""multilingual/eval/run_eval.py

Evaluates a model on the SFT val set and/or the 10 provided validation samples.

Supports:
  - Merged HF models (full model on HF or local)
  - Local LoRA adapters (best_adapter folder from training)

The custom chat template from multilingual/assets/chat_template.jinja
is applied to the tokenizer when --use_custom_template is set (default True).

Usage:
    # Eval merged model from HF
    python -m multilingual.eval.run_eval \
        --model_name cs-552-2026-mystery-machine/multilingual_model \
        --val_file multilingual/data/sft/v9/mixed/val_mixed.jsonl \
        --output_dir multilingual/results/eval/v9

    # Eval local LoRA adapter (best_adapter folder)
    python -m multilingual.eval.run_eval \
        --model_name multilingual/results/checkpoints/sft_multilingual_v9_distilled/best_adapter \
        --val_file multilingual/data/sft/v9/mixed/val_mixed.jsonl \
        --output_dir multilingual/results/eval/v9_adapter

    # INCLUDE val only
    python -m multilingual.eval.run_eval \
        --model_name multilingual/results/checkpoints/sft_multilingual_v9_distilled/best_adapter \
        --val_file multilingual/data/sft/v9/mixed/include_val.jsonl \
        --output_dir multilingual/results/eval/v9_include

    # 10 provided examples only
    python -m multilingual.eval.run_eval \
        --model_name multilingual/results/checkpoints/sft_multilingual_v9_distilled/best_adapter \
        --provided_only \
        --output_dir multilingual/results/eval/v9_provided
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from multilingual.config import (
    DEFAULT_MAX_NEW_TOKENS_BY_TASK,
    DEFAULT_MODEL_NAME,
    REPO,
    SYSTEM_PROMPT,
)
from multilingual.eval.evaluation import evaluate_examples, summarize_results
from multilingual.generation import generate_answer
from multilingual.utils import save_jsonl

BASE_MODEL       = "Qwen/Qwen3-1.7B"
CHAT_TEMPLATE    = REPO / "multilingual" / "assets" / "chat_template.jinja"

# ---------------------------------------------------------------------------
# Model loading — supports merged models and LoRA adapters
# ---------------------------------------------------------------------------

def _is_lora_adapter(model_path: Path) -> bool:
    """Check if the path is a LoRA adapter (has adapter_config.json)."""
    return (model_path / "adapter_config.json").exists()


def load_model_and_tokenizer_for_eval(
    model_name: str,
    hf_token: str,
    use_custom_template: bool = True,
):
    """
    Load model + tokenizer for evaluation.

    If model_name is a LoRA adapter path, loads Qwen3-1.7B base and applies
    the adapter via PEFT. Otherwise loads as a full model (local or HF).
    """
    from peft import PeftModel

    model_path = Path(model_name).resolve()
    is_local   = model_path.exists()
    is_adapter = is_local and _is_lora_adapter(model_path)

    if is_adapter:
        print(f"[eval] Loading LoRA adapter from {model_path}")
        print(f"[eval] Base model: {BASE_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, token=hf_token, trust_remote_code=True,
        )
        # Disable bitsandbytes dispatch to avoid CUDA setup crash
        try:
            import peft.tuners.lora.model as lora_model
            lora_model.is_bnb_available = lambda: False
            lora_model.is_bnb_4bit_available = lambda: False
        except Exception:
            pass
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            token=hf_token,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, str(model_path))
        model = model.merge_and_unload()
    else:
        # Local full model or HF repo name
        load_name = str(model_path) if is_local else model_name
        print(f"[eval] Loading full model from {load_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            load_name, token=hf_token, trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            load_name,
            token=hf_token,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    if use_custom_template and CHAT_TEMPLATE.exists():
        print(f"[eval] Applying custom chat template: {CHAT_TEMPLATE}")
        tokenizer.chat_template = CHAT_TEMPLATE.read_text(encoding="utf-8")
    else:
        print("[eval] Using tokenizer's default chat template")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model.eval()
    return model, tokenizer

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _sft_to_eval_examples(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    examples = []
    for r in records:
        prompt = r.get("prompt") or (
            r["messages"][1]["content"] if r.get("messages") else ""
        )
        examples.append({
            "id":        r.get("id", ""),
            "dataset":   r.get("dataset", "unknown"),
            "language":  r.get("language", "unknown"),
            "task_type": r.get("task_type", "mcq"),
            "prompt":    prompt,
            "answer":    r.get("answer", ""),
            "metadata":  r.get("metadata", {}),
        })
    return examples


def _provided_to_eval_examples(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "id":        f"provided_{i}",
            "dataset":   "provided",
            "language":  "mixed",
            "task_type": "mcq",
            "prompt":    r["prompt"],
            "answer":    r["answer"],
            "metadata":  {},
        }
        for i, r in enumerate(records)
    ]

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME,
                        help="HF model name, local merged model path, or best_adapter path.")
    parser.add_argument("--val_file",   type=Path, default=None,
                        help="SFT-format JSONL val file (val_mixed.jsonl or include_val.jsonl).")
    parser.add_argument("--output_dir", type=Path,
                        default=REPO / "multilingual" / "results" / "eval")
    parser.add_argument("--n",          type=int, default=None,
                        help="Cap total examples from val_file (None = all).")
    parser.add_argument("--eval_provided",  action="store_true",
                        help="Also eval on the 10 provided validation samples.")
    parser.add_argument("--provided_only",  action="store_true",
                        help="Only eval on the 10 provided samples.")
    parser.add_argument("--provided_file",  type=Path,
                        default=REPO / "validation_samples" / "multilingual.jsonl")
    parser.add_argument("--no_custom_template", action="store_true",
                        help="Skip applying assets/chat_template.jinja.")
    parser.add_argument("--mcq_max_new_tokens",  type=int,
                        default=DEFAULT_MAX_NEW_TOKENS_BY_TASK["mcq"])
    parser.add_argument("--math_max_new_tokens", type=int,
                        default=DEFAULT_MAX_NEW_TOKENS_BY_TASK["math"])
    parser.add_argument("--do_sample",   action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p",       type=float, default=0.95)
    parser.add_argument("--top_k",       type=int,   default=20)
    parser.add_argument("--overwrite",   action="store_true")
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    hf_token = os.environ.get("HF_TOKEN", "")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_new_tokens = {
        "mcq":  args.mcq_max_new_tokens,
        "math": args.math_max_new_tokens,
    }

    print("=" * 60)
    print(f"Model:      {args.model_name}")
    print(f"Val file:   {args.val_file}")
    print(f"Output:     {args.output_dir}")
    print(f"Sampling:   do_sample={args.do_sample}")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer_for_eval(
        args.model_name,
        hf_token,
        use_custom_template=not args.no_custom_template,
    )

    generate_kwargs = dict(
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    def _generate(model, tokenizer, prompt, task_type, max_new_tokens):
        return generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            task_type=task_type,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

    all_dfs = []

    # ── Val file ──────────────────────────────────────────────────────────────
    if not args.provided_only and args.val_file is not None:
        out_path = args.output_dir / "val_results.jsonl"
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path} — use --overwrite to re-run")
            all_dfs.append(pd.read_json(out_path, lines=True))
        else:
            records  = _load_jsonl(args.val_file)
            if args.n:
                records = records[:args.n]
            examples = _sft_to_eval_examples(records)
            print(f"Loaded {len(examples)} examples from {args.val_file.name}")

            # Group by dataset + language for per-dataset reporting
            by_group: Dict = {}
            for ex in examples:
                key = (ex["dataset"], ex["language"])
                by_group.setdefault(key, []).append(ex)

            group_dfs = []
            for (dataset, language), group_examples in sorted(by_group.items()):
                df = evaluate_examples(
                    examples=group_examples,
                    dataset_name=dataset,
                    language=language,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens_by_task=max_new_tokens,
                    generate_fn=_generate,
                )
                group_dfs.append(df)

            if group_dfs:
                df_val = pd.concat(group_dfs, ignore_index=True)
                save_jsonl(df_val, out_path)
                all_dfs.append(df_val)

    # ── Provided examples ─────────────────────────────────────────────────────
    if args.eval_provided or args.provided_only:
        out_path = args.output_dir / "provided_results.jsonl"
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path} — use --overwrite to re-run")
            all_dfs.append(pd.read_json(out_path, lines=True))
        elif not args.provided_file.exists():
            print(f"[WARN] Provided file not found: {args.provided_file}")
        else:
            records  = _load_jsonl(args.provided_file)
            examples = _provided_to_eval_examples(records)
            print(f"\nEvaluating {len(examples)} provided examples...")
            df_prov = evaluate_examples(
                examples=examples,
                dataset_name="provided",
                language="mixed",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens_by_task=max_new_tokens,
                generate_fn=_generate,
            )
            save_jsonl(df_prov, out_path)
            all_dfs.append(df_prov)

    # ── Summary ───────────────────────────────────────────────────────────────
    if not all_dfs:
        print("No results produced.")
        return

    df_all  = pd.concat(all_dfs, ignore_index=True)
    summary = summarize_results(df_all)

    save_jsonl(df_all, args.output_dir / "all_results.jsonl")
    summary_path = args.output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("SUMMARY (per dataset/language)")
    print("=" * 60)
    print(summary.to_string(index=False))

    overall = df_all["correct"].mean()
    n_correct = int(df_all["correct"].sum())
    print(f"\nOverall strict accuracy: {overall:.3f} ({n_correct}/{len(df_all)})")

    # Per-language summary
    print("\nPer-language accuracy:")
    for lang, grp in df_all.groupby("language"):
        acc = grp["correct"].mean()
        print(f"  {lang:10s}: {acc:.3f} ({int(grp['correct'].sum())}/{len(grp)})")

    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()