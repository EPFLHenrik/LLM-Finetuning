#!/usr/bin/env python3
"""Evaluate multiple saved checkpoints (or adapters) from a folder.

Usage examples:
  python scripts/evaluate_training.py \
    --checkpoints-dir /scratch/workspace/henrik/standard-project-m2-mystery-machine/models/commonsense_qa \
    --validation-file data/preprocessing/commonsense_qa/validation.jsonl \
    --base-model /path/to/base/model \
    --max-new-tokens 512

The script will try to use vLLM if `--use-vllm` is passed and vllm is importable.
Otherwise it falls back to HuggingFace transformers + PEFT for adapter checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate.extract_answer import extract_boxed_answer_with_fallbacks

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

# Try importing vllm but allow fallback
try:
    import vllm  # type: ignore
    from vllm import LLM, SamplingParams  # type: ignore
    _HAS_VLLM = True
except Exception:
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore
    _HAS_VLLM = False

PROMPT_KEYS = ("prompt", "question", "input", "text")
ANSWER_KEYS = ("answer", "reference", "label", "target", "answerKey")

PENALTY_BY_STRATEGY = {
    "boxed_exact_capital": 0.0,
    "boxed_contains_capital": 0.5,
    "last_letter_capital": 1.0,
    "tail10_capital": 1.5,
    "no_answer_found": 3.0,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _get_first(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            value = str(row[key]).strip()
            if value:
                return value
    raise SystemExit(f"Missing one of {keys} in row: {row}")


def _evaluation_penalty(strategy: str) -> float:
    return PENALTY_BY_STRATEGY.get(strategy, PENALTY_BY_STRATEGY["no_answer_found"])


def _score_prediction(prediction: str, reference: str, extract_fn) -> tuple[bool, float, str, str | None]:
    extracted, strategy = extract_fn(prediction)
    reference = reference.strip()

    if reference and len(reference) == 1 and reference.isalpha() and reference.isupper():
        correct = extracted is not None and extracted.upper() == reference.upper()
        penalty = 0.0 if correct else _evaluation_penalty(strategy)
        return correct, penalty, strategy, extracted

    import re

    boxed_match = re.search(r"\\boxed\{(.*)\}", prediction)
    if boxed_match:
        candidate = boxed_match.group(1).strip()
        # best-effort equivalence
        correct = candidate.strip().lower() == reference.strip().lower()
    else:
        candidate = extracted
        correct = candidate is not None and candidate.strip().lower() == reference.strip().lower()
    penalty = 0.0 if correct else _evaluation_penalty(strategy)
    return correct, penalty, strategy, extracted


def _print_eval_summary(eval_report: dict[str, Any]) -> None:
    n = eval_report.get("n_examples", 0)
    accuracy = eval_report.get("accuracy", 0.0)
    mean_penalty = eval_report.get("mean_penalty", 0.0)
    strategy_counts = eval_report.get("strategy_counts", {})
    details = eval_report.get("details", [])

    boxed_count = 0
    extracted_count = 0
    for d in details:
        pred = d.get("prediction", "") or ""
        extracted, strategy = extract_boxed_answer_with_fallbacks(pred)
        if strategy.startswith("boxed"):
            boxed_count += 1
        if extracted is not None:
            extracted_count += 1

    print("\n=== Evaluation Summary ===")
    print(f"n_examples: {n}")
    print(f"accuracy: {accuracy:.4f}")
    print(f"mean_penalty: {mean_penalty:.4f}")
    print("strategy_counts:")
    for k, v in strategy_counts.items():
        print(f"  {k}: {v}")
    print(f"predictions_with_boxed_answer: {boxed_count}")
    print(f"predictions_with_extracted_answer: {extracted_count}")
    print("==========================\n")


def _log_eval_report_to_wandb(eval_report: dict[str, Any], *, prefix: str, step: int | None = None) -> None:
    if wandb.run is None:
        return
    metrics = {
        f"{prefix}/accuracy": eval_report.get("accuracy", 0.0),
        f"{prefix}/mean_penalty": eval_report.get("mean_penalty", 0.0),
        f"{prefix}/n_examples": eval_report.get("n_examples", 0),
    }
    wandb.log(metrics, step=step)
    wandb.run.summary.update(metrics)


class TransformerEvaluator:
    def __init__(self, model_path: Path | str, tokenizer, device: str):
        self.device = device
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16), trust_remote_code=True)
        self.model.to(device)
        self.model.eval()

    def generate_once(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        with torch.no_grad():
            generated = self.model.generate(**inputs, **gen_kwargs)
        prompt_length = inputs["input_ids"].shape[-1]
        completion = generated[0][prompt_length:]
        return self.tokenizer.decode(completion, skip_special_tokens=False)


class VLLMEvaluator:
    def __init__(self, model_path: Path | str, tokenizer, device: str, sampling_params: Any):
        # model_path should be a local model snapshot vLLM can read
        self.model_path = str(model_path)
        self.tokenizer = tokenizer
        self.device = device
        self.sampling_params = sampling_params
        self.llm = LLM(model=self.model_path)

    def generate_once(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
        params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
        # vLLM generate returns a generator of responses; grab first
        gen = self.llm.generate(prompt, sampling_params=params)
        # Iterate one result
        for batch in gen:
            # each batch may have multiple outputs; take first
            if batch.outputs:
                return batch.outputs[0].text
        return ""


def evaluate_rows(evaluator, tokenizer, rows, extract_fn, *, max_new_tokens, temperature, top_p):
    correct_count = 0
    penalty_sum = 0.0
    strategy_counts: Counter[str] = Counter()
    details: list[dict[str, Any]] = []

    for row in rows:
        prompt = _get_first(row, PROMPT_KEYS)
        reference = _get_first(row, ANSWER_KEYS)
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            messages = [{"role": "user", "content": prompt}]
        # Use tokenizer's chat template if available
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prediction = evaluator.generate_once(input_text, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        try:
            from evaluate.extract_answer import extract_boxed_answer_with_fallbacks
            extract_fn_local = extract_boxed_answer_with_fallbacks
        except Exception:
            def extract_fn_local(x):
                return (None, "no_answer_found")
        correct, penalty, strategy, extracted = _score_prediction(prediction, reference, extract_fn_local)
        correct_count += int(correct)
        penalty_sum += penalty
        strategy_counts[strategy] += 1
        details.append({
            "prompt": prompt,
            "reference": reference,
            "prediction": prediction,
            "extracted": extracted,
            "strategy": strategy,
            "correct": correct,
            "penalty": penalty,
        })

    n_examples = len(rows)
    return {
        "n_examples": n_examples,
        "accuracy": correct_count / n_examples if n_examples else 0.0,
        "mean_penalty": penalty_sum / n_examples if n_examples else 0.0,
        "strategy_counts": dict(strategy_counts),
        "details": details,
    }


def find_candidate_checkpoints(base_dir: Path) -> list[Path]:
    # include the directory itself if it looks like a model
    candidates: list[Path] = []
    def looks_like_model(d: Path) -> bool:
        for fname in ("pytorch_model.bin", "model.safetensors", "config.json", "checkpoint.pt", "adapter_config.json"):
            if (d / fname).exists():
                return True
        return False

    if looks_like_model(base_dir):
        candidates.append(base_dir)
    for child in sorted(base_dir.iterdir()):
        if child.is_dir() and looks_like_model(child):
            candidates.append(child)
    return candidates


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints-dir", required=True, type=Path)
    p.add_argument("--base-model", type=Path, default=None, help="Base model path used to load adapters when checkpoints are PEFT adapters")
    p.add_argument("--validation-file", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results/checkpoint_evals"))
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--use-vllm", action="store_true")
    p.add_argument("--skip-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="cs552-mystery-machine")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-key", type=str, default=None)
    p.add_argument("--max-eval-samples", type=int, default=0)
    args = p.parse_args()

    if not args.checkpoints_dir.exists():
        raise SystemExit("checkpoints-dir not found")
    if not args.validation_file.exists():
        raise SystemExit("validation-file not found")

    rows = _read_jsonl(args.validation_file)
    if args.max_eval_samples > 0:
        rows = rows[: args.max_eval_samples]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_wandb and args.wandb_key:
        try:
            wandb.login(key=args.wandb_key)
        except Exception as exc:
            print(f"[warn] wandb.login() failed: {exc}")
    if not args.skip_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=args.checkpoints_dir.name)

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model) if args.base_model is not None else str(args.checkpoints_dir), use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # attempt to set chat_template if available
    try:
        tokenizer.chat_template = tokenizer.chat_template or ""
    except Exception:
        pass

    candidates = find_candidate_checkpoints(args.checkpoints_dir)
    if not candidates:
        print(f"No model-like subfolders found under {args.checkpoints_dir}; treating subdirs as adapter dirs instead.")
        candidates = [p for p in sorted(args.checkpoints_dir.iterdir()) if p.is_dir()]

    results = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for ck in candidates:
        print(f"Evaluating checkpoint: {ck}")
        evaluator = None
        label = ck.name
        # Decide how to load the candidate: adapter (unmerged) or full model
        try:
            model_files = {p.name for p in ck.iterdir()} if ck.is_dir() else set()
            is_adapter = "adapter_config.json" in model_files or "adapter_model.safetensors" in model_files

            if is_adapter:
                # Load unmerged adapter via PEFT using the provided base model
                if args.base_model is None:
                    raise SystemExit(f"Found adapter checkpoint {ck} but --base-model is required to load adapters")
                if PeftModel is None:
                    raise SystemExit("PEFT is required to load adapters; install peft")
                base_model = AutoModelForCausalLM.from_pretrained(str(args.base_model), torch_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16), trust_remote_code=True)
                model_with_adapter = PeftModel.from_pretrained(base_model, str(ck))
                model_with_adapter.to(device)
                model_with_adapter.eval()

                class _AdapterEval:
                    def __init__(self, model, tokenizer, device):
                        self.model = model
                        self.tokenizer = tokenizer
                        self.device = device

                    def generate_once(self, prompt, max_new_tokens, temperature, top_p):
                        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                        gen_kwargs = {
                            "max_new_tokens": max_new_tokens,
                            "do_sample": temperature > 0,
                            "pad_token_id": self.tokenizer.eos_token_id,
                            "eos_token_id": self.tokenizer.eos_token_id,
                        }
                        if temperature > 0:
                            gen_kwargs["temperature"] = temperature
                            gen_kwargs["top_p"] = top_p
                        with torch.no_grad():
                            generated = self.model.generate(**inputs, **gen_kwargs)
                        prompt_length = inputs["input_ids"].shape[-1]
                        completion = generated[0][prompt_length:]
                        return self.tokenizer.decode(completion, skip_special_tokens=False)

                evaluator = _AdapterEval(model_with_adapter, tokenizer, device)
            else:
                # Full model: use vLLM if requested and available, otherwise transformers
                if args.use_vllm and _HAS_VLLM:
                    evaluator = VLLMEvaluator(ck, tokenizer, device, sampling_params=None)
                else:
                    evaluator = TransformerEvaluator(ck, tokenizer, device)

        except Exception as exc:
            print(f"[warn] Could not instantiate evaluator for {ck}: {exc}")
            traceback.print_exc()
            continue

        eval_report = evaluate_rows(evaluator, tokenizer, rows, None, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)
        eval_report["checkpoint"] = str(ck)
        out_path = args.output_dir / f"evaluation_{label}.json"
        out_path.write_text(json.dumps(eval_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        hist_path = args.output_dir / "evaluation_history.jsonl"
        with hist_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(eval_report, ensure_ascii=False) + "\n")
        _print_eval_summary(eval_report)
        _log_eval_report_to_wandb(eval_report, prefix="evaluation/checkpoint", step=None)
        results.append(eval_report)

    # write combined summary
    summary_path = args.output_dir / "evaluation_summary.json"
    summary = {"n_checkpoints": len(results), "results": results}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not args.skip_wandb:
        wandb.finish()

    print(f"Wrote evaluations to {args.output_dir}")


if __name__ == "__main__":
    main()
