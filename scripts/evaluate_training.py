#!/usr/bin/env python3
"""Evaluate every merged training checkpoint in a folder.

This is the checkpoint-oriented sibling of `eval/validate_general_knowledge_vllm.py`.
It keeps the same vLLM-first inference flow, but loops over every merged checkpoint
folder and writes per-checkpoint detail + summary JSON reports.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import wandb
from evaluate.extract_answer import extract_boxed_answer_with_fallbacks
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM, SamplingParams

    _HAS_VLLM = True
except Exception:
    LLM = None  # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment]
    _HAS_VLLM = False

PROMPT_KEYS = ("prompt", "question", "input", "text")
ANSWER_KEYS = ("answer", "reference", "label", "target", "answerKey")

BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise SystemExit(f"{path}:{lineno}: each row must be a JSON object")
            rows.append(item)
    if not rows:
        raise SystemExit(f"{path}: file is empty")
    return rows


def _get_first(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in row and row[key] is not None:
            value = str(row[key]).strip()
            if value:
                return value
    raise SystemExit(f"Missing one of {keys} in row: {row}")


def _format_prompt(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=None)
    return prompt


def _evaluation_penalty(strategy: str) -> float:
    penalties = {
        "boxed_exact_capital": 0.0,
        "boxed_contains_capital": 0.5,
        "last_letter_capital": 1.0,
        "tail10_capital": 1.5,
        "no_answer_found": 3.0,
    }
    return penalties.get(strategy, penalties["no_answer_found"])


def _score_prediction(prediction: str, reference: str) -> tuple[bool, float, str, str | None]:
    extracted, strategy = extract_boxed_answer_with_fallbacks(prediction)
    reference = reference.strip()

    if reference and len(reference) == 1 and reference.isalpha() and reference.isupper():
        correct = extracted is not None and extracted.upper() == reference.upper()
        return correct, 0.0 if correct else _evaluation_penalty(strategy), strategy, extracted

    boxed_match = BOX_RE.search(prediction)
    if boxed_match:
        candidate = boxed_match.group(1).strip()
    else:
        candidate = extracted

    correct = candidate is not None and candidate.strip().lower() == reference.lower()
    return correct, 0.0 if correct else _evaluation_penalty(strategy), strategy, extracted


def _print_eval_summary(eval_report: dict[str, Any]) -> None:
    n = eval_report.get("n_examples", 0)
    accuracy = eval_report.get("accuracy", 0.0)
    mean_penalty = eval_report.get("mean_penalty", 0.0)
    strategy_counts = eval_report.get("strategy_counts", {})

    boxed_count = 0
    extracted_count = 0
    for detail in eval_report.get("details", []):
        prediction = detail.get("prediction", "") or ""
        extracted, strategy = extract_boxed_answer_with_fallbacks(prediction)
        if strategy.startswith("boxed"):
            boxed_count += 1
        if extracted is not None:
            extracted_count += 1

    print("\n=== Evaluation Summary ===")
    print(f"n_examples: {n}")
    print(f"accuracy: {accuracy:.4f}")
    print(f"mean_penalty: {mean_penalty:.4f}")
    print("strategy_counts:")
    for key, value in strategy_counts.items():
        print(f"  {key}: {value}")
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
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=dtype, trust_remote_code=True)
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
    def __init__(self, model_path: Path | str, tokenizer):
        self.model_path = str(model_path)
        self.tokenizer = tokenizer
        self.llm = LLM(model=self.model_path, tokenizer=self.model_path, trust_remote_code=True)

    def generate_batch(self, prompts: list[str], max_new_tokens: int, temperature: float, top_p: float) -> list[str]:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.llm.generate(prompts, sampling_params=sampling_params)
        completions: list[str] = []
        for request_output in outputs:
            if request_output.outputs:
                completions.append(request_output.outputs[0].text)
            else:
                completions.append("")
        return completions


def evaluate_rows(
    evaluator,
    tokenizer,
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    correct_count = 0
    penalty_sum = 0.0
    strategy_counts: Counter[str] = Counter()
    details: list[dict[str, Any]] = []

    prompts = [_format_prompt(tokenizer, _get_first(row, PROMPT_KEYS)) for row in rows]
    if hasattr(evaluator, "generate_batch"):
        predictions = evaluator.generate_batch(prompts, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    else:
        predictions = [
            evaluator.generate_once(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
            for prompt in prompts
        ]

    for row, prompt, prediction in zip(rows, prompts, predictions):
        reference = _get_first(row, ANSWER_KEYS)
        correct, penalty, strategy, extracted = _score_prediction(prediction, reference)
        correct_count += int(correct)
        penalty_sum += penalty
        strategy_counts[strategy] += 1
        details.append(
            {
                "prompt": _get_first(row, PROMPT_KEYS),
                "reference": reference,
                "prediction": prediction,
                "extracted": extracted,
                "strategy": strategy,
                "correct": correct,
                "penalty": penalty,
            }
        )

    n_examples = len(rows)
    return {
        "n_examples": n_examples,
        "accuracy": correct_count / n_examples if n_examples else 0.0,
        "mean_penalty": penalty_sum / n_examples if n_examples else 0.0,
        "strategy_counts": dict(strategy_counts),
        "details": details,
    }


def find_candidate_checkpoints(base_dir: Path) -> list[Path]:
    def looks_like_model(d: Path) -> bool:
        return any((d / name).exists() for name in ("model.safetensors", "pytorch_model.bin", "config.json", "generation_config.json"))

    candidates: list[Path] = []
    if looks_like_model(base_dir):
        candidates.append(base_dir)
    candidates.extend(sorted(child for child in base_dir.iterdir() if child.is_dir() and looks_like_model(child)))
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all merged checkpoints in a folder.")
    parser.add_argument("--checkpoints-dir", required=True, type=Path)
    parser.add_argument("--validation-file", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/checkpoint_evals"))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--skip-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="cs552-mystery-machine")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-key", type=str, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    args = parser.parse_args()

    if not args.checkpoints_dir.exists():
        raise SystemExit("checkpoints-dir not found")
    if not args.validation_file.exists():
        raise SystemExit("validation-file not found")

    rows = _read_jsonl(args.validation_file)
    if args.max_eval_samples > 0:
        rows = rows[: args.max_eval_samples]

    candidates = find_candidate_checkpoints(args.checkpoints_dir)
    if not candidates:
        raise SystemExit(f"No merged checkpoints found under {args.checkpoints_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_wandb:
        try:
            wandb.login(key=args.wandb_key)
        except Exception as exc:
            print(f"[warn] wandb.login() failed: {exc}")
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=args.checkpoints_dir.name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: list[dict[str, Any]] = []

    for checkpoint_index, ckpt_dir in enumerate(candidates, start=1):
        print(f"[progress] starting checkpoint {checkpoint_index}/{len(candidates)}: {ckpt_dir}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=True, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            if args.use_vllm and _HAS_VLLM:
                evaluator = VLLMEvaluator(ckpt_dir, tokenizer)
            else:
                evaluator = TransformerEvaluator(ckpt_dir, tokenizer, device)

            eval_report = evaluate_rows(
                evaluator,
                tokenizer,
                rows,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except Exception as exc:
            print(f"[warn] Could not evaluate {ckpt_dir}: {exc}")
            traceback.print_exc()
            continue

        eval_report["checkpoint"] = str(ckpt_dir)
        label = ckpt_dir.name
        out_path = args.output_dir / f"evaluation_{label}.json"
        out_path.write_text(json.dumps(eval_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        hist_path = args.output_dir / "evaluation_history.jsonl"
        with hist_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(eval_report, ensure_ascii=False) + "\n")

        _print_eval_summary(eval_report)
        _log_eval_report_to_wandb(eval_report, prefix="evaluation/checkpoint")
        results.append(eval_report)

    summary_path = args.output_dir / "evaluation_summary.json"
    summary = {"n_checkpoints": len(results), "results": results}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not args.skip_wandb:
        wandb.finish()

    print(f"Wrote evaluations to {args.output_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
