#!/usr/bin/env python3
"""Emulate the nightly CI for the general knowledge snapshot only.

This script mirrors the local CI flow for `validation_samples/general_knowledge.jsonl`:
1. Load a local checkpoint from `models/qwen3_baseline` by default.
2. Run sampled inference on the general knowledge JSONL file.
3. Store all completions in a generations JSONL file.
4. Score the results with the repository's own extraction / pass@k logic.
5. Write a JSON report and print a concise summary.

The local validation snapshot is not the hidden eval set, but this is a useful
end-to-end dry run for prompt formatting, sampling, extraction, and scoring.
"""

from __future__ import annotations

import argparse
import json
import copy
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate.score import BENCHMARK_TO_METHOD, score_generations


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "qwen3_baseline"
DEFAULT_SAMPLES = Path(__file__).resolve().parents[1] / "validation_samples" / "general_knowledge.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "nightly_ci_general_knowledge"
DEFAULT_GENERATIONS = DEFAULT_OUTPUT_DIR / "generations.jsonl"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "report.json"


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def load_samples(samples_path: Path) -> list[dict]:
    rows: list[dict] = []
    for item in read_jsonl(samples_path):
        rows.append(
            {
                "benchmark": "general_knowledge",
                "source_file": samples_path.name,
                "prompt": item["prompt"],
                "answer": str(item["answer"]),
            }
        )
    return rows


def format_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def assert_qwen_family(model, tokenizer, model_dir: Path) -> None:
    model_type = str(getattr(model.config, "model_type", "")).lower()
    architectures = [str(item) for item in getattr(model.config, "architectures", [])]
    tokenizer_name = tokenizer.__class__.__name__.lower()

    print(f"Detected model_type={model_type!r}, architectures={architectures}, tokenizer={tokenizer_name!r}")

    looks_qwen = "qwen" in model_type or any("qwen" in item.lower() for item in architectures) or "qwen" in tokenizer_name
    if not looks_qwen:
        raise SystemExit(
            f"Checkpoint at {model_dir} does not look Qwen-based: "
            f"model_type={model_type!r}, architectures={architectures}, tokenizer={tokenizer_name!r}"
        )


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def generate_completion(
    model,
    tokenizer,
    formatted_prompt: str,
    generation_config: GenerationConfig,
    max_new_tokens: int,
    seed: int,
    device: torch.device,
) -> str:
    batch = tokenizer(formatted_prompt, return_tensors="pt")
    batch = move_batch_to_device(batch, device)

    local_config = copy.deepcopy(generation_config)
    local_config.do_sample = True
    local_config.max_new_tokens = max_new_tokens
    local_config.pad_token_id = tokenizer.eos_token_id
    local_config.eos_token_id = tokenizer.eos_token_id

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        output = model.generate(**batch, generation_config=local_config)

    generated_tokens = output[0, batch["input_ids"].shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Emulate the nightly CI on the validation snapshot.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Local checkpoint directory.")
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES), help="Path to the general knowledge JSONL file.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for generations and report artifacts.")
    parser.add_argument("--generations", default=None, help="Optional path for the combined generations JSONL file.")
    parser.add_argument("--report", default=None, help="Optional path for the combined JSON report.")
    parser.add_argument("--num-completions", type=int, default=2, help="Sampled completions per problem.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum new tokens per completion.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for sampling.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Execution device.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on examples for a faster dry run.")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    samples_path = Path(args.samples)
    output_dir = Path(args.output_dir)
    generations_path = Path(args.generations) if args.generations else DEFAULT_GENERATIONS
    report_path = Path(args.report) if args.report else DEFAULT_REPORT

    print(f"Loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Loading model from {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch_dtype)
    model.eval()
    assert_qwen_family(model, tokenizer, model_dir)

    if args.device == "cpu":
        device = torch.device("cpu")
        model = model.to(device)
    elif args.device == "cuda":
        device = torch.device("cuda")
        model = model.to(device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
    else:
        device = torch.device("cpu")

    generation_config = GenerationConfig.from_pretrained(model_dir)

    samples = load_samples(samples_path)
    if not samples:
        raise SystemExit(f"No validation samples found in {samples_path}.")

    if args.limit is not None:
        samples = samples[: args.limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    generations_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    per_benchmark_items: dict[str, list[dict]] = {}
    with generations_path.open("w", encoding="utf-8") as generations_handle:
        for sample_index, sample in enumerate(samples, start=1):
            benchmark = sample["benchmark"]
            prompt = sample["prompt"]
            gold = sample["answer"]
            formatted_prompt = format_prompt(tokenizer, prompt)
            completions: list[str] = []

            for completion_index in range(args.num_completions):
                seed = args.seed + sample_index * 10_000 + completion_index
                completion = generate_completion(
                    model=model,
                    tokenizer=tokenizer,
                    formatted_prompt=formatted_prompt,
                    generation_config=generation_config,
                    max_new_tokens=args.max_new_tokens,
                    seed=seed,
                    device=device,
                )
                completions.append(completion)

            row = {
                "benchmark": benchmark,
                "source_file": sample["source_file"],
                "prompt": prompt,
                "answer": gold,
                "completions": completions,
            }
            generations_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            per_benchmark_items.setdefault(benchmark, []).append(row)

            print(f"Generated {args.num_completions} completions for {benchmark} example {len(per_benchmark_items[benchmark])}")

    report: dict[str, object] = {
        "model_dir": str(model_dir),
        "samples": str(samples_path),
        "num_completions": args.num_completions,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "benchmark": "general_knowledge",
        "method": BENCHMARK_TO_METHOD["knowledge"],
        "generations_path": str(generations_path),
    }
    scored = score_generations(per_benchmark_items["general_knowledge"], BENCHMARK_TO_METHOD["knowledge"])
    report["result"] = {
        "n_problems": scored["n_problems"],
        "n_completions": scored["n_completions"],
        "metrics": scored["metrics"],
        "detailed_results": scored["detailed_results"],
    }

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("=== Nightly CI Emulation Summary ===")
    metrics = report["result"]["metrics"]
    metric_text = ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    print(f"general_knowledge: {metric_text} (n={report['result']['n_problems']}, n_completions={report['result']['n_completions']}, method={report['method']})")
    print(f"Wrote generations to {generations_path}")
    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())