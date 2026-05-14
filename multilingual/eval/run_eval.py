"""multilingual/run_eval.py"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional
import pandas as pd

from multilingual.config import (
    DEFAULT_MAX_NEW_TOKENS_BY_TASK,
    DEFAULT_MODEL_NAME,
    REPO,
    TARGET_LANGUAGES,
)
from multilingual.eval.evaluation import evaluate_examples, summarize_results
from multilingual.generation import generate_answer, load_model_and_tokenizer
from multilingual.loaders.ceval_loader import load_ceval_examples
from multilingual.loaders.include_loader import load_include_examples
from multilingual.loaders.mgsm_loader import load_mgsm_examples
from multilingual.loaders.milu_loader import load_milu_examples
from multilingual.loaders.mmmlu_loader import load_mmmlu_examples
from multilingual.utils import save_jsonl


def parse_n(value: str) -> Optional[int]:
    if value.lower() in {"full", "none", "all"}:
        return None
    n = int(value)
    if n <= 0:
        raise ValueError("--n_per_language must be a positive integer or 'full'")
    return n


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a Qwen3 model on the multilingual val split."
    )
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REPO / "multilingual" / "results" / "eval"),
    )
    parser.add_argument(
        "--datasets", nargs="+",
        choices=["include", "mgsm", "mmmlu", "milu", "ceval"],
        default=["include", "mgsm", "mmmlu", "milu", "ceval"],
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=list(TARGET_LANGUAGES.keys()),
        default=list(TARGET_LANGUAGES.keys()),
    )
    parser.add_argument(
        "--n_per_language", type=str, default="50",
        help="Examples per language/dataset from the val split, or 'full'.",
    )
    parser.add_argument("--mcq_max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS_BY_TASK["mcq"])
    parser.add_argument("--math_max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS_BY_TASK["math"])
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save_official_format", action="store_true",
        help=(
            "Also save a JSONL per dataset/language in evaluate/score.py format "
            "to <output_dir>/official/. Pipe into: "
            "python -m evaluate.score --generations <file> --benchmark multilingual"
        ),
    )
    return parser.parse_args()


def _save_official_jsonl(df: pd.DataFrame, path: Path) -> None:
    """Write {prompt, answer, completions:[output]} JSONL from a results df."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({
                "prompt":      row["prompt"],
                "answer":      row["gold"],
                "completions": [row["output"]],
            }, ensure_ascii=False) + "\n")
    print(f"Saved official format: {path}")


def main():
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("Set HF_TOKEN before running: export HF_TOKEN=hf_xxx")

    n = parse_n(args.n_per_language)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_new_tokens = {
        "mcq":  args.mcq_max_new_tokens,
        "math": args.math_max_new_tokens,
    }

    print("=" * 60)
    print(f"Model:             {args.model_name}")
    print(f"Datasets:          {args.datasets}")
    print(f"Languages:         {args.languages}")
    print(f"N/lang:            {n if n is not None else 'full'}")
    print(f"Sampling:          do_sample={args.do_sample}, T={args.temperature}, "
          f"top_p={args.top_p}, top_k={args.top_k}")
    print(f"Save official fmt: {args.save_official_format}")
    print(f"Output:            {output_dir}")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer(args.model_name, hf_token)

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

    all_results = []

    for language in args.languages:
        for dataset_name in args.datasets:
            out_path = output_dir / f"{dataset_name}_{language}.jsonl"

            if out_path.exists() and not args.overwrite:
                print(f"[SKIP] {out_path}")
                df = pd.read_json(out_path, lines=True)
                all_results.append(df)
                continue

            if dataset_name == "include":
                examples = load_include_examples(language, n=n, split="val")
            elif dataset_name == "mgsm":
                examples = load_mgsm_examples(language, n=n, split="val")
            elif dataset_name == "mmmlu":
                examples = load_mmmlu_examples(language, n=n, split="val")
            elif dataset_name == "milu":
                if language != "hindi":
                    continue
                examples = load_milu_examples(n=n, split="val", hf_token=hf_token)
            elif dataset_name == "ceval":
                if language != "chinese":
                    continue
                examples = load_ceval_examples(n=n, split="val")

            if not examples:
                continue

            df = evaluate_examples(
                examples=examples,
                dataset_name=dataset_name,
                language=language,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens_by_task=max_new_tokens,
                generate_fn=_generate,
            )
            save_jsonl(df, out_path)

            if args.save_official_format:
                _save_official_jsonl(
                    df,
                    output_dir / "official" / f"{dataset_name}_{language}.jsonl",
                )

            all_results.append(df)

    if not all_results:
        print("No results produced.")
        return

    df_all = pd.concat(all_results, ignore_index=True)
    summary = summarize_results(df_all)

    save_jsonl(df_all, output_dir / "all_results.jsonl")
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")
    print(summary.to_string(index=False))

    if args.save_official_format:
        print(f"\nOfficial format files in: {output_dir / 'official'}/")
        print("Score with:")
        print("  python -m evaluate.score --generations <file> --benchmark multilingual")


if __name__ == "__main__":
    main()