import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from multilingual.config import (
    DEFAULT_MAX_NEW_TOKENS_BY_TASK,
    DEFAULT_MODEL_NAME,
    REPO,
    TARGET_LANGUAGES,
)
from multilingual.evaluation import evaluate_examples, summarize_results
from multilingual.generation import generate_answer, load_model_and_tokenizer
from multilingual.include_loader import load_include_examples
from multilingual.mgsm_loader import load_mgsm_examples
from multilingual.utils import save_jsonl


def parse_n_per_language(value: str) -> Optional[int]:
    if value.lower() in {"full", "none", "all"}:
        return None
    n = int(value)
    if n <= 0:
        raise ValueError("--n_per_language must be a positive integer or 'full'")
    return n


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen model on multilingual datasets.")

    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="HF model name or local path.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REPO / "multilingual" / "results" / "base_qwen_modular"),
        help="Directory where JSONL and CSV results are saved.",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["include", "mgsm"],
        default=["include", "mgsm"],
        help="Datasets to evaluate.",
    )

    parser.add_argument(
        "--languages",
        nargs="+",
        choices=list(TARGET_LANGUAGES.keys()),
        default=list(TARGET_LANGUAGES.keys()),
        help="Languages to evaluate.",
    )

    parser.add_argument(
        "--n_per_language",
        type=str,
        default="5",
        help="Number of examples per language/dataset, or 'full'.",
    )

    parser.add_argument(
        "--mcq_max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS_BY_TASK["mcq"],
    )

    parser.add_argument(
        "--math_max_new_tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS_BY_TASK["math"],
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-dataset/language JSONL files.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN is not set. Run `export HF_TOKEN=hf_xxx` before launching the script."
        )

    n_per_language = parse_n_per_language(args.n_per_language)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_new_tokens_by_task = {
        "mcq": args.mcq_max_new_tokens,
        "math": args.math_max_new_tokens,
    }

    print("Configuration")
    print(f"  model_name: {args.model_name}")
    print(f"  datasets: {args.datasets}")
    print(f"  languages: {args.languages}")
    print(f"  n_per_language: {n_per_language if n_per_language is not None else 'full'}")
    print(f"  output_dir: {output_dir}")
    print(f"  max_new_tokens_by_task: {max_new_tokens_by_task}")

    model, tokenizer = load_model_and_tokenizer(args.model_name, hf_token)

    all_results = []

    for language in args.languages:
        if language not in TARGET_LANGUAGES:
            raise ValueError(f"Unknown language: {language}")

        if "include" in args.datasets:
            include_path = output_dir / f"include_{language}.jsonl"

            if include_path.exists() and not args.overwrite:
                print(f"[SKIP] Existing file: {include_path}")
                df_include = pd.read_json(include_path, lines=True)
                all_results.append(df_include)
            else:
                include_examples = load_include_examples(language, n_per_language)

                if len(include_examples) > 0:
                    df_include = evaluate_examples(
                        examples=include_examples,
                        dataset_name="include",
                        language=language,
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens_by_task=max_new_tokens_by_task,
                        generate_fn=generate_answer,
                    )
                    save_jsonl(df_include, include_path)
                    all_results.append(df_include)

        if "mgsm" in args.datasets:
            mgsm_path = output_dir / f"mgsm_{language}.jsonl"

            if mgsm_path.exists() and not args.overwrite:
                print(f"[SKIP] Existing file: {mgsm_path}")
                df_mgsm = pd.read_json(mgsm_path, lines=True)
                all_results.append(df_mgsm)
            else:
                mgsm_examples = load_mgsm_examples(language, n_per_language)

                if len(mgsm_examples) > 0:
                    df_mgsm = evaluate_examples(
                        examples=mgsm_examples,
                        dataset_name="mgsm",
                        language=language,
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens_by_task=max_new_tokens_by_task,
                        generate_fn=generate_answer,
                    )
                    save_jsonl(df_mgsm, mgsm_path)
                    all_results.append(df_mgsm)

    if len(all_results) == 0:
        print("No results produced.")
        return

    df_all = pd.concat(all_results, ignore_index=True)
    summary = summarize_results(df_all)

    all_results_path = output_dir / "all_results.jsonl"
    summary_path = output_dir / "summary.csv"

    save_jsonl(df_all, all_results_path)
    summary.to_csv(summary_path, index=False)

    print(f"Saved summary: {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()