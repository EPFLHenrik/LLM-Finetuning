#!/usr/bin/env python3
"""Aggregate evaluation_summary.json files from benchmarks into a JSONL summary."""

import argparse
import json
from pathlib import Path

KEEP_FIELDS = {"n_examples", "accuracy", "mean_penalty", "strategy_counts"}


def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark evaluation results.")
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results/full_sft_grpo/benchmarks",
        help="Directory containing per-benchmark subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file path (default: <benchmarks-dir>/../all_benchmarks_summary.jsonl).",
    )
    args = parser.parse_args()

    benchmarks_dir = args.benchmarks_dir.resolve()
    if not benchmarks_dir.is_dir():
        raise SystemExit(f"Benchmarks directory not found: {benchmarks_dir}")

    output_path = args.output or benchmarks_dir.parent / "all_benchmarks_summary.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with open(output_path, "w") as out:
        for summary_file in sorted(benchmarks_dir.glob("*/evaluation_summary.json")):
            with open(summary_file) as f:
                data = json.load(f)

            for result in data.get("results", []):
                row = {"name": summary_file.parent.name}
                row.update({k: result[k] for k in KEEP_FIELDS if k in result})
                out.write(json.dumps(row) + "\n")
                n_written += 1

    if n_written == 0:
        raise SystemExit("No evaluation_summary.json files found.")

    print(f"Wrote {n_written} lines to {output_path}")


if __name__ == "__main__":
    main()
