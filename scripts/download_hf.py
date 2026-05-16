#!/usr/bin/env python3
"""Download a Hugging Face dataset and write each split as JSONL.

Usage:
    python scripts/download_hf.py --hf-token <token> --dataset-path <dataset> [--dataset-config <config>] --output-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def dataset_slug(dataset_path: str, dataset_config: str | None) -> str:
    slug = dataset_path.replace("/", "__").replace(":", "__")
    if dataset_config:
        slug += f"__{dataset_config.replace('/', '__').replace(':', '__').replace(' ', '_')}"
    return slug


def serialize_row(row: dict) -> str:
    try:
        return json.dumps(row, ensure_ascii=False)
    except TypeError:
        simple = {
            key: (value if isinstance(value, (str, int, float, bool, list, dict, type(None))) else str(value))
            for key, value in row.items()
        }
        return json.dumps(simple, ensure_ascii=False)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token", required=True, help="Hugging Face token")
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Hugging Face dataset path, for example tau/commonsense_qa",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Optional Hugging Face dataset config/subset, for example ARC-Challenge",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the downloaded JSONL files",
    )
    args = parser.parse_args(argv)

    if args.hf_token:
        os.environ.setdefault("HF_TOKEN", args.hf_token)
        os.environ.setdefault("HF_HUB_TOKEN", args.hf_token)

    from datasets import load_dataset

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset_path if not args.dataset_config else f"{args.dataset_path} ({args.dataset_config})"
    print(f"Loading dataset {dataset_name} from Hugging Face...")
    if args.dataset_config:
        ds = load_dataset(args.dataset_path, args.dataset_config)
    else:
        ds = load_dataset(args.dataset_path)

    for split, dataset in ds.items():
        out_path = outdir / f"{split}.jsonl"
        print(f"Writing {len(dataset)} rows to {out_path}")
        with out_path.open("w", encoding="utf-8") as fout:
            for row in dataset:
                fout.write(serialize_row(row) + "\n")

    print("Done.")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))