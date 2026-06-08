#!/usr/bin/env python3
"""Download a Hugging Face dataset and write each split as JSONL.

Usage:
    python scripts/download_hf.py --hf-token <token> --dataset-path <dataset> [--dataset-config <config>|all] --output-dir <dir>
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


def _sanitize(value: object) -> object:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return {"__type__": type(value).__name__}


def serialize_row(row: dict) -> str:
    try:
        return json.dumps(row, ensure_ascii=False)
    except TypeError:
        return json.dumps(_sanitize(row), ensure_ascii=False)


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
        help="Optional Hugging Face dataset config/subset, for example ARC-Challenge, or 'all' to download every config",
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

    from datasets import get_dataset_config_names, load_dataset

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset_path if not args.dataset_config else f"{args.dataset_path} ({args.dataset_config})"
    print(f"Loading dataset {dataset_name} from Hugging Face...")
    if args.dataset_config == "all":
        config_names = get_dataset_config_names(args.dataset_path)
        if not config_names:
            raise SystemExit(f"No configs found for dataset {args.dataset_path}")
        print(f"Found {len(config_names)} configs: {', '.join(config_names)}")
        for config_name in config_names:
            config_outdir = outdir / config_name
            config_outdir.mkdir(parents=True, exist_ok=True)
            print(f"Loading config {config_name}...")
            ds = load_dataset(args.dataset_path, config_name)
            for split, dataset in ds.items():
                out_path = config_outdir / f"{split}.jsonl"
                print(f"Writing {len(dataset)} rows to {out_path}")
                with out_path.open("w", encoding="utf-8") as fout:
                    for row in dataset:
                        fout.write(serialize_row(row) + "\n")
    elif args.dataset_config:
        ds = load_dataset(args.dataset_path, args.dataset_config)
        for split, dataset in ds.items():
            out_path = outdir / f"{split}.jsonl"
            print(f"Writing {len(dataset)} rows to {out_path}")
            with out_path.open("w", encoding="utf-8") as fout:
                for row in dataset:
                    fout.write(serialize_row(row) + "\n")
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