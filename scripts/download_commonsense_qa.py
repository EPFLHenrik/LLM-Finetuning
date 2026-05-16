#!/usr/bin/env python3
"""Download the HF dataset `tau/commonsense_qa` and write raw JSONL files.

Writes one file per split under `/data/datasets/commonsense_qa/<split>.jsonl`.

Usage:
  python scripts/download_commonsense_qa.py --output-dir /data/datasets/commonsense_qa

If you need an auth token, set `--hf-token` or export `HF_TOKEN`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (optional)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/datasets/commonsense_qa"),
        help="Directory to write the downloaded JSONL files",
    )
    args = parser.parse_args(argv)

    if args.hf_token:
        os.environ.setdefault("HF_TOKEN", args.hf_token)
        os.environ.setdefault("HF_HUB_TOKEN", args.hf_token)

    from datasets import load_dataset

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset tau/commonsense_qa from Hugging Face...")
    ds = load_dataset("tau/commonsense_qa")

    for split, dataset in ds.items():
        out_path = outdir / f"{split}.jsonl"
        print(f"Writing {len(dataset)} rows to {out_path}")
        with out_path.open("w", encoding="utf-8") as fout:
            for row in dataset:
                # Ensure row is JSON-serializable; convert non-serializable items to str
                try:
                    json_str = json.dumps(row, ensure_ascii=False)
                except TypeError:
                    # fallback: convert values to primitives
                    simple = {k: (v if isinstance(v, (str, int, float, bool, list, dict, type(None))) else str(v)) for k, v in row.items()}
                    json_str = json.dumps(simple, ensure_ascii=False)
                fout.write(json_str + "\n")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
