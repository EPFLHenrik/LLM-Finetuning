import argparse
from pathlib import Path

from preprocessing.utils.config import DEFAULT_OUTPUT_PATH


def parse_generation_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate short SafetyBench rationales with a teacher model."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Which split to process (train/test)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Max tokens per rationale",
    )
    parser.add_argument(
        "--from_id",
        type=int,
        default=None,
        help="Start index (inclusive) in shuffled dataset",
    )
    parser.add_argument(
        "--to_id",
        type=int,
        default=None,
        help="End index (exclusive) in shuffled dataset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing output dataset (if any)",
    )
    return parser.parse_args()