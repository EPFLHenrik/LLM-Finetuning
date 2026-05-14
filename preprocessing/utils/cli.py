import argparse
from pathlib import Path
from constants.global_variables import TrainingConstants

def parse_generation_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reasoning rationales from a teacher model to distill into a student model. This script handles both the primary generation and fallback generation."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input Hugging Face dataset directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Which split to process (train/test). If None, the program considers --input as being the directory containing the data.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=TrainingConstants.MAX_NEW_TOKENS,
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