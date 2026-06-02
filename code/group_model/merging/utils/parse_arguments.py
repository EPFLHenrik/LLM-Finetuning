import argparse
from pathlib import Path

def parse_merge_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", default="checkpoints/Qwen3-1.7B-baseline", type=str)
    parser.add_argument("--adapters", nargs="+", required=True, help="Paths to LoRA adapters to merge, or full models, depending on the script run.")
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional weights. Defaults to uniform.",
    )

    parser.add_argument("--dtype", type=str, default="bfloat16")

    args = parser.parse_args()

    if args.weights is None:
        args.weights = [1.0 / len(args.adapters)] * len(args.adapters)

    if len(args.weights) != len(args.adapters):
        raise ValueError("--weights must match number of --adapters")

    args.output_path = str(Path(args.output_path))
    args.config_path = str(Path(args.output_path).with_suffix(".yaml"))

    return args