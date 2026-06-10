"""Run the math benchmark on a model checkpoint.

Thin CLI wrapper that delegates to the weighted multi-source evaluation
defined in `code.math_model.evaluation.evaluate`.

Two benchmark profiles are supported:

  - broad  (~290 problems): MATH-500 stratified + MATH-500-hard + AIME 24/25
                            + GSM8K. The profile used for fast ranking.
  - full   (~400 problems): broad + OlympiadBench + Hendrycks-MATH (level 4-5).
                            Used for the final model-selection decision.

Both profiles compute weighted pass@1 and pass@8 at the 16K budget via
Math-Verify on `\\boxed{}` extraction.
"""
from __future__ import annotations

import argparse
import sys

from .evaluate import main as evaluate_main


VALID_BENCHMARKS = ["broad", "full", "quick"]
# Profiles run by --run_all, in order.
RUN_ALL_PROFILES = ["broad", "full"]


def _invoke(name: str, checkpoint: str, profile: str, n: int, budget: str) -> None:
    """Invoke `evaluate.main()` by patching sys.argv with the equivalent flags."""
    argv = [
        "evaluate",
        "--model", checkpoint,
        "--profile", profile,
        "--budget", budget,
        "--n", str(n),
        "--thinking", "on",
    ]
    print("\n=========================================")
    print(f"  benchmark:  {profile}")
    print(f"  name:       {name}")
    print(f"  checkpoint: {checkpoint}")
    print("=========================================\n")
    old_argv = sys.argv
    sys.argv = argv
    try:
        evaluate_main()
    finally:
        sys.argv = old_argv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True,
                    help="Short label for the run (used in the saved result path).")
    ap.add_argument("--checkpoint", required=True,
                    help="HF model id or local path of the checkpoint to evaluate.")
    ap.add_argument("--benchmark", default=None, choices=VALID_BENCHMARKS,
                    help="Which benchmark profile to run (broad / full / quick).")
    ap.add_argument("--run_all", action="store_true",
                    help="Run the broad then full profiles in sequence "
                         "(useful for the final candidate).")
    ap.add_argument("--n", type=int, default=8,
                    help="Completions per problem. 8 (default) computes pass@8.")
    ap.add_argument("--budget", default="final", choices=["ci", "final"],
                    help="Token budget: 'final' (16K, default) or 'ci' (4K).")
    args = ap.parse_args()

    if not args.run_all and args.benchmark is None:
        ap.error("Specify either --benchmark <profile> or --run_all.")

    profiles = RUN_ALL_PROFILES if args.run_all else [args.benchmark]
    for profile in profiles:
        _invoke(args.name, args.checkpoint, profile, args.n, args.budget)


if __name__ == "__main__":
    main()
