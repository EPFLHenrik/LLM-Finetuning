"""Run the math benchmark on a model checkpoint.

Thin CLI wrapper that delegates to the calibrated multi-source evaluation
defined in `code.math_model.evaluation.eval_v2`. Mirrors the safety model's
`benchmark.py` interface so the team-wide invocation convention is consistent.

Two benchmark profiles are supported:

  - broad  (~290 problems): MATH-500 stratified + MATH-500-hard + AIME 24/25
                            + GSM8K. The profile used for fast ranking.
  - full   (~400 problems): broad + OlympiadBench + Hendrycks-MATH (level 4-5).
                            Used for the final model-selection decision.

Both profiles compute weighted pass@1 and pass@8 at the 16K final-grading
budget via Math-Verify on `\\boxed{}` extraction.
"""
from __future__ import annotations

import argparse
import sys

from .evaluation.eval_v2 import main as eval_v2_main


VALID_BENCHMARKS = ["broad", "full", "quick"]


def _invoke(name: str, checkpoint: str, profile: str, n: int, budget: str) -> None:
    """Invoke `eval_v2.main()` by patching sys.argv with the equivalent flags."""
    argv = [
        "benchmark",
        "--model", checkpoint,
        "--profile", profile,
        "--budget", budget,
        "--n", str(n),
        "--thinking", "on",
    ]
    print(f"\n=========================================")
    print(f"  benchmark: {profile}")
    print(f"  name:      {name}")
    print(f"  checkpoint: {checkpoint}")
    print(f"=========================================\n")
    old_argv = sys.argv
    sys.argv = argv
    try:
        eval_v2_main()
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
                    help="Run every benchmark profile in sequence "
                         "(broad then full; useful for the final candidate).")
    ap.add_argument("--n", type=int, default=8,
                    help="Completions per problem. 8 (default) computes pass@8.")
    ap.add_argument("--budget", default="final", choices=["ci", "final"],
                    help="Token budget: 'final' (16K, default — matches grading) "
                         "or 'ci' (4K, matches nightly CI).")
    args = ap.parse_args()

    if not args.run_all and args.benchmark is None:
        ap.error("Specify either --benchmark <profile> or --run_all.")

    profiles = VALID_BENCHMARKS if args.run_all else [args.benchmark]
    for profile in profiles:
        _invoke(args.name, args.checkpoint, profile, args.n, args.budget)


if __name__ == "__main__":
    main()
