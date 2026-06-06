"""Run all active preprocessing scripts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path("/scratch/workspace/emma/standard-project-m2-mystery-machine")

ACTIVE_PREPROCESSING_MODULES = [
    "multilingual.data_pipeline.preprocessing.preprocess_ceval",
    "multilingual.data_pipeline.preprocessing.preprocess_cmmlu",
    "multilingual.data_pipeline.preprocessing.preprocess_cultsportqa",
    "multilingual.data_pipeline.preprocessing.preprocess_global_piqa",
    "multilingual.data_pipeline.preprocessing.preprocess_head_qa",
    "multilingual.data_pipeline.preprocessing.preprocess_include",
    "multilingual.data_pipeline.preprocessing.preprocess_mera",
    "multilingual.data_pipeline.preprocessing.preprocess_milu",
    "multilingual.data_pipeline.preprocessing.preprocess_mmlu_prox",
    "multilingual.data_pipeline.preprocessing.preprocess_global_mmlu",
    "multilingual.data_pipeline.preprocessing.preprocess_polizia_di_stato",
    "multilingual.data_pipeline.preprocessing.preprocess_quaderno_prefettizia",
]

OPTIONAL_PREPROCESSING_MODULES = [
    "multilingual.data_pipeline.preprocessing.preprocess_exams",
]

BENCHMARK_MODULES = [
    "multilingual.data_pipeline.preprocessing.build_benchmark",
]


def run_module(module: str) -> None:
    print()
    print(f"Running {module}")
    print("=" * (len(module) + 8))
    subprocess.run([sys.executable, "-m", module], cwd=REPO_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-exams", action="store_true")
    parser.add_argument("--build-benchmark", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    modules = list(ACTIVE_PREPROCESSING_MODULES)

    if args.include_exams:
        modules.extend(OPTIONAL_PREPROCESSING_MODULES)

    if args.build_benchmark:
        modules.extend(BENCHMARK_MODULES)

    if args.dry_run:
        for module in modules:
            print(module)
        return 0

    for module in modules:
        run_module(module)

    print()
    print("All requested preprocessing scripts completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
