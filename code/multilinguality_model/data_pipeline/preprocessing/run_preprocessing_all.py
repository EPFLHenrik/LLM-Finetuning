"""code/multilinguality_model/data_pipeline/preprocessing/run_preprocessing_all.py

Run active preprocessing modules.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from multilinguality_model.paths import get_repo_root


REPO_ROOT = get_repo_root()
PACKAGE = __package__

ACTIVE_PREPROCESSING_MODULES = [
    "preprocess_ceval",
    "preprocess_cmmlu",
    "preprocess_cultsportqa",
    "preprocess_global_piqa",
    "preprocess_head_qa",
    "preprocess_include",
    "preprocess_mera",
    "preprocess_milu",
    "preprocess_mmlu_prox",
    "preprocess_global_mmlu",
    "preprocess_polizia_di_stato",
    "preprocess_quaderno_prefettizia",
    "preprocess_exams",
]

OPTIONAL_PREPROCESSING_MODULES = []

BENCHMARK_MODULES = [
    "build_benchmark",
]


def full_module_name(module_name: str) -> str:
    if PACKAGE is None:
        raise RuntimeError("This script must be run as a module with python -m.")
    return f"{PACKAGE}.{module_name}"


def run_module(module_name: str) -> None:
    module = full_module_name(module_name)

    print()
    print(f"Running {module}")
    print("=" * (len(module) + 8))

    subprocess.run(
        [sys.executable, "-m", module],
        cwd=REPO_ROOT,
        check=True,
    )


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
        for module_name in modules:
            print(full_module_name(module_name))
        return 0

    for module_name in modules:
        run_module(module_name)

    print()
    print("All requested preprocessing scripts completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
