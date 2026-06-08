from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_package_dir() -> Path:
    return Path(__file__).resolve().parent


def get_data_dir() -> Path:
    return get_repo_root() / "data" / "multilinguality"


def get_processed_data_dir() -> Path:
    return get_data_dir() / "processed"


def get_distilled_data_dir() -> Path:
    return get_data_dir() / "distilled"


def get_sft_data_dir() -> Path:
    return get_data_dir() / "sft"


def get_benchmark_data_dir() -> Path:
    return get_data_dir() / "benchmark"


def get_checkpoints_dir() -> Path:
    return get_repo_root() / "checkpoints" / "multilinguality_model"


def get_runs_dir() -> Path:
    return get_repo_root() / "runs" / "multilinguality_model"


def get_configs_dir() -> Path:
    return get_package_dir() / "configs"


def get_assets_dir() -> Path:
    return get_package_dir() / "assets"


def get_benchmarks_data_dir() -> Path:
    return get_data_dir() / "benchmarks"

def get_raw_data_dir() -> Path:
    return get_data_dir() / "raw"

def get_sft_configs_dir() -> Path:
    return get_configs_dir() / "sft"
