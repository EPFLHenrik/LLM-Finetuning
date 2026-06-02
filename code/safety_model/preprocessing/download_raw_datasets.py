import os
from pathlib import Path
from urllib.request import urlretrieve

from datasets import get_dataset_split_names, load_dataset

token = os.getenv("HF_TOKEN")

datasets_to_download = [
    # Datasets - Benchmarks Only
    ("advbench", ("AlignmentResearch/AdvBench", None)),
    ("jbb-behaviors", ("JailbreakBench/JBB-Behaviors", "behaviors")),
    ("SGXSTest", ("walledai/SGXSTest", None)),
    ("wildguardmix", ("allenai/wildguardmix", "wildguardtest")),
    ("wildjailbreak_eval", ("allenai/wildjailbreak", "eval", {"delimiter": "\t", "keep_default_na": False})),
    
    # Datasets
    # Safetybench => https://github.com/thu-coai/SafetyBench
    ("wildguardmix", ("allenai/wildguardmix", "wildguardtrain")),
    # Finally not used in the training, only the evaluation set is used for benchmarking (wildjailbreak_eval).
    ("wildjailbreak", ("allenai/wildjailbreak", "train", {"delimiter": "\t", "keep_default_na": False})),
]

raw_dir = Path("data/safety/raw")
raw_dir.mkdir(parents=True, exist_ok=True)

for i, (name, (repo, config, *extra_args)) in enumerate(datasets_to_download):
    print(
        f"\n{i+1}/{len(datasets_to_download)} Downloading {name} from {repo} with config {config}..."
    )

    try:
        split_names = get_dataset_split_names(repo, config_name=config)
    except Exception as e:
        print(f"\033[91mError fetching splits for {name}: {e}\033[0m")
        continue

    print(f"Available splits: {split_names}")

    dataset_dict = {}
    for split in split_names:
        out_dir = raw_dir / name / split
        if out_dir.exists():
            print(f"Skipping {name}:{split} because it already exists at {out_dir}")
            continue

        try:
            if extra_args:
                ds = load_dataset(repo, config, split=split, token=token, **extra_args[0])
            else:
                ds = load_dataset(repo, config, split=split, token=token)
            dataset_dict[split] = ds
            out_dir.parent.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(out_dir))
            print(f"Saved {name}:{split} -> {out_dir}")
        except Exception as e:
            print(f"\033[91mSkipping {name}:{split} because of error: {e}\033[0m")


# Download safetybench separately since it requires a different parsing logic and we only use the test set for benchmarking, not the training set.

safetybench_en_dir = Path("data/safety/raw/safetybench/en")
safetybench_en_dir.mkdir(parents=True, exist_ok=True)

safetybench_files = {
    "test_en.json": "https://raw.githubusercontent.com/thu-coai/SafetyBench/960833692a1ec82b191ff3e902daca7993c35cc1/opensource_data/test_en.json",
    "test_answers_en.json": "https://raw.githubusercontent.com/thu-coai/SafetyBench/960833692a1ec82b191ff3e902daca7993c35cc1/opensource_data/test_answers_en.json",
}

for filename, url in safetybench_files.items():
    target = safetybench_en_dir / filename
    if target.exists():
        print(f"Skipping SafetyBench file because it already exists at {target}")
        continue

    try:
        urlretrieve(url, target)
        print(f"Saved SafetyBench file -> {target}")
    except Exception as e:
        print(f"\033[91mFailed downloading SafetyBench file {filename}: {e}\033[0m")

