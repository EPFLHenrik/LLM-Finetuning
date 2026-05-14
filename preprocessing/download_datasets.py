import os
from pathlib import Path

from datasets import get_dataset_split_names, load_dataset

token = os.getenv("HF_TOKEN")

datasets_to_download = [
    # BENCHMARKS
    ("qwen3guardtest", ("Qwen/Qwen3GuardTest", None)),
    ("advbench", ("AlignmentResearch/AdvBench", None)),
    # ("safetybench", ("thu-coai/SafetyBench", "test")), # Bad as we can't see the labels
    ("jbb-behaviors", ("JailbreakBench/JBB-Behaviors", "behaviors")),
    ("SGXSTest", ("walledai/SGXSTest", None)),
    # ("truthfulqa_generation", ("truthfulqa/truthful_qa", "generation")),
    # ("truthfulqa_multiple_choice", ("truthfulqa/truthful_qa", "multiple_choice")),
    # SFT Datasets
    ("wildguardmix", ("allenai/wildguardmix", "wildguardtrain")),
    ("wildguardmix", ("allenai/wildguardmix", "wildguardtest")),
    ("wildjailbreak", ("allenai/wildjailbreak", "train", {"delimiter": "\t", "keep_default_na": False})),
    ("wildjailbreak_eval", ("allenai/wildjailbreak", "eval", {"delimiter": "\t", "keep_default_na": False})),
    ("safechain", ("UWNSL/SafeChain", None)),
    # DPO Datasets
    ("pku-safeRLHF", ("PKU-Alignment/PKU-SafeRLHF", "default")),
    ("hh-rlhf", ("Anthropic/hh-rlhf", None)),
]

raw_dir = Path("data/raw")
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
