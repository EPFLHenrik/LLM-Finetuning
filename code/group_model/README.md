# Group Model Reproducibility Guide

## Goal

Reproduce the final submitted group model by:
- merging existing post-trained model
- evaluating this merged model with benchmarks from all categories (math, safety, general knowledge, multilinguality)

## Before You Start

- Run all commands from the repository root.
- Ensure your Python environment is activated and project dependencies are installed (see [requirements.txt](../../requirements.txt)).


## 1. Benchmarks Downloading

This step downloads all raw safety datasets required by preprocessing and benchmarking.

`python -m code.group_model.download_benchmarks`

## 2. Model Merging

This stage merges existing models (merging_targets) into one model.
Merging techniques include:
- `full_dare_task_arithmetic.py`
- `full_dare_ties.py`
- `full_slerp.py`
- `full_task_arithmetic.py`
- `full_ties.py`

The arguments are:
- `--base_model`, path to base model, by default "checkpoints/Qwen3-1.7B-baseline".
- `--merging_targets`, paths to merge targets (specialized models).
- `--output_path`, output path where the merged model will be stored
- `--weights`, weigths attributed to each merge target, by default uniform.
- `--density`, density factor for compatible merging techniques (like TIES), by default 0.8.

### Examples
This first example requires the reproduction of the 4 individual specialists first, so please follow these 4 readme's first before running this command.
```
python -m code.group_model.merging.full_task_arithmetic --merging_targets checkpoints/math_model/final checkpoints/general_knowledge_model/sft_grpo_dpo checkpoints/multilinguality_model/best_model checkpoints/safety_model/dpo --output_path checkpoints/merged_model
```
Alternatively the individual models from hf can be downloaded and used for the merge
```python
from huggingface_hub import snapshot_download

models = {
    "checkpoints/math_model/hf_model":              "cs-552-2026-mystery-machine/math_model",
    "checkpoints/general_knowledge_model/hf_model": "cs-552-2026-mystery-machine/general_knowledge_model",
    "checkpoints/multilinguality_model/hf_model":   "cs-552-2026-mystery-machine/multilingual_model",
    "checkpoints/safety_model/hf_model":            "cs-552-2026-mystery-machine/safety_model",
}

for local_dir, repo_id in models.items():
    print(f"Downloading {repo_id} -> {local_dir}")
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
```
Then run the merge on the downloaded models:
```
python -m code.group_model.merging.full_task_arithmetic \
    --merging_targets \
        checkpoints/math_model/hf_model \
        checkpoints/general_knowledge_model/hf_model \
        checkpoints/multilinguality_model/hf_model \
        checkpoints/safety_model/hf_model \
    --output_path checkpoints/merged_model
```

## 3. Model Benchmarking

To evaluate any checkpoint on any benchmarks, `code/group_model/benchmark.py` script has to be run with the following arguments:
- `--name`, name of the model benchmark, by default the model checkpoint path.
- `--benchmark`, name of the benchmark in `data/group_model/benchmark/`.
- `--checkpoint`, path of the model to be evaluated.
- `--limit`, maximum number of samples per benchmark to evaluate. If not specified, the entire benchmark is considered.

replace `--benchmark <...>` with `--run_all` to run all benchmarks present in `data/group_model/benchmark/`.

### Example:
```
python -m code.group_model.benchmark --name merged_model --checkpoint checkpoints/merged_model --run_all --limit 100
```
