## Safety Model Reproducibility Guide

This document describes the exact pipeline used to reproduce the submitted safety model.

The commands below are intentionally kept in the original order and should be executed sequentially.

## Goal

Reproduce the final submitted safety model by:
- preparing safety datasets,
- generating distilled reasoning data,
- training an SFT adapter,
- creating DPO preference pairs,
- training the final DPO adapter.

## Before You Start

- Run all commands from the repository root.
- Ensure your Python environment is activated and project dependencies are installed (see [requirements.txt](../../requirements.txt)).


## 1. Datasets Downloading

This step downloads all raw safety datasets required by preprocessing and benchmarking.

`python -m code.safety_model.preprocessing.download_raw_datasets`

## 2. Datasets Preprocessing

### Extraction and MCQ formatting

This stage converts raw data into the project-specific processed formats used for SFT, DPO, and evaluation.

**Training Datasets**

```
python -m code.safety_model.preprocessing.extraction.wildguardmix.preprocess
python -m code.safety_model.preprocessing.extraction.safetybench.preprocess
```

**Benchmarks**


```
python -m code.safety_model.preprocessing.extraction.advbench.benchmark 
python -m code.safety_model.preprocessing.extraction.jbb-behaviors.benchmark
python -m code.safety_model.preprocessing.extraction.sgxstest.benchmark

# For WildJailBreak, the training split is not used
python -m code.safety_model.preprocessing.extraction.wildjailbreak.preprocess
```

### Distillation

This step generates reasoning traces for selected training splits, merges generated shards, then restores the original test split.

#### Distillation for `wildguardmix/binary_choice`

``` 
python -m code.safety_model.preprocessing.distillation.reasoning_generation \
    --input data/safety/processed/sft/wildguardmix/binary_choice \
    --output data/safety/processed/sft/distillation/wildguardmix/binary_choice \
    --split train --from_id 0 --to_id 3000 && \

python -m code.safety_model.preprocessing.distillation.merge \ 
    data/safety/processed/sft/distillation/wildguardmix/binary_choice && \

cp -r \
    data/safety/processed/sft/wildguardmix/binary_choice/test \
    data/safety/processed/sft/distillation/wildguardmix/binary_choice/
```

#### Distillation for `wildguardmix/subcategory_classification`

``` 
python -m code.safety_model.preprocessing.distillation.reasoning_generation \
    --input data/safety/processed/sft/wildguardmix/subcategory_classification \
    --output data/safety/processed/sft/distillation/wildguardmix/subcategory_classification \
    --split train --from_id 0 --to_id 5000 && \

python -m code.safety_model.preprocessing.distillation.merge \
    data/safety/processed/sft/distillation/wildguardmix/subcategory_classification && \

cp -r \
    data/safety/processed/sft/wildguardmix/subcategory_classification/test \
    data/safety/processed/sft/distillation/wildguardmix/subcategory_classification/
```

#### Distillation for `wildguardmix/safety_combo`

``` 
python -m code.safety_model.preprocessing.distillation.reasoning_generation \
    --input data/safety/processed/sft/wildguardmix/safety_combo \
    --output data/safety/processed/sft/distillation/wildguardmix/safety_combo \
    --split train --from_id 0 --to_id 5000 && \

python -m code.safety_model.preprocessing.distillation.merge \
    data/safety/processed/sft/distillation/wildguardmix/safety_combo && \

cp -r \
    data/safety/processed/sft/wildguardmix/safety_combo/test \
    data/safety/processed/sft/distillation/wildguardmix/safety_combo/
```

#### Distillation for `safetybench`

``` 
python -m code.safety_model.preprocessing.distillation.reasoning_generation \
    --input data/safety/processed/sft/safetybench \
    --output data/safety/processed/sft/distillation/safetybench \
    --split train --from_id 0 --to_id 5000 && \

python -m code.safety_model.preprocessing.distillation.merge \
    data/safety/processed/sft/distillation/safetybench && \

cp -r \
    data/safety/processed/sft/safetybench/test \
    data/safety/processed/sft/distillation/safetybench/
```

## 3. Downloading Baseline Model (Qwen3-1.7B)

This creates/downloads the baseline checkpoint used as the base for adapter training.

`python -m code.safety_model.huggingface.base_model_checkpoint_creation`

## 4. SFT

Train the first-stage LoRA adapter on distilled safety datasets.

```
python -m code.safety_model.sft \
    --run_name safety_model/sft \
    --datasets \
        distillation/wildguardmix/binary_choice \
        distillation/wildguardmix/subcategory_classification \
        distillation/wildguardmix/safety_combo \
        distillation/safetybench \
    --train-sizes 3000 5000 5000 5000 \
    --epochs 5
```

Expected artifact:
- `checkpoints/safety_model/sft/LoRA`
- `checkpoints/safety_model/sft/hf_model`

## 5. DPO Pair Creation

Generate preference pairs from the SFT model outputs. Run each block in order.

```
python -m code.safety_model.preprocessing.dpo.pair_generation \
    --checkpoint checkpoints/safety_model/sft/hf_model \
    --dataset_path data/safety/processed/sft/wildguardmix/binary_choice/train \
    --output_path data/safety/processed/dpo/wildguardmix/binary_choice/train \
    --start_index 3000 \
    --end_index 6000 && \

cp -r \
    data/safety/processed/sft/wildguardmix/binary_choice/test \
    data/safety/processed/dpo/wildguardmix/binary_choice/
```

```
python -m code.safety_model.preprocessing.dpo.pair_generation \
    --checkpoint checkpoints/safety_model/sft/hf_model \
    --dataset_path data/safety/processed/sft/wildguardmix/subcategory_classification/train \
    --output_path data/safety/processed/dpo/wildguardmix/subcategory_classification/train \
    --start_index 5000 \
    --end_index 10000 && \

cp -r \
    data/safety/processed/sft/wildguardmix/subcategory_classification/test \
    data/safety/processed/dpo/wildguardmix/subcategory_classification/
```

```
python -m code.safety_model.preprocessing.dpo.pair_generation \
    --checkpoint checkpoints/safety_model/sft/hf_model \
    --dataset_path data/safety/processed/sft/wildguardmix/safety_combo/train \
    --output_path data/safety/processed/dpo/wildguardmix/safety_combo/train \
    --start_index 5000 \
    --end_index 10000 && \

cp -r \
    data/safety/processed/sft/wildguardmix/safety_combo/test \
    data/safety/processed/dpo/wildguardmix/safety_combo/
```

```
python -m code.safety_model.preprocessing.dpo.pair_generation \
    --checkpoint checkpoints/safety_model/sft/hf_model \
    --dataset_path data/safety/processed/sft/safetybench/train \
    --output_path data/safety/processed/dpo/safetybench/train \
    --start_index 0 \
    --end_index 10000 && \

cp -r \
    data/safety/processed/sft/safetybench/test \
    data/safety/processed/dpo/safetybench/
```

## 6. DPO training

Train the final DPO adapter starting from the SFT adapter and baseline model.

```
python -m code.safety_model.dpo \
    --adapter-path checkpoints/safety_model/sft/LoRA \
    --base-model checkpoints/Qwen3-1.7B-baseline \
    --datasets \
        safetybench \
        wildguardmix/binary_choice \
        wildguardmix/subcategory_classification \ 
        wildguardmix/safety_combo \
    --best-checkpoint-dir checkpoints/safety_model/dpo \ 
    --drop-missing-chosen \
    --epochs 3 \
    --run_name dpo
```

Expected artifact:
- `checkpoints/safety_model/dpo`


## 7. Benchmarking
To evaluate any checkpoint on the safety benchmarks, run the following command:

`python -m code.safety_model.benchmark --name <my-model-name> --benchmark <benchmark_name> --checkpoint <checkpoint_dir>`

replace `--benchmark <...>` with `--run_all` to run all benchmarks present in data/safety/processed/benchmark/.

Example:
`python -m code.safety_model.benchmark --name <my-model-name> --benchmark <benchmark_name> --checkpoint <checkpoint_dir>`