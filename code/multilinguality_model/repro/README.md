# Reproducing the best multilingual model

This folder contains the scripts to reproduce the current best multilingual model.

## Overview

Run the full pipeline with one script, or execute the steps individually. 

- `reproduce_best_model.sh`: downloads data and benchmarks from Hugging Face, trains three source models, merges them, and evaluates the final model.
- Individual scripts: run each step separately for more control.

The three source models are `v1` (best joing multilingual SFT checkpoint), `v20-russian` (Russian specialist), and `v20-hindi` (Hindi specialist).

## Run everything

From the repository root:

```
cd /scratch/workspace/emma/standard-project-m3-mystery-machine
source setup_env.sh
bash code/multilinguality_model/repro/reproduce_best_model.sh
```

For a detached run:

```
nohup bash code/multilinguality_model/repro/reproduce_best_model.sh > logs/reproduce_best_model.log 2>&1 &
```

## Run step by step

```
bash code/multilinguality_model/repro/00_install_repro_requirements.sh
bash code/multilinguality_model/repro/01_sync_repro_data.sh
bash code/multilinguality_model/repro/03_train_export_all_sources.sh
bash code/multilinguality_model/repro/04_merge_best_model.sh
bash code/multilinguality_model/repro/05_eval_best_model.sh
```

Train source models individually:

```
bash code/multilinguality_model/repro/02_train_export_source_model.sh v1
bash code/multilinguality_model/repro/02_train_export_source_model.sh v20-russian
bash code/multilinguality_model/repro/02_train_export_source_model.sh v20-hindi
```

## Output

- Final model: `checkpoints/multilinguality_model/best_model`
- Evaluation results: `runs/benchmark/results/multilinguality_model/best_model`

## Notes

- Models are saved locally only.
- Merge config: `code/multilinguality_model/configs/merging/best_model_ties.yaml`
- Evaluation uses vLLM and requires a GPU pod.
