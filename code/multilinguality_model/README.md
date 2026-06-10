# Multilingual component

This folder contains the multilingual dataset, training, and evaluation code used for the M3 submission. It focuses on reproducible data preparation, multilingual SFT training, merge workflows and vLLM-based evaluation.

## Overview

- `config.py`: shared prompts, constants, and task budgets.
- `paths.py`: repo-relative data and checkpoint path helpers.
- `utils.py`: helper functions shared by loaders, training, and evaluation.
- `assets/`: prompt templates for model generation and evaluation.
- `configs/`: merge configuration template.
- `data_pipeline/`: download, parse, preprocess, benchmark, and distill datasets.
- `loaders/`: dataset loaders that convert processed JSONL into a unified example schema.
- `train/`: SFT mixture build, training entry points, and export utilities.
- `eval/`: evaluation and scoring logic, including vLLM inference.
- `merging/`: model merging experiments and PEFT merge helpers.
- `repro/`: the reproduction workflow that fetches data from HF, starts trainings, merges models, and evaluates results.

## Reproducibility and `repro/`

The `repro/` folder is the main reproduction entry point for this component.
It is the recommended path when you want to reproduce the current best multilingual model from scratch.

Use the all-in-one reproduction script:

```bash
cd /scratch/workspace/emma/standard-project-m3-mystery-machine
source setup_env.sh
bash code/multilinguality_model/repro/reproduce_best_model.sh
```

This script performs:
- install reproduction requirements,
- sync processed datasets and benchmarks from Hugging Face,
- train the three source models,
- merge them into the final best model,
- evaluate the merged model.

### Repro scripts

- `repro/00_install_repro_requirements.sh` — install reproduction dependencies.
- `repro/01_sync_repro_data.sh` — download processed datasets and benchmarks from HF.
- `repro/02_train_export_source_model.sh` — train one source model and export it.
- `repro/03_train_export_all_sources.sh` — train all source models sequentially.
- `repro/04_merge_best_model.sh` — merge the trained source models.
- `repro/05_eval_best_model.sh` — evaluate the merged model.
- `repro/requirements.txt` — reproduction-specific Python dependencies.

## Data and Hugging Face storage

No dataset files are committed to this repository. The `data/` folder is git-ignored and will be empty on a fresh clone. All data artifacts live on the Hugging Face dataset repo:

**[`cs-552-2026-mystery-machine/multilingual`](https://huggingface.co/datasets/cs-552-2026-mystery-machine/multilingual)**

| Folder | Contents |
|---|---|
| [`processed_data/`](https://huggingface.co/datasets/cs-552-2026-mystery-machine/multilingual/tree/main/processed_data) | Preprocessed JSONL files, one per source dataset |
| [`benchmarks/`](https://huggingface.co/datasets/cs-552-2026-mystery-machine/multilingual/tree/main/benchmarks) | Evaluation benchmark splits used by the eval pipeline |
| [`sft_data/`](https://huggingface.co/datasets/cs-552-2026-mystery-machine/multilingual/tree/main/sft_data) | Final SFT mixtures used for training |
| [`raw_data/`](https://huggingface.co/datasets/cs-552-2026-mystery-machine/multilingual/tree/main/raw_data) | Raw source files before preprocessing |
| [`distilled_data/`](https://huggingface.co/datasets/cs-552-2026-mystery-machine/multilingual/tree/main/distilled_data) | Distilled reasoning traces (generated but not used in the final SFT mixture) |

The `repro/` scripts handle syncing the relevant subsets locally before training and evaluation:
- `repro/01_sync_repro_data.sh` fetches `processed_data/`, `benchmarks/`, and `sft_data/` into `data/multilinguality/`.
- `data_pipeline/download_processed_data.py` and `data_pipeline/download_distilled_data.py` can fetch individual subsets on demand. The main local data root is `data/multilinguality/`.


## Repository tree

```
multilinguality_model/
├── .gitkeep                          
├── README.md
├── __init__.py
├── config.py
├── paths.py
├── utils.py
├── setup_env.sh                       # environment setup for reproduction/local workflow
├── assets/                            # prompt templates and chat formatting assets
│   ├── README.md
│   ├── chat_template.jinja
│   ├── chat_template_multilingual-augmented.jinja
│   ├── chat_template_multilingual-nonthinking.jinja
│   ├── chat_template_multilingual-thinking.jinja
│   └── generation_config.json
├── configs/
│   └── merging/
│       └── best_model_ties.yaml       # merge configuration used by repro workflow
├── data_pipeline/                     # data download, parsing, preprocessing, distillation
│   ├── README.md
│   ├── __init__.py
│   ├── download_distilled_data.py
│   ├── download_processed_data.py
│   ├── distillation/
│   │   ├── __init__.py
│   │   └── distill_fallback.py        # main distillation pipeline (not used for final selected models)
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── filter_italian_mcq.py
│   │   ├── generate_italian_civil_code_mcq.py
│   │   ├── generate_rubq_mcq.py
│   │   └── generate_spain_boe_mcq.py
│   ├── parsing/
│   │   ├── __init__.py
│   │   ├── download_utils.py
│   │   ├── merge_boe_jsonl.py
│   │   ├── parse_ccse.py
│   │   ├── parse_demre.py
│   │   ├── parse_polizia_di_stato.py
│   │   ├── parse_quaderno_prefettizia.py
│   │   ├── parse_spain_boe_constitucion_espanola.py
│   │   └── parse_spanish_regional_statutes.py
│   └── preprocessing/
│       ├── build_benchmark.py
│       ├── preprocess_ceval.py
│       ├── preprocess_cmmlu.py
│       ├── preprocess_cultsportqa.py
│       ├── preprocess_exams.py
│       ├── preprocess_global_mmlu.py
│       ├── preprocess_global_piqa.py
│       ├── preprocess_head_qa.py
│       ├── preprocess_include.py
│       ├── preprocess_mera.py
│       ├── preprocess_milu.py
│       ├── preprocess_mmlu_prox.py
│       ├── preprocess_polizia_di_stato.py
│       ├── preprocess_quaderno_prefettizia.py
│       └── run_preprocessing_all.py
├── eval/                              # evaluation and scoring logic, including vLLM runner
│   ├── README.md
│   ├── __init__.py
│   ├── aggregate_results.py
│   ├── benchmarks.py
│   ├── eval_utils.py
│   ├── evaluation.py
│   ├── extract_answer.py
│   ├── pass_at_k.py
│   ├── run_eval_vllm.py
│   ├── score.py
│   └── bash/
│       ├── eval_do_sample.sh
│       ├── eval_group_models_hugo.sh
│       ├── eval_loop.sh
│       └── eval_merged.sh
├── generation.py                     # shared generation utilities
├── loaders/                          # dataset loaders for unified example creation
│   ├── README.md
│   ├── __init__.py
│   ├── ccse_loader.py
│   ├── ceval_loader.py
│   ├── cmmlu_loader.py
│   ├── cultsportqa_loader.py
│   ├── demre_loader.py
│   ├── exams_loader.py
│   ├── global_mmlu_loader.py
│   ├── global_piqa_loader.py
│   ├── head_qa_loader.py
│   ├── include_loader.py
│   ├── italian_civil_code_loader.py
│   ├── mera_loader.py
│   ├── mgsm_loader.py
│   ├── milu_loader.py
│   ├── mmlu_prox_loader.py
│   ├── polizia_di_stato_loader.py
│   ├── quaderno_prefettizia_economia_politica_loader.py
│   ├── quaderno_prefettizia_loader.py
│   ├── recarve_utils.py              # shared split/recarve utilities
│   ├── rubq_loader.py
│   ├── spain_boe_legislation_loader.py
│   └── belebele_loader.py
├── merging/                          # model merge experiments and PEFT merge helpers
│   ├── README.md
│   ├── full_slerp.py
│   ├── full_task_arithmetic.py
│   ├── full_ties.py
│   ├── LoRA_dare_linear.py
│   ├── LoRA_dare_ties.py
│   ├── LoRA_task_arithmetic.py
│   ├── LoRA_ties.py
│   └── utils/
│       ├── parse_arguments.py
│       └── peft_merge.py
├── repro/                            # reproduction workflow and HF sync scripts
│   ├── README.md
│   ├── reproduce_best_model.sh
│   ├── 00_install_repro_requirements.sh
│   ├── 01_sync_repro_data.sh
│   ├── 02_train_export_source_model.sh
│   ├── 03_train_export_all_sources.sh
│   ├── 04_merge_best_model.sh
│   ├── 05_eval_best_model.sh
│   └── requirements.txt
├── train/                            # SFT dataset building, training, and export
│   ├── build_distilled_sft_data.py
│   ├── build_sft_augmented_mcq.py
│   ├── build_sft_data.py
│   ├── export.py
│   ├── export_full.py
│   └── sft.py
└── tests/                            # unit tests and data invariant checks
    ├── README.md
    └── test_sft_data.py
```


## Evaluation

Evaluation now uses the vLLM pipeline in `eval/run_eval_vllm.py`.
This is the current inference path for offline benchmark evaluation.

The evaluator:
- builds chat prompts from raw MCQ prompts using the shared system prompt,
- sends batched requests to vLLM,
- extracts the final `\boxed{...}` answer,
- computes accuracy and diagnostic metrics.

## Training and SFT data building

The training folder constructs SFT mixtures from processed and distilled examples.
Key build scripts:

- `train/build_sft_data.py`: build mixed SFT datasets from processed source datasets.
- `train/build_distilled_sft_data.py`: build SFT datasets from distilled reasoning examples.
- `train/build_sft_augmented_mcq.py`: build augmented MCQ mixtures with distractor shuffling.
- `train/sft.py`: main training entry point for adapter/SFT runs.

The final SFT data format is chat-style JSONL compatible with the prompt templates in `assets/`.

## Notes

- `repro/` is the primary reproduction entry point for this component.
- Upon following the reproduction scripts in `repro/`, model checkpoints are saved under `checkpoints/multilinguality_model/`.
- The merged best model is saved to `checkpoints/multilinguality_model/best_model`.
- Merge config used by the reproduction workflow: `code/multilinguality_model/configs/merging/best_model_ties.yaml`.
- Fetched processed data, SFT mixtures, and benchmarks are saved under `data/multilinguality/`.
- Evaluation uses vLLM and requires a GPU pod.
