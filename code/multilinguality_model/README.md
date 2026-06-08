# Multilingual component

This folder contains the multilingual dataset, training, and evaluation code used for the M3 submission. It focuses on reproducible data preparation, multilingual SFT training, merge workflows and vLLM-based evaluation.

## Overview

- `config.py`: shared prompts, constants, and task budgets.
- `paths.py`: repo-relative data and checkpoint path helpers.
- `utils.py`: helper functions shared by loaders, training, and evaluation.
- `assets/`: prompt templates for model generation and evaluation.
- `configs/`: SFT and merge configuration templates.
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

## Current dataset coverage

This package supports the following multilingual datasets and source categories:

- `include`
- `mgsm`
- `global_mmlu`
- `global_piqa`
- `ceval`
- `cmmlu`
- `milu`
- `mera`
- `head_qa`
- `polizia_di_stato`
- `quaderno_prefettizia`
- `spain_boe_legislation`
- `italian_civil_code`
- `rubq`
- `demre`
- `exams` (legacy/reproducibility)
- `mmlu_prox`
- `belebele` (loader present)

The pipeline uses processed JSONL files under `data/multilinguality/` and can fetch raw, processed, and distilled artifacts from the HF dataset repo if needed.

## Repository tree

```
multilinguality_model/
├── assets/                              # prompt templates for chat formatting
│   ├── chat_template.jinja
│   ├── chat_template_multilingual-thinking.jinja
│   ├── chat_template_multilingual-nonthinking.jinja
│   └── chat_template_multilingual-augmented.jinja
├── configs/                             # training and merging configuration templates
│   ├── merging/
│   │   └── best_model_ties.yaml         # merge configuration used by repro workflow
│   └── sft/
│       ├── v10.json
│       ├── v11.json
│       ├── v12.json
│       ├── v13.json
│       ├── v14.json
│       ├── v15.json
│       ├── v16.json
│       ├── v17_10k.json
│       ├── v17_20k.json
│       ├── v18.json
│       ├── v19_15k.json
│       ├── v19_20k.json
│       ├── v20-chinese.json
│       ├── v20-hindi.json
│       ├── v20-italian.json
│       ├── v20-russian.json
│       ├── v20-spanish.json
│       └── debug_test.json
├── data_pipeline/                       # data download, parse, preprocess, benchmark, distill
│   ├── distillation/
│   │   ├── distill_fallback.py          # main distillation pipeline
│   │   └── distill_1stage.py            # one-stage distillation path
│   ├── generation/
│   │   ├── filter_italian_mcq.py        # filter generated Italian MCQs
│   │   ├── generate_italian_civil_code_mcq.py
│   │   ├── generate_rubq_mcq.py
│   │   └── generate_spain_boe_mcq.py
│   ├── parsing/
│   │   ├── download_utils.py            # HF raw file download helper
│   │   ├── merge_boe_jsonl.py
│   │   ├── parse_ccse.py
│   │   ├── parse_demre.py
│   │   ├── parse_polizia_di_stato.py
│   │   ├── parse_quaderno_prefettizia.py
│   │   ├── parse_spain_boe_constitucion_espanola.py
│   │   └── parse_spanish_regional_statutes.py
│   └── preprocessing/
│       ├── build_benchmark.py           # benchmark split and manifest builder
│       ├── preprocess_ceval.py
│       ├── preprocess_cmmlu.py
│       ├── preprocess_cultsportqa.py
│       ├── preprocess_exams.py          # legacy reproduction preprocessing
│       ├── preprocess_global_mmlu.py
│       ├── preprocess_global_piqa.py
│       ├── preprocess_head_qa.py
│       ├── preprocess_include.py
│       ├── preprocess_mera.py
│       ├── preprocess_milu.py
│       ├── preprocess_mmlu_prox.py
│       ├── preprocess_polizia_di_stato.py
│       ├── preprocess_quaderno_prefettizia.py
│       └── run_preprocessing_all.py     # run all preprocessing
├── eval/                                # evaluation and scoring tools
│   ├── run_eval_vllm.py                 # vLLM inference evaluator
│   ├── score.py                         # scoring for generated completions
│   ├── aggregate_results.py             # merge multiple run summaries
│   ├── extract_answer.py                # answer extraction and normalization
│   ├── pass_at_k.py                     # pass@k analysis
│   ├── benchmarks.py                    # benchmark-specific logic
│   └── bash/                            # example shell wrappers
├── generation.py                        # common generation utilities
├── loaders/                             # dataset loaders for unified example creation
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
│   ├── quaderno_prefettizia_loader.py
│   ├── recarve_utils.py                 # shared split/recarve utilities
│   ├── rubq_loader.py
│   ├── spain_boe_legislation_loader.py
│   └── belebele_loader.py
├── merging/                             # model merge experiments and merge helpers
│   ├── full_slerp.py
│   ├── full_task_arithmetic.py
│   ├── full_ties.py
│   ├── LoRA_dare_linear.py
│   ├── LoRA_dare_ties.py
│   ├── LoRA_task_arithmetic.py
│   ├── LoRA_ties.py
│   └── utils/                           # shared merge argument helpers
├── repro/                               # reproduction workflow and HF sync scripts
│   ├── reproduce_best_model.sh          # full end-to-end reproduction script
│   ├── 00_install_repro_requirements.sh # install reproduction dependencies
│   ├── 01_sync_repro_data.sh           # download processed datasets and benchmarks from HF
│   ├── 02_train_export_source_model.sh # train a single source model
│   ├── 03_train_export_all_sources.sh  # train all source models sequentially
│   ├── 04_merge_best_model.sh          # merge the trained source models
│   ├── 05_eval_best_model.sh           # evaluate the merged model
│   └── requirements.txt                # reproduction-specific deps
├── train/                               # SFT data build, training, and export
│   ├── build_sft_data.py                # build mixed SFT datasets
│   ├── build_distilled_sft_data.py      # build SFT from distilled examples
│   ├── build_sft_augmented_mcq.py       # augmented MCQ mixture builder
│   ├── sft.py                           # main supervised fine-tuning entry point
│   ├── export.py                        # export adapter artifacts and tokenizer files
│   ├── export_full.py                   # export full checkpoint or adapter bundle
│   ├── check_sft_stats.py               # SFT dataset statistics checks
│   ├── inspect_formatted_sft.py         # inspect final chat-formatted SFT examples
│   └── push_base_tokenizer.py           # tokenizer sync helper
├── tests/                               # unit tests and data invariant checks
│   └── test_sft_data.py
├── __init__.py
├── config.py
├── paths.py
├── utils.py
└── README.md
```

## Data and HF fetch behavior

This package is designed to work from a clean repo clone with no large dataset artifacts committed.
The main local data root is `data/multilinguality/`.

- `data_pipeline/download_processed_data.py` pulls processed JSONL from `cs-552-2026-mystery-machine/multilingual`.
- `data_pipeline/download_distilled_data.py` pulls distilled JSONL from the same HF dataset repo.
- `data_pipeline/parsing/download_utils.py` can fetch raw source files into `data/multilinguality/raw/`.

Because `data/*` is ignored, the repo itself contains code only; data is downloaded locally on demand.

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
- Final model checkpoints are saved under `checkpoints/multilinguality_model/`.
- The merged best model is saved to `checkpoints/multilinguality_model/best_model`.
- Evaluation results are saved to `runs/benchmark/results/multilinguality_model/best_model`.
- Merge config used by the reproduction workflow: `code/multilinguality_model/configs/merging/best_model_ties.yaml`.
- Evaluation uses vLLM and requires a GPU pod.
