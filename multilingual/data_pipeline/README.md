# Data Pipeline

This folder contains the data preparation pipeline for the multilingual MCQA project. It covers four stages:

1. Parsing raw documents into structured intermediate records.
2. Generating synthetic MCQA data from source documents.
3. Preprocessing public and project-specific datasets into the common JSONL schema.
4. Distilling, filtering, and decontaminating training data.

The common processed dataset format is JSONL, with one example per line. Most processed records follow this schema:

- `id`: record identifier.
- `dataset`: dataset name.
- `language`: one of `russian`, `chinese`, `italian`, `spanish`, `hindi`.
- `split`: dataset split, usually `all`, `train`, `val`, `eval`, or `test`.
- `task_type`: usually `mcq`.
- `prompt`: multiple-choice question with answer options.
- `answer`: gold answer letter.
- `metadata`: source-specific metadata.

The preprocessing scripts are intended to be reproducible. Running them should overwrite only the specific output files they produce, not delete or rebuild entire folders.

## Folder tree

data_pipeline/
├── distillation/                                  # Teacher-model distillation, fallback handling, and contamination removal.
│   ├── old/                                      # Deprecated or archived scripts.
│   │   └── distill_2stage.py
│   ├── __init__.py                               # Marks the folder as a Python package.
│   ├── distill_1stage.py                         # Runs one-stage teacher distillation.
│   ├── distill_fallback.py                       # Runs fallback distillation for examples not handled by the main path.
│   ├── distill_fallback.sh                       # Shell wrapper for fallback distillation.
│   ├── distilled_counts.py                       # Counts and summarizes distilled examples.
│   ├── not_sure_collect.py                       # Collects examples marked uncertain or not usable by the teacher pipeline.
│   ├── remove_benchmark_contamination.py         # Removes examples overlapping with held-out benchmark source IDs.
│   ├── remove_gold_fallback.py                   # Removes or filters fallback examples that leak gold labels.
│   └── transfer.py                               # Transfers or reorganizes distilled data between folders/formats.
├── generation/                                   # Synthetic MCQA generation from legal/regional source material.
│   ├── __init__.py                               # Marks the folder as a Python package.
│   ├── filter_italian_mcq.py                     # Applies judge/model filtering to generated Italian MCQs.
│   ├── generate_italian_civil_code_mcq.py        # Generates Italian Civil Code MCQs.
│   ├── generate_rubq_mcq.py                      # Generates Russian RuBQ-based MCQs.
│   ├── generate_spain_boe_mcq.py                 # Generates Spanish BOE legislation MCQs.
│   └── run_mcq_generation.sh                     # Shell wrapper for MCQ generation jobs.
├── parsing/                                      # Parsers for raw PDFs, scraped files, and source-specific raw data.
│   ├── __init__.py                               # Marks the folder as a Python package.
│   ├── merge_spain_boe_legislation.py            # Merges generated/parsed Spanish BOE legislation outputs.
│   ├── parse_ccse.py                             # Parses CCSE Spanish citizenship exam material.
│   ├── parse_demre.py                            # Parses DEMRE-style Spanish-language exam material.
│   ├── parse_polizia_di_stato.py                 # Parses Polizia di Stato raw material into intermediate MCQ records.
│   ├── parse_quaderno_prefettizia.py             # Parses Quaderno Prefettizia raw material into intermediate MCQ records.
│   ├── parse_spain_boe_constitucion_espanola.py  # Parses the Spanish Constitution from BOE source material.
│   ├── parse_spanish_regional_statutes.py        # Parses Spanish regional statute PDFs/source files.
│   ├── run_boe_parsing.sh                        # Shell wrapper for BOE parsing scripts.
│   └── run_italian_mcq_parsing.sh                # Shell wrapper for Italian MCQ parsing scripts.
├── preprocessing/                                # Reproducible conversion of datasets into processed JSONL files.
│   ├── build_benchmark.py                        # Builds the held-out benchmark files from processed datasets.
│   ├── preprocess_ceval.py                       # Preprocesses CEval Chinese MCQA data.
│   ├── preprocess_cmmlu.py                       # Preprocesses CMMLU Chinese MCQA data.
│   ├── preprocess_cultsportqa.py                 # Preprocesses CultSportQA Chinese/Hindi sports MCQA data.
│   ├── preprocess_exams.py                       # Preprocesses deprecated EXAMS data; kept for reproducibility, not active use.
│   ├── preprocess_global_mmlu.py                 # Preprocesses Global MMLU for all target languages.
│   ├── preprocess_global_piqa.py                 # Preprocesses Global PIQA for all target languages.
│   ├── preprocess_head_qa.py                     # Preprocesses HEAD-QA Spanish and Italian medical MCQA data.
│   ├── preprocess_include.py                     # Preprocesses INCLUDE for all target languages.
│   ├── preprocess_mera.py                        # Preprocesses selected MERA Russian MCQA subsets.
│   ├── preprocess_milu.py                        # Preprocesses MILU Hindi MCQA data.
│   ├── preprocess_mmlu_prox.py                   # Preprocesses MMLU-ProX and creates safe train/eval splits.
│   ├── preprocess_polizia_di_stato.py            # Rebuilds Polizia di Stato initial parsed/cleaned JSONL.
│   ├── preprocess_quaderno_prefettizia.py        # Rebuilds Quaderno Prefettizia initial parsed/cleaned JSONL files.
│   └── run_preprocessing_all.py                  # Runs all active preprocessing scripts, optionally including EXAMS/benchmark build.
├── __init__.py                                   # Marks data_pipeline as a Python package.
└── README.md                                     # This file.

## Processed data conventions

The main processed data folder is:

- `multilingual/data/processed/`

Preprocessing scripts write directly into this folder. They overwrite their own target files only.

For judge-filtered datasets, the convention is:

- `*_initial_parsed.jsonl`: cleaned candidate pool produced by parsing/preprocessing.
- `*_filter_all.jsonl`: candidate pool after judge annotations are attached.
- `*_filter_rejected.jsonl`: judge-rejected records.
- `<dataset>.jsonl`: final judge-kept dataset.

For example:

- `polizia_di_stato_initial_parsed.jsonl` is produced by `preprocess_polizia_di_stato.py`.
- `polizia_di_stato_filtering_all.jsonl`, `polizia_di_stato_rejected.jsonl`, and `polizia_di_stato.jsonl` are kept from the judge-filtering stage.
- `quaderno_prefettizia_initial_parsed.jsonl` and `quaderno_prefettizia_economia_politica_initial_parsed.jsonl` are produced by `preprocess_quaderno_prefettizia.py`.
- `quaderno_prefettizia.jsonl` and `quaderno_prefettizia_economia_politica.jsonl` are final judge-kept files.

## Running preprocessing

Run all active preprocessing scripts:

    python -m multilingual.data_pipeline.preprocessing.run_preprocessing_all

Run all active preprocessing scripts and rebuild benchmarks:

    python -m multilingual.data_pipeline.preprocessing.run_preprocessing_all --build-benchmark

List the scripts that would be run without executing them:

    python -m multilingual.data_pipeline.preprocessing.run_preprocessing_all --dry-run

Include deprecated EXAMS preprocessing:

    python -m multilingual.data_pipeline.preprocessing.run_preprocessing_all --include-exams

EXAMS is kept for reproducibility but is not part of the active benchmark.

## Benchmark construction

The benchmark builder writes to:

- `multilingual/data/benchmarks/`

The current active benchmark contains:

- 500 Russian examples.
- 500 Chinese examples.
- 500 Hindi examples.
- 500 Italian examples.
- 500 Spanish examples.

Total:

- 2500 examples in `benchmark_all.jsonl`.

The benchmark components are:

- Russian: INCLUDE, Global MMLU, MERA.
- Chinese: INCLUDE, Global MMLU, CEval, CMMLU.
- Hindi: INCLUDE, Global MMLU, MILU.
- Italian: INCLUDE, Global MMLU.
- Spanish: INCLUDE, Global MMLU, HEAD-QA.

The benchmark builder also creates:

- by-language benchmark files.
- by-dataset benchmark files.
- a 500-example small benchmark.
- a separate MMLU-ProX hard benchmark.

Run benchmark construction directly:

    python -m multilingual.data_pipeline.preprocessing.build_benchmark

## Validation notes

Most preprocessing scripts were converted from notebooks and checked by comparing generated JSONL files against the existing processed files. Comparisons were done by exact relative filename and JSON-normalized content.

For same-name processed files, the intended check is:

- same number of records;
- same ordered JSON-normalized hash;
- same content and same order.

Files with different names are not compared automatically. This matters for judge-filtered datasets, where the preprocessing scripts generate `*_initial_parsed.jsonl`, while final accepted datasets are saved under shorter names such as `polizia_di_stato.jsonl`.

## Important cautions

Do not rerun generation or judge-filtering scripts unless intentionally regenerating synthetic datasets or judge decisions.

Do not delete `processed/` before running preprocessing. The preprocessing scripts are designed to overwrite their own files, not reconstruct every artifact in the processed folder.

Before major changes, create a backup of `multilingual/data/processed/` and `multilingual/data/benchmarks/`.
