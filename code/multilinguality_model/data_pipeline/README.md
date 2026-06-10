# Data Pipeline

This folder contains dataset preparation and synthetic MCQ generation code for the multilingual pipeline.

It is organized into four main sections:
- `parsing/`: parse raw PDFs and source files into intermediate JSONL.
- `generation/`: generate and filter MCQ candidates from source material.
- `preprocessing/`: convert public datasets and parsed sources into standardized processed files.
- `distillation/`: create distilled reasoning examples and teacher-traced data.

## Key outputs

- `data/multilinguality/processed/`: standardized processed dataset JSONL.
- `data/multilinguality/distilled/`: distilled reasoning examples.
- `data/multilinguality/benchmarks/`: benchmark splits and evaluation subsets.
- `data/multilinguality/raw/`: raw source inputs.

## Notes

- The JSONL format is the main intermediate format.
- Each preprocessing script is intended to overwrite only its own target files.
- This README covers the whole data pipeline; subfolders do not each have separate README files.
- For reproduction, running this pipeline is **not** required. SFT data mixtures and benchmarks are fetched from HF by `repro/01_sync_repro_data.sh`. Full processed datasets can also be pulled separately via `data_pipeline/download_processed_data.py` if needed.

## Run the pipeline

Run all active preprocessing scripts:

```bash
python -m multilinguality_model.data_pipeline.preprocessing.run_preprocessing_all
```

Rebuild benchmarks as well:

```bash
python -m multilinguality_model.data_pipeline.preprocessing.run_preprocessing_all --build-benchmark
```

Dry-run without executing:

```bash
python -m multilinguality_model.data_pipeline.preprocessing.run_preprocessing_all --dry-run
```