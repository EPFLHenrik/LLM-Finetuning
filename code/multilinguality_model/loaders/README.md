# Loaders

This folder contains dataset loader modules for the multilingual pipeline.
Each loader converts processed JSONL files into a unified example format used by SFT build scripts.

All loaders return examples with these fields:
- `id`
- `source_id`
- `dataset`
- `language`
- `task_type`
- `prompt`
- `answer`
- `metadata`

Files in this folder cover core multilingual datasets such as INCLUDE, Global MMLU, Global PIQA, MILU, CEval, HEAD-QA, MERA, and others.
Most modules read local processed files under `data/multilinguality/processed/` and some also support fallback loading from Hugging Face when local files are missing. Run `data_pipeline/download_processed_data.py` to fetch them from HF and save them under `data/multilinguality/processed/` before using the loaders.
