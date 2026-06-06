#!/usr/bin/env bash
# multilingual/data_pipeline/generation/run_mcq_generation.sh
#
# Reference commands for teacher-based MCQ generation and filtering.
#
# This file documents how the project-specific generated datasets were created.
# Commands are intentionally commented out to avoid accidentally launching
# expensive API jobs. Copy and run the relevant command manually when needed.
#
# Before running any command:
#   export CSCS_SERVING_API="sk-rc-..."
#
# Run from repository root:
#   cd /scratch/workspace/emma/standard-project-m2-mystery-machine


# ============================================================
# 1. Italian Civil Code MCQ generation
# ============================================================
# Output folder:
#   multilingual/data/processed/italian_civil_code/
#
# Main output files:
#   italian_civil_code_all.jsonl
#   italian_civil_code_generated.jsonl
#   italian_civil_code_fail.jsonl


# Dry run:
python -m multilingual.data_pipeline.generation.generate_italian_civil_code_mcq \
  --output_dir multilingual/data/processed/italian_civil_code \
  --concurrency 5 \
  --max_valid 20 \
  --dry_run \
  --debug_n 20

# Full run:
python -m multilingual.data_pipeline.generation.generate_italian_civil_code_mcq \
  --output_dir multilingual/data/processed/italian_civil_code \
  --concurrency 20 \
  --max_valid 2000


# ============================================================
# 2. Spain BOE legislation MCQ generation
# ============================================================
#
# Source folder:
#   multilingual/data/raw/spain_boe_legislation/
#
# Output folder:
#   multilingual/data/processed/spain_boe_legislation/
#
# Main output files:
#   <source_file>_all.jsonl
#   <source_file>_generated.jsonl
#   <source_file>_fail.jsonl
#   spain_boe_legislation_generated.jsonl


# Dry run:
python -m multilingual.data_pipeline.generation.generate_spain_boe_mcq \
  --input_dir multilingual/data/raw/spain_boe_legislation \
  --output_base multilingual/data/processed \
  --concurrency 5 \
  --dry_run \
  --debug_n 5


# Full run:
python -m multilingual.data_pipeline.generation.generate_spain_boe_mcq \
  --input_dir multilingual/data/raw/spain_boe_legislation \
  --output_base multilingual/data/processed \
  --concurrency 20


# ============================================================
# 3. RuBQ MCQ generation
# ============================================================
#
# Input file:
#   multilingual/data/raw/rubq/RuBQ_2.0.json
#
# Output folder:
#   multilingual/data/processed/rubq/
#
# Main output files:
#   rubq_all.jsonl
#   rubq_generated.jsonl
#   rubq_fail.jsonl


# Dry run:
python -m multilingual.data_pipeline.generation.generate_rubq_mcq \
  --input multilingual/data/raw/rubq/RuBQ_2.0.json \
  --output_dir multilingual/data/processed/rubq \
  --concurrency 5 \
  --dry_run \
  --debug_n 20

# Full run:
python -m multilingual.data_pipeline.generation.generate_rubq_mcq \
  --input multilingual/data/raw/rubq/RuBQ_2.0.json \
  --output_dir multilingual/data/processed/rubq \
  --concurrency 20 \
  --max_valid 1000


# ============================================================
# 4. Italian generated/parsed MCQ judge filtering
# ============================================================
#
# This stage filters candidate Italian MCQs from parsed/generated source files.
# It produces final judge-kept datasets and rejected/all-filtered audit files.
#
# Polizia di Stato:
#   input:
#     polizia_di_stato_initial_parsed.jsonl
#   outputs:
#     polizia_di_stato.jsonl
#     polizia_di_stato_rejected.jsonl
#     polizia_di_stato_filtering_all.jsonl


# Dry run:
python -m multilingual.data_pipeline.generation.filter_italian_mcq \
  --input multilingual/data/processed/polizia_di_stato/polizia_di_stato_initial_parsed.jsonl \
  --output_dir multilingual/data/processed/polizia_di_stato \
  --output_prefix polizia_di_stato \
  --dry_run \
  --debug_n 20

# Full run:
python -m multilingual.data_pipeline.generation.filter_italian_mcq \
  --input multilingual/data/processed/polizia_di_stato/polizia_di_stato_initial_parsed.jsonl \
  --output_dir multilingual/data/processed/polizia_di_stato \
  --output_prefix polizia_di_stato \
  --concurrency 20


# Quaderno Prefettizia:
#   input:
#     quaderno_prefettizia_initial_parsed.jsonl
#   outputs:
#     quaderno_prefettizia.jsonl
#     quaderno_prefettizia_rejected.jsonl
#     quaderno_prefettizia_filter_all.jsonl


# Dry run:
python -m multilingual.data_pipeline.generation.filter_italian_mcq \
  --input multilingual/data/processed/quaderno_prefettizia/quaderno_prefettizia_initial_parsed.jsonl \
  --output_dir multilingual/data/processed/quaderno_prefettizia \
  --output_prefix quaderno_prefettizia \
  --dry_run \
  --debug_n 20

# Full run:
python -m multilingual.data_pipeline.generation.filter_italian_mcq \
  --input multilingual/data/processed/quaderno_prefettizia/quaderno_prefettizia_initial_parsed.jsonl \
  --output_dir multilingual/data/processed/quaderno_prefettizia \
  --output_prefix quaderno_prefettizia \
  --concurrency 20


# Quaderno Prefettizia Economia Politica:
#   input:
#     quaderno_prefettizia_economia_politica_initial_parsed.jsonl
#   outputs:
#     quaderno_prefettizia_economia_politica.jsonl
#     quaderno_prefettizia_economia_politica_rejected.jsonl
#     quaderno_prefettizia_economia_politica_filter_all.jsonl


# Dry run:
python -m multilingual.data_pipeline.generation.filter_italian_mcq \
  --input multilingual/data/processed/quaderno_prefettizia/quaderno_prefettizia_economia_politica_initial_parsed.jsonl \
  --output_dir multilingual/data/processed/quaderno_prefettizia \
  --output_prefix quaderno_prefettizia_economia_politica \
  --dry_run \
  --debug_n 20

# Full run:
python -m multilingual.data_pipeline.generation.filter_italian_mcq \
  --input multilingual/data/processed/quaderno_prefettizia/quaderno_prefettizia_economia_politica_initial_parsed.jsonl \
  --output_dir multilingual/data/processed/quaderno_prefettizia \
  --output_prefix quaderno_prefettizia_economia_politica \
  --concurrency 20
