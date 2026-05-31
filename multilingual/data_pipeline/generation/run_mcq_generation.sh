#!/bin/bash
# multilingual/data_pipeline/generation/run_mcq_generation.sh
# Example commands for teacher-based MCQ generation.
# Before running:
#   export CSCS_SERVING_API="sk-rc-..."

set -e


# ============================================================
# 1. Italian Civil Code MCQ generation
# ============================================================

# Dry run
python -m multilingual.data_pipeline.generation.generate_italian_civil_code_mcq \
    --output_dir multilingual/data/processed/italian_civil_code \
    --concurrency 5 \
    --max_valid 20 \
    --dry_run

# Full run
python -m multilingual.data_pipeline.generation.generate_italian_civil_code_mcq \
    --output_dir multilingual/data/processed/italian_civil_code \
    --concurrency 20 \
    --max_valid 2000


# ============================================================
# 2. Spain BOE legislation MCQ generation
# ============================================================

# Dry run
python -m multilingual.data_pipeline.generation.generate_spain_boe_mcq \
    --input_dir multilingual/data/raw/spain_boe_legislation \
    --output_base multilingual/data/processed \
    --concurrency 5 \
    --dry_run

# Full run
python -m multilingual.data_pipeline.generation.generate_spain_boe_mcq \
    --input_dir multilingual/data/raw/spain_boe_legislation \
    --output_base multilingual/data/processed \
    --concurrency 20