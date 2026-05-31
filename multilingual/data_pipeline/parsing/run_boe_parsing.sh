#!/bin/bash
# multilingual/data_pipeline/parsing/run_boe_parsing.sh

# ============================================================
# 1. Parse the Spanish Constitution
# ============================================================

python -m multilingual.data_pipeline.parsing.parse_spain_boe_constitucion_espanola \
  --pdf multilingual/data/raw/spanish-extra/spain_boe_constitucion_espanola_consolidada.pdf \
  --output multilingual/data/raw/spain_boe_legislation/spain_boe_constitucion_espanola.jsonl


# ============================================================
# 2. Parse the regional autonomy statutes
# ============================================================

python -m multilingual.data_pipeline.parsing.parse_spanish_regional_statutes \
  --input-dir multilingual/data/raw/spanish-extra \
  --output-root multilingual/data/raw