#!/bin/bash
# multilingual/data_pipeline/parsing/run_italian_mcq_parsing.sh


# ============================================================
# 1. Parse Quaderno Prefettizia MCQs
# ============================================================

python -m multilingual.data_pipeline.parsing.parse_quaderno_prefettizia \
  --pdf multilingual/data/raw/italian_mcq_pdfs/1.quaderno-delle-domande-A.pdf \
  --output multilingual/data/processed/quaderno_prefettizia/quaderno_prefettizia.jsonl \
  --dataset quaderno_prefettizia \
  --source quaderno_prefettizia \
  --seed 42


# ============================================================
# 2. Parse Polizia di Stato MCQs
# ============================================================

python -m multilingual.data_pipeline.parsing.parse_polizia_di_stato \
  --pdf multilingual/data/raw/italian_mcq_pdfs/2.polizia-di-stato-banca-dati.pdf \
  --output multilingual/data/processed/polizia_di_stato/polizia_di_stato.jsonl \
  --dataset polizia_di_stato \
  --source polizia_di_stato \
  --seed 42