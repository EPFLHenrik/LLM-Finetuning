python /scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/distillation/parse_spain_boe_constitucion_espanola.py \
  --pdf /scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/data/raw/spanish-extra/spain_boe_constitucion_espanola_consolidada.pdf \
  --output /scratch/workspace/emma/standard-project-m2-mystery-machine/multilingual/data/raw/spain_boe_constitucion_espanola/spain_boe_constitucion_espanola.jsonl

# parse the regional statutes
python multilingual/distillation/parse_spanish_regional_statutes.py \
  --input-dir multilingual/data/raw/spanish-extra \
  --output-root multilingual/data/raw