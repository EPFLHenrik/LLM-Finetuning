python /mnt/data/parse_spain_boe_constitucion_espanola.py \
  --pdf /mnt/data/spain_boe_constitucion_espanola_consolidada.pdf \
  --output /mnt/data/multilingual/data/raw/spain_boe_constitucion_espanola/spain_boe_constitucion_espanola.jsonl

python multilingual/distillation/parse_spanish_regional_statutes.py \
  --input-dir multilingual/data/raw/spanish-extra

# Parse only one or a few files
python multilingual/distillation/parse_spanish_regional_statutes.py \
  --input-dir multilingual/data/raw/spanish-extra \
  --only cataluna galicia

# Test without writing
python multilingual/distillation/parse_spanish_regional_statutes.py \
  --input-dir multilingual/data/raw/spanish-extra \
  --dry-run