# spain BOE legislation
python -m multilingual.distillation.generate_spain_boe_mcq \
    --input_dir multilingual/data/raw/spain_boe_legislation \
    --output_base multilingual/data/processed \
    --concurrency 20