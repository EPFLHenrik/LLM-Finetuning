# distilled example check
python -m multilingual.train.inspect_formatted_sft \
  --train_file multilingual/data/sft/v8/mixed/train_mixed.jsonl \
  --distilled \
  --idx 0

# non-distilled
python -m multilingual.train.inspect_formatted_sft \
  --train_file multilingual/data/sft/v8/mixed/train_mixed.jsonl \
  --idx 0