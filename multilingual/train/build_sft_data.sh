# debug
python -m multilingual.train.build_sft_data --debug

# full
python -m multilingual.train.build_sft_data \
  --max_per_language 1000 \
  --max_per_dataset 5000 \
  --overwrite

# then, test
pytest multilingual/tests/test_sft_data.py -v