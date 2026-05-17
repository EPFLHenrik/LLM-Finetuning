# debug
python -m multilingual.train.build_sft_data --debug

# full
python -m multilingual.train.build_sft_data --overwrite

# then, test
pytest multilingual/tests/test_sft_data.py -v