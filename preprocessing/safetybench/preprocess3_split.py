from constants import global_variables
from datasets import load_from_disk



RATIONALES_DS_PATH = "data/processed/sft/safetybench/rationales"
COT_DS_PATH = "data/processed/sft/safetybench/CoT"

TARGET_TRAIN_RATIONALES_PATH = RATIONALES_DS_PATH + "/train"
TARGET_TEST_RATIONALES_PATH = RATIONALES_DS_PATH + "/test"
TARGET_TRAIN_COT_PATH = COT_DS_PATH + "/train"
TARGET_TEST_COT_PATH = COT_DS_PATH + "/test"
TARGET_BENCHMARK_PATH = "data/processed/benchmark/safetybench"

# Datasets are already shuffled, so we can just split by index.
rationales_ds = load_from_disk(RATIONALES_DS_PATH)
cot_ds = load_from_disk(COT_DS_PATH)

bench_size = 1000
test_size = 100
train_size = len(rationales_ds) - bench_size - test_size

print(f"Splitting datasets into train ({train_size} examples), test ({test_size} examples), and benchmark ({bench_size} examples)...")
rationales_train = rationales_ds.select(range(0, train_size))
rationales_test = rationales_ds.select(range(train_size, train_size + test_size))
rationales_bench = rationales_ds.select(range(train_size + test_size, len(rationales_ds)))

cot_train = cot_ds.select(range(0, train_size))
cot_test = cot_ds.select(range(train_size, train_size + test_size))


print("Safety Check:")
print(f"Theoretical Train size: {train_size}, actual: {len(rationales_train)}")
print(f"Theoretical Test size: {test_size}, actual: {len(rationales_test)}")
print(f"Theoretical Bench size: {bench_size}, actual: {len(rationales_bench)}")

rationales_train.save_to_disk(TARGET_TRAIN_RATIONALES_PATH)
rationales_test.save_to_disk(TARGET_TEST_RATIONALES_PATH)

cot_train.save_to_disk(TARGET_TRAIN_COT_PATH)
cot_test.save_to_disk(TARGET_TEST_COT_PATH)

# remove [global_variables.DatasetFieldNames.SFT_TARGET, global_variables.DistillationFieldNames.TEACHER_MODEL]
rationales_bench = rationales_bench.remove_columns([global_variables.DatasetFieldNames.SFT_TARGET, global_variables.DistillationFieldNames.TEACHER_MODEL])
rationales_bench.save_to_disk(TARGET_BENCHMARK_PATH)





