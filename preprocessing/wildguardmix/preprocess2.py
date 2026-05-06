from datasets import load_from_disk
import shutil

SUBCAT_MCQ_DS = "data/processed/rlvr/wildguardmix/subcategory_mcq/test"
COMBO_DS = "data/processed/rlvr/wildguardmix/analysis_combo_mcq/test"
TEMP = SUBCAT_MCQ_DS + "_temp"
TEMP2 = COMBO_DS + "_temp"
BENCH_OUTPUT = "data/processed/benchmark/wildguardmix/subcategory_mcq"
BENCH_OUTPUT_COMBO = "data/processed/benchmark/wildguardmix/analysis_combo_mcq"

# Datasets are already shuffled, so we can just split by index.
ds = load_from_disk(SUBCAT_MCQ_DS)
ds.shuffle(seed=42)  # Shuffle again to ensure randomness before splitting, even if already shuffled
ds2 = load_from_disk(COMBO_DS)
ds2.shuffle(seed=42)  # Shuffle again to ensure randomness before splitting, even if already shuffled

bench_size = 1000
test_size = len(ds) - bench_size

print(f"Splitting dataset into test ({test_size} examples) and benchmark ({bench_size} examples)...")

bench_ds = ds.select(range(bench_size))
test_ds = ds.select(range(bench_size, bench_size + test_size))

bench_ds2 = ds2.select(range(bench_size))
test_ds2 = ds2.select(range(bench_size, bench_size + test_size))

print(f"Saving benchmark dataset to {BENCH_OUTPUT}...")
bench_ds.save_to_disk(BENCH_OUTPUT)
bench_ds2.save_to_disk(BENCH_OUTPUT_COMBO)

print(f"Saving test dataset to {TEMP}...")
test_ds.save_to_disk(TEMP)
test_ds2.save_to_disk(TEMP2)

del ds
del bench_ds
del test_ds
del ds2
del bench_ds2
del test_ds2

ds_new = load_from_disk(TEMP)
ds_new.save_to_disk(SUBCAT_MCQ_DS)

ds_new2 = load_from_disk(TEMP2)
ds_new2.save_to_disk(COMBO_DS)

shutil.rmtree(TEMP)
shutil.rmtree(TEMP2)
