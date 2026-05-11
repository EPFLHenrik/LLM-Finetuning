from datasets import load_from_disk
import shutil

SUBCAT_MCQ_DS = "data/processed/sft/wildguardmix/subcategory_mcq_answers_only/test"
COMBO_DS = "data/processed/sft/wildguardmix/safety_combo_mcq_answers_only/test"
BINARY_SFT = "data/processed/sft/wildguardmix/binary_choice/test"
TEMP = SUBCAT_MCQ_DS + "_temp"
TEMP2 = COMBO_DS + "_temp"
TEMP3 = BINARY_SFT + "_temp"
BENCH_OUTPUT = "data/processed/benchmark/wildguardmix/subcategory_mcq"
BENCH_OUTPUT_COMBO = "data/processed/benchmark/wildguardmix/safety_combo_mcq"
BENCH_OUTPUT_BINARY = "data/processed/benchmark/wildguardmix/binary_choice"

# Datasets are already shuffled, so we can just split by index.
ds = load_from_disk(SUBCAT_MCQ_DS)
ds.shuffle(seed=42)  # Shuffle again to ensure randomness before splitting, even if already shuffled
ds2 = load_from_disk(COMBO_DS)
ds2.shuffle(seed=42)  # Shuffle again to ensure randomness before splitting, even if already shuffled
ds3 = load_from_disk(BINARY_SFT)
ds3.shuffle(seed=42)  # Shuffle again to ensure randomness before splitting, even if

bench_size = 150
test_size = len(ds) - bench_size

bench_size2 = 150
test_size2 = len(ds2) - bench_size2

bench_size3 = 750
test_size3 = len(ds3) - bench_size3

print(f"Splitting dataset into test ({test_size} examples) and benchmark ({bench_size} examples)...")

bench_ds = ds.select(range(bench_size))
test_ds = ds.select(range(bench_size, bench_size + test_size))
bench_ds2 = ds2.select(range(bench_size2))
test_ds2 = ds2.select(range(bench_size2, bench_size2 + test_size2))
bench_ds3 = ds3.select(range(bench_size3))
test_ds3 = ds3.select(range(bench_size3, bench_size3 + test_size3))

bench_ds.save_to_disk(BENCH_OUTPUT)
test_ds.save_to_disk(TEMP)
bench_ds2.save_to_disk(BENCH_OUTPUT_COMBO)
test_ds2.save_to_disk(TEMP2)
bench_ds3.save_to_disk(BENCH_OUTPUT_BINARY)
test_ds3.save_to_disk(TEMP3)

print("Summary:")
print(f"  Subcategory MCQ: original test set had {len(ds)} examples, split into {len(bench_ds)} benchmark examples and {len(test_ds)} test examples.")
print(f"  Safety Combo MCQ: original test set had {len(ds2)} examples, split into {len(bench_ds2)} benchmark examples and {len(test_ds2)} test examples.")
print(f"  Binary Choice SFT: original test set had {len(ds3)} examples, split into {len(bench_ds3)} benchmark examples and {len(test_ds3)} test examples.")

del ds
del bench_ds
del test_ds
del ds2
del bench_ds2
del test_ds2
del ds3
del bench_ds3
del test_ds3

ds_new = load_from_disk(TEMP)
ds_new.save_to_disk(SUBCAT_MCQ_DS)

ds_new2 = load_from_disk(TEMP2)
ds_new2.save_to_disk(COMBO_DS)

ds_new3 = load_from_disk(TEMP3)
ds_new3.save_to_disk(BINARY_SFT)

shutil.rmtree(TEMP)
shutil.rmtree(TEMP2)
shutil.rmtree(TEMP3)
