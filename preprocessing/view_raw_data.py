# Objective: View the .arrow files first 3 rows to understand the data structure and content.
import json
from pathlib import Path

from datasets import load_from_disk

raw_dir = Path(
    "data/processed/rlvr/wildguardmix/analysis_combo_mcq/test"
)  # Change this path to the specific dataset split you want to view
dataset = load_from_disk(raw_dir)
# shuffle the dataset to get a random sample of rows
dataset = dataset.shuffle(seed=42)

print(f"Loaded: {raw_dir}")
print(f"Rows: {len(dataset)}  |  Columns: {dataset.column_names}\n")

SEP = "─" * 72

start = 500
N = 100

end = start + N

start = max(0, start)
end = min(len(dataset), end)


for i in range(start, end):
    row = dataset[i]
    print(f"{SEP}")
    print(f"  Row {i}")
    print(SEP)
    for col in dataset.column_names:
        value = row.get(col, "")
        if isinstance(value, (dict, list)):
            value_str = json.dumps(value, indent=2)[:500]  # Truncate long values
        else:
            value_str = str(value)
        print(f"\033[32m{col.upper()}\033[0m\n")
        print(f"{value_str}\n\n")

datatype_histogram = {}
key = "data_type"
for row in dataset:
    dt = row.get(key, "UNKNOWN")
    datatype_histogram[dt] = datatype_histogram.get(dt, 0) + 1

print(f"{SEP}")
print(f"Data Type Distribution:")
for dt, count in datatype_histogram.items():
    print(f"  {dt:<15} {count}")