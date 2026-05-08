# Objective: View the .arrow files first 3 rows to understand the data structure and content.
import argparse
import json
from pathlib import Path
from datetime import datetime
import re

from datasets import load_from_disk

parser = argparse.ArgumentParser()
parser.add_argument("--ds", required=True, help="Path to the dataset directory")
args = parser.parse_args()

HISTORY_FILE = Path(__file__).with_name("view_raw_data_requests.txt")
MAX_HISTORY_ENTRIES = 10


def append_request_history(request_path: str, history_file: Path, max_entries: int = 10) -> None:
    if history_file.exists():
        entries = [line.strip() for line in history_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        entries = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entries.append(f"{timestamp} | --ds {request_path}")
    entries = entries[-max_entries:]
    history_file.write_text("\n".join(entries) + "\n", encoding="utf-8")


append_request_history(args.ds, HISTORY_FILE, MAX_HISTORY_ENTRIES)


def normalize_for_filename(path_value: str) -> str:
    normalized = path_value.replace("\\", "/").strip("/")
    normalized = re.sub(r"[^A-Za-z0-9._/-]+", "_", normalized)
    normalized = normalized.replace("/", "__")
    return normalized or "dataset"


OUTPUT_DIR = Path(__file__).with_name("view_raw_data_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / f"{normalize_for_filename(args.ds)}.txt"

output_lines = []


def emit(text: str, file_text: str | None = None) -> None:
    print(text)
    output_lines.append(text if file_text is None else file_text)


raw_dir = Path(args.ds)
dataset = load_from_disk(raw_dir)
# shuffle the dataset to get a random sample of rows
dataset = dataset.shuffle(seed=42)

emit(f"Loaded: {raw_dir}")
emit(f"Rows: {len(dataset)}  |  Columns: {dataset.column_names}\n")

SEP = "─" * 72

start = 0
N = 10

end = start + N

start = max(0, start)
end = min(len(dataset), end)


for i in range(start, end):
    row = dataset[i]
    emit(f"{SEP}")
    emit(f"  Row {i}")
    emit(SEP)
    for col in dataset.column_names:
        value = row.get(col, "")
        if isinstance(value, (dict, list)):
            value_str = json.dumps(value, indent=2)
        else:
            value_str = str(value)
        emit(f"\033[32m{col.upper()}\033[0m\n", f"{col.upper()}\n")
        emit(f"{value_str}\n\n")

datatype_histogram = {}
key = "data_type"
for row in dataset:
    dt = row.get(key, "UNKNOWN")
    datatype_histogram[dt] = datatype_histogram.get(dt, 0) + 1

emit(f"{SEP}")
emit("Data Type Distribution:")
for dt, count in datatype_histogram.items():
    emit(f"  {dt:<15} {count}")

emit(f"Rows: {len(dataset)}  |  Columns: {dataset.column_names}\n")

OUTPUT_FILE.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
emit(f"Saved output to: {OUTPUT_FILE}")