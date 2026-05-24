import json
from pathlib import Path
from pprint import pprint
from collections import Counter

from datasets import Dataset
from constants.global_variables import DatasetFieldNames

RAW_VALIDATION_DIR = Path("data/raw/validation_samples")
OUTPUT_BASE_PATH = Path("data/processed/benchmark/validation_samples")

# Keep the task order explicit and stable for reproducible outputs.
TASK_NAMES = ["general_knowledge", "math", "multilingual", "safety"]


def preprocess_task(task_name: str):
    jsonl_path = RAW_VALIDATION_DIR / f"{task_name}.jsonl"
    output_path = OUTPUT_BASE_PATH / task_name

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            records.append({
                DatasetFieldNames.BENCHMARK_INPUT: item["prompt"],
                DatasetFieldNames.BENCHMARK_ANSWER: item["answer"],
                # Keep other fields if needed
            })

    dataset = Dataset.from_list(records)
    dataset.save_to_disk(output_path)

    return {
        "task": task_name,
        "total_samples": len(dataset),
        "answer_distribution": dict(Counter(dataset[DatasetFieldNames.BENCHMARK_ANSWER])),
    }


def preprocess():
    for task_name in TASK_NAMES:
        stats = preprocess_task(task_name)
        pprint(stats)
    
if __name__ == "__main__":
    preprocess()