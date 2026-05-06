import json
from pathlib import Path
from pprint import pprint
from typing import Counter
from datasets import Dataset
from constants.global_variables import DatasetFieldNames

jsonl_path = Path("data/raw/validation_samples/safety.jsonl")
output_path = Path("data/processed/benchmark/validation_samples")

def preprocess():
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

    # Print some stats about the dataset
    stats = {
        "total_samples": len(dataset),
        "answer_distribution": dict(Counter(dataset[DatasetFieldNames.BENCHMARK_ANSWER])),
    }

    pprint(stats)
    
if __name__ == "__main__":
    preprocess()