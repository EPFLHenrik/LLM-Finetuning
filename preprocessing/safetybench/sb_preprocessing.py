import json
from pathlib import Path
from typing import List
import random
from constants.global_variables import DatasetFieldNames, Random

from datasets import Dataset

ROOT = Path(__file__).resolve().parents[2]
questions_path = ROOT / "data" / "raw" / "safetybench" / "en" / "test_en.json"
answers_path = ROOT / "data" / "raw" / "safetybench" / "en" / "test_answers_en.json"
output_path = ROOT / "data" / "processed" / "sft" / "safetybench" / "mcq"
benchmark_output_path = ROOT / "data" / "processed" / "benchmark" / "safetybench"

with questions_path.open("r", encoding="utf-8") as f:
    questions = json.load(f)

with answers_path.open("r", encoding="utf-8") as f:
    answers = json.load(f)

if len(questions) != len(answers):
    raise ValueError(
        f"Mismatched question/answer counts: {len(questions)} questions vs {len(answers)} answers"
    )

letters = ["A", "B", "C", "D", "E"]

def format_mcq_prompt(question_prompt, options: List[str], answer_index: int):
    return f"{question_prompt}\n\n" \
           + "\n".join([f"{letters[idx]}) {option}" for idx, option in enumerate(options)])


# Combine questions and answers into a single list of dictionaries on the same index
bench_size = 1000
test_size = 100
train_size = len(questions) - bench_size - test_size

# Deterministic shuffle based on global seed
shuffled_indices = list(range(len(questions)))
random.Random(Random.SEED).shuffle(shuffled_indices)


output_path.parent.mkdir(parents=True, exist_ok=True)
benchmark_output_path.parent.mkdir(parents=True, exist_ok=True)

offset = 0
for split, start_idx in [("train", train_size), ("test", test_size)]:
    ds = []
    for i in range(start_idx):
        original_idx = shuffled_indices[offset + i]
        question = questions[original_idx]
        answer = answers.get(str(original_idx), "No answer found")
        ds.append({
            DatasetFieldNames.SFT_INPUT: format_mcq_prompt(question["question"], question["options"], int(answer["answer"])),
            DatasetFieldNames.SFT_TARGET: f"\\boxed{{{letters[int(answer['answer'])]}}}",
            DatasetFieldNames.SFT_ANSWER: letters[int(answer["answer"])]
        })
    dataset = Dataset.from_list(ds)
    dataset.save_to_disk(output_path / split)
    offset += start_idx

# Process benchmark split separately to only include the answer as the target
benchmark_ds = []
for i in range(bench_size):
    original_idx = shuffled_indices[offset + i]
    question = questions[original_idx]
    answer = answers.get(str(original_idx), "No answer found")
    benchmark_ds.append({
        DatasetFieldNames.BENCHMARK_INPUT: format_mcq_prompt(question["question"], question["options"], int(answer["answer"])),
        DatasetFieldNames.BENCHMARK_ANSWER: letters[int(answer["answer"])]
    })
benchmark_dataset = Dataset.from_list(benchmark_ds)
benchmark_dataset.save_to_disk(benchmark_output_path)