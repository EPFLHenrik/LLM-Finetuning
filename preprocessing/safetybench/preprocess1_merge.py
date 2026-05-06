import json
from pathlib import Path
from typing import List
from constants.global_variables import DatasetFieldNames

from datasets import Dataset

ROOT = Path(__file__).resolve().parents[2]
questions_path = ROOT / "data" / "raw" / "safetybench" / "en" / "test_en.json"
answers_path = ROOT / "data" / "raw" / "safetybench" / "en" / "test_answers_en.json"
output_path = ROOT / "data" / "processed" / "sft" / "safetybench" / "only_answers"

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
processed_dataset = []
for i in range(len(questions)):
    question = questions[i]
    answer = answers.get(str(i), "No answer found")
    processed_dataset.append({
        DatasetFieldNames.SFT_ANSWER: format_mcq_prompt(question["question"], question["options"], int(answer["answer"])),
        DatasetFieldNames.SFT_TARGET: f"\\boxed{{{letters[int(answer['answer'])]}}}",
        DatasetFieldNames.SFT_ANSWER: letters[int(answer["answer"])]
        })

dataset = Dataset.from_list(processed_dataset)
output_path.parent.mkdir(parents=True, exist_ok=True)
dataset.save_to_disk(str(output_path))