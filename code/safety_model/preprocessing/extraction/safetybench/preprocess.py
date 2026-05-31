import argparse
import json
import hashlib
import random
from pathlib import Path
from typing import List
from pprint import pprint

from datasets import Dataset, DatasetDict

from code.safety_model.constants.global_variables import DatasetFieldNames, Random


SEED = Random.SEED
LETTERS = [chr(ord("A") + i) for i in range(26)]

OUT_SFT = Path("data/safety/processed/sft/safetybench/mcq")
OUT_BENCH = Path("data/safety/processed/benchmark/safetybench")


def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def build_options(options: List[str]) -> str:
    return "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options))


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_records(questions, answers):
    if len(questions) != len(answers):
        raise ValueError(
            f"Mismatched question/answer counts: {len(questions)} questions vs {len(answers)} answers"
        )

    records = []
    for idx, question in enumerate(questions):
        answer_obj = answers.get(str(idx))
        if answer_obj is None:
            raise ValueError(f"Missing answer for question index {idx}")

        answer_index = int(answer_obj["answer"])
        options = question.get("options", [])
        prompt = (question.get("question") or "").strip()

        if not prompt:
            raise ValueError(f"Empty question text at index {idx}")
        if not options:
            raise ValueError(f"No options found at index {idx}")
        if not (0 <= answer_index < len(options)):
            raise ValueError(
                f"Answer index {answer_index} out of range for question {idx} with {len(options)} options"
            )
        if len(options) > len(LETTERS):
            raise ValueError(f"Too many options at index {idx}: {len(options)} > {len(LETTERS)}")

        prompt_key = normalize_text(prompt + " || " + " || ".join(options))
        source_uid = hashlib.md5(prompt_key.encode("utf-8")).hexdigest()

        records.append(
            {
                "original_idx": idx,
                "source_uid": source_uid,
                "question": prompt,
                "options": options,
                "answer_index": answer_index,
                "answer_letter": LETTERS[answer_index],
                "category": question.get("category"),
                "sub_category": question.get("sub_category"),
            }
        )
    return records


def format_mcq_prompt(question_prompt: str, options: List[str]) -> str:
    return f"{question_prompt}\n\n{build_options(options)}"


def to_sft_dataset(records):
    rows = []
    for row in records:
        rows.append(
            {
                DatasetFieldNames.SFT_INPUT: format_mcq_prompt(row["question"], row["options"]),
                DatasetFieldNames.SFT_TARGET: f"The correct answer is {row['answer_letter']}.\n\n\\boxed{{{row['answer_letter']}}}",
                DatasetFieldNames.SFT_ANSWER: row["answer_letter"],
                "task_type": "mcq",
                "source_uid": row["source_uid"],
                "original_idx": row["original_idx"],
                "num_options": len(row["options"]),
                "category": row.get("category"),
                "sub_category": row.get("sub_category"),
            }
        )
    return Dataset.from_list(rows)


def to_benchmark_dataset(records):
    rows = []
    for row in records:
        rows.append(
            {
                DatasetFieldNames.BENCHMARK_INPUT: format_mcq_prompt(row["question"], row["options"]),
                DatasetFieldNames.BENCHMARK_ANSWER: row["answer_letter"],
                "task_type": "mcq",
                "source_uid": row["source_uid"],
                "original_idx": row["original_idx"],
                "num_options": len(row["options"]),
                "category": row.get("category"),
                "sub_category": row.get("sub_category"),
            }
        )
    return Dataset.from_list(rows)


def stratified_split_records(records, test_ratio: float):
    buckets = {}
    for row in records:
        key = (row.get("category") or "__none__", row.get("sub_category") or "__none__", len(row["options"]))
        buckets.setdefault(key, []).append(row)

    train_rows, test_rows = [], []
    for key, bucket in buckets.items():
        bucket = list(bucket)
        random.Random(SEED).shuffle(bucket)
        n_test = int(len(bucket) * test_ratio)
        if n_test == 0 and len(bucket) > 1:
            n_test = 1
        test_rows.extend(bucket[:n_test])
        train_rows.extend(bucket[n_test:])

    random.Random(SEED).shuffle(train_rows)
    random.Random(SEED + 1).shuffle(test_rows)
    return train_rows, test_rows


def print_stats(name, records):
    counts = {}
    for row in records:
        key = row.get("category") or "__none__"
        counts[key] = counts.get(key, 0) + 1
    print(f"{name}: size={len(records)} category_dist={counts}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions_path", type=str, default="data/raw/safetybench/en/test_en.json")
    parser.add_argument("--answers_path", type=str, default="data/raw/safetybench/en/test_answers_en.json")
    parser.add_argument("--internal_test_ratio", type=float, default=0.05)
    parser.add_argument("--benchmark_ratio", type=float, default=0.05)
    args = parser.parse_args()

    questions_path = Path(args.questions_path)
    answers_path = Path(args.answers_path)

    questions = load_json(questions_path)
    answers = load_json(answers_path)
    records = parse_records(questions, answers)

    print_stats("all_records", records)

    internal_pool, benchmark_rows = stratified_split_records(records, args.benchmark_ratio)
    train_rows, test_rows = stratified_split_records(internal_pool, args.internal_test_ratio)

    print_stats("internal_pool", internal_pool)
    print_stats("train_rows", train_rows)
    print_stats("test_rows", test_rows)
    print_stats("benchmark_rows", benchmark_rows)

    OUT_SFT.parent.mkdir(parents=True, exist_ok=True)
    OUT_BENCH.parent.mkdir(parents=True, exist_ok=True)

    DatasetDict(
        train=to_sft_dataset(train_rows),
        test=to_sft_dataset(test_rows),
    ).save_to_disk(str(OUT_SFT))

    to_benchmark_dataset(benchmark_rows).save_to_disk(str(OUT_BENCH))

    pprint(
        {
            "train": len(train_rows),
            "test": len(test_rows),
            "benchmark": len(benchmark_rows),
        }
    )


if __name__ == "__main__":
    main()