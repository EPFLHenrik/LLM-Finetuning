# multilingual/train/inspect_formatted_sft.py

from pathlib import Path
import argparse
import json
from transformers import AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-1.7B"


def load_jsonl(path: Path):
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file",
        type=Path,
        default=Path("multilingual/data/sft/v8/mixed/train_mixed.jsonl"),
    )
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--distilled", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    records = load_jsonl(args.train_file)

    if args.distilled:
        records = [
            r for r in records
            if r.get("metadata", {}).get("distilled") is True
        ]

    if not records:
        raise ValueError("No matching records found.")

    ex = records[args.idx]

    print("\n--- Metadata ---")
    print("dataset:", ex.get("dataset"))
    print("language:", ex.get("language"))
    print("answer:", ex.get("answer"))
    print("source_id:", ex.get("source_id"))
    print("distilled:", ex.get("metadata", {}).get("distilled", False))

    print("\n--- Sample formatted example ---")
    print(tokenizer.apply_chat_template(
        ex["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    ))


if __name__ == "__main__":
    main()