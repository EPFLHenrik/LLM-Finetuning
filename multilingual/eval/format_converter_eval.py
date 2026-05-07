from __future__ import annotations

import argparse
import json
from pathlib import Path


def get_gold(row: dict) -> str:
    for key in ["answer", "gold", "reference"]:
        if key in row and row[key] is not None:
            return str(row[key])
    raise KeyError(f"No gold answer field found. Available keys: {list(row.keys())}")


def get_completion(row: dict) -> str:
    # Case 1: already official
    if "completions" in row:
        comps = row["completions"]
        if isinstance(comps, list) and comps:
            return str(comps[0])

    # Case 2: previous custom format
    for key in ["output", "completion", "generation", "model_output"]:
        if key in row and row[key] is not None:
            return str(row[key])

    raise KeyError(f"No completion/output field found. Available keys: {list(row.keys())}")


def convert_jsonl(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)

            official_row = {
                "prompt": row.get("prompt", ""),
                "answer": get_gold(row),
                "completions": [get_completion(row)],
            }

            fout.write(json.dumps(official_row, ensure_ascii=False) + "\n")
            n += 1

    print(f"Converted {n} rows")
    print(f"Saved official generations to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    convert_jsonl(args.input, args.output)


if __name__ == "__main__":
    main()