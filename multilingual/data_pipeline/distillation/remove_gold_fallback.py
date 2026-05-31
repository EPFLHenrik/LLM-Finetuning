from __future__ import annotations

import argparse
import json
from pathlib import Path


def is_gold_fallback(example: dict) -> bool:
    distill = example.get("distill", {})
    return distill.get("mode") == "gold_fallback"


def process_file(path: Path, removed_path: Path, dry_run: bool = False) -> tuple[int, int]:
    kept = []
    removed = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} line {line_num}: {e}") from e

            if is_gold_fallback(ex):
                ex["_removed_from_file"] = str(path)
                removed.append(ex)
            else:
                kept.append(ex)

    if not dry_run:
        # Rewrite original _distilled.jsonl with only clean examples.
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            for ex in kept:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        tmp_path.replace(path)

        # Append removed gold_fallback examples to one shared file.
        if removed:
            with removed_path.open("a", encoding="utf-8") as f:
                for ex in removed:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return len(kept), len(removed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove gold_fallback examples from all *_distilled.jsonl files in a folder."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Folder containing *_distilled.jsonl files.",
    )
    parser.add_argument(
        "--removed_file",
        type=Path,
        default=None,
        help="File where removed gold_fallback examples are appended. Default: input_dir/all_removed_gold_fallback.jsonl",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively for *_distilled.jsonl files.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report counts; do not modify files.",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    removed_path = args.removed_file or (input_dir / "all_removed_gold_fallback.jsonl")
    removed_path.parent.mkdir(parents=True, exist_ok=True)

    pattern = "**/*_distilled.jsonl" if args.recursive else "*_distilled.jsonl"
    files = sorted(input_dir.glob(pattern))

    # Avoid processing the removal log itself if its name happens to match.
    files = [p for p in files if p.resolve() != removed_path.resolve()]

    if not files:
        print(f"No *_distilled.jsonl files found in {input_dir}")
        return

    total_kept = 0
    total_removed = 0

    print(f"Found {len(files)} distilled files.")
    print(f"Removed examples will be appended to: {removed_path}")
    print()

    for path in files:
        kept, removed = process_file(path, removed_path, dry_run=args.dry_run)
        total_kept += kept
        total_removed += removed
        print(f"{path}: kept={kept}, removed_gold_fallback={removed}")

    print()
    print(f"Total kept: {total_kept}")
    print(f"Total removed gold_fallback: {total_removed}")

    if args.dry_run:
        print("Dry run only. No files were modified.")
    else:
        print("Done. Original distilled files were rewritten without gold_fallback examples.")
        print("Removed examples were appended to all_removed_gold_fallback.jsonl.")


if __name__ == "__main__":
    main()