"""Rename Spain BOE distilled files and update dataset field inside records.

Renames both filenames and the 'dataset' field in every record.
Handles v2_fallback/ and v2_fallback/sft_1/.

Usage:
    python rename_spain_boe.py \
        --distill_dir multilingual/data/distilled/v2_fallback \
        --dry_run
"""

import argparse
import json
from pathlib import Path

RENAMES = {
    "spain_boe_constitucion_espanola": "spain_boe_constitucion",
    "spain_boe_estatuto_andalucia":    "spain_boe_andalucia",
    "spain_boe_estatuto_cataluna":     "spain_boe_cataluna",
    "spain_boe_estatuto_canarias":     "spain_boe_canarias",
    "spain_boe_estatuto_comunitat_valenciana": "spain_boe_valencia",
    "spain_boe_estatuto_galicia":      "spain_boe_galicia",
    "spain_boe_estatuto_madrid":       "spain_boe_madrid",
    "spain_boe_estatuto_pais_vasco":   "spain_boe_pais_vasco",
}

SUFFIXES = ["_all.jsonl", "_distilled.jsonl", "_fail.jsonl",
            "_all_shuffle_meta.json", "_generated.jsonl"]


def rename_files(directory: Path, dry_run: bool) -> None:
    for old_name, new_name in RENAMES.items():
        for suffix in SUFFIXES:
            old_path = directory / f"{old_name}{suffix}"
            new_path = directory / f"{new_name}{suffix}"
            if old_path.exists():
                print(f"  {'[DRY]' if dry_run else ''} {old_path.name} → {new_path.name}")
                if not dry_run:
                    old_path.rename(new_path)


def update_records(directory: Path, dry_run: bool) -> None:
    for old_name, new_name in RENAMES.items():
        for suffix in ["_all.jsonl", "_distilled.jsonl", "_fail.jsonl", "_generated.jsonl"]:
            path = directory / f"{new_name}{suffix}"
            if not path.exists():
                continue
            records = []
            changed = 0
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    if r.get("dataset") == old_name:
                        r["dataset"] = new_name
                        changed += 1
                    records.append(r)
            if changed:
                print(f"  {'[DRY]' if dry_run else ''} {path.name}: updated {changed} records")
                if not dry_run:
                    with open(path, "w", encoding="utf-8") as f:
                        for r in records:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distill_dir", type=Path,
                        default=Path("multilingual/data/distilled/v2_fallback"))
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    dirs = [args.distill_dir, args.distill_dir / "sft_1"]

    for d in dirs:
        if not d.exists():
            print(f"Skipping (not found): {d}")
            continue
        print(f"\n── {d} ──")
        rename_files(d, args.dry_run)
        update_records(d, args.dry_run)

    print("\nDone." if not args.dry_run else "\n[DRY RUN] No changes made.")


if __name__ == "__main__":
    main()