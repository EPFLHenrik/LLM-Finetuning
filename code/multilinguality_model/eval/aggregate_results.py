"""code/multilinguality_model/eval/aggregate_results.py

Aggregate multilingual eval result folders into summary files.
"""

from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from multilinguality_model.paths import get_repo_root

COLUMNS = [
    "timestamp_utc",
    "model_name",
    "benchmark",
    "benchmark_subcategory",
    "language",
    "think_mode",
    "use_vllm",
    "limit",
    "n",
    "pass@1",
    "box_compliance_rate",
    "pass@1_among_extracted",
    "fallback_accuracy",
    "hit_max_tokens_rate",
    "avg_generated_tokens",
    "results_path",
]


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _infer_model_name(result_dir: Path) -> str:
    return result_dir.name


def _infer_benchmark(path: Path) -> tuple[str, str]:
    name = path.stem
    if name == "provided_results":
        return "validation_samples", "multilingual"
    if name == "val_results":
        return "multilingual", "val"
    if name == "all_results":
        return "multilingual", "all"
    return "multilingual", name


def _bool_from_name(text: str, default: bool) -> bool:
    lowered = text.lower()
    if "nonthink" in lowered or "no_think" in lowered or "nothink" in lowered:
        return False
    if "think" in lowered:
        return True
    return default


def _summarize_df(
    df: pd.DataFrame,
    result_path: Path,
    result_root: Path,
    default_think_mode: bool,
    use_vllm: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    if df.empty:
        return []

    model_name = _infer_model_name(result_path.parent)
    benchmark, benchmark_subcategory = _infer_benchmark(result_path)
    think_mode = _bool_from_name(str(result_path), default_think_mode)
    timestamp = datetime.fromtimestamp(result_path.stat().st_mtime, tz=timezone.utc).isoformat()
    rows = []

    if "box_compliant" not in df.columns and "missing_box" in df.columns:
        df["box_compliant"] = ~df["missing_box"]
    if "box_compliant" not in df.columns:
        df["box_compliant"] = df.get("predicted", pd.Series([None] * len(df))).notna()
    if "correct" not in df.columns:
        return []

    group_cols = ["dataset", "language"]
    for col in group_cols:
        if col not in df.columns:
            df[col] = ""

    for (dataset, language), group in df.groupby(group_cols, dropna=False):
        n = int(len(group))
        if n == 0:
            continue
        boxed = group[group["box_compliant"].astype(bool)]
        pass_at_1 = float(group["correct"].mean())
        box_rate = float(group["box_compliant"].mean())
        pass_extracted = float(boxed["correct"].mean()) if len(boxed) else None

        fallback = None
        if "fallback_correct" in group.columns:
            fallback = float(group["fallback_correct"].mean())

        hit_max = None
        if "hit_max_tokens" in group.columns:
            hit_max = float(group["hit_max_tokens"].mean())

        avg_tokens = None
        if "n_generated_tokens" in group.columns:
            avg_tokens = float(group["n_generated_tokens"].mean())

        rows.append({
            "timestamp_utc": timestamp,
            "model_name": model_name,
            "benchmark": benchmark,
            "benchmark_subcategory": str(dataset or benchmark_subcategory),
            "language": str(language or ""),
            "think_mode": think_mode,
            "use_vllm": use_vllm,
            "limit": limit,
            "n": n,
            "pass@1": pass_at_1,
            "box_compliance_rate": box_rate,
            "pass@1_among_extracted": pass_extracted,
            "fallback_accuracy": fallback,
            "hit_max_tokens_rate": hit_max,
            "avg_generated_tokens": avg_tokens,
            "results_path": str(result_path.relative_to(result_root.parent.parent.parent) if result_path.is_relative_to(result_root.parent.parent.parent) else result_path),
        })

    return rows


def aggregate_results(
    result_root: Path,
    output_json: Path,
    output_csv: Path,
    default_think_mode: bool,
    use_vllm: bool,
    limit: int | None,
) -> None:
    rows = []
    for path in sorted(result_root.rglob("*results.jsonl")):
        if path.name == "all_results.jsonl":
            continue
        df = _read_jsonl(path)
        rows.extend(_summarize_df(df, path, result_root, default_think_mode, use_vllm, limit))

    output = {"columns": COLUMNS, "rows": rows}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    df_out = pd.DataFrame(rows, columns=COLUMNS)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"Wrote {len(rows)} rows to {output_json}")
    print(f"Wrote CSV to {output_csv}")


def parse_args() -> argparse.Namespace:
    repo = get_repo_root()
    default_root = repo / "runs" / "benchmark" / "results" / "multilinguality_model"
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", type=Path, default=default_root)
    parser.add_argument("--output_json", type=Path, default=default_root / "combined_results.json")
    parser.add_argument("--output_csv", type=Path, default=default_root / "combined_results.csv")
    parser.add_argument("--default_think_mode", action="store_true")
    parser.add_argument("--use_vllm", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    aggregate_results(
        result_root=args.result_root,
        output_json=args.output_json,
        output_csv=args.output_csv,
        default_think_mode=args.default_think_mode,
        use_vllm=args.use_vllm,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
