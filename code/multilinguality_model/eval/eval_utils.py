"""multilingual/eval/eval_utils.py"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List
import pandas as pd


def make_summary(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for keys, grp in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {
            col: value
            for col, value in zip(group_cols, keys)
        }
        row.update({
            "n": int(len(grp)),
            "correct": int(grp["correct"].sum()),
            "accuracy": round(float(grp["correct"].mean()), 4),
            "box_compliance_rate": round(float(grp["box_compliant"].mean()), 4) if "box_compliant" in grp.columns else None,
        })
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=group_cols + ["n", "correct", "accuracy"])
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def save_eval_summaries(df_all: pd.DataFrame, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_by_dataset_language = make_summary(df_all, ["dataset", "language"])
    summary_by_language = make_summary(df_all, ["language"])
    summary_by_dataset = make_summary(df_all, ["dataset"])
    paths = {
        "summary_by_dataset_language": output_dir / "summary_by_dataset_language.csv",
        "summary_by_language": output_dir / "summary_by_language.csv",
        "summary_by_dataset": output_dir / "summary_by_dataset.csv",
        "summary_overall": output_dir / "summary_overall.json",
        "summary": output_dir / "summary.csv",
    }
    summary_by_dataset_language.to_csv(paths["summary_by_dataset_language"], index=False)
    summary_by_language.to_csv(paths["summary_by_language"], index=False)
    summary_by_dataset.to_csv(paths["summary_by_dataset"], index=False)
    summary_by_dataset_language.to_csv(paths["summary"], index=False)
    overall = {
        "n": int(len(df_all)),
        "correct": int(df_all["correct"].sum()),
        "accuracy": round(float(df_all["correct"].mean()), 4),
        "box_compliance_rate": round(float(df_all["box_compliant"].mean()), 4) if "box_compliant" in df_all.columns else None,
    }

    with paths["summary_overall"].open("w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)

    return {
        "summary_by_dataset_language": summary_by_dataset_language,
        "summary_by_language": summary_by_language,
        "summary_by_dataset": summary_by_dataset,
        "overall": overall,
        "paths": paths,
    }


def format_eval_summary_text(
    *,
    model_name: str,
    output_dir: Path,
    summary_by_dataset_language: pd.DataFrame,
    summary_by_language: pd.DataFrame,
    summary_by_dataset: pd.DataFrame,
    overall: dict,
    runtime_seconds: float | None = None,
) -> str:
    lines = []

    lines.append("=" * 60)
    lines.append("EVALUATION SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Model:      {model_name}")
    lines.append(f"Output dir: {output_dir}")

    if runtime_seconds is not None:
        lines.append(f"Runtime seconds: {runtime_seconds:.2f}")
        lines.append(f"Runtime minutes: {runtime_seconds / 60:.2f}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("SUMMARY OVERALL")
    lines.append("=" * 60)
    lines.append(json.dumps(overall, ensure_ascii=False, indent=2))

    lines.append("")
    lines.append("=" * 60)
    lines.append("SUMMARY BY LANGUAGE")
    lines.append("=" * 60)
    lines.append(summary_by_language.to_string(index=False))

    lines.append("")
    lines.append("=" * 60)
    lines.append("SUMMARY BY DATASET")
    lines.append("=" * 60)
    lines.append(summary_by_dataset.to_string(index=False))

    lines.append("")
    lines.append("=" * 60)
    lines.append("SUMMARY BY DATASET/LANGUAGE")
    lines.append("=" * 60)
    lines.append(summary_by_dataset_language.to_string(index=False))

    lines.append("")
    lines.append("=" * 60)
    lines.append("FILES")
    lines.append("=" * 60)

    for path in sorted(output_dir.glob("*")):
        if path.is_file():
            lines.append(path.name)

    lines.append("=" * 60)

    return "\n".join(lines) + "\n"


def save_summary_txt(
    *,
    model_name: str,
    output_dir: Path,
    summary_by_dataset_language: pd.DataFrame,
    summary_by_language: pd.DataFrame,
    summary_by_dataset: pd.DataFrame,
    overall: dict,
    runtime_seconds: float | None = None,
) -> Path:
    text = format_eval_summary_text(
        model_name=model_name,
        output_dir=output_dir,
        summary_by_dataset_language=summary_by_dataset_language,
        summary_by_language=summary_by_language,
        summary_by_dataset=summary_by_dataset,
        overall=overall,
        runtime_seconds=runtime_seconds,
    )

    path = output_dir / "summary.txt"
    path.write_text(text, encoding="utf-8")
    return path