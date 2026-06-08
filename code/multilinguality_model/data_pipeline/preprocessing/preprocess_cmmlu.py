"""Preprocess CMMLU into clean multilingual MCQA JSONL files."""

from pathlib import Path
import csv
import io
import json
import zipfile

import pandas as pd
from huggingface_hub import hf_hub_download
from multilinguality_model.paths import get_processed_data_dir, get_data_dir


DATASET_NAME = "haonan-li/cmmlu"
ZIP_NAME = "cmmlu_v1_0_1.zip"
OUTPUT_DIR = get_processed_data_dir() / "cmmlu"
VALID_ANSWERS = {"A", "B", "C", "D"}
REQUIRED_COLUMNS = {"Question", "A", "B", "C", "D", "Answer"}
SOURCE_TO_PROJECT_SPLIT = {
    "dev": "eval",
    "test": "train",
}
CATEGORY_OVERRIDES = {
    "college_actuarial_science": "STEM",
    "high_school_geography": "Social Science",
    "high_school_politics": "Social Science",
    "journalism": "Humanities",
    "sports_science": "Other",
}
REGIONALITY_OVERRIDES = {
    "high_school_politics": "regional",
    "high_school_geography": "regional",
    "journalism": "regional_implicit",
    "sports_science": "regional_implicit",
}


def infer_category(subject):
    s = str(subject)

    if any(
        key in s
        for key in [
            "mathematics",
            "physics",
            "chemistry",
            "biology",
            "computer",
            "machine_learning",
            "engineering",
            "electrical",
            "agronomy",
            "anatomy",
            "astronomy",
            "genetics",
            "virology",
            "medicine",
            "medical",
            "clinical",
            "nutrition",
            "food_science",
        ]
    ):
        return "STEM"

    if any(
        key in s
        for key in [
            "law",
            "jurisprudence",
            "accounting",
            "management",
            "marketing",
            "business",
            "economics",
            "actuarial",
            "project_management",
            "teacher_qualification",
            "education",
        ]
    ):
        return "Law & Professional"

    if any(
        key in s
        for key in [
            "history",
            "literature",
            "chinese",
            "arts",
            "philosophy",
            "world_religions",
            "ancient_chinese",
            "modern_chinese",
        ]
    ):
        return "Humanities"

    if any(
        key in s
        for key in [
            "sociology",
            "ethnology",
            "politics",
            "foreign_policy",
            "marxist",
            "security",
            "public_relations",
            "human_sexuality",
            "psychology",
            "civil_service",
            "driving_rule",
            "moral",
        ]
    ):
        return "Social Science"

    return "Other"


def infer_regionality(subject):
    s = str(subject)

    if any(
        key in s
        for key in [
            "chinese",
            "china",
            "civil_service",
            "driving_rule",
            "foreign_policy",
            "chinese_history",
            "modern_chinese",
            "ancient_chinese",
            "marxist_theory",
            "ethnology",
            "traditional_chinese_medicine",
            "college_law",
            "professional_law",
            "jurisprudence",
            "legal_and_moral_basis",
        ]
    ):
        return "regional"

    if any(
        key in s
        for key in [
            "law",
            "accounting",
            "teacher_qualification",
            "education",
            "economics",
            "management",
            "marketing",
            "business",
            "medicine",
            "clinical",
            "psychology",
            "public_relations",
            "security",
            "sociology",
            "food_culture",
        ]
    ):
        return "regional_implicit"

    return "region_agnostic"


def make_prompt(row):
    return (
        f"{row['question']}\n\n"
        f"A) {row['A']}\n"
        f"B) {row['B']}\n"
        f"C) {row['C']}\n"
        f"D) {row['D']}"
    )


def safe_id_part(text):
    return (
        str(text)
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
    )


def make_record(row):
    subject = safe_id_part(row["subject"])
    project_split = safe_id_part(row["split"])
    source_split = safe_id_part(row["source_split"])
    source_idx = int(row["source_idx"])

    return {
        "id": f"cmmlu_chinese_{project_split}_{subject}_{source_split}_{source_idx}",
        "dataset": "cmmlu",
        "language": "chinese",
        "split": row["split"],
        "task_type": "mcq",
        "prompt": make_prompt(row),
        "answer": row["answer"],
        "metadata": {
            "subject": row["subject"],
            "category": row["category"],
            "regionality": row["regionality"],
            "source_split": row["source_split"],
            "source_idx": source_idx,
            "source_path": row["source_path"],
            "n_examples_in_subject_split": int(row["n_examples_in_subject_split"]),
        },
    }


def validate_record(record):
    assert set(record) == {
        "id",
        "dataset",
        "language",
        "split",
        "task_type",
        "prompt",
        "answer",
        "metadata",
    }
    assert record["id"].startswith("cmmlu_chinese_")
    assert record["dataset"] == "cmmlu"
    assert record["language"] == "chinese"
    assert record["split"] in {"train", "eval"}
    assert record["task_type"] == "mcq"
    assert record["answer"] in VALID_ANSWERS
    assert "\n\nA) " in record["prompt"]
    assert "\nB) " in record["prompt"]
    assert "\nC) " in record["prompt"]
    assert "\nD) " in record["prompt"]
    assert "\\boxed" not in record["prompt"]

    metadata = record["metadata"]
    assert set(metadata) == {
        "subject",
        "category",
        "regionality",
        "source_split",
        "source_idx",
        "source_path",
        "n_examples_in_subject_split",
    }
    assert metadata["category"] in {
        "STEM",
        "Humanities",
        "Social Science",
        "Law & Professional",
        "Other",
    }
    assert metadata["regionality"] in {
        "regional",
        "regional_implicit",
        "region_agnostic",
    }
    assert metadata["source_split"] in {"dev", "test"}
    assert isinstance(metadata["source_idx"], int)
    assert isinstance(metadata["n_examples_in_subject_split"], int)


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_raw_rows():
    zip_path = hf_hub_download(
        repo_id=DATASET_NAME,
        filename=ZIP_NAME,
        repo_type="dataset",
    )

    rows = []
    failed_files = []

    with zipfile.ZipFile(zip_path) as z:
        csv_names = sorted(
            name for name in z.namelist() if name.lower().endswith(".csv")
        )

        unexpected_splits = sorted(
            {
                Path(name).parts[0]
                for name in csv_names
                if Path(name).parts[0] not in SOURCE_TO_PROJECT_SPLIT
            }
        )
        assert not unexpected_splits, f"Unexpected CMMLU source splits: {unexpected_splits}"

        for name in csv_names:
            path = Path(name)
            source_split = path.parts[0]
            subject = path.stem

            if source_split not in SOURCE_TO_PROJECT_SPLIT:
                rows.append(
                    {
                        "question": "",
                        "A": "",
                        "B": "",
                        "C": "",
                        "D": "",
                        "answer": "",
                        "subject": subject,
                        "source_split": source_split,
                        "source_idx": -1,
                        "source_path": name,
                        "n_examples_in_subject_split": 0,
                    }
                )
                continue

            with z.open(name) as f:
                text_f = io.TextIOWrapper(f, encoding="utf-8-sig")
                reader = csv.DictReader(text_f)
                fieldnames = reader.fieldnames or []

                missing_cols = sorted(REQUIRED_COLUMNS - set(fieldnames))
                if missing_cols:
                    failed_files.append((name, missing_cols))
                    continue

                file_rows = list(reader)
                n_examples = len(file_rows)

                for idx, row in enumerate(file_rows):
                    rows.append(
                        {
                            "question": row.get("Question", ""),
                            "A": row.get("A", ""),
                            "B": row.get("B", ""),
                            "C": row.get("C", ""),
                            "D": row.get("D", ""),
                            "answer": row.get("Answer", ""),
                            "subject": subject,
                            "source_split": source_split,
                            "source_idx": idx,
                            "source_path": name,
                            "n_examples_in_subject_split": n_examples,
                        }
                    )

    if failed_files:
        print(f"Warning: failed CMMLU files: {len(failed_files)}")

    return rows


def clean_dataframe(df):
    text_columns = [
        "question",
        "A",
        "B",
        "C",
        "D",
        "answer",
        "subject",
        "source_split",
        "source_path",
    ]

    for col in text_columns:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["answer"] = df["answer"].str.upper()

    df = df[
        (df["question"] != "")
        & (df["A"] != "")
        & (df["B"] != "")
        & (df["C"] != "")
        & (df["D"] != "")
        & (df["answer"].isin(VALID_ANSWERS))
    ].copy()

    df["category"] = df["subject"].apply(infer_category)
    df["regionality"] = df["subject"].apply(infer_regionality)

    df["category"] = df["subject"].map(CATEGORY_OVERRIDES).fillna(df["category"])
    df["regionality"] = df["subject"].map(REGIONALITY_OVERRIDES).fillna(
        df["regionality"]
    )

    dedup_columns = ["subject", "question", "A", "B", "C", "D"]
    df = df.drop_duplicates(subset=dedup_columns, keep="first").reset_index(drop=True)

    assert df.duplicated(subset=dedup_columns).sum() == 0

    df["split"] = df["source_split"].map(SOURCE_TO_PROJECT_SPLIT)
    missing_splits = sorted(df.loc[df["split"].isna(), "source_split"].unique())
    assert not missing_splits, f"Missing CMMLU split mappings: {missing_splits}"

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_rows = load_raw_rows()
    df = clean_dataframe(pd.DataFrame(raw_rows))
    records = [make_record(row) for _, row in df.iterrows()]

    ids = [record["id"] for record in records]
    assert len(ids) == len(set(ids)), "Duplicate CMMLU ids found."

    for record in records:
        validate_record(record)

    all_path = OUTPUT_DIR / "cmmlu_chinese_all.jsonl"
    eval_path = OUTPUT_DIR / "cmmlu_chinese_eval.jsonl"
    train_path = OUTPUT_DIR / "cmmlu_chinese_train.jsonl"

    eval_records = [record for record in records if record["split"] == "eval"]
    train_records = [record for record in records if record["split"] == "train"]

    write_jsonl(records, all_path)
    write_jsonl(eval_records, eval_path)
    write_jsonl(train_records, train_path)

    print(
        f"CMMLU: wrote {len(records)} records "
        f"({len(train_records)} train, {len(eval_records)} eval) to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
