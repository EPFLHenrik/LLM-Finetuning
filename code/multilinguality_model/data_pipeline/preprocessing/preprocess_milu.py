"""Preprocess MILU Hindi into clean multilingual MCQA JSONL files."""

from pathlib import Path
import json
import os

import pandas as pd
from datasets import load_dataset
from multilinguality_model.paths import get_processed_data_dir, get_data_dir


DATASET_NAME = "ai4bharat/MILU"
DATA_DIR = "Hindi"
OUTPUT_DIR = get_processed_data_dir() / "milu"

SPLITS_TO_LOAD = ["test", "validation"]
TARGET_TO_ANSWER = {"option1": "A", "option2": "B", "option3": "C", "option4": "D"}
VALID_ANSWERS = {"A", "B", "C", "D"}
VALID_DOMAINS = {
    "Arts & Humanities",
    "Business Studies",
    "Engineering & Tech",
    "Environmental Sciences",
    "Health & Medicine",
    "Law & Governance",
    "Science",
    "Social Sciences",
}
VALID_REGIONALITIES = {"regional", "regional_implicit", "region_agnostic"}

SUBJECT_TO_REGIONALITY = {
    "Arts and Culture": "regional",
    "Defense and Security": "regional",
    "Geography": "regional",
    "History": "regional",
    "Language Studies": "regional",
    "Law and Ethics": "regional",
    "Literature and Linguistics": "regional",
    "Politics and Governance": "regional",
    "Public Administration": "regional",
    "Religion and Spirituality": "regional",
    "Agriculture": "regional_implicit",
    "Anthropology": "regional_implicit",
    "Architecture and Design": "regional_implicit",
    "Business and Management": "regional_implicit",
    "Economics": "regional_implicit",
    "Education": "regional_implicit",
    "Energy and Power": "regional_implicit",
    "Environmental Science": "regional_implicit",
    "Ethics and Human Rights": "regional_implicit",
    "Finance and Investment": "regional_implicit",
    "Food Science": "regional_implicit",
    "Health and Medicine": "regional_implicit",
    "International Relations": "regional_implicit",
    "Media and Communication": "regional_implicit",
    "Music and Performing Arts": "regional_implicit",
    "Psychology": "regional_implicit",
    "Social Welfare and Development": "regional_implicit",
    "Sociology": "regional_implicit",
    "Sports and Recreation": "regional_implicit",
    "Technology and Innovation": "regional_implicit",
    "Transportation and Logistics": "regional_implicit",
    "Astronomy and Astrophysics": "region_agnostic",
    "Biology": "region_agnostic",
    "Chemistry": "region_agnostic",
    "Computer Science": "region_agnostic",
    "Earth Sciences": "region_agnostic",
    "Engineering": "region_agnostic",
    "Information Technology": "region_agnostic",
    "Logical Reasoning": "region_agnostic",
    "Materials Science": "region_agnostic",
    "Physics": "region_agnostic",
}


def make_prompt(row):
    return (
        f"{row['question']}\n\n"
        f"A) {row['option1']}\n"
        f"B) {row['option2']}\n"
        f"C) {row['option3']}\n"
        f"D) {row['option4']}"
    )


def safe_id_part(text):
    return str(text).strip().replace("/", "_").replace("\\", "_").replace(" ", "_").replace("-", "_").replace(":", "_")


def make_record(row, idx):
    source_split = safe_id_part(row["source_split"])
    source_idx = int(row["source_idx"])
    return {
        "id": f"milu_hindi_all_{idx:06d}_{source_split}_{source_idx}",
        "dataset": "milu",
        "language": "hindi",
        "split": "all",
        "task_type": "mcq",
        "prompt": make_prompt(row),
        "answer": row["answer"],
        "metadata": {
            "subject": row["subject"],
            "domain": row["domain"],
            "regionality": row["regionality"],
            "is_translated": bool(row["is_translated"]),
            "source_split": row["source_split"],
            "source_idx": source_idx,
        },
    }


def validate_record(record):
    assert set(record) == {"id", "dataset", "language", "split", "task_type", "prompt", "answer", "metadata"}
    assert isinstance(record["id"], str)
    assert record["id"].startswith("milu_hindi_all_")
    assert record["dataset"] == "milu"
    assert record["language"] == "hindi"
    assert record["split"] == "all"
    assert record["task_type"] == "mcq"
    assert record["answer"] in VALID_ANSWERS
    assert "\n\nA) " in record["prompt"]
    assert "\nB) " in record["prompt"]
    assert "\nC) " in record["prompt"]
    assert "\nD) " in record["prompt"]
    metadata = record["metadata"]
    assert set(metadata) == {"subject", "domain", "regionality", "is_translated", "source_split", "source_idx"}
    assert metadata["domain"] in VALID_DOMAINS
    assert metadata["regionality"] in VALID_REGIONALITIES
    assert isinstance(metadata["is_translated"], bool)
    assert metadata["source_split"] in {"test", "validation"}
    assert isinstance(metadata["source_idx"], int)


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_raw_dataframe():
    hf_token = os.environ.get("HF_TOKEN")
    assert hf_token is not None and hf_token.strip(), "HF_TOKEN is not set in the environment."
    rows = []
    failed_loads = []
    for source_split in SPLITS_TO_LOAD:
        try:
            ds = load_dataset(DATASET_NAME, data_dir=DATA_DIR, split=source_split, token=hf_token)
            for idx, ex in enumerate(ds):
                rows.append({
                    "question": ex.get("question", ""),
                    "option1": ex.get("option1", ""),
                    "option2": ex.get("option2", ""),
                    "option3": ex.get("option3", ""),
                    "option4": ex.get("option4", ""),
                    "target": ex.get("target", ""),
                    "subject": ex.get("subject", ""),
                    "domain": ex.get("domain", ""),
                    "is_translated": ex.get("is_translated", None),
                    "source_split": source_split,
                    "source_idx": idx,
                })
        except Exception as exc:
            failed_loads.append((source_split, repr(exc)))
    if failed_loads:
        print(f"Warning: failed MILU loads: {failed_loads}")
    return pd.DataFrame(rows)


def clean_dataframe(df_raw):
    df = df_raw.copy()
    text_columns = ["question", "option1", "option2", "option3", "option4", "target", "subject", "domain", "source_split"]
    for col in text_columns:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df["target"] = df["target"].str.lower()
    df["answer"] = df["target"].map(TARGET_TO_ANSWER)
    df["is_translated"] = df["is_translated"].fillna(False).astype(bool)
    df["regionality"] = df["subject"].map(SUBJECT_TO_REGIONALITY)
    missing_regionality = sorted(df.loc[df["regionality"].isna(), "subject"].unique())
    assert not missing_regionality, f"Missing MILU regionality labels: {missing_regionality}"
    df = df[
        (df["question"] != "")
        & (df["option1"] != "")
        & (df["option2"] != "")
        & (df["option3"] != "")
        & (df["option4"] != "")
        & (df["answer"].isin(VALID_ANSWERS))
    ].copy()
    df = df.reset_index(drop=True)
    dedup_columns = ["question", "option1", "option2", "option3", "option4"]
    df = df.drop_duplicates(subset=dedup_columns, keep="first").reset_index(drop=True)
    assert df.duplicated(subset=dedup_columns).sum() == 0
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_raw = load_raw_dataframe()
    df = clean_dataframe(df_raw)
    records = [make_record(row, idx) for idx, row in df.reset_index(drop=True).iterrows()]
    ids = [record["id"] for record in records]
    assert len(ids) == len(set(ids)), "Duplicate MILU ids found."
    for record in records:
        validate_record(record)

    train_records = [record for record in records if record["metadata"]["source_split"] == "test"]
    val_records = [record for record in records if record["metadata"]["source_split"] == "validation"]

    write_jsonl(records, OUTPUT_DIR / "milu_hindi_all.jsonl")
    write_jsonl(train_records, OUTPUT_DIR / "milu_hindi_train.jsonl")
    write_jsonl(val_records, OUTPUT_DIR / "milu_hindi_val.jsonl")

    regionality_to_all = {
        "regional": OUTPUT_DIR / "milu_hindi_regional.jsonl",
        "regional_implicit": OUTPUT_DIR / "milu_hindi_regional_implicit.jsonl",
        "region_agnostic": OUTPUT_DIR / "milu_hindi_region_agnostic.jsonl",
    }
    regionality_to_train = {
        "regional": OUTPUT_DIR / "milu_hindi_train_regional.jsonl",
        "regional_implicit": OUTPUT_DIR / "milu_hindi_train_regional_implicit.jsonl",
        "region_agnostic": OUTPUT_DIR / "milu_hindi_train_region_agnostic.jsonl",
    }

    for regionality, path in regionality_to_all.items():
        subset = [record for record in records if record["metadata"]["regionality"] == regionality]
        write_jsonl(subset, path)

    for regionality, path in regionality_to_train.items():
        subset = [record for record in train_records if record["metadata"]["regionality"] == regionality]
        write_jsonl(subset, path)

    print(f"MILU: wrote {len(records)} records to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
