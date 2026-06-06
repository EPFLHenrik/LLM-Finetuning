"""Preprocess INCLUDE into clean multilingual MCQA JSONL files."""

from pathlib import Path
import json
import os

import pandas as pd
from datasets import load_dataset


REPO_ROOT = Path("/scratch/workspace/emma/standard-project-m2-mystery-machine")
OUTPUT_DIR = REPO_ROOT / "multilingual" / "data" / "processed" / "include"

DATASET_NAME = "CohereLabs/include-base-44"
LANGUAGE_CONFIGS = {
    "italian": "Italian",
    "spanish": "Spanish",
    "russian": "Russian",
    "chinese": "Chinese",
    "hindi": "Hindi",
}
SPLITS = ["validation", "test"]
ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D", "0": "A", "1": "B", "2": "C", "3": "D"}
REGIONALITY_MAP = {
    "culture": "regional",
    "culture-related": "regional",
    "region explicit": "regional",
    "region-explicit": "regional",
    "region implicit": "regional_implicit",
    "region-implicit": "regional_implicit",
    "agnostic": "region_agnostic",
}
VALID_ANSWERS = {"A", "B", "C", "D"}
VALID_REGIONALITIES = {"regional", "regional_implicit", "region_agnostic"}


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def map_answer(x):
    if pd.isna(x):
        return ""
    if x in ANSWER_MAP:
        return ANSWER_MAP[x]
    x_str = str(x).strip()
    return ANSWER_MAP.get(x_str, x_str.upper())


def make_prompt(row):
    return (
        f"{clean_text(row['question'])}\n\n"
        f"A) {clean_text(row['A'])}\n"
        f"B) {clean_text(row['B'])}\n"
        f"C) {clean_text(row['C'])}\n"
        f"D) {clean_text(row['D'])}"
    )


def safe_id_part(x):
    return str(x).strip().replace(" ", "_").replace("/", "_")


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_raw_dataframe():
    frames = []
    hf_token = os.environ.get("HF_TOKEN")
    for language, config in LANGUAGE_CONFIGS.items():
        for split in SPLITS:
            ds = load_dataset(DATASET_NAME, config, split=split, token=hf_token)
            df_split = ds.to_pandas()
            df_split["language"] = language
            df_split["source_split"] = split
            df_split["source_idx"] = range(len(df_split))
            frames.append(df_split)
    return pd.concat(frames, ignore_index=True)


def clean_dataframe(df_raw):
    df = df_raw.copy()
    df["question"] = df["question"].apply(clean_text)
    df["option_a"] = df["option_a"].apply(clean_text)
    df["option_b"] = df["option_b"].apply(clean_text)
    df["option_c"] = df["option_c"].apply(clean_text)
    df["option_d"] = df["option_d"].apply(clean_text)
    df["answer"] = df["answer"].apply(map_answer)
    df["regional_feature"] = df["regional_feature"].apply(clean_text)
    df["regionality"] = df["regional_feature"].map(REGIONALITY_MAP)
    unknown_features = sorted(df.loc[df["regionality"].isna(), "regional_feature"].dropna().unique())
    assert not unknown_features, f"Unknown INCLUDE regional_feature values: {unknown_features}"
    df = df[
        df["question"].ne("")
        & df["option_a"].ne("")
        & df["option_b"].ne("")
        & df["option_c"].ne("")
        & df["option_d"].ne("")
    ].copy()
    df = df.rename(columns={"option_a": "A", "option_b": "B", "option_c": "C", "option_d": "D"})
    df = df[df["answer"].isin(VALID_ANSWERS)].copy()
    df = df.drop_duplicates(subset=["language", "question", "A", "B", "C", "D"]).reset_index(drop=True)
    return df


def make_record(row, idx):
    source_split = safe_id_part(row["source_split"])
    source_idx = int(row["source_idx"])
    language = row["language"]
    return {
        "id": f"include_{language}_all_{source_split}_{source_idx}",
        "dataset": "include",
        "language": language,
        "split": "all",
        "task_type": "mcq",
        "prompt": make_prompt(row),
        "answer": row["answer"],
        "metadata": {
            "country": clean_text(row.get("country", "")),
            "domain": clean_text(row.get("domain", "")),
            "subject": clean_text(row.get("subject", "")),
            "level": clean_text(row.get("level", "")),
            "regional_feature": clean_text(row.get("regional_feature", "")),
            "regionality": row["regionality"],
            "source_split": row["source_split"],
            "source_idx": source_idx,
        },
    }


def validate_record(record):
    assert record["dataset"] == "include"
    assert record["language"] in LANGUAGE_CONFIGS
    assert record["split"] == "all"
    assert record["task_type"] == "mcq"
    assert record["answer"] in VALID_ANSWERS
    assert record["metadata"]["regionality"] in VALID_REGIONALITIES
    assert "\n\nA) " in record["prompt"]
    assert "\nB) " in record["prompt"]
    assert "\nC) " in record["prompt"]
    assert "\nD) " in record["prompt"]
    assert record["prompt"].strip()
    assert isinstance(record["metadata"]["source_idx"], int)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_raw = load_raw_dataframe()
    df = clean_dataframe(df_raw)
    records = [make_record(row, idx) for idx, row in df.iterrows()]
    for record in records:
        validate_record(record)
    for language in LANGUAGE_CONFIGS:
        language_records = [record for record in records if record["language"] == language]
        ids = [record["id"] for record in language_records]
        assert len(ids) == len(set(ids)), f"Duplicate ids for {language}"
        write_jsonl(language_records, OUTPUT_DIR / f"include_{language}_all.jsonl")
        regional_records = [record for record in language_records if record["metadata"]["regionality"] == "regional"]
        regional_implicit_records = [record for record in language_records if record["metadata"]["regionality"] == "regional_implicit"]
        write_jsonl(regional_records, OUTPUT_DIR / f"include_{language}_regional.jsonl")
        write_jsonl(regional_implicit_records, OUTPUT_DIR / f"include_{language}_regional_implicit.jsonl")
    print(f"INCLUDE: wrote outputs for {len(LANGUAGE_CONFIGS)} languages to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
