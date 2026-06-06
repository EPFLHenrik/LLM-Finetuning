"""Preprocess Global MMLU into clean multilingual MCQA JSONL files."""

from pathlib import Path
import json
import os
import random

import numpy as np
import pandas as pd
from datasets import load_dataset


REPO_ROOT = Path("/scratch/workspace/emma/standard-project-m2-mystery-machine")
OUTPUT_DIR = REPO_ROOT / "multilingual" / "data" / "processed" / "global_mmlu"

DATASET_NAME = "CohereLabs/Global-MMLU"
LANGUAGE_CONFIGS = {
    "russian": "ru",
    "chinese": "zh",
    "italian": "it",
    "spanish": "es",
    "hindi": "hi",
}
SOURCE_SPLIT = "test"
RANDOM_SEED = 42
VALID_ANSWERS = {"A", "B", "C", "D"}
VALID_REGIONALITIES = {"regional", "regional_implicit", "region_agnostic"}
REGIONALITY_MAP = {
    "CS": "regional",
    "CC": "regional_implicit",
    "CA": "region_agnostic",
    "-": "region_agnostic",
    "": "region_agnostic",
}
VERY_HIGH_PRIORITY_SUBJECTS = {
    "professional_law",
    "international_law",
    "jurisprudence",
    "high_school_government_and_politics",
    "high_school_geography",
    "high_school_world_history",
    "high_school_us_history",
    "high_school_european_history",
    "prehistory",
    "world_religions",
    "sociology",
    "security_studies",
    "us_foreign_policy",
}
HIGH_PRIORITY_SUBJECTS = {
    "philosophy",
    "moral_scenarios",
    "moral_disputes",
    "business_ethics",
    "public_relations",
    "management",
    "marketing",
}
MEDIUM_PRIORITY_SUBJECTS = {
    "miscellaneous",
    "professional_psychology",
    "high_school_psychology",
    "human_aging",
    "human_sexuality",
    "nutrition",
    "clinical_knowledge",
    "professional_medicine",
    "college_medicine",
    "medical_genetics",
    "virology",
    "professional_accounting",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "econometrics",
    "logical_fallacies",
    "global_facts",
}
LOW_PRIORITY_SUBJECTS = {
    "elementary_mathematics",
    "high_school_mathematics",
    "high_school_statistics",
    "college_mathematics",
    "abstract_algebra",
    "formal_logic",
    "conceptual_physics",
    "high_school_physics",
    "college_physics",
    "high_school_chemistry",
    "college_chemistry",
    "high_school_biology",
    "college_biology",
    "anatomy",
    "astronomy",
    "electrical_engineering",
    "machine_learning",
    "computer_security",
    "college_computer_science",
    "high_school_computer_science",
}
LANGUAGE_SHUFFLE_SEEDS = {
    "russian": 42,
    "chinese": 43,
    "italian": 44,
    "spanish": 45,
    "hindi": 46,
}


def subject_keep_weight(subject):
    if subject in VERY_HIGH_PRIORITY_SUBJECTS:
        return 0.85
    if subject in HIGH_PRIORITY_SUBJECTS:
        return 0.65
    if subject in MEDIUM_PRIORITY_SUBJECTS:
        return 0.35
    if subject in LOW_PRIORITY_SUBJECTS:
        return 0.15
    return 0.25


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def make_prompt(row):
    return (
        f"{clean_text(row['question'])}\n\n"
        f"A) {clean_text(row['A'])}\n"
        f"B) {clean_text(row['B'])}\n"
        f"C) {clean_text(row['C'])}\n"
        f"D) {clean_text(row['D'])}"
    )


def safe_id_part(x):
    return clean_text(x).replace(" ", "_").replace("/", "_").replace("\\", "_")


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_raw_dataframe():
    frames = []
    hf_token = os.environ.get("HF_TOKEN")
    for language, config in LANGUAGE_CONFIGS.items():
        ds = load_dataset(DATASET_NAME, config, split=SOURCE_SPLIT, token=hf_token)
        df_lang = ds.to_pandas()
        df_lang["language"] = language
        df_lang["source_split"] = SOURCE_SPLIT
        df_lang["source_idx"] = range(len(df_lang))
        frames.append(df_lang)
    return pd.concat(frames, ignore_index=True)


def clean_dataframe(df_raw):
    df = df_raw.copy()
    df["question"] = df["question"].apply(clean_text)
    df["option_a"] = df["option_a"].apply(clean_text)
    df["option_b"] = df["option_b"].apply(clean_text)
    df["option_c"] = df["option_c"].apply(clean_text)
    df["option_d"] = df["option_d"].apply(clean_text)
    df["answer"] = df["answer"].apply(clean_text).str.upper()
    df["cultural_sensitivity_label"] = df["cultural_sensitivity_label"].apply(clean_text)
    df["regionality"] = df["cultural_sensitivity_label"].map(REGIONALITY_MAP)
    unknown_labels = sorted(df.loc[df["regionality"].isna(), "cultural_sensitivity_label"].dropna().unique())
    assert not unknown_labels, f"Unknown cultural_sensitivity_label values: {unknown_labels}"
    df = df[
        df["question"].ne("")
        & df["option_a"].ne("")
        & df["option_b"].ne("")
        & df["option_c"].ne("")
        & df["option_d"].ne("")
    ].copy()
    df = df[df["answer"].isin(VALID_ANSWERS)].copy()
    df = df.rename(columns={"option_a": "A", "option_b": "B", "option_c": "C", "option_d": "D"})
    df = df.drop_duplicates(subset=["language", "question", "A", "B", "C", "D"]).reset_index(drop=True)
    return df


def apply_subject_sampling(df_clean):
    df = df_clean.copy()
    df["subject_keep_weight"] = df["subject"].apply(subject_keep_weight)
    df.loc[df["regionality"] == "regional", "subject_keep_weight"] = 1.0
    rng = np.random.default_rng(RANDOM_SEED)
    df["sampling_random_value"] = rng.random(len(df))
    df["keep_by_subject_weight"] = df["sampling_random_value"] <= df["subject_keep_weight"]
    df_kept = df[df["keep_by_subject_weight"]].copy().reset_index(drop=True)
    df_discarded = df[~df["keep_by_subject_weight"]].copy().reset_index(drop=True)
    df_discarded_nonregional = df_discarded[df_discarded["regionality"] != "regional"].copy().reset_index(drop=True)
    return df_kept, df_discarded_nonregional


def make_record(row, idx):
    sample_id = safe_id_part(row.get("sample_id", ""))
    if sample_id == "":
        sample_id = f"{row['source_split']}_{int(row['source_idx'])}"
    return {
        "id": f"global_mmlu_{row['language']}_{sample_id}",
        "dataset": "global_mmlu",
        "language": row["language"],
        "split": "all",
        "task_type": "mcq",
        "prompt": make_prompt(row),
        "answer": row["answer"],
        "metadata": {
            "subject": clean_text(row.get("subject", "")),
            "subject_category": clean_text(row.get("subject_category", "")),
            "cultural_sensitivity_label": clean_text(row.get("cultural_sensitivity_label", "")),
            "regionality": row["regionality"],
            "region": clean_text(row.get("region", "")),
            "country": clean_text(row.get("country", "")),
            "sample_id": clean_text(row.get("sample_id", "")),
            "source_split": row["source_split"],
            "source_idx": int(row["source_idx"]),
            "subject_keep_weight": float(row["subject_keep_weight"]),
        },
    }


def validate_record(record):
    assert record["dataset"] == "global_mmlu"
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


def build_records(df_kept, df_discarded_nonregional):
    kept_records = [make_record(row, idx) for idx, row in df_kept.iterrows()]
    discarded_nonregional_records = [make_record(row, idx) for idx, row in df_discarded_nonregional.iterrows()]
    records_by_id = {}
    for record in kept_records:
        records_by_id[record["id"]] = record
    for record in discarded_nonregional_records:
        if record["id"] in records_by_id:
            raise ValueError(f"Duplicate recovered id: {record['id']}")
        records_by_id[record["id"]] = record
    records = list(records_by_id.values())
    for record in records:
        validate_record(record)
    return records


def save_outputs(records):
    for language in LANGUAGE_CONFIGS:
        language_records = [record for record in records if record["language"] == language]
        ids = [record["id"] for record in language_records]
        assert len(ids) == len(set(ids)), f"Duplicate ids for {language}"
        regional_records = [record for record in language_records if record["metadata"]["regionality"] == "regional"]
        write_jsonl(regional_records, OUTPUT_DIR / f"global_mmlu_{language}_regional.jsonl")
        write_jsonl(language_records, OUTPUT_DIR / f"global_mmlu_{language}_all.jsonl")
    for language in LANGUAGE_CONFIGS:
        all_path = OUTPUT_DIR / f"global_mmlu_{language}_all.jsonl"
        nonregional_path = OUTPUT_DIR / f"global_mmlu_{language}_shuffled_region_agnostic.jsonl"
        all_records = read_jsonl(all_path)
        nonregional_records = [record for record in all_records if record["metadata"]["regionality"] != "regional"]
        ids = [record["id"] for record in nonregional_records]
        assert len(ids) == len(set(ids)), f"Duplicate ids in {nonregional_path}"
        rng = random.Random(LANGUAGE_SHUFFLE_SEEDS[language])
        rng.shuffle(nonregional_records)
        write_jsonl(nonregional_records, nonregional_path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_raw = load_raw_dataframe()
    df_clean = clean_dataframe(df_raw)
    df_kept, df_discarded_nonregional = apply_subject_sampling(df_clean)
    records = build_records(df_kept, df_discarded_nonregional)
    save_outputs(records)
    print(f"Global MMLU: wrote outputs for {len(LANGUAGE_CONFIGS)} languages to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
